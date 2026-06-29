import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn.dense import dense_mincut_pool


def get_spatio_temporal_edges(T=8, N=256, K=8):
    """
    Precompute ONCE (static for all videos):
      - Spatial KNN edges (undirected, within each frame)
      - Temporal exact-match edges (bidirectional: same patch ↔ same patch in adjacent frame)
    """
    # ====================== SPATIAL KNN EDGES ======================
    grid_size = int(N ** 0.5)
    if grid_size * grid_size != N:
        raise ValueError(f"Patch count {N} must be perfect square")

    indices = torch.arange(N)
    rows = indices // grid_size
    cols = indices % grid_size
    positions = torch.stack([rows, cols], dim=1).float()

    diff = positions.unsqueeze(1) - positions.unsqueeze(0)
    dist_matrix = torch.sqrt((diff ** 2).sum(dim=-1))
    dist_matrix.fill_diagonal_(float("inf"))
    _, knn_idxs = torch.topk(dist_matrix, k=K, dim=-1, largest=False)

    src_global, dst_global = [], []
    for i in range(T):
        row_idx = torch.arange(N).unsqueeze(1).expand(-1, K)
        adj = torch.zeros((N, N), dtype=torch.bool, device=knn_idxs.device)
        adj[row_idx, knn_idxs] = True
        adj = adj | adj.t()
        src, dst = torch.where(adj)
        offset = i * N
        src_global.append(src + offset)
        dst_global.append(dst + offset)

    spatial_src = torch.cat(src_global)
    spatial_dst = torch.cat(dst_global)

    # ====================== TEMPORAL EXACT-MATCH EDGES ======================
    # Bidirectional: frame_t ↔ frame_{t+1} for same patch index
    temporal_src, temporal_dst = [], []
    for t in range(T - 1):
        patch_idx = torch.arange(N)
        offset_curr = t * N
        offset_next = (t + 1) * N

        # Forward:  frame_t → frame_{t+1}
        temporal_src.append(patch_idx + offset_curr)
        temporal_dst.append(patch_idx + offset_next)

        # Backward: frame_{t+1} → frame_t
        temporal_src.append(patch_idx + offset_next)
        temporal_dst.append(patch_idx + offset_curr)

    temporal_src = torch.cat(temporal_src)
    temporal_dst = torch.cat(temporal_dst)

    # ====================== COMBINE BOTH ======================
    src = torch.cat([spatial_src, temporal_src])
    dst = torch.cat([spatial_dst, temporal_dst])
    return src, dst


class VideoFeatureExtractor(nn.Module):
    def __init__(self, vit_name='dinov2_vits14', feature_dim=384, weight_path=None):
        super().__init__()

        # Weight path is mandatory
        if weight_path is None:
            raise ValueError(
                "weight_path must be provided. "
                "Pass the path to your fine_tune.py checkpoint (best_vit_checkpoint.pth)."
            )

        self.vit = torch.hub.load('facebookresearch/dinov2', vit_name)

        # ── Load weights from fine_tune.py checkpoint ──────────────────────
        ckpt = torch.load(weight_path, map_location='cpu', weights_only=False)

        if 'best_auc' in ckpt:
            print(f'best auc of vit is: {ckpt['best_auc']}')
        if 'epoch' in ckpt:
            print(f'best auc epoch is: {ckpt['epoch']}')

        # fine_tune.py saves under "model_state_dict" (full DeepfakeDetector)
        # or optionally "vit_state_dict" (bare ViT, no prefix) if you used
        # the improved save block. Handle both transparently.
        if "vit_state_dict" in ckpt:
            # Already stripped of "vit." prefix — load directly
            vit_state_dict = ckpt["vit_state_dict"]
        else:
            state_dict = ckpt.get("model_state_dict", ckpt)
            # Strip the "vit." prefix added by DeepfakeDetector
            vit_state_dict = {
                k.replace("vit.", "", 1): v
                for k, v in state_dict.items()
                if k.startswith("vit.")
            }

        missing, unexpected = self.vit.load_state_dict(vit_state_dict, strict=False)
        if missing:
            print(f"[VideoFeatureExtractor] Missing keys : {missing}")
        if unexpected:
            print(f"[VideoFeatureExtractor] Unexpected keys: {unexpected}")

        epoch = ckpt.get("epoch", "?")
        best_auc = ckpt.get("best_auc", "?")
        print(f"[VideoFeatureExtractor] Loaded checkpoint — epoch={epoch}, best_auc={best_auc}")

        # ── Freeze ALL parameters (feature extractor, no updates) ──────────
        for p in self.vit.parameters():
            p.requires_grad = False

        # ── Eval mode — disables dropout / BN running stats updates ────────
        self.vit.eval()

        self.D = feature_dim

    def forward(self, x: torch.Tensor):
        # Ensure eval mode is enforced even if someone calls model.train() upstream
        self.vit.eval()

        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)

        with torch.no_grad():
            features = self.vit.forward_features(x_flat)

        cls_tokens = features['x_norm_clstoken']
        patch_tokens = features['x_norm_patchtokens']

        return {
            'cls': cls_tokens.reshape(B, T, self.D),
            'patch': patch_tokens.reshape(B, T * 256, self.D)
        }

    def train(self, mode: bool = True):
        # Prevent any external .train() call from affecting the frozen ViT
        super().train(mode)
        self.vit.eval()   # always keep ViT in eval regardless
        return self


class FusedModel(nn.Module):
    """
    Final Fused Model with static spatio-temporal edges.
    - Temporal directed edges (exact same-patch connections) now provide clear positional/temporal information to GAT.
    - Removed learnable graph_pos_embed (after MinCut pooling) from graph path as requested.
    - Kept BiLSTM temporal_pos_embed (CLS path) unchanged.
    """
    def __init__(
        self,
        vit_name: str = 'dinov2_vits14',
        feature_dim: int = 384,
        dropout: float = 0.2,
        num_of_frames: int = 8,
        num_gcn_layers: int = 2,
        num_clusters: int = 512,
        num_transformer_blocks: int = 1,
        num_heads: int = 8,
        mlp_dim: int = 512,
        vit_weight_path=None,
    ):
        super().__init__()
        self.num_of_frames = num_of_frames
        self.feature_dim = feature_dim
        self.num_clusters = num_clusters

        self.vit = VideoFeatureExtractor(vit_name=vit_name, weight_path=vit_weight_path)

        # ====================== BiLSTM Path (CLS tokens) ======================
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_of_frames, feature_dim))
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

        self.bilstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=feature_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0
        )

        # ====================== Graph Path ======================
        # Precompute static spatio-temporal edges ONCE
        src, dst = get_spatio_temporal_edges(T=num_of_frames, N=256, K=8)
        self.register_buffer('edge_index', torch.stack([src, dst], dim=0))   # [2, total_edges]

        self.gcn_layers = nn.ModuleList()
        head_dim = feature_dim // num_heads
        for _ in range(num_gcn_layers):
            self.gcn_layers.append(
                GATv2Conv(
                    in_channels=feature_dim,
                    out_channels=head_dim,
                    heads=num_heads,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=True,
                )
            )

        self.assign_net = nn.Linear(feature_dim, num_clusters)

        # Graph-level CLS token (still kept)
        self.class_token_graph = nn.Parameter(torch.zeros(1, 1, feature_dim))

        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=num_heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ) for _ in range(num_transformer_blocks)
        ])

        # ASIF-style fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        feats = self.vit(x)
        cls_features = feats['cls']
        patch_features = feats['patch']                     # [B, T*256, D]

        # ====================== BiLSTM Path ======================
        cls_features = cls_features + self.temporal_pos_embed
        lstm_out, _ = self.bilstm(cls_features)
        bilstm_feat = lstm_out.mean(dim=1)

        # ====================== Graph Path ======================
        data_list = []
        for i in range(B):
            data = Data(x=patch_features[i], edge_index=self.edge_index)
            data_list.append(data)

        batched = Batch.from_data_list(data_list)
        x_g = batched.x
        edge_index = batched.edge_index
        batch_idx = batched.batch

        for gcn in self.gcn_layers:
            x_g = gcn(x_g, edge_index)
            x_g = F.relu(x_g)

        # Reshape back to [B, T*256, D]
        x_g = torch.split(x_g, split_size_or_sections=patch_features.shape[1])
        x_g = torch.stack(x_g, dim=0)

        # MinCut Pooling
        s = self.assign_net(x_g)
        adj = to_dense_adj(edge_index, batch=batch_idx)
        x_pooled, _, mincut_loss, ortho_loss = dense_mincut_pool(x_g, adj, s)

        # Graph-level CLS token (no positional embedding anymore)
        cls_token = self.class_token_graph.expand(B, 1, -1)
        x_g = torch.cat([cls_token, x_pooled], dim=1)

        for block in self.transformer_blocks:
            x_g = block(x_g)

        graph_feat = x_g[:, 0]

        # ====================== Fusion ======================
        combined = torch.cat([bilstm_feat, graph_feat], dim=1)
        logits = self.fusion_layer(combined)

        return logits, mincut_loss, ortho_loss