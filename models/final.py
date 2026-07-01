import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv


def get_spatio_temporal_edges(T=8, N=256, K=8):
    """
    Precompute ONCE (static for all videos) the GRAPH STRUCTURE (which nodes connect):
      - Spatial KNN edges (undirected, within each frame)
      - Temporal exact-match edges (bidirectional: same patch ↔ same patch in adjacent frame)

    NOTE: this only returns the static src/dst index pairs. The actual edge
    ATTRIBUTE (cosine similarity between the connected patch features) is
    computed dynamically inside the model's forward pass, since it depends
    on the input video's features, not just the fixed topology.
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

        if weight_path is None:
            raise ValueError(
                "weight_path must be provided. "
                "Pass the path to your fine_tune.py checkpoint (best_vit_checkpoint.pth)."
            )

        self.vit = torch.hub.load('facebookresearch/dinov2', vit_name)

        ckpt = torch.load(weight_path, map_location='cpu', weights_only=False)

        if 'best_auc' in ckpt:
            print(f"best auc of vit is: {ckpt['best_auc']}")
        if 'epoch' in ckpt:
            print(f"best auc epoch is: {ckpt['epoch']}")

        if "vit_state_dict" in ckpt:
            vit_state_dict = ckpt["vit_state_dict"]
        else:
            state_dict = ckpt.get("model_state_dict", ckpt)
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

        for p in self.vit.parameters():
            p.requires_grad = False

        self.vit.eval()
        self.D = feature_dim

    def forward(self, x: torch.Tensor):
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
        super().train(mode)
        self.vit.eval()
        return self


class FusedModel(nn.Module):
    """
    Fused Model with static spatio-temporal edge STRUCTURE and
    dynamic, feature-dependent edge ATTRIBUTES.

    Graph path: DINOv2 patch features → GATv2Conv layers (with cosine-
    similarity edge attributes) → per-frame mean pool [B, T, D] → prepend
    graph CLS token → transformer encoder blocks → graph_feat.

    MinCut pooling has been removed entirely. The per-frame mean pool is a
    lossless compression of the 256 spatial patches per frame, retaining all
    temporal granularity and costing zero learnable parameters. The
    transformer then attends across T frame-level summaries rather than K
    learned clusters, so the model is both simpler and more interpretable.
    """
    def __init__(
        self,
        vit_name: str = 'dinov2_vits14',
        feature_dim: int = 384,
        dropout: float = 0.2,
        num_of_frames: int = 8,
        num_gcn_layers: int = 2,
        num_transformer_blocks: int = 1,
        num_heads: int = 8,
        mlp_dim: int = 512,
        vit_weight_path=None,
    ):
        super().__init__()
        self.num_of_frames = num_of_frames
        self.feature_dim = feature_dim
        self.N_patches = 256  # DINOv2-vits14 produces 16x16 = 256 patches for 224x224 input

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
        # Precompute static spatio-temporal edge STRUCTURE once.
        src, dst = get_spatio_temporal_edges(T=num_of_frames, N=self.N_patches, K=8)
        self.register_buffer('edge_index', torch.stack([src, dst], dim=0))  # [2, E]

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
                    edge_dim=1,  # scalar cosine-similarity edge attribute
                )
            )

        # Graph-level CLS token
        self.class_token_graph = nn.Parameter(torch.zeros(1, 1, feature_dim))

        # Positional embedding over [graph-CLS, frame_1 ... frame_T].
        # After per-frame mean pooling we have T frame tokens + 1 CLS token.
        self.graph_pos_embed = nn.Parameter(torch.zeros(1, num_of_frames + 1, feature_dim))
        nn.init.trunc_normal_(self.graph_pos_embed, std=0.02)

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
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

    @staticmethod
    def _cosine_edge_attr(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between connected node features for every
        edge in edge_index. Returns shape [E, 1].
        """
        src, dst = edge_index
        x_norm = F.normalize(x, p=2, dim=-1, eps=1e-8)
        sim = (x_norm[src] * x_norm[dst]).sum(dim=-1, keepdim=True)  # [E, 1]
        return sim

    def _encode_branches(self, x: torch.Tensor):
        B = x.shape[0]
        feats = self.vit(x)
        cls_features = feats['cls']    # [B, T, D]
        patch_features = feats['patch']  # [B, T*256, D]

        # ====================== BiLSTM Path ======================
        cls_features = cls_features + self.temporal_pos_embed
        lstm_out, _ = self.bilstm(cls_features)
        bilstm_feat = lstm_out.mean(dim=1)   # [B, D]

        # ====================== Graph Path ======================
        # Build one PyG Data object per sample in the batch.
        data_list = []
        for i in range(B):
            data = Data(x=patch_features[i], edge_index=self.edge_index)
            data_list.append(data)

        batched = Batch.from_data_list(data_list)
        x_g = batched.x           # [B*T*256, D]
        edge_index = batched.edge_index

        # Dynamic edge attributes recomputed after every GCN layer.
        edge_attr = self._cosine_edge_attr(x_g, edge_index)

        for gcn in self.gcn_layers:
            x_g = gcn(x_g, edge_index, edge_attr=edge_attr)
            x_g = F.gelu(x_g)
            edge_attr = self._cosine_edge_attr(x_g, edge_index)

        # ── Per-frame mean pooling (replaces MinCut pooling) ──────────────
        # Reshape [B*T*256, D] → [B, T, 256, D] → mean over N=256 patches
        # → [B, T, D].  No learnable parameters, no adjacency materialisation.
        x_g = x_g.view(B, self.num_of_frames, self.N_patches, self.feature_dim)
        x_g = x_g.mean(dim=2)  # [B, T, D]

        # Prepend graph CLS token → [B, T+1, D], add positional embedding.
        cls_token = self.class_token_graph.expand(B, 1, -1)
        x_g = torch.cat([cls_token, x_g], dim=1)   # [B, T+1, D]
        x_g = x_g + self.graph_pos_embed

        for block in self.transformer_blocks:
            x_g = block(x_g)

        graph_feat = x_g[:, 0]   # [B, D]  — graph CLS output
        return bilstm_feat, graph_feat

    def _fuse(self, bilstm_feat: torch.Tensor, graph_feat: torch.Tensor):
        combined = torch.cat([bilstm_feat, graph_feat], dim=1)
        return self.fusion_layer(combined)

    def forward(
        self,
        x: torch.Tensor,
        branch_ablation: str | None = None,
        return_branch_logits: bool = False,
    ):
        bilstm_feat, graph_feat = self._encode_branches(x)

        if branch_ablation == 'cls':
            bilstm_feat = torch.zeros_like(bilstm_feat)
        elif branch_ablation == 'graph':
            graph_feat = torch.zeros_like(graph_feat)

        logits = self._fuse(bilstm_feat, graph_feat)

        if return_branch_logits:
            cls_only_logits   = self._fuse(bilstm_feat, torch.zeros_like(graph_feat))
            graph_only_logits = self._fuse(torch.zeros_like(bilstm_feat), graph_feat)
            return logits, {
                'bilstm_feat': bilstm_feat,
                'graph_feat': graph_feat,
                'cls_only_logits': cls_only_logits,
                'graph_only_logits': graph_only_logits,
            }

        return logits