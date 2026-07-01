import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn.dense import dense_mincut_pool


def get_spatio_temporal_edges(T=8, N=256, K=8):
    """
    Precompute ONCE (static for all videos) the GRAPH STRUCTURE (which nodes connect):
      - Spatial KNN edges (undirected, within each frame)
      - Temporal exact-match edges (bidirectional: same patch ↔ same patch in adjacent frame)
    """
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

    temporal_src, temporal_dst = [], []
    for t in range(T - 1):
        patch_idx = torch.arange(N)
        offset_curr = t * N
        offset_next = (t + 1) * N

        temporal_src.append(patch_idx + offset_curr)
        temporal_dst.append(patch_idx + offset_next)

        temporal_src.append(patch_idx + offset_next)
        temporal_dst.append(patch_idx + offset_curr)

    temporal_src = torch.cat(temporal_src)
    temporal_dst = torch.cat(temporal_dst)

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
    Single shared learnable CLS token drives BOTH branches, so fusion is
    implicit rather than via concat/prepend-then-attend-separately:

    Stage 1 (temporal): the CLS token is the QUERY into a multihead
    attention over the T per-frame DINOv2 CLS tokens (K/V). This replaces
    the BiLSTM entirely. Output: an updated CLS vector carrying a
    temporal summary of the video.

    Stage 2 (spatial/graph): patch tokens go through GATv2Conv layers
    (dynamic cosine-similarity edge attributes, recomputed each layer),
    then dense MinCut pooling (like the old model) compresses
    T*256 nodes down to num_clusters soft-cluster nodes.

    Stage 3 (fusion): the SAME CLS vector from Stage 1 is prepended to
    the pooled cluster nodes and the sequence is run through transformer
    encoder blocks. Self-attention lets the CLS token (already carrying
    temporal info) mix with the pooled spatial/graph info. The final
    decision is read off that same CLS token's output.
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
        self.N_patches = 256

        self.vit = VideoFeatureExtractor(vit_name=vit_name, weight_path=vit_weight_path)

        # ====================== Single shared learnable CLS token ======================
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ====================== Stage 1: temporal MHA (replaces BiLSTM) ======================
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_of_frames, feature_dim))
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_norm = nn.LayerNorm(feature_dim)

        # ====================== Stage 2: Graph path (GAT + MinCut pool) ======================
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

        self.assign_net = nn.Linear(feature_dim, num_clusters)

        # ====================== Stage 3: fusion transformer ======================
        # Sequence = [cls_token, cluster_1 ... cluster_{num_clusters}]
        seq_len = num_clusters + 1
        self.fusion_pos_embed = nn.Parameter(torch.zeros(1, seq_len, feature_dim))
        nn.init.trunc_normal_(self.fusion_pos_embed, std=0.02)

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

        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim, 384),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(384, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

    @staticmethod
    def _cosine_edge_attr(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        x_norm = F.normalize(x, p=2, dim=-1, eps=1e-8)
        sim = (x_norm[src] * x_norm[dst]).sum(dim=-1, keepdim=True)  # [E, 1]
        return sim

    def _encode_fused(self, x: torch.Tensor, branch_ablation: str | None = None):
        B = x.shape[0]
        feats = self.vit(x)
        cls_features = feats['cls']       # [B, T, D]
        patch_features = feats['patch']   # [B, T*256, D]

        # ====================== Stage 1: temporal MHA ======================
        cls_features = cls_features + self.temporal_pos_embed  # [B, T, D]

        if branch_ablation == 'cls':
            cls_features = torch.zeros_like(cls_features)

        cls_q = self.cls_token.expand(B, -1, -1)   # [B, 1, D]
        temporal_out, _ = self.temporal_attn(
            query=cls_q, key=cls_features, value=cls_features
        )                                            # [B, 1, D]
        cls_vec = self.temporal_norm(cls_q + temporal_out)  # [B, 1, D] residual + LN

        # ====================== Stage 2: Graph path ======================
        data_list = []
        for i in range(B):
            data = Data(x=patch_features[i], edge_index=self.edge_index)
            data_list.append(data)

        batched = Batch.from_data_list(data_list)
        x_g = batched.x            # [B*T*256, D]
        edge_index = batched.edge_index
        batch_idx = batched.batch

        edge_attr = self._cosine_edge_attr(x_g, edge_index)
        for gcn in self.gcn_layers:
            x_g = gcn(x_g, edge_index, edge_attr=edge_attr)
            x_g = F.gelu(x_g)
            edge_attr = self._cosine_edge_attr(x_g, edge_index)

        # [B*T*256, D] -> [B, T*256, D]
        x_g = torch.split(x_g, split_size_or_sections=patch_features.shape[1])
        x_g = torch.stack(x_g, dim=0)

        # MinCut pooling: T*256 nodes -> num_clusters nodes
        s = self.assign_net(x_g)
        adj = to_dense_adj(edge_index, batch=batch_idx)
        x_pooled, _, mincut_loss, ortho_loss = dense_mincut_pool(x_g, adj, s)  # [B, num_clusters, D]

        if branch_ablation == 'graph':
            x_pooled = torch.zeros_like(x_pooled)

        # ====================== Stage 3: fusion via shared CLS token ======================
        x_fused = torch.cat([cls_vec, x_pooled], dim=1)  # [B, num_clusters+1, D]
        x_fused = x_fused + self.fusion_pos_embed

        for block in self.transformer_blocks:
            x_fused = block(x_fused)

        fused_feat = x_fused[:, 0]   # [B, D] — fused CLS output

        return fused_feat, cls_vec.squeeze(1), x_pooled.mean(dim=1), mincut_loss, ortho_loss

    def forward(
        self,
        x: torch.Tensor,
        branch_ablation: str | None = None,
        return_branch_logits: bool = False,
    ):
        fused_feat, cls_vec, graph_feat_summary, mincut_loss, ortho_loss = \
            self._encode_fused(x, branch_ablation=branch_ablation)

        logits = self.fusion_layer(fused_feat)

        if return_branch_logits:
            # Diagnostic-only: re-run with one branch ablated. Expensive
            # (re-runs GAT + MinCut + transformer), same caveat as before.
            cls_only_feat, _, _, _, _ = self._encode_fused(x, branch_ablation='graph')
            graph_only_feat, _, _, _, _ = self._encode_fused(x, branch_ablation='cls')
            cls_only_logits = self.fusion_layer(cls_only_feat)
            graph_only_logits = self.fusion_layer(graph_only_feat)
            return logits, mincut_loss, ortho_loss, {
                'cls_feat': cls_vec,
                'graph_feat': graph_feat_summary,
                'cls_only_logits': cls_only_logits,
                'graph_only_logits': graph_only_logits,
            }

        return logits, mincut_loss, ortho_loss