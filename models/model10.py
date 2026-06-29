import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


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
            print(f"best auc of vit is: {ckpt['best_auc']}")
        if 'epoch' in ckpt:
            print(f"best auc epoch is: {ckpt['epoch']}")

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
        tau_s: float = 0.6,
        tau_t: float = 0.6,
        graph_eps: float = 1e-4,
        local_window: int = 3,
        spectral_hidden_dim: int = 64,
        vit_weight_path=None,
    ):
        super().__init__()
        self.num_of_frames = num_of_frames
        self.feature_dim = feature_dim
        self.num_clusters = num_clusters
        self.tau_s = tau_s
        self.tau_t = tau_t
        self.graph_eps = graph_eps
        self.local_window = local_window

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
        self.consistency_gnn_layers = nn.ModuleList()
        self.inconsistency_gnn_layers = nn.ModuleList()
        head_dim = feature_dim // num_heads
        for _ in range(num_gcn_layers):
            self.consistency_gnn_layers.append(
                GATv2Conv(
                    in_channels=feature_dim,
                    out_channels=head_dim,
                    heads=num_heads,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=True,
                )
            )
            self.inconsistency_gnn_layers.append(
                GATv2Conv(
                    in_channels=feature_dim,
                    out_channels=head_dim,
                    heads=num_heads,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=True,
                )
            )

        self.spatial_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
        )

        self.spectral_filter_mlp = nn.Sequential(
            nn.Linear(1, spectral_hidden_dim),
            nn.GELU(),
            nn.Linear(spectral_hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.graph_projector = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
        )

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

    def _local_negative_edges(self, t: int, num_patches: int, grid_size: int, device: torch.device):
        half = self.local_window // 2
        src, dst = [], []
        offset = t * num_patches
        for idx in range(num_patches):
            r = idx // grid_size
            c = idx % grid_size
            for rr in range(max(0, r - half), min(grid_size, r + half + 1)):
                for cc in range(max(0, c - half), min(grid_size, c + half + 1)):
                    nbr = rr * grid_size + cc
                    if nbr == idx:
                        continue
                    src.append(offset + idx)
                    dst.append(offset + nbr)
                    src.append(offset + nbr)
                    dst.append(offset + idx)

        if not src:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        return torch.tensor([src, dst], dtype=torch.long, device=device)

    def _build_graphs_for_sample(self, patch_tokens: torch.Tensor):
        # patch_tokens: [T, N, D]
        T, N, _ = patch_tokens.shape
        device = patch_tokens.device
        grid_size = int(N ** 0.5)
        if grid_size * grid_size != N:
            raise ValueError(f"Patch count {N} must be a perfect square")

        x_norm = patch_tokens / (patch_tokens.norm(dim=-1, keepdim=True) + self.graph_eps)

        cons_src, cons_dst = [], []
        incon_src, incon_dst = [], []
        spatial_adjs = []

        # Spatial consistency graph A^(t)
        for t in range(T):
            sim = torch.matmul(x_norm[t], x_norm[t].transpose(0, 1))
            sim.fill_diagonal_(0.0)
            spatial_adjs.append(sim)

            edge_mask = sim >= self.tau_s
            src, dst = torch.where(edge_mask)
            if src.numel() > 0:
                cons_src.append(src + t * N)
                cons_dst.append(dst + t * N)

            neg_edges = self._local_negative_edges(t=t, num_patches=N, grid_size=grid_size, device=device)
            if neg_edges.numel() > 0:
                incon_src.append(neg_edges[0])
                incon_dst.append(neg_edges[1])

        # Temporal consistency + inconsistency edges between adjacent frames
        for t in range(T - 1):
            idx = torch.arange(N, device=device)
            offset_curr = t * N
            offset_next = (t + 1) * N

            a_curr = spatial_adjs[t]
            a_next = spatial_adjs[t + 1]
            struct_sim = F.cosine_similarity(a_curr, a_next, dim=-1)
            feat_sim = F.cosine_similarity(x_norm[t], x_norm[t + 1], dim=-1)
            score = 0.5 * (struct_sim + feat_sim)

            keep = score >= self.tau_t
            if keep.any():
                keep_idx = idx[keep]
                cons_src.append(keep_idx + offset_curr)
                cons_dst.append(keep_idx + offset_next)
                cons_src.append(keep_idx + offset_next)
                cons_dst.append(keep_idx + offset_curr)

            # Temporal inconsistency graph A_bar (paper-style negative links)
            incon_src.append(idx + offset_curr)
            incon_dst.append(idx + offset_next)
            incon_src.append(idx + offset_next)
            incon_dst.append(idx + offset_curr)

        if cons_src:
            edge_index_cons = torch.stack([torch.cat(cons_src), torch.cat(cons_dst)], dim=0)
        else:
            node_idx = torch.arange(T * N, device=device)
            edge_index_cons = torch.stack([node_idx, node_idx], dim=0)

        if incon_src:
            edge_index_incon = torch.stack([torch.cat(incon_src), torch.cat(incon_dst)], dim=0)
        else:
            node_idx = torch.arange(T * N, device=device)
            edge_index_incon = torch.stack([node_idx, node_idx], dim=0)

        return edge_index_cons, edge_index_incon

    def _spectral_pool(self, x_nodes: torch.Tensor, edge_index_cons: torch.Tensor):
        # x_nodes: [TN, D]
        num_nodes = x_nodes.shape[0]
        device = x_nodes.device

        adj = torch.zeros((num_nodes, num_nodes), device=device, dtype=x_nodes.dtype)
        adj[edge_index_cons[0], edge_index_cons[1]] = 1.0
        adj = 0.5 * (adj + adj.transpose(0, 1))

        deg = adj.sum(dim=1)
        inv_sqrt_deg = torch.pow(deg + self.graph_eps, -0.5)
        lap = torch.eye(num_nodes, device=device, dtype=x_nodes.dtype) - (
            inv_sqrt_deg.unsqueeze(1) * adj * inv_sqrt_deg.unsqueeze(0)
        )

        try:
            evals, evecs = torch.linalg.eigh(lap)
            phi = self.spectral_filter_mlp(evals.unsqueeze(-1)).squeeze(-1)
            filtered = evecs @ (phi.unsqueeze(-1) * (evecs.transpose(0, 1) @ x_nodes))
            return filtered.mean(dim=0)
        except RuntimeError:
            # Fallback for rare decomposition instability
            return x_nodes.mean(dim=0)

    def _encode_graph_sample(self, patch_tokens_flat: torch.Tensor):
        T = self.num_of_frames
        TN, _ = patch_tokens_flat.shape
        N = TN // T
        if N * T != TN:
            raise ValueError("Patch token count is not divisible by num_of_frames")

        patch_tokens = patch_tokens_flat.view(T, N, self.feature_dim)
        edge_index_cons, edge_index_incon = self._build_graphs_for_sample(patch_tokens)

        x_nodes = patch_tokens_flat
        x_cons = x_nodes
        x_incon = x_nodes

        for layer_cons, layer_incon in zip(self.consistency_gnn_layers, self.inconsistency_gnn_layers):
            x_cons = F.relu(layer_cons(x_cons, edge_index_cons))
            x_incon = F.relu(layer_incon(x_incon, edge_index_incon))

        z_spatial_nodes = self.spatial_mlp(torch.cat([x_cons, x_incon], dim=-1))
        z_spatial = z_spatial_nodes.mean(dim=0)
        z_spectral = self._spectral_pool(x_nodes, edge_index_cons)

        z_graph = torch.cat([z_spatial, z_spectral], dim=-1)
        graph_feat = self.graph_projector(z_graph)
        return graph_feat

    def _encode_branches(self, x: torch.Tensor):
        B = x.shape[0]
        feats = self.vit(x)
        cls_features = feats['cls']
        patch_features = feats['patch']                     # [B, T*256, D]

        # ====================== BiLSTM Path ======================
        cls_features = cls_features + self.temporal_pos_embed
        lstm_out, _ = self.bilstm(cls_features)
        bilstm_feat = lstm_out.mean(dim=1)

        # ====================== Graph Path ======================
        graph_feat = []
        for i in range(B):
            graph_feat.append(self._encode_graph_sample(patch_features[i]))
        graph_feat = torch.stack(graph_feat, dim=0)

        # Keep interface identical to existing train/test code.
        mincut_loss = torch.zeros((), device=x.device)
        ortho_loss = torch.zeros((), device=x.device)
        return bilstm_feat, graph_feat, mincut_loss, ortho_loss

    def _fuse(self, bilstm_feat: torch.Tensor, graph_feat: torch.Tensor):
        combined = torch.cat([bilstm_feat, graph_feat], dim=1)
        return self.fusion_layer(combined)

    def forward(
        self,
        x: torch.Tensor,
        branch_ablation: str | None = None,
        return_branch_logits: bool = False,
    ):
        bilstm_feat, graph_feat, mincut_loss, ortho_loss = self._encode_branches(x)

        if branch_ablation == 'cls':
            bilstm_feat = torch.zeros_like(bilstm_feat)
        elif branch_ablation == 'graph':
            graph_feat = torch.zeros_like(graph_feat)

        # ====================== Fusion ======================
        logits = self._fuse(bilstm_feat, graph_feat)

        if return_branch_logits:
            cls_only_logits = self._fuse(bilstm_feat, torch.zeros_like(graph_feat))
            graph_only_logits = self._fuse(torch.zeros_like(bilstm_feat), graph_feat)
            return logits, mincut_loss, ortho_loss, {
                'bilstm_feat': bilstm_feat,
                'graph_feat': graph_feat,
                'cls_only_logits': cls_only_logits,
                'graph_only_logits': graph_only_logits,
            }

        return logits, mincut_loss, ortho_loss