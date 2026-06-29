import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


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

        if 'vit_state_dict' in ckpt:
            vit_state_dict = ckpt['vit_state_dict']
        else:
            state_dict = ckpt.get('model_state_dict', ckpt)
            vit_state_dict = {
                k.replace('vit.', '', 1): v
                for k, v in state_dict.items()
                if k.startswith('vit.')
            }

        missing, unexpected = self.vit.load_state_dict(vit_state_dict, strict=False)
        if missing:
            print(f"[VideoFeatureExtractor] Missing keys : {missing}")
        if unexpected:
            print(f"[VideoFeatureExtractor] Unexpected keys: {unexpected}")

        epoch = ckpt.get('epoch', '?')
        best_auc = ckpt.get('best_auc', '?')
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

        patch_tokens = features['x_norm_patchtokens']
        return {
            'patch': patch_tokens.reshape(B, T * 256, self.D)
        }

    def train(self, mode: bool = True):
        super().train(mode)
        self.vit.eval()
        return self


class FusedModel(nn.Module):
    """
    Graph-only model variant.
    Uses patch-token graph construction and GAT/spectral encoding only.
    """
    def __init__(
        self,
        vit_name: str = 'dinov2_vits14',
        feature_dim: int = 384,
        dropout: float = 0.2,
        num_of_frames: int = 8,
        num_gcn_layers: int = 3,
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

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
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
        T, N, _ = patch_tokens.shape
        device = patch_tokens.device
        grid_size = int(N ** 0.5)
        if grid_size * grid_size != N:
            raise ValueError(f"Patch count {N} must be a perfect square")

        x_norm = patch_tokens / (patch_tokens.norm(dim=-1, keepdim=True) + self.graph_eps)

        cons_src, cons_dst = [], []
        incon_src, incon_dst = [], []
        spatial_adjs = []

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
        return self.graph_projector(z_graph)

    def _encode_branches(self, x: torch.Tensor):
        B = x.shape[0]
        feats = self.vit(x)
        patch_features = feats['patch']

        graph_feat = []
        for i in range(B):
            graph_feat.append(self._encode_graph_sample(patch_features[i]))
        graph_feat = torch.stack(graph_feat, dim=0)

        zero_loss = torch.zeros((), device=x.device)
        return graph_feat, zero_loss, zero_loss

    def forward(
        self,
        x: torch.Tensor,
        branch_ablation: str | None = None,
        return_branch_logits: bool = False,
    ):
        graph_feat, mincut_loss, ortho_loss = self._encode_branches(x)
        logits = self.classifier(graph_feat)

        if return_branch_logits:
            return logits, mincut_loss, ortho_loss, {
                'graph_feat': graph_feat,
                'graph_only_logits': logits,
            }

        return logits, mincut_loss, ortho_loss