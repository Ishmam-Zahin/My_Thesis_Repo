import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.dense import dense_mincut_pool


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
    Graph model:
      - Dynamic similarity-based spatial-temporal graph construction  (current model)
      - Dual GATv2Conv stacks for consistency / inconsistency modeling (current model)
      - spatial_mlp to merge dual-GAT node features                   (current model)
      - MinCut pooling                                                 (old model)
      - Graph CLS token + Transformer encoder blocks                  (old model)
      - Linear classifier                                              (old model)
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

        # ── ViT backbone ──────────────────────────────────────────────────────
        self.vit = VideoFeatureExtractor(vit_name=vit_name, weight_path=vit_weight_path)

        # ── Dual GAT stacks (current model) ───────────────────────────────────
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
                    edge_dim=1,            # scalar similarity weight per edge
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
                    edge_dim=1,            # scalar −1 weight per edge
                )
            )

        # Merge consistency + inconsistency node features → single feature_dim
        self.spatial_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
        )

        # ── MinCut pooling components (old model) ─────────────────────────────
        # Soft-assignment network: maps each node to one of num_clusters clusters
        self.assign_net = nn.Linear(feature_dim, num_clusters)

        # ── Transformer encoder on pooled clusters (old model) ────────────────
        # Learnable graph-level CLS token
        self.class_token_graph = nn.Parameter(torch.zeros(1, 1, feature_dim))

        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=num_heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_transformer_blocks)
        ])

        # ── Classifier (old model) ────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    # =========================================================================
    # Graph construction helpers (current model — unchanged)
    # =========================================================================

    def _local_negative_edges(
        self,
        t: int,
        num_patches: int,
        grid_size: int,
        device: torch.device,
    ):
        """
        Returns local-window inconsistency edges with weight −1 each,
        matching the paper's negative-edge convention for spatial differentials.
        """
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
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_attr  = torch.empty((0,),   dtype=torch.float, device=device)
            return edge_index, edge_attr

        edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
        edge_attr  = torch.full((edge_index.shape[1],), -1.0, device=device)  # paper: −1
        return edge_index, edge_attr

    def _build_graphs_for_sample(self, patch_tokens: torch.Tensor):
        """
        Returns edge indices AND edge weights for both graph streams.

        Consistency  edges: weight = cosine similarity score  (paper: A^(t)_{i,j} = SIM(...))
        Inconsistency edges: weight = −1 everywhere           (paper: A(v_i^t, v_i^{t+1}) = −1)

        Returns
        -------
        edge_index_cons  : [2, E_cons]   long
        edge_attr_cons   : [E_cons]      float  — similarity scores in [tau_s, 1]
        edge_index_incon : [2, E_incon]  long
        edge_attr_incon  : [E_incon]     float  — all −1.0
        """
        T, N, _ = patch_tokens.shape
        device = patch_tokens.device
        grid_size = int(N ** 0.5)
        if grid_size * grid_size != N:
            raise ValueError(f"Patch count {N} must be a perfect square")

        x_norm = patch_tokens / (patch_tokens.norm(dim=-1, keepdim=True) + self.graph_eps)

        cons_src,   cons_dst,   cons_w   = [], [], []
        incon_src,  incon_dst,  incon_w  = [], [], []
        spatial_adjs = []

        # ── Spatial (within-frame) edges ──────────────────────────────────────
        for t in range(T):
            sim = torch.matmul(x_norm[t], x_norm[t].transpose(0, 1))
            sim.fill_diagonal_(0.0)
            spatial_adjs.append(sim)

            edge_mask = sim >= self.tau_s
            src, dst = torch.where(edge_mask)
            if src.numel() > 0:
                cons_src.append(src + t * N)
                cons_dst.append(dst + t * N)
                cons_w.append(sim[src, dst])           # similarity score as weight

            # Spatial inconsistency: local-window negative edges (weight = −1)
            neg_ei, neg_ew = self._local_negative_edges(
                t=t, num_patches=N, grid_size=grid_size, device=device
            )
            if neg_ei.numel() > 0:
                incon_src.append(neg_ei[0])
                incon_dst.append(neg_ei[1])
                incon_w.append(neg_ew)

        # ── Temporal (cross-frame) edges ──────────────────────────────────────
        for t in range(T - 1):
            idx = torch.arange(N, device=device)
            offset_curr = t * N
            offset_next = (t + 1) * N

            a_curr = spatial_adjs[t]
            a_next = spatial_adjs[t + 1]
            struct_sim = F.cosine_similarity(a_curr, a_next, dim=-1)
            feat_sim   = F.cosine_similarity(x_norm[t], x_norm[t + 1], dim=-1)
            score = 0.5 * (struct_sim + feat_sim)      # paper: S^(t)(v_i^t, v_i^{t+1})

            # Consistency temporal edges: weight = the combined score
            keep = score >= self.tau_t
            if keep.any():
                keep_idx = idx[keep]
                keep_scores = score[keep]
                # bidirectional — same score for both directions
                cons_src.append(keep_idx + offset_curr)
                cons_dst.append(keep_idx + offset_next)
                cons_w.append(keep_scores)
                cons_src.append(keep_idx + offset_next)
                cons_dst.append(keep_idx + offset_curr)
                cons_w.append(keep_scores)

            # Inconsistency temporal edges: weight = −1 (paper's negative edges)
            neg_w = torch.full((N,), -1.0, device=device)
            incon_src.append(idx + offset_curr)
            incon_dst.append(idx + offset_next)
            incon_w.append(neg_w)
            incon_src.append(idx + offset_next)
            incon_dst.append(idx + offset_curr)
            incon_w.append(neg_w)

        # ── Assemble ──────────────────────────────────────────────────────────
        if cons_src:
            edge_index_cons = torch.stack(
                [torch.cat(cons_src), torch.cat(cons_dst)], dim=0
            )
            edge_attr_cons = torch.cat(cons_w)
        else:
            # Fallback: self-loops with weight 1.0 so GAT doesn't error
            node_idx = torch.arange(T * N, device=device)
            edge_index_cons = torch.stack([node_idx, node_idx], dim=0)
            edge_attr_cons  = torch.ones(T * N, device=device)

        if incon_src:
            edge_index_incon = torch.stack(
                [torch.cat(incon_src), torch.cat(incon_dst)], dim=0
            )
            edge_attr_incon = torch.cat(incon_w)
        else:
            node_idx = torch.arange(T * N, device=device)
            edge_index_incon = torch.stack([node_idx, node_idx], dim=0)
            edge_attr_incon  = torch.full((T * N,), -1.0, device=device)

        return edge_index_cons, edge_attr_cons, edge_index_incon, edge_attr_incon

    # =========================================================================
    # Per-sample graph encoding
    # =========================================================================

    def _encode_graph_sample(self, patch_tokens_flat: torch.Tensor):
        """
        Args:
            patch_tokens_flat: [T*N, D]

        Returns:
            x_pooled:    [num_clusters, D]  — MinCut-pooled cluster features
            mincut_loss: scalar
            ortho_loss:  scalar
        """
        T = self.num_of_frames
        TN, _ = patch_tokens_flat.shape
        N = TN // T
        if N * T != TN:
            raise ValueError("Patch token count is not divisible by num_of_frames")

        patch_tokens = patch_tokens_flat.view(T, N, self.feature_dim)
        edge_index_cons, edge_attr_cons, edge_index_incon, edge_attr_incon = \
            self._build_graphs_for_sample(patch_tokens)

        # GATv2Conv expects edge_attr shape [E, edge_dim] → unsqueeze to [E, 1]
        ea_cons  = edge_attr_cons.unsqueeze(-1)
        ea_incon = edge_attr_incon.unsqueeze(-1)

        # ── Dual GAT (current model) ──────────────────────────────────────────
        x_cons  = patch_tokens_flat
        x_incon = patch_tokens_flat

        for layer_cons, layer_incon in zip(
            self.consistency_gnn_layers, self.inconsistency_gnn_layers
        ):
            x_cons  = F.relu(layer_cons(x_cons,   edge_index_cons,  ea_cons))
            x_incon = F.relu(layer_incon(x_incon, edge_index_incon, ea_incon))

        # Merge dual-GAT outputs → [T*N, D]
        x_nodes = self.spatial_mlp(torch.cat([x_cons, x_incon], dim=-1))

        # ── MinCut pooling (old model) ────────────────────────────────────────
        num_nodes = T * N
        x_nodes_3d = x_nodes.unsqueeze(0)                       # [1, T*N, D]
        s = self.assign_net(x_nodes_3d)                         # [1, T*N, num_clusters]

        # Build dense adjacency from consistency edges — use similarity scores as weights
        # (paper uses A^(t)_{i,j} = SIM(...), not binary 1/0)
        adj = torch.zeros(
            (1, num_nodes, num_nodes),
            device=x_nodes.device,
            dtype=x_nodes.dtype,
        )
        adj[0, edge_index_cons[0], edge_index_cons[1]] = edge_attr_cons
        adj = 0.5 * (adj + adj.transpose(1, 2))                 # symmetrise

        x_pooled, _, mincut_loss, ortho_loss = dense_mincut_pool(x_nodes_3d, adj, s)
        # x_pooled: [1, num_clusters, D]

        return x_pooled.squeeze(0), mincut_loss, ortho_loss      # [num_clusters, D]

    # =========================================================================
    # Branch encoding (over full batch)
    # =========================================================================

    def _encode_branches(self, x: torch.Tensor):
        B = x.shape[0]
        feats = self.vit(x)
        patch_features = feats['patch']                          # [B, T*256, D]

        pooled_list = []
        total_mincut_loss = torch.zeros((), device=x.device)
        total_ortho_loss  = torch.zeros((), device=x.device)

        for i in range(B):
            x_pooled, mc, ort = self._encode_graph_sample(patch_features[i])
            pooled_list.append(x_pooled)
            total_mincut_loss = total_mincut_loss + mc
            total_ortho_loss  = total_ortho_loss  + ort

        x_g = torch.stack(pooled_list, dim=0)                   # [B, num_clusters, D]

        # ── Transformer encoder with graph CLS token (old model) ─────────────
        cls_token = self.class_token_graph.expand(B, 1, -1)     # [B, 1, D]
        x_g = torch.cat([cls_token, x_g], dim=1)                # [B, 1+num_clusters, D]

        for block in self.transformer_blocks:
            x_g = block(x_g)

        graph_feat = x_g[:, 0]                                   # [B, D]

        return graph_feat, total_mincut_loss, total_ortho_loss

    # =========================================================================
    # Forward
    # =========================================================================

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