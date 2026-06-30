import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.dense import dense_mincut_pool


# =============================================================================
# VideoFeatureExtractor
# Extracts both CLS tokens (for BiLSTM) and patch tokens (for graph)
# =============================================================================

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

        epoch    = ckpt.get('epoch',    '?')
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

        cls_tokens   = features['x_norm_clstoken']
        patch_tokens = features['x_norm_patchtokens']

        return {
            'cls':   cls_tokens.reshape(B, T, self.D),
            'patch': patch_tokens.reshape(B, T * 256, self.D)
        }

    def train(self, mode: bool = True):
        super().train(mode)
        self.vit.eval()
        return self


# =============================================================================
# AdaptiveFusion — FIXED VERSION with Competing Softmax
# =============================================================================

class AdaptiveFusion(nn.Module):
    """
    Improved adaptive fusion with:
      1. Competing softmax contributions (not independent sigmoids)
      2. Entropy regularization to prevent collapse
      3. Temperature scaling for smooth learning
      4. Joint contribution network that sees both branches
    """
    def __init__(
        self,
        bilstm_dim:   int = 384,
        graph_dim:    int = 384,
        common_dim:   int = 512,
        num_classes:  int = 2,
        num_heads:    int = 8,
        dropout:      float = 0.3,
        temperature:  float = 1.0,
    ):
        super().__init__()

        # ── Step 1: Project each branch to a common dimension ─────────────────
        self.bilstm_proj = nn.Sequential(
            nn.Linear(bilstm_dim, common_dim),
            nn.BatchNorm1d(common_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(common_dim, common_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.graph_proj = nn.Sequential(
            nn.Linear(graph_dim, common_dim),
            nn.BatchNorm1d(common_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(common_dim, common_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Step 2: Cross-attention between both projected features ───────────
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=common_dim,
            num_heads=num_heads,
            dropout=dropout * 0.5,
            batch_first=True,
        )

        # ── Step 3: NEW — Joint contribution network with competing softmax ───
        # Instead of two independent sigmoid heads, use one network that outputs
        # competing logits, forcing the model to choose between branches
        self.contribution_net = nn.Sequential(
            nn.Linear(common_dim * 2, 256),  # Takes BOTH branch features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 2),  # Output 2 logits: [bilstm_logit, graph_logit]
        )
        
        # Temperature for controlling sharpness of contribution distribution
        self.register_buffer('temperature', torch.tensor(temperature))

        # ── Step 4: Fusion strategy MLP ───────────────────────────────────────
        self.fusion_strategy = nn.Sequential(
            nn.Linear(common_dim * 2 + 2, common_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(common_dim, common_dim),
        )

        # ── Step 5: Final classifier ──────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(common_dim, common_dim),
            nn.BatchNorm1d(common_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(common_dim, common_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(common_dim // 2, num_classes),
        )

    def forward(
        self,
        bilstm_feat: torch.Tensor,   # [B, bilstm_dim]
        graph_feat:  torch.Tensor,   # [B, graph_dim]
    ):
        # ── Step 1: Project to common space ───────────────────────────────────
        bilstm_proj = self.bilstm_proj(bilstm_feat)   # [B, common_dim]
        graph_proj  = self.graph_proj(graph_feat)      # [B, common_dim]

        # ── Step 2: Cross-attention ───────────────────────────────────────────
        stacked = torch.stack([bilstm_proj, graph_proj], dim=1)   # [B, 2, common_dim]
        attended, attention_weights = self.cross_attention(stacked, stacked, stacked)
        attended_bilstm = attended[:, 0]   # [B, common_dim]
        attended_graph  = attended[:, 1]   # [B, common_dim]

        # ── Step 3: NEW — Competing contribution scoring ──────────────────────
        # Concatenate BOTH features so the network sees them together
        combined_for_contrib = torch.cat([attended_bilstm, attended_graph], dim=-1)
        
        # Get contribution logits (raw scores before softmax)
        contrib_logits = self.contribution_net(combined_for_contrib)  # [B, 2]
        
        # Apply temperature-scaled softmax to force competition
        # Temperature > 1: smoother distribution
        # Temperature < 1: sharper distribution
        contributions = F.softmax(contrib_logits / self.temperature, dim=-1)
        bilstm_w = contributions[:, 0:1]  # [B, 1]
        graph_w  = contributions[:, 1:2]  # [B, 1]
        
        # Compute entropy to monitor contribution balance
        # High entropy (~0.69 for binary) = balanced 50/50
        # Low entropy (~0) = collapsed to one branch
        entropy = -(contributions * torch.log(contributions + 1e-8)).sum(dim=-1).mean()

        # ── Step 4: Weight features by their contributions ────────────────────
        weighted_bilstm = attended_bilstm * bilstm_w   # [B, common_dim]
        weighted_graph  = attended_graph  * graph_w     # [B, common_dim]

        # ── Step 5: Fusion strategy MLP ───────────────────────────────────────
        fusion_input = torch.cat(
            [weighted_bilstm, weighted_graph, bilstm_w, graph_w], dim=1
        )   # [B, common_dim*2 + 2]
        combined = self.fusion_strategy(fusion_input)   # [B, common_dim]

        # ── Step 6: Classify ──────────────────────────────────────────────────
        logits = self.classifier(combined)   # [B, num_classes]

        # Detailed contribution statistics
        contributions_dict = {
            'bilstm_contribution': bilstm_w.mean().item(),
            'graph_contribution':  graph_w.mean().item(),
            'entropy':             entropy.item(),
            'bilstm_logit':        contrib_logits[:, 0].mean().item(),
            'graph_logit':         contrib_logits[:, 1].mean().item(),
            'bilstm_std':          bilstm_w.std().item(),
            'graph_std':           graph_w.std().item(),
        }

        return logits, contributions_dict, attention_weights, entropy


# =============================================================================
# FusedModel — Main Model with Both Branches
# =============================================================================

class FusedModel(nn.Module):
    def __init__(
        self,
        vit_name:               str   = 'dinov2_vits14',
        feature_dim:            int   = 384,
        dropout:                float = 0.2,
        num_of_frames:          int   = 8,
        num_gcn_layers:         int   = 3,
        num_clusters:           int   = 512,
        num_transformer_blocks: int   = 1,
        num_heads:              int   = 8,
        mlp_dim:                int   = 512,
        tau_s:                  float = 0.6,
        tau_t:                  float = 0.6,
        graph_eps:              float = 1e-4,
        local_window:           int   = 3,
        vit_weight_path               = None,
        fusion_temperature:     float = 1.0,
    ):
        super().__init__()
        self.num_of_frames = num_of_frames
        self.feature_dim   = feature_dim
        self.num_clusters  = num_clusters
        self.tau_s         = tau_s
        self.tau_t         = tau_t
        self.graph_eps     = graph_eps
        self.local_window  = local_window

        # ── Shared ViT backbone ───────────────────────────────────────────────
        self.vit = VideoFeatureExtractor(
            vit_name=vit_name,
            feature_dim=feature_dim,
            weight_path=vit_weight_path,
        )

        # =====================================================================
        # BiLSTM Branch
        # =====================================================================
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, num_of_frames, feature_dim)
        )
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

        self.bilstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=feature_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0,
        )

        # =====================================================================
        # Graph Branch
        # =====================================================================

        # Dual GATv2Conv stacks (consistency / inconsistency)
        self.consistency_gnn_layers   = nn.ModuleList()
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
                    edge_dim=1,
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
                    edge_dim=1,
                )
            )

        # Merge consistency + inconsistency node features
        self.spatial_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
        )

        # MinCut pooling soft-assignment network
        self.assign_net = nn.Linear(feature_dim, num_clusters)

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

        # =====================================================================
        # Adaptive Fusion — IMPROVED VERSION
        # =====================================================================
        self.adaptive_fusion = AdaptiveFusion(
            bilstm_dim=feature_dim,
            graph_dim=feature_dim,
            common_dim=512,
            num_classes=2,
            num_heads=8,
            dropout=0.3,
            temperature=fusion_temperature,
        )

        # Contribution history for monitoring
        self.contribution_history = []

    # =========================================================================
    # Graph branch helpers
    # =========================================================================

    def _local_negative_edges(
        self,
        t:           int,
        num_patches: int,
        grid_size:   int,
        device:      torch.device,
    ):
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
            edge_index = torch.empty((2, 0), dtype=torch.long,  device=device)
            edge_attr  = torch.empty((0,),   dtype=torch.float, device=device)
            return edge_index, edge_attr

        edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
        edge_attr  = torch.full((edge_index.shape[1],), -1.0,   device=device)
        return edge_index, edge_attr

    def _build_graphs_for_sample(self, patch_tokens: torch.Tensor):
        T, N, _ = patch_tokens.shape
        device    = patch_tokens.device
        grid_size = int(N ** 0.5)
        if grid_size * grid_size != N:
            raise ValueError(f"Patch count {N} must be a perfect square")

        x_norm = patch_tokens / (patch_tokens.norm(dim=-1, keepdim=True) + self.graph_eps)

        cons_src,  cons_dst,  cons_w  = [], [], []
        incon_src, incon_dst, incon_w = [], [], []
        spatial_adjs = []

        # ── Spatial (within-frame) edges ──────────────────────────────────────
        for t in range(T):
            sim = torch.matmul(x_norm[t], x_norm[t].transpose(0, 1))
            sim.fill_diagonal_(0.0)
            spatial_adjs.append(sim)

            edge_mask = sim >= self.tau_s
            src, dst  = torch.where(edge_mask)
            if src.numel() > 0:
                cons_src.append(src + t * N)
                cons_dst.append(dst + t * N)
                cons_w.append(sim[src, dst])

            neg_ei, neg_ew = self._local_negative_edges(
                t=t, num_patches=N, grid_size=grid_size, device=device
            )
            if neg_ei.numel() > 0:
                incon_src.append(neg_ei[0])
                incon_dst.append(neg_ei[1])
                incon_w.append(neg_ew)

        # ── Temporal (cross-frame) edges ──────────────────────────────────────
        for t in range(T - 1):
            idx          = torch.arange(N, device=device)
            offset_curr  = t * N
            offset_next  = (t + 1) * N

            a_curr      = spatial_adjs[t]
            a_next      = spatial_adjs[t + 1]
            struct_sim  = F.cosine_similarity(a_curr, a_next, dim=-1)
            feat_sim    = F.cosine_similarity(x_norm[t], x_norm[t + 1], dim=-1)
            score       = 0.5 * (struct_sim + feat_sim)

            keep = score >= self.tau_t
            if keep.any():
                keep_idx    = idx[keep]
                keep_scores = score[keep]
                cons_src.append(keep_idx + offset_curr)
                cons_dst.append(keep_idx + offset_next)
                cons_w.append(keep_scores)
                cons_src.append(keep_idx + offset_next)
                cons_dst.append(keep_idx + offset_curr)
                cons_w.append(keep_scores)

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
            node_idx        = torch.arange(T * N, device=device)
            edge_index_cons = torch.stack([node_idx, node_idx], dim=0)
            edge_attr_cons  = torch.ones(T * N, device=device)

        if incon_src:
            edge_index_incon = torch.stack(
                [torch.cat(incon_src), torch.cat(incon_dst)], dim=0
            )
            edge_attr_incon = torch.cat(incon_w)
        else:
            node_idx         = torch.arange(T * N, device=device)
            edge_index_incon = torch.stack([node_idx, node_idx], dim=0)
            edge_attr_incon  = torch.full((T * N,), -1.0, device=device)

        return edge_index_cons, edge_attr_cons, edge_index_incon, edge_attr_incon

    def _encode_graph_sample(self, patch_tokens_flat: torch.Tensor):
        T  = self.num_of_frames
        TN, _ = patch_tokens_flat.shape
        N  = TN // T
        if N * T != TN:
            raise ValueError("Patch token count is not divisible by num_of_frames")

        patch_tokens = patch_tokens_flat.view(T, N, self.feature_dim)
        edge_index_cons, edge_attr_cons, edge_index_incon, edge_attr_incon = \
            self._build_graphs_for_sample(patch_tokens)

        ea_cons  = edge_attr_cons.unsqueeze(-1)
        ea_incon = edge_attr_incon.unsqueeze(-1)

        # ── Dual GAT ──────────────────────────────────────────────────────────
        x_cons  = patch_tokens_flat
        x_incon = patch_tokens_flat

        for layer_cons, layer_incon in zip(
            self.consistency_gnn_layers, self.inconsistency_gnn_layers
        ):
            x_cons  = F.relu(layer_cons(x_cons,   edge_index_cons,  ea_cons))
            x_incon = F.relu(layer_incon(x_incon, edge_index_incon, ea_incon))

        x_nodes = self.spatial_mlp(torch.cat([x_cons, x_incon], dim=-1))

        # ── MinCut pooling ────────────────────────────────────────────────────
        num_nodes   = T * N
        x_nodes_3d  = x_nodes.unsqueeze(0)
        s           = self.assign_net(x_nodes_3d)

        adj = torch.zeros(
            (1, num_nodes, num_nodes),
            device=x_nodes.device,
            dtype=x_nodes.dtype,
        )
        adj[0, edge_index_cons[0], edge_index_cons[1]] = edge_attr_cons
        adj = 0.5 * (adj + adj.transpose(1, 2))

        x_pooled, _, mincut_loss, ortho_loss = dense_mincut_pool(x_nodes_3d, adj, s)

        return x_pooled.squeeze(0), mincut_loss, ortho_loss

    # =========================================================================
    # Branch encoding
    # =========================================================================

    def _encode_branches(self, x: torch.Tensor):
        B     = x.shape[0]
        feats = self.vit(x)

        cls_features   = feats['cls']
        patch_features = feats['patch']

        # ── BiLSTM Branch ─────────────────────────────────────────────────────
        cls_features = cls_features + self.temporal_pos_embed
        lstm_out, _  = self.bilstm(cls_features)
        bilstm_feat  = lstm_out.mean(dim=1)

        # ── Graph Branch ──────────────────────────────────────────────────────
        pooled_list       = []
        total_mincut_loss = torch.zeros((), device=x.device)
        total_ortho_loss  = torch.zeros((), device=x.device)

        for i in range(B):
            x_pooled, mc, ort = self._encode_graph_sample(patch_features[i])
            pooled_list.append(x_pooled)
            total_mincut_loss = total_mincut_loss + mc
            total_ortho_loss  = total_ortho_loss  + ort

        x_g = torch.stack(pooled_list, dim=0)

        cls_token = self.class_token_graph.expand(B, 1, -1)
        x_g       = torch.cat([cls_token, x_g], dim=1)

        for block in self.transformer_blocks:
            x_g = block(x_g)

        graph_feat = x_g[:, 0]

        return bilstm_feat, graph_feat, total_mincut_loss, total_ortho_loss

    # =========================================================================
    # Forward
    # =========================================================================

    def forward(
        self,
        x:                    torch.Tensor,
        branch_ablation:      str | None = None,
        return_branch_logits: bool       = False,
    ):
        bilstm_feat, graph_feat, mincut_loss, ortho_loss = self._encode_branches(x)

        # Optional branch ablation (zero out one branch for analysis)
        if branch_ablation == 'cls':
            bilstm_feat = torch.zeros_like(bilstm_feat)
        elif branch_ablation == 'graph':
            graph_feat = torch.zeros_like(graph_feat)

        # ── Adaptive Fusion (now returns entropy) ─────────────────────────────
        logits, contributions, attention_weights, entropy = self.adaptive_fusion(
            bilstm_feat, graph_feat
        )

        # Track contributions for monitoring
        self.contribution_history.append(contributions)
        if len(self.contribution_history) % 100 == 0:
            last100      = self.contribution_history[-100:]
            avg_bilstm   = sum(c['bilstm_contribution'] for c in last100) / 100
            avg_graph    = sum(c['graph_contribution']  for c in last100) / 100
            avg_entropy  = sum(c['entropy']             for c in last100) / 100
            avg_b_logit  = sum(c['bilstm_logit']        for c in last100) / 100
            avg_g_logit  = sum(c['graph_logit']         for c in last100) / 100
            
            print(
                f"📊 Contributions (last 100): "
                f"BiLSTM={avg_bilstm:.3f}, Graph={avg_graph:.3f}, "
                f"Entropy={avg_entropy:.3f} | "
                f"Logits: BiLSTM={avg_b_logit:.3f}, Graph={avg_g_logit:.3f}"
            )

        if return_branch_logits:
            # Run each branch independently for analysis
            zero_bilstm = torch.zeros_like(bilstm_feat)
            zero_graph  = torch.zeros_like(graph_feat)
            bilstm_only_logits, _, _, _ = self.adaptive_fusion(bilstm_feat, zero_graph)
            graph_only_logits,  _, _, _ = self.adaptive_fusion(zero_bilstm, graph_feat)

            return logits, mincut_loss, ortho_loss, {
                'bilstm_feat':          bilstm_feat,
                'graph_feat':           graph_feat,
                'contributions':        contributions,
                'attention_weights':    attention_weights,
                'entropy':              entropy,
                'bilstm_only_logits':   bilstm_only_logits,
                'graph_only_logits':    graph_only_logits,
            }

        return logits, mincut_loss, ortho_loss, entropy


# =============================================================================
# Helper function for gradient analysis (optional, for debugging)
# =============================================================================

def get_branch_gradients(model):
    """
    Analyzes gradient flow to each branch.
    Useful for debugging gradient competition.
    """
    bilstm_grad_norm = 0.0
    graph_grad_norm = 0.0
    fusion_grad_norm = 0.0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if 'bilstm' in name or 'temporal_pos_embed' in name:
                bilstm_grad_norm += grad_norm
            elif 'gnn' in name or 'spatial_mlp' in name or 'assign_net' in name or 'class_token_graph' in name or 'transformer_blocks' in name:
                graph_grad_norm += grad_norm
            elif 'adaptive_fusion' in name:
                fusion_grad_norm += grad_norm
    
    return {
        'bilstm_grad': bilstm_grad_norm,
        'graph_grad': graph_grad_norm,
        'fusion_grad': fusion_grad_norm,
    }