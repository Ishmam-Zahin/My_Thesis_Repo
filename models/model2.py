import torch
from torch import nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn.dense import dense_mincut_pool


class VideoFeatureExtractor(nn.Module):
    """
    DINOv2 ViT that extracts BOTH patch tokens AND cls token.
    - Only last 2 transformer blocks are fully unfrozen
    - ALL LayerNorm layers (25 of them) are unfrozen
    """
    def __init__(self, vit_name='dinov2_vits14', feature_dim=384, total_nodes=256, weight_path=None):
        super().__init__()
        self.vit = torch.hub.load('facebookresearch/dinov2', vit_name)

        if weight_path is not None:
            ckpt = torch.load(weight_path, map_location='cpu', weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            vit_state_dict = {
                k.replace("vit.", "", 1): v
                for k, v in state_dict.items()
                if k.startswith("vit.")
            }
            missing, unexpected = self.vit.load_state_dict(vit_state_dict, strict=False)
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

        # === UNFROZEN STRATEGY ===
        # 1. Freeze everything first
        for p in self.vit.parameters():
            p.requires_grad = False

        # 2. Unfreeze ONLY the last 2 transformer blocks (fully)
        for block in self.vit.blocks[-2:]:
            for p in block.parameters():
                p.requires_grad = True

        # 3. Unfreeze ALL LayerNorm layers in the entire model
        for name, module in self.vit.named_modules():
            if isinstance(module, nn.LayerNorm):
                for p in module.parameters():
                    p.requires_grad = True

        # Keep in eval mode (recommended for DINOv2)
        self.vit.eval()

        self.D = feature_dim
        self.total_nodes = total_nodes

    def forward(self, x: torch.Tensor):
        """
        Returns BOTH patch tokens and cls tokens
        """
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)          # [B*T, 3, 224, 224]

        features = self.vit.forward_features(x_flat)

        patch_tokens = features['x_norm_patchtokens']   # [B*T, 256, 384]
        cls_tokens   = features['x_norm_clstoken']      # [B*T, 384]

        # Reshape to per-video
        patch_tokens = patch_tokens.reshape(B, T * self.total_nodes, self.D)
        cls_tokens   = cls_tokens.reshape(B, T, self.D)

        return patch_tokens, cls_tokens


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8, mlp_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
            add_zero_attn=False
        )
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        x_norm = self.ln2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        return x


class MyModel(nn.Module):
    """
    Updated model with:
      - Only last 2 ViT blocks + ALL LayerNorms unfrozen
      - CLS token added AFTER GAT layers (before MinCut)
      - Learnable positional embedding before Transformer
    """
    def __init__(
        self,
        vit_name: str = 'dinov2_vits14',
        feature_dim: int = 384,
        num_gcn_layers: int = 2,
        num_clusters: int = 512,
        num_transformer_blocks: int = 1,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.2,
        num_of_frames=8,
        num_of_nodes_per_frame=256,
        num_of_temporal_edge_per_node=4,
        video_spatial_src_edges=None,
        video_spatial_dst_edges=None,
        vit_weight_path=None,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_clusters = num_clusters
        self.num_transformer_blocks = num_transformer_blocks
        self.num_heads = num_heads
        self.num_of_frames = num_of_frames
        self.num_of_nodes_per_frame = num_of_nodes_per_frame
        self.num_of_temporal_edge_per_node = num_of_temporal_edge_per_node
        self.video_spatial_src_edges = video_spatial_src_edges
        self.video_spatial_dst_edges = video_spatial_dst_edges

        # Feature extractor
        self.vit = VideoFeatureExtractor(
            vit_name=vit_name,
            weight_path=vit_weight_path
        )

        assert feature_dim % num_heads == 0
        head_dim = feature_dim // num_heads

        self.gcn_layers = nn.ModuleList()
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

        # Learnable class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, feature_dim))

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_clusters, feature_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=feature_dim,
                heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout
            ) for _ in range(num_transformer_blocks)
        ])

        # Final layers
        self.norm = nn.LayerNorm(feature_dim)
        self.classifier = nn.Linear(feature_dim, 2)

    def add_temporal_edges(self, frame_patches, edge_src_global, edge_dst_global):
        patches = frame_patches.view(self.num_of_frames, self.num_of_nodes_per_frame, -1)
        frame_patch_norm = F.normalize(patches, dim=-1)

        for t in range(self.num_of_frames - 1):
            patch_curr = frame_patch_norm[t]
            patch_next = frame_patch_norm[t + 1]
            sim = torch.mm(patch_curr, patch_next.t())
            _, top_idx = torch.topk(sim, k=self.num_of_temporal_edge_per_node, dim=-1)

            edge_temporal_src_local = torch.arange(
                self.num_of_nodes_per_frame, device=frame_patches.device
            ).unsqueeze(1).expand(-1, self.num_of_temporal_edge_per_node).reshape(-1)
            edge_temporal_dst_local = top_idx.reshape(-1)

            edge_temporal_src_local += (t * self.num_of_nodes_per_frame)
            edge_temporal_dst_local += ((t + 1) * self.num_of_nodes_per_frame)

            edge_src_global.append(torch.cat([edge_temporal_src_local, edge_temporal_dst_local]))
            edge_dst_global.append(torch.cat([edge_temporal_dst_local, edge_temporal_src_local]))

        return torch.stack([torch.cat(edge_src_global), torch.cat(edge_dst_global)], dim=0)

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        num_nodes_per_video = x.shape[1] * 256

        # 1. Extract patch features + CLS tokens
        patch_features, cls_features = self.vit(x)          # [B, T*N, D], [B, T, D]

        # === Use pure patch features for graph + GAT (CLS added later) ===
        features = patch_features

        # 2. Build PyG batch + GCN/GAT layers
        data_list = []
        for i in range(B):
            edge_index = self.add_temporal_edges(
                features[i],
                edge_src_global=[self.video_spatial_src_edges],
                edge_dst_global=[self.video_spatial_dst_edges],
            )
            data = Data(x=features[i], edge_index=edge_index)
            data_list.append(data)

        batched_data = Batch.from_data_list(data_list)
        x = batched_data.x
        edge_index = batched_data.edge_index
        batch = batched_data.batch

        # 3. GAT layers
        for gcn in self.gcn_layers:
            x = gcn(x, edge_index)
            x = F.relu(x)

        # 4. Reshape back to per-video
        x = torch.split(x, split_size_or_sections=num_nodes_per_video)
        x = torch.stack(x, dim=0)                                # [B, T*N, D]

        # === ADD CLS TOKEN HERE (AFTER GAT, BEFORE MINCUT) ===
        cls_features = cls_features.unsqueeze(2).expand(-1, -1, self.num_of_nodes_per_frame, -1)
        cls_features = cls_features.reshape(B, -1, self.feature_dim)   # [B, T*N, D]
        x = x + cls_features                                           # ← Global context injected

        # 5. Min-cut pooling
        s = self.assign_net(x)
        adj = to_dense_adj(edge_index, batch=batch)
        x_pooled, adj_pooled, mincut_loss, ortho_loss = dense_mincut_pool(x, adj, s)

        # 6. Transformer classifier
        cls_tokens = self.class_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x_pooled), dim=1)
        x = x + self.pos_embed

        for block in self.transformer_blocks:
            x = block(x)

        x = x[:, 0]
        x = self.norm(x)
        logits = self.classifier(x)

        return (logits, mincut_loss, ortho_loss)