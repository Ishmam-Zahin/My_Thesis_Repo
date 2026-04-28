import torch
from torch import nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn.dense import dense_mincut_pool


class VideoFeatureExtractor(nn.Module):
    """
    Frozen DINOv2 ViT that extracts patch tokens for each frame of a video.
    Input: [B, T, 3, H, W]
    Output: [B, T*N, D] where N=256, D=384 for dinov2_vits14
    """
    def __init__(self, vit_name = 'dinov2_vitb14', feature_dim = 384, total_nodes = 256):
        super().__init__()
        self.vit = torch.hub.load('facebookresearch/dinov2', vit_name)
        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.eval()
        self.D =  feature_dim
        self.total_nodes = total_nodes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)  # [B*T, 3, 224, 224]
        features = self.vit.forward_features(x_flat)
        patch_tokens = features['x_norm_patchtokens']  # [B*T, N=256, D=384]
        # Reshape back to per-video
        return patch_tokens.reshape(B, T * self.total_nodes, self.D)  # [B, T*N, D]


class TransformerBlock(nn.Module):
    """
    Standard pre-norm Transformer block (matches paper equations 5-10).
    Uses Multihead Self-Attention (MSA) + MLP with residual connections.
    """
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
        # x: [B, seq_len, dim]
        # Pre-norm MSA + residual
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Pre-norm MLP + residual
        x_norm = self.ln2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        return x


class MyModel(nn.Module):
    """
    Completed model inspired by the paper:
    - DINOv2 ViT feature extractor (frozen, self-supervised)
    - Graph construction (via create_frame_graph)
    - Multi-layer GCN (local structure)
    - Min-cut pooling (node reduction, as in paper)
    - Transformer blocks (global dependencies via MSA + MLP)
    - Class token + final classifier (binary: normal vs deepfake)
    """
    def __init__(
        self,
        vit_name: str = 'dinov2_vits14',
        feature_dim: int = 384,
        num_gcn_layers: int = 2,
        num_clusters: int = 512,          # pooled nodes after min-cut
        num_transformer_blocks: int = 1,  # as tested in paper ablation (TB=3)
        num_heads: int = 8,               # paper: 8 self-attention heads
        mlp_dim: int = 512,               # paper: MLP size of 512
        dropout: float = 0.2,
        num_of_frames = 8,
        num_of_nodes_per_frame = 256,
        num_of_temporal_edge_per_node = 4,
        video_spatial_src_edges = None,
        video_spatial_dst_edges = None,

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

        # Feature extractor (frozen DINOv2)
        self.vit = VideoFeatureExtractor(vit_name=vit_name)

        assert feature_dim % num_heads == 0
        head_dim = feature_dim // num_heads
        self.gcn_layers = nn.ModuleList()
        for _ in range(num_gcn_layers):
            self.gcn_layers.append(
                GATv2Conv(
                    in_channels=feature_dim,         # 768
                    out_channels=head_dim,           # 48 (per head)
                    heads=num_heads,                 # 16
                    concat=True,                     # concat → output is 48*16 = 768 ✅
                    dropout=dropout,
                    add_self_loops=True,
                )
            )
        

        # Assignment network for min-cut pooling
        self.assign_net = nn.Linear(feature_dim, num_clusters)

        # Learnable class token + positional embeddings
        # (class token + pooled nodes; adjacency is already captured by GCN)
        self.class_token = nn.Parameter(torch.zeros(1, 1, feature_dim))

        # Transformer blocks (global dependencies)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=feature_dim,
                heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout
            )
            for _ in range(num_transformer_blocks)
        ])

        # Final layers
        self.norm = nn.LayerNorm(feature_dim)
        self.classifier = nn.Linear(feature_dim, 2)  # binary: normal / deepfake
    

    def add_temporal_edges(
            self,
            frame_patches,
            edge_src_global,
            edge_dst_global
    ):
        patches = frame_patches.view(self.num_of_frames, self.num_of_nodes_per_frame, -1)
        frame_patch_norm = F.normalize(patches, dim = -1)

        for t in range(self.num_of_frames - 1):
            patch_curr = frame_patch_norm[t]
            patch_next = frame_patch_norm[t + 1]
            sim = torch.mm(patch_curr, patch_next.t())
            _, top_idx = torch.topk(sim, k = self.num_of_temporal_edge_per_node, dim = -1)
            edge_temporal_src_local = torch.arange(self.num_of_nodes_per_frame, device = frame_patches.device).unsqueeze(1).expand(-1, self.num_of_temporal_edge_per_node).reshape(-1)
            edge_temporal_dst_local = top_idx.reshape(-1)
            edge_temporal_src_local += (t * self.num_of_nodes_per_frame)
            edge_temporal_dst_local += ((t + 1) * self.num_of_nodes_per_frame)
            edge_src_global.append(torch.cat([edge_temporal_src_local, edge_temporal_dst_local]))
            edge_dst_global.append(torch.cat([edge_temporal_dst_local, edge_temporal_src_local]))
        
        return torch.stack([torch.cat(edge_src_global), torch.cat(edge_dst_global)], dim = 0)

    def forward(self, x: torch.Tensor):
        """
        x:          video tensor [B, T, 3, H, W]
        edge_index: graph edges from create_frame_graph (same structure for all videos)
        Returns:    logits [B, 2]
        """
        B = x.shape[0]
        num_nodes_per_video = x.shape[1] * 256  # T * N_patches

        # 1. Extract patch features (ViT)
        features = self.vit(x)  # [B, T*N, D] = [B, num_nodes_per_video, D]

        # 2. Build PyG batch (each video = one graph)
        data_list = []
        for i in range(B):
            edge_index = self.add_temporal_edges(
                features[i],
                edge_src_global = [self.video_spatial_src_edges],
                edge_dst_global = [self.video_spatial_dst_edges],
            )
            data = Data(x=features[i], edge_index=edge_index)
            data_list.append(data)
        batched_data = Batch.from_data_list(data_list)

        x = batched_data.x                  # [B * num_nodes_per_video, D]
        edge_index = batched_data.edge_index
        batch = batched_data.batch

        # 3. GCN layers (local structure)
        for gcn in self.gcn_layers:
            x = gcn(x, edge_index)
            x = F.relu(x)

        # 4. Reshape back to per-video
        x = torch.split(x, split_size_or_sections=num_nodes_per_video)
        x = torch.stack(x, dim=0)           # [B, num_nodes_per_video, D]

        # 5. Min-cut pooling (node reduction, as in paper)
        s = self.assign_net(x)              # [B, num_nodes_per_video, num_clusters]
        adj = to_dense_adj(edge_index, batch=batch)  # [B, num_nodes_per_video, num_nodes_per_video]
        x_pooled, adj_pooled, mincut_loss, ortho_loss = dense_mincut_pool(
            x, adj, s
        )                                   # x_pooled: [B, num_clusters, D]

        # 6. Transformer classifier (global dependencies)
        # Add class token
        cls_tokens = self.class_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, x_pooled), dim=1)     # [B, 1 + num_clusters, D]

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Take class token output
        x = x[:, 0]                         # [B, D]

        # Final classification head
        x = self.norm(x)
        logits = self.classifier(x)         # [B, 2]

        # (Optional: you can return mincut_loss + ortho_loss for training)
        # return logits, mincut_loss, ortho_loss
        return (logits, mincut_loss, ortho_loss)
