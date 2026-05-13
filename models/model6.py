import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn.dense import dense_mincut_pool


def get_spatial_edges(T=8, N=256, K=8):
    """Precompute spatial KNN edges once (same for all videos)"""
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
    return torch.cat(src_global), torch.cat(dst_global)


class VideoFeatureExtractor(nn.Module):
    def __init__(self, vit_name='dinov2_vits14', feature_dim=384, weight_path=None):
        super().__init__()
        self.vit = torch.hub.load('facebookresearch/dinov2', vit_name)
        if weight_path is not None:
            ckpt = torch.load(weight_path, map_location='cpu', weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            vit_state_dict = {k.replace("vit.", "", 1): v for k, v in state_dict.items() if k.startswith("vit.")}
            missing, unexpected = self.vit.load_state_dict(vit_state_dict, strict=False)

        for p in self.vit.parameters():
            p.requires_grad = False
        for block in self.vit.blocks[-2:]:
            for p in block.parameters():
                p.requires_grad = True
        for name, module in self.vit.named_modules():
            if isinstance(module, nn.LayerNorm):
                for p in module.parameters():
                    p.requires_grad = True

        self.vit.eval()
        self.D = feature_dim

    def forward(self, x: torch.Tensor):
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        features = self.vit.forward_features(x_flat)
        cls_tokens = features['x_norm_clstoken']
        patch_tokens = features['x_norm_patchtokens']
        return {
            'cls': cls_tokens.reshape(B, T, self.D),
            'patch': patch_tokens.reshape(B, T * 256, self.D)
        }


class FusedModel(nn.Module):
    """
    Final Fused Model with MinCut losses returned + Focal Loss support
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

        # BiLSTM Path
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

        # Graph Path
        src, dst = get_spatial_edges(T=num_of_frames, N=256, K=8)
        self.register_buffer('spatial_src', src)
        self.register_buffer('spatial_dst', dst)

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

        self.class_token_graph = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.graph_pos_embed = nn.Parameter(torch.zeros(1, 1 + num_clusters, feature_dim))
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
        patch_features = feats['patch']

        # BiLSTM Path
        cls_features = cls_features + self.temporal_pos_embed
        lstm_out, _ = self.bilstm(cls_features)
        bilstm_feat = lstm_out.mean(dim=1)

        # Graph Path
        data_list = []
        for i in range(B):
            edge_index = self._add_temporal_edges(patch_features[i])
            data = Data(x=patch_features[i], edge_index=edge_index)
            data_list.append(data)

        batched = Batch.from_data_list(data_list)
        x_g = batched.x
        edge_index = batched.edge_index
        batch_idx = batched.batch

        for gcn in self.gcn_layers:
            x_g = gcn(x_g, edge_index)
            x_g = F.relu(x_g)

        x_g = torch.split(x_g, split_size_or_sections=patch_features.shape[1])
        x_g = torch.stack(x_g, dim=0)

        s = self.assign_net(x_g)
        adj = to_dense_adj(edge_index, batch=batch_idx)
        x_pooled, _, mincut_loss, ortho_loss = dense_mincut_pool(x_g, adj, s)

        cls_token = self.class_token_graph.expand(B, 1, -1)
        x_g = torch.cat([cls_token, x_pooled], dim=1)
        x_g = x_g + self.graph_pos_embed

        for block in self.transformer_blocks:
            x_g = block(x_g)

        graph_feat = x_g[:, 0]

        # ASIF fusion
        combined = torch.cat([bilstm_feat, graph_feat], dim=1)
        logits = self.fusion_layer(combined)

        return logits, mincut_loss, ortho_loss

    def _add_temporal_edges(self, frame_patches):
        patches = frame_patches.view(self.num_of_frames, 256, -1)
        frame_patch_norm = F.normalize(patches, dim=-1)
        edge_src, edge_dst = [], []
        for t in range(self.num_of_frames - 1):
            sim = torch.mm(frame_patch_norm[t], frame_patch_norm[t + 1].t())
            _, top_idx = torch.topk(sim, k=4, dim=-1)
            src_local = torch.arange(256, device=frame_patches.device).unsqueeze(1).expand(-1, 4).reshape(-1)
            dst_local = top_idx.reshape(-1)
            offset_curr = t * 256
            offset_next = (t + 1) * 256
            edge_src.append(torch.cat([src_local + offset_curr, dst_local + offset_next]))
            edge_dst.append(torch.cat([dst_local + offset_next, src_local + offset_curr]))
        temporal_edge = torch.stack([torch.cat(edge_src), torch.cat(edge_dst)], dim=0)
        spatial_edge = torch.stack([self.spatial_src, self.spatial_dst], dim=0).to(temporal_edge.device)
        return torch.cat([spatial_edge, temporal_edge], dim=1)