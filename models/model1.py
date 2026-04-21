import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class Vit(nn.Module):
    def __init__(self, vit_name='dinov2_vits14'):
        super().__init__()
        self.vit = torch.hub.load('facebookresearch/dinov2', vit_name)
        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.eval()

    def forward(self, x):
        # Returns only patch tokens: shape (B, N, D)
        # For dinov2_vits14 on 224×224 → N = 256, D = 384
        features = self.vit.forward_features(x)
        return features['x_norm_patchtokens']


class TransformerBlock(nn.Module):
    """Exact pre-norm Transformer block matching the paper (Eq. 6-7)."""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
            add_zero_attn=False,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),                    # paper uses ReLU in GCN; consistent here
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, d_model)
        # Pre-norm + residual exactly as in paper
        attn_input = self.ln1(x)
        attn_out, _ = self.self_attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + attn_out

        mlp_input = self.ln2(x)
        mlp_out = self.mlp(mlp_input)
        x = x + mlp_out
        return x


class My_Model(nn.Module):
    def __init__(
        self,
        vit_name: str = "dinov2_vits14",
        k_neighbour: int = 8,
        num_gcn_layers: int = 3,          # paper ablation best = 3; text mentions "a" but we follow best reported
        num_transformer_blocks: int = 3,  # paper main implementation
        d_model: int = 256,               # exact value from paper §IV-B
        nhead: int = 8,                   # paper
        dim_feedforward: int = 512,       # paper
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vit_name = vit_name
        self.vit = Vit(vit_name=self.vit_name)

        # DINOv2 vits14 patch tokens are 384-dim; project to paper's d_model
        self.vit_dim = 384
        self.d_model = d_model
        self.feature_proj = (
            nn.Linear(self.vit_dim, d_model)
            if self.vit_dim != d_model
            else nn.Identity()
        )

        self.k = k_neighbour

        # ------------------- GCN (paper Eq. 4) -------------------
        self.gcn_layers = nn.ModuleList()
        for _ in range(num_gcn_layers):
            self.gcn_layers.append(GCNConv(d_model, d_model))

        # ------------------- Transformer (paper Eq. 5-10) -------------------
        self.class_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_transformer_blocks)
            ]
        )

        # ------------------- Classifier -------------------
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, 2)  # logits: [real, fake]

    def create_graph_edge_index(
        self,
        patch_features: torch.Tensor,
        k: int | None = None,
    ) -> torch.Tensor:
        """
        Exactly reproduces the paper's undirected K-NN spatial graph (§III-A).
        - Nodes = image patches arranged on a square grid.
        - Edges = K nearest neighbors by spatial (row, col) distance.
        - Symmetrized → undirected graph.
        - No self-loops (GCNConv adds them automatically).
        - Fully batched with node offset.
        """
        if k is None:
            k = self.k

        # Handle single-image input
        if patch_features.dim() == 2:
            patch_features = patch_features.unsqueeze(0)

        B, N, D = patch_features.shape
        device = patch_features.device

        # Grid layout (paper uses fixed 16×16 = 256 patches for 224×224)
        grid_size = int(N ** 0.5)
        if grid_size * grid_size != N:
            raise ValueError(f"Patch count {N} must be perfect square (got {grid_size}×{grid_size})")

        # Patch (row, col) coordinates - identical for every image
        indices = torch.arange(N, device=device)
        rows = indices // grid_size
        cols = indices % grid_size
        positions = torch.stack([rows, cols], dim=1).float()  # (N, 2)

        # Pairwise Euclidean distances
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # (N, N, 2)
        dist_matrix = torch.sqrt((diff ** 2).sum(dim=-1))      # (N, N)
        dist_matrix.fill_diagonal_(float("inf"))               # no self-loops

        # K nearest neighbors (directed)
        _, knn_idx = torch.topk(dist_matrix, k=k, dim=1, largest=False)  # (N, K)

        # Adjacency matrix (bool = 4× memory saving)
        row_idx = torch.arange(N, device=device).unsqueeze(1).expand(-1, k)
        adj = torch.zeros((N, N), dtype=torch.bool, device=device)
        adj[row_idx, knn_idx] = True

        # Symmetrize → undirected graph (Aij = Aji) as required by the paper
        adj = adj | adj.t()

        # PyG edge_index (2, E)
        src, dst = torch.where(adj)
        edge_index_single = torch.stack([src, dst], dim=0)  # (2, E)

        # Batch handling
        if B == 1:
            return edge_index_single.long()

        edge_list = []
        for b in range(B):
            offset = b * N
            edge_list.append(edge_index_single + offset)
        return torch.cat(edge_list, dim=1).long()  # (2, B*E)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W) - typically 224×224, normalized as required by DINOv2
        Returns: logits (B, 2)
        """
        B = x.shape[0]

        # 1. Self-supervised ViT feature extractor (paper §III-B)
        patch_features = self.vit(x)                    # (B, N, vit_dim)
        patch_features = self.feature_proj(patch_features)  # (B, N, d_model)

        # 2. Build graph (paper §III-A)
        edge_index = self.create_graph_edge_index(patch_features)  # (2, B*E)

        # 3. Flatten for GNN
        N = patch_features.shape[1]
        node_features = patch_features.view(B * N, self.d_model)  # (B*N, d_model)

        # 4. Multi-layer GCN (paper Eq. 4)
        h = node_features
        for gcn in self.gcn_layers:
            h = gcn(h, edge_index)
            h = torch.relu(h)

        # 5. Reshape back per image
        h = h.view(B, N, self.d_model)

        # 6. Add learnable class token (standard ViT-style for classification)
        class_tokens = self.class_token.expand(B, -1, -1)  # (B, 1, d_model)
        z = torch.cat([class_tokens, h], dim=1)            # (B, 1+N, d_model)

        # 7. Graph-Transformer blocks (paper Eq. 5-10)
        for block in self.transformer_blocks:
            z = block(z)

        # 8. Classification head (use class token)
        cls_feat = z[:, 0, :]          # (B, d_model)
        cls_feat = self.dropout(cls_feat)
        logits = self.classifier(cls_feat)  # (B, 2)

        return logits