import torch
from torch import nn
from torchvision import transforms
import numpy
import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from torchinfo import summary
from tqdm import tqdm

def get_transforms():
    transform = transforms.Compose([
        transforms.Resize(size = (224, 224)),
        transforms.ToTensor(),
    ])
    return transform

def load_paths(root):
    paths = list(root.glob('**/frames/*/*.png'))
    paths.sort()
    return paths

def load_image(path):
    img = Image.open(path).convert('RGB')
    return img

def load_vit(vit_name):
    dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', vit_name)
    return dinov2_vits14

def create_adjacency_matrix(
    patch_features: torch.Tensor, 
    k: int = 8
) -> torch.Tensor:
    """
    Creates the undirected adjacency matrix for the deepfake graph 
    exactly as described in the paper (Section III.A - Deepfake Graph Representation).
    
    - Each patch is a node.
    - Nodes are connected based on **spatial proximity** (K nearest neighbors in the 2D grid).
    - The graph is undirected → the matrix is symmetrized.
    - Self-loops are NOT included here (they are added later in the GCN as Ã = A + I).
    
    Args:
        patch_features (torch.Tensor): Shape (N, D) - the patch token features 
                                       returned by your ViT (e.g. dinov2_vits14 'x_norm_patchtokens').
        k (int): Number of nearest neighbors (default 8, as used in the best ablation result 
                 in Table 7 of the paper).
    
    Returns:
        torch.Tensor: Adjacency matrix A of shape (N, N) with values 0 or 1 (float32).
    """
    N = patch_features.shape[0]
    device = patch_features.device
    dtype = torch.float32

    # ------------------------------------------------------------------
    # 1. Determine the grid layout (paper uses 16×16 = 256 patches)
    # ------------------------------------------------------------------
    grid_size = int(N ** 0.5)
    if grid_size * grid_size != N:
        raise ValueError(
            f"Number of patches ({N}) must be a perfect square for the ViT grid layout. "
            f"Got grid_size = {grid_size}."
        )

    # ------------------------------------------------------------------
    # 2. Compute (row, col) position for every patch (0-based)
    # ------------------------------------------------------------------
    indices = torch.arange(N, device=device)
    rows = indices // grid_size
    cols = indices % grid_size
    positions = torch.stack([rows, cols], dim=1).float()   # (N, 2)

    # ------------------------------------------------------------------
    # 3. Pairwise Euclidean distances between patch centers
    # ------------------------------------------------------------------
    # (N, 1, 2) - (1, N, 2) → (N, N, 2)
    diff = positions.unsqueeze(1) - positions.unsqueeze(0)
    dist_matrix = torch.sqrt(torch.sum(diff ** 2, dim=-1))   # (N, N)

    # Exclude self-loops (distance to self = 0)
    dist_matrix.fill_diagonal_(float('inf'))

    # ------------------------------------------------------------------
    # 4. Find K nearest neighbors for each patch
    # ------------------------------------------------------------------
    # topk returns the smallest distances (nearest)
    _, knn_idx = torch.topk(dist_matrix, k=k, dim=1, largest=False)   # (N, K)

    # ------------------------------------------------------------------
    # 5. Build adjacency matrix (directed k-NN first)
    # ------------------------------------------------------------------
    adj = torch.zeros((N, N), dtype=dtype, device=device)

    # Efficient scatter: set A[i, knn_idx[i]] = 1 for every row i
    row_idx = torch.arange(N, device=device).unsqueeze(1).expand(-1, k)   # (N, K)
    adj[row_idx, knn_idx] = 1.0

    # ------------------------------------------------------------------
    # 6. Symmetrize → make it undirected (Aij = Aji) as required by the paper
    # ------------------------------------------------------------------
    adj = (adj + adj.t() > 0).float()

    return adj


def get_save_path(original_path: Path, model_name: str) -> Path:
    parts = list(original_path.parts)

    # Replace "frames" with "matrix/<model_name>"
    new_parts = []
    for part in parts:
        if part == "frames":
            new_parts.append("matrix")
            new_parts.append(model_name)
        else:
            new_parts.append(part)

    new_path = Path(*new_parts)

    # Replace .png → .pth
    new_path = new_path.with_suffix('.pth')

    # Ensure directory exists
    new_path.parent.mkdir(parents=True, exist_ok=True)

    return new_path


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'----------------- current device is: {device} ---------------------------')
    parser = argparse.ArgumentParser(description = 'matrix_creation script')
    parser.add_argument('--path', type = str, required = True, help = 'enter the dataset path')
    parser.add_argument('--vit-name', type = str, required = True, help = 'enter the vit name')
    args = parser.parse_args()
    root = Path(args.path)
    vit_name = args.vit_name
    paths = load_paths(root)
    transform = get_transforms()
    vit = load_vit(vit_name)
    vit.to(device)
    vit.eval()

    for path in tqdm(paths, desc="Processing images"):
        img = load_image(path)
        img_t = transform(img)
        img_t = img_t.unsqueeze(0)
        img_t = img_t.to(device)
        with torch.inference_mode():
            features = vit.forward_features(img_t)['x_norm_patchtokens'].cpu().squeeze()
        adj = create_adjacency_matrix(features)
        save_path = get_save_path(path, vit_name)
        torch.save(adj, save_path)
    

    print('finished!!!')





if __name__ == '__main__':
    main()
