import torch


def get_spatial_edges(
        T = 8,
        N = 256,
        K = 8,
):
    grid_size = int(N ** 0.5)
    if grid_size * grid_size != N:
        raise ValueError(f"Patch count {N} must be perfect square (got {grid_size}×{grid_size})")
    indices = torch.arange(N)
    rows = indices // grid_size
    cols = indices % grid_size
    positions = torch.stack([rows, cols], dim=1).float()
    diff = positions.unsqueeze(1) - positions.unsqueeze(0)
    dist_matrix = torch.sqrt((diff ** 2).sum(dim=-1))
    dist_matrix.fill_diagonal_(float("inf"))
    _, knn_idxs = torch.topk(dist_matrix, k = K, dim = -1, largest = False)
    src_global = []
    dst_global = []
    for i in range(T):
        row_idx = torch.arange(N).unsqueeze(1).expand(-1, K)
        adj = torch.zeros((N, N), dtype=torch.bool)
        adj[row_idx, knn_idxs] = True
        adj = adj | adj.t()
        src, dst = torch.where(adj)
        offset = i * N
        src = src + offset
        dst = dst + offset
        src_global.append(src)
        dst_global.append(dst)
    
    return (torch.cat(src_global), torch.cat(dst_global))
