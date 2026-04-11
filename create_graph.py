import torch


def create_frame_graph(
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
        temporal_src = []
        temporal_dst = []

        for t in range(T - 1):
            for p in range(N):
                node_curr = t * N + p
                node_next = (t + 1) * N + p
                temporal_src.extend([node_curr, node_next])
                temporal_dst.extend([node_next, node_curr])

        if temporal_src:
            temporal_src = torch.tensor(temporal_src, dtype=torch.long)
            temporal_dst = torch.tensor(temporal_dst, dtype=torch.long)
            src_global.append(temporal_src)
            dst_global.append(temporal_dst)
        src_global = torch.cat(src_global, dim = 0)
        dst_global = torch.cat(dst_global,dim = 0)
        edge_indexes = torch.stack([src_global, dst_global], dim = 0)

        return edge_indexes