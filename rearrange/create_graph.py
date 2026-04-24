from pathlib import Path
import json
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import numpy as np
import torch.nn.functional as F



class VideoFeatureExtractor(nn.Module):
    """
    Frozen DINOv2 ViT that extracts patch tokens for each frame of a video.
    Input: [B, T, 3, H, W]
    Output: [B, T*N, D] where N=256, D=384 for dinov2_vits14
    """
    def __init__(self, vit_name: str = 'dinov2_vits14'):
        super().__init__()
        self.vit = torch.hub.load('facebookresearch/dinov2', vit_name)
        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.eval()
        self.D = 384  # dinov2_vits14 patch token dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.vit.forward_features(x)
        patch_tokens = features['x_norm_patchtokens']
        return patch_tokens


def load_dataset_json(dataset_json_path):
    with open(dataset_json_path, 'r') as f:
        data =json.load(f)
    
    return (data['train'], data['test'])

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def load_images(video_path, transform):
    frames = []
    for frame_path in video_path:
        img = Image.open(frame_path).convert("RGB")
        img_t = transform(img)
        frames.append(img_t)

    return torch.stack(frames)

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

def add_temporal_edges(
        frame_patches,
        edge_src_global,
        edge_dst_global,
        K = 3,
        T = 8,
        N = 256,
):
    frame_patch_norm = F.normalize(frame_patches, dim = -1)

    for t in range(T - 1):
        patch_curr = frame_patch_norm[t]
        patch_next = frame_patch_norm[t + 1]
        sim = torch.mm(patch_curr, patch_next.t())
        _, top_idx = torch.topk(sim, k = K, dim = -1)
        edge_temporal_src_local = torch.arange(N, device = frame_patches.device).unsqueeze(1).expand(-1, K).reshape(-1)
        edge_temporal_dst_local = top_idx.reshape(-1)
        edge_temporal_src_local += (t * N)
        edge_temporal_dst_local += ((t + 1) * N)
        edge_src_global.append(torch.cat([edge_temporal_src_local, edge_temporal_dst_local]))
        edge_dst_global.append(torch.cat([edge_temporal_dst_local, edge_temporal_src_local]))
    
    return (torch.cat(edge_src_global), torch.cat(edge_dst_global))

def main():
    dataset_json_root = '/home/zahin/Desktop/My_Thesis_Repo/rearrange/dataset_json'
    dataset_name = 'FaceForensics++'
    dataset_json_path = Path(dataset_json_root, dataset_name + '.json')
    train_videos, test_videos = load_dataset_json(dataset_json_path)
    edge_src_global, edge_dst_global = get_spatial_edges()
    transform = get_transforms()

    vit = VideoFeatureExtractor()

    for video in train_videos:
        frames = load_images(video, transform)
        patches = vit(frames)
        edge_temporal_src_global, edge_temporal_dst_global = add_temporal_edges(patches, [edge_src_global], [edge_dst_global])
        print(edge_temporal_src_global.shape)
        print(edge_temporal_dst_global.shape)
        print(edge_temporal_src_global.dtype)
        print(edge_temporal_dst_global.dtype)
        break
    



if __name__ == '__main__':
    main()