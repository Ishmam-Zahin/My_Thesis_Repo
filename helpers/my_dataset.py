from torch.utils.data import Dataset
from PIL import Image
import torch

class MyDataset(Dataset):
    def __init__(self, videos_paths, videos_labels, transform = None):
        super().__init__()
        self.videos_paths = videos_paths
        self.videos_labels = videos_labels
        self.transform = transform
    
    def __len__(self):
        return len(self.videos_paths)
    
    def __getitem__(self, index):
        frames_paths = self.videos_paths[index]
        label = self.videos_labels[index]

        frames = []
        for fp in frames_paths:
            img = Image.open(fp).convert("RGB")
            img_t = self.transform(img) if self.transform else img
            frames.append(img_t)

        video = torch.stack(frames, dim=0)
        label = torch.tensor(label, dtype=torch.long)
        return video, label