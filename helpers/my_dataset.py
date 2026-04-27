import random
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image
import torch

class MyDataset(Dataset):
    def __init__(self, videos_paths, videos_labels, transform=None, aug_prob=0.5, test = False):
        self.videos_paths = videos_paths
        self.videos_labels = videos_labels
        self.transform = transform
        self.aug_prob = aug_prob
        self.test = test
    
    def __len__(self):
        return len(self.videos_paths)

    def __getitem__(self, index):
        frames_paths = self.videos_paths[index]
        label = self.videos_labels[index]

        # decide once per video
        if not self.test:
            aug_apply = random.random() < self.aug_prob
        else:
            aug_apply = False

        # sample params once
        if aug_apply:
            do_flip = random.random() < 0.5
            brightness = random.uniform(0.9, 1.1)
            contrast = random.uniform(0.9, 1.1)
            do_blur = random.random() < 0.3
            sigma = random.uniform(0.1, 1.0)

        frames = []
        for fp in frames_paths:
            img = Image.open(fp).convert("RGB")

            # 🔹 Step 1: Resize FIRST
            img = TF.resize(img, (256, 256))

            # 🔹 Step 2: Augment (same params)
            if aug_apply:
                if do_flip:
                    img = TF.hflip(img)
                img = TF.adjust_brightness(img, brightness)
                img = TF.adjust_contrast(img, contrast)
                if do_blur:
                    img = TF.gaussian_blur(img, kernel_size=3, sigma=sigma)

            # 🔹 Step 3: ToTensor + Normalize
            img = TF.to_tensor(img)
            img = TF.normalize(
                img,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )

            frames.append(img)

        video = torch.stack(frames, dim=0)
        label = torch.tensor(label, dtype=torch.long)

        return video, label