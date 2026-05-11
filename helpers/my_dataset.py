import random
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image
import torch

class MyDataset(Dataset):
    def __init__(self, videos_paths, videos_labels, transform=None, aug_prob=0.65, test=False):
        self.videos_paths = videos_paths
        self.videos_labels = videos_labels
        self.transform = transform
        self.aug_prob = aug_prob          # increased a bit
        self.test = test

    def __len__(self):
        return len(self.videos_paths)

    def __getitem__(self, index):
        frames_paths = self.videos_paths[index]
        label = self.videos_labels[index]

        # Decide augmentation once per video (very important for your graph model)
        aug_apply = (not self.test) and (random.random() < self.aug_prob)

        # Sample augmentation parameters once per video
        if aug_apply:
            do_flip = random.random() < 0.5
            do_rotate = random.random() < 0.4
            angle = random.uniform(-12, 12) if do_rotate else 0

            brightness = random.uniform(0.85, 1.15)
            contrast = random.uniform(0.85, 1.15)
            saturation = random.uniform(0.85, 1.15)
            hue = random.uniform(-0.06, 0.06)

            do_blur = random.random() < 0.25
            sigma = random.uniform(0.3, 0.8)   # much milder than before

        frames = []
        for fp in frames_paths:
            img = Image.open(fp).convert("RGB")

            # === AUGMENTATIONS (geometric first) ===
            if aug_apply:
                if do_flip:
                    img = TF.hflip(img)
                if do_rotate:
                    img = TF.rotate(img, angle)

                # Color jitter
                img = TF.adjust_brightness(img, brightness)
                img = TF.adjust_contrast(img, contrast)
                img = TF.adjust_saturation(img, saturation)
                img = TF.adjust_hue(img, hue)

                if do_blur:
                    img = TF.gaussian_blur(img, kernel_size=3, sigma=sigma)

            # Resize AFTER geometric transforms
            img = TF.resize(img, (224, 224))

            # To tensor + normalize
            img = TF.to_tensor(img)
            img = TF.normalize(img, mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

            frames.append(img)

        video = torch.stack(frames, dim=0)
        label = torch.tensor(label, dtype=torch.long)

        return video, label