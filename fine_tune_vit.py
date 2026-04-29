import torch
from pathlib import Path
import json
import tqdm
from sklearn.model_selection import train_test_split
from torch import nn
import random
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image


def load_json(path):
    with open(path) as f:
        data = json.load(f)
    
    return (data.get('train', []), data.get('test', []))

def determine_labels(path):
    real_words = ['real', 'original']
    fake_words = ['fake', 'manipulated', 'synthesis']
    path_str = str(path).lower()

    if 'dfdc' in path_str:
        label = int(path_str.split('+')[1])
        return label

    for word in real_words:
        if word in path_str:
            return 0
    for word in fake_words:
        if word in path_str:
            return 1
    raise Exception('unable to determine video label')

def process_videos(videos, imgs, labels, is_dfdc = False):
    for video in tqdm.tqdm(videos):
        for frame in video:
            label = determine_labels(frame)
            if is_dfdc:
                frame = Path(str(frame).split('+')[0])
            imgs.append(frame)
            labels.append(label)

def split_videos(videos):
    train, test = train_test_split(videos, test_size = 0.2, random_state = 42, shuffle = True)
    
    return (train, test)



class MyDataset(Dataset):
    def __init__(self, img_paths, img_labels, transform=None, aug_prob=0.5, test = False):
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.transform = transform
        self.aug_prob = aug_prob
        self.test = test
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        frames_path = self.img_paths[index]
        label = self.img_labels[index]

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

        img = Image.open(frames_path).convert("RGB")

        # 🔹 Step 1: Resize FIRST
        img = TF.resize(img, (224, 224))

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
            
        label = torch.tensor(label, dtype=torch.long)

        return img, label





class VideoFeatureExtractor(nn.Module):
    """
    Frozen DINOv2 ViT that extracts patch tokens for each frame of a video.
    Input: [B, T, 3, H, W]
    Output: [B, T*N, D] where N=256, D=384 for dinov2_vits14
    """
    def __init__(self, vit_name='dinov2_vits14', feature_dim=384, total_nodes=256):
        super().__init__()
        self.vit = torch.hub.load('facebookresearch/dinov2', vit_name)

        # Freeze everything first
        for p in self.vit.parameters():
            p.requires_grad = False

        # Unfreeze last 2 transformer blocks
        if hasattr(self.vit, "blocks"):
            for block in self.vit.blocks[-2:]:
                for p in block.parameters():
                    p.requires_grad = True

        # Usually also fine to unfreeze the final norm
        if hasattr(self.vit, "norm"):
            for p in self.vit.norm.parameters():
                p.requires_grad = True

        self.D = feature_dim
        self.total_nodes = total_nodes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)  # [B*T, 3, 224, 224]
        features = self.vit.forward_features(x_flat)
        patch_tokens = features['x_norm_patchtokens']  # [B*T, N=256, D=384]
        # Reshape back to per-video
        return patch_tokens.reshape(B, T * self.total_nodes, self.D)  # [B, T*N, D]
    

def main():
    ff_json_path = Path('/home/cse/Desktop/zahin_thesis_work/My_Thesis_Repo/rearrange/dataset_json/FaceForensics++.json')
    celeb_json_path = Path('/home/cse/Desktop/zahin_thesis_work/My_Thesis_Repo/rearrange/dataset_json/Celeb-DF-v2.json')
    dfdc_json_path = Path('/home/cse/Desktop/zahin_thesis_work/My_Thesis_Repo/rearrange/dataset_json/DFDC.json')
    train_imgs = []
    train_labels = []
    test_imgs = []
    test_labels = []

    train_videos, test_videos = load_json(ff_json_path)
    process_videos(train_videos, train_imgs, train_labels)
    process_videos(test_videos, test_imgs, test_labels)
    print('finished ff++')
    
    _, test_videos = load_json(celeb_json_path)
    train_videos, test_videos = split_videos(test_videos)
    process_videos(train_videos, train_imgs, train_labels)
    process_videos(test_videos, test_imgs, test_labels)
    print('finished celeb')

    _, test_videos = load_json(dfdc_json_path)
    train_videos, test_videos = split_videos(test_videos)
    process_videos(train_videos, train_imgs, train_labels, is_dfdc = True)
    process_videos(test_videos, test_imgs, test_labels, is_dfdc = True)
    print('finished dfdc')

    train_dataset = MyDataset(train_imgs, train_labels, test = False)
    test_dataset = MyDataset(test_imgs, test_labels, test = True)

    weight_save_path = Path('/home/cse/Desktop/zahin_thesis_work/My_Thesis_Repo/vit_weights')

    EPOCHS = 10
    BATCH_SIZE = 32
    LR = 0.00001


    



if __name__ == '__main__':
    main()