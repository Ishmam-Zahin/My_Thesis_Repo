from pathlib import Path
import random
import argparse
from sklearn.model_selection import train_test_split
import tqdm
import numpy as np
import json

def load_frames(video_paths, k=8):
    frames_per_video_paths = []
    skip = 0
    for i, path in enumerate(tqdm.tqdm(video_paths)):
        frames = sorted(path.glob('*'), key=lambda x: int(x.stem))
        total = len(frames)
        if total < k:
            skip += 1
            continue
        indexes = np.linspace(0, total - 1, k)
        indexes = np.round(indexes).astype(np.int16)
        selected = [frames[i] for i in indexes]
        frames_per_video_paths.append(selected)
    print(f'total video skipped due to less frames: {skip}')
    return frames_per_video_paths

def save_dictionary(train_videos_paths, test_videos_paths, save_dir, dataset_name, labels):
    dc = {
        'train': [[str(p) + f'+{labels[i]}' for p in video] for i, video in enumerate(train_videos_paths)],
        'test': [[str(p) + f'+{labels[i]}' for p in video] for i, video in enumerate(test_videos_paths)],
    }
    final_path = Path(save_dir) / f"{dataset_name}.json"
    with open(final_path, 'w') as f:
        json.dump(dc, f, indent=2)
    print(f"JSON saved at: {final_path}")


def get_labels(video_paths, metadata):
    labels = []
    for video in video_paths:
        video_name = video.name + '.mp4'
        desc = metadata.get(video_name, None)
        if desc is None:
            raise Exception('video not found in metadata')
        labels.append(int(desc['is_fake']))
    return labels

def main():
    dataset_name = 'DFDC'
    dataset_root = '/home/cse/Desktop/zahin_thesis_work/My_Thesis_Repo/datasets'
    dataset_path = Path(dataset_root, dataset_name, 'test')
    metadata_path = Path(dataset_path, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    video_paths = list(dataset_path.glob('**/frames/*'))
    labels = get_labels(video_paths, metadata)
    frames_per_video_paths = load_frames(video_paths)
    save_dir = '/home/cse/Desktop/zahin_thesis_work/My_Thesis_Repo/rearrange/dataset_json'
    save_dictionary([], frames_per_video_paths, save_dir, dataset_name, labels)
    
    print('finished')
    



if __name__ == '__main__':
    main()