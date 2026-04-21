from pathlib import Path
import random
import argparse
from sklearn.model_selection import train_test_split
import tqdm
import numpy as np
import json


def load_real_videos(root, dataset_name, seed=42):
    dataset_path = Path(root, dataset_name)

    if dataset_name == 'FaceForensics++':
        real_videos_folder_name = 'original_sequences'
        dataset_real_videos_path = Path(dataset_path, real_videos_folder_name)
        paths = list(dataset_real_videos_path.glob('**/frames/*'))
    else:
        paths = list(dataset_path.glob('**/frames/*'))

    random.seed(seed)
    random.shuffle(paths)
    return paths


def split_real_videos(real_videos_paths, test_size=0.2, seed=42):
    train_paths, test_paths = train_test_split(
        real_videos_paths, test_size=test_size, shuffle=True, random_state=seed
    )
    return train_paths, test_paths


def load_fake_videos(real_videos_paths, root, dataset_name, seed=42):
    fake_videos_paths = []
    if dataset_name == 'FaceForensics++':
        dataset_path = Path(root, dataset_name)
        fake_videos_folder_name = 'manipulated_sequences'

        dataset_fake_videos_path = Path(dataset_path, fake_videos_folder_name)

        for path in tqdm.tqdm(real_videos_paths):
            video_label = path.stem
            tmp = list(dataset_fake_videos_path.glob(f'**/frames/{video_label}_*'))
            fake_videos_paths.extend(tmp)

    fake_videos_paths.extend(real_videos_paths)              # add real videos too

    random.seed(seed)
    random.shuffle(fake_videos_paths)
    return fake_videos_paths


def load_frames(video_paths, k=8):
    frames_per_video_paths = []
    skip = 0
    for path in tqdm.tqdm(video_paths):
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


def save_dictionary(train_videos_paths, test_videos_paths, save_dir, dataset_name):
    dc = {
        'train': [[str(p) for p in video] for video in train_videos_paths],
        'test': [[str(p) for p in video] for video in test_videos_paths],
    }
    final_path = Path(save_dir) / f"{dataset_name}.json"
    with open(final_path, 'w') as f:
        json.dump(dc, f, indent=2)
    print(f"JSON saved at: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset JSON for deepfake detection")
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory of the datasets (e.g. /home/zahin/Desktop/My_Thesis_Repo/datasets)')
    parser.add_argument('--dataset-name', type=str, required=True,
                        help='Name of the dataset (e.g. FaceForensics++ or UADFV)')
    parser.add_argument('--json-dir', type=str, required=True,
                        help='Directory where the output JSON file will be saved')
    parser.add_argument('--test-size', type=float, required=False, default=0.2,
                        help='Directory where the output JSON file will be saved')

    args = parser.parse_args()


    real_videos_paths = load_real_videos(args.root, args.dataset_name)
    print(f'total real videos found: {len(real_videos_paths)}')

    if args.dataset_name == 'FaceForensics++':
        train_real_videos_paths, test_real_videos_paths = split_real_videos(
            real_videos_paths, test_size=args.test_size
        )
    else:
        train_real_videos_paths = []
        test_real_videos_paths = real_videos_paths

    print(f'total train real videos: {len(train_real_videos_paths)}')
    print(f'total test real videos: {len(test_real_videos_paths)}')

    train_videos_paths = load_fake_videos(train_real_videos_paths, args.root, args.dataset_name)
    test_videos_paths = load_fake_videos(test_real_videos_paths, args.root, args.dataset_name)

    print(f'final total train videos: {len(train_videos_paths)}')
    print(f'final total test videos: {len(test_videos_paths)}')

    train_videos_paths = load_frames(train_videos_paths)
    test_videos_paths = load_frames(test_videos_paths)

    save_dictionary(train_videos_paths, test_videos_paths, args.json_dir, args.dataset_name)
    print('JSON file created successfully!!!')


if __name__ == '__main__':
    main()