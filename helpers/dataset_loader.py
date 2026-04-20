from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import random


def load_video_paths(dataset_name, root):
    dataset_path = Path(root, dataset_name)
    video_paths = list(dataset_path.glob('**/frames/*'))
    return video_paths

def load_frames(video_paths, k = 8): # k = number of frames per video
    frames_per_video_paths = []
    for path in video_paths:
        frames = sorted(path.glob('*'), key = lambda x: int(x.stem))
        total = len(frames)
        indexes = np.linspace(0, total - 1, k)
        indexes = np.round(indexes).astype(np.int16)
        selected = [frames[i] for i in indexes]
        frames_per_video_paths.append(selected)
    return frames_per_video_paths

def determine_video_label(path):
    path = str(path).lower()
    real_words = ['real', 'original']
    fake_words = ['fake', 'manipulated', 'synthesis']

    for word in real_words:
        if word in path:
            return 0
    for word in fake_words:
        if word in path:
            return 1
    raise Exception('unknown label found!!!')

def video_groupping(video_paths, dataset_name):
    groupped_videos_paths = dict()
    for path in video_paths:
        video_label = determine_video_label(path)
        items = path.parts
        index = items.index('frames')
        group_label_index = index - 1
        if dataset_name == 'FaceForensics++':
            group_label_index -= 1
        group_label = items[group_label_index]
        if group_label not in groupped_videos_paths:
            groupped_videos_paths[group_label] = [path]
        else:
            groupped_videos_paths[group_label].append(path)
    return groupped_videos_paths

def split_dataset(groupped_videos_paths, test_size = 0.2, seed = 42):
    X_train = []
    X_test = []
    for label in groupped_videos_paths:
        paths = np.array(groupped_videos_paths[label])
        X_train_local, X_test_local = train_test_split(
            paths,
            test_size = test_size,
            random_state = seed,
            shuffle = True,
        )
        X_train.extend(X_train_local)
        X_test.extend(X_test_local)
    random.seed(seed)
    random.shuffle(X_train)
    random.shuffle(X_test)
    return (X_train, X_test)

def main():
    root = Path('/home/cse/Desktop/zahin_thesis_work/My_Thesis_Repo/datasets')
    name = 'FaceForensics++'
    video_paths = load_video_paths(name, root)
    groupped_videos_paths = video_groupping(video_paths, name)
    X_train, X_test = split_dataset(groupped_videos_paths)
    X_train = load_frames(X_train)
    X_test = load_frames(X_test)
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    print(X_train.shape)
    print(X_test.shape)
    print(X_train[:5])







if __name__ == "__main__":
    main()