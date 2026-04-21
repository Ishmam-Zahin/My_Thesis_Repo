import json
from pathlib import Path
from helpers.my_dataset import MyDataset

def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    train_videos_paths = data['train']
    test_videos_paths = data['test']
    return (train_videos_paths, test_videos_paths)

def determine_labels(path):
    real_words = ['real', 'original']
    fake_words = ['fake', 'manipulated', 'synthesis']
    path_str = str(path).lower()

    for word in real_words:
        if word in path_str:
            return 0
    for word in fake_words:
        if word in path_str:
            return 1
    raise Exception('unable to determine video label')

def get_labels(videos_paths):
    videos_labels = []
    for video in videos_paths:
        label = determine_labels(video[0])
        videos_labels.append(label)
    return videos_labels

def get_dataset(
        json_root,
        dataset_name,
        transform,
):
    json_path = Path(json_root, dataset_name + '.json')
    train_videos_paths, test_videos_paths = load_json(json_path)
    train_videos_labels = get_labels(train_videos_paths)
    test_videos_labels = get_labels(test_videos_paths)
    
    if train_videos_paths:
        train_dataset = MyDataset(train_videos_paths, train_videos_labels, transform = transform)
        test_dataset = MyDataset(test_videos_paths, test_videos_labels, transform = transform)
    else:
        train_dataset = None
        test_dataset = MyDataset(test_videos_paths, test_videos_labels, transform = transform)
    
    return (train_dataset, test_dataset)


def main():
    root = Path('/home/zahin/Desktop/My_Thesis_Repo/rearrange/dataset_json/FaceForensics++.json')
    train_videos_paths, test_videos_paths = load_json(root)
    train_videos_labels = get_labels(train_videos_paths)
    test_videos_labels = get_labels(test_videos_paths)
    

if __name__ == "__main__":
    main()