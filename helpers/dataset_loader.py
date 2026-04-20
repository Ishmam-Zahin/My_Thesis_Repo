from pathlib import Path


def load_paths(dataset_name, root):
    dataset_path = Path(root, dataset_name)
    video_paths = list(dataset_path.glob('**/frames/*'))
    print(len(video_paths))
    print(video_paths[:10])

def main():
    root = Path('/home/cse/Desktop/zahin_thesis_work/My_Thesis_Repo/datasets')
    name = 'FaceForensics++'
    load_paths(name, root)





if __name__ == "__main__":
    main()