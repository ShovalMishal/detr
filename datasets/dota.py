from pathlib import Path
import datasets.transforms as T
from dota_dataset import DotaDataset


def build_dota(image_set, args):
    root = Path(args.dataset_path)
    assert root.exists(), f'provided DOTA path {root} does not exist'
    PATHS = {
        "train": (root / "train/images", root / "train/annfiles"),
        "val": (root / "val/images", root / "val/annfiles"),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = DotaDataset(img_folder, ann_file, transform=make_dota_transforms(image_set))
    return dataset

def make_dota_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')