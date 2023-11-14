from pathlib import Path
import datasets.transforms as T
from .coco import CocoDetection
from .dota_dataset import DotaDataset


def build_dota(image_set, args):
    root = Path(args.dataset_path)
    assert root.exists(), f'provided DOTA path {root} does not exist'
    PATHS = {
        "train": (root / "train/images",root / "annotations" / f'train.json'),
        "val": (root / "val/images", root / "annotations" / f'val.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_dota_transforms(image_set), return_masks=False)
    return dataset



def make_dota_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            # T.RandomV TODO
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')