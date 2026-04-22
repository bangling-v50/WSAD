import os
from enum import Enum
from typing import  List, Tuple, Optional

import PIL
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMG_EXTS = (".png", ".jpg")


class DatasetSplit(Enum):
    TRAIN = "train"
    TEST  = "test"


def _list_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(IMG_EXTS) and os.path.isfile(os.path.join(folder, f))
    ])


def _basename_noext(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


class DatasetLoader(torch.utils.data.Dataset):
    def __init__(
        self,
        source: str,
        imagesize: int = 224,
        split: DatasetSplit = DatasetSplit.TRAIN,
        **kwargs,
    ):
        super().__init__()
        self.source = source
        self.split = split

        self.data_to_iterate: List[Tuple[str, str, Optional[str]]] = self._build_index()

        self.transform_img = transforms.Compose([
            transforms.Resize((imagesize,imagesize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize((imagesize,imagesize), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

        self.imagesize = (3, imagesize, imagesize)
    
    def _find_mask_path(self, mask_dir: str, image_path: str) -> str:
        stem = _basename_noext(image_path)
        candidate_names = [f"{stem}_mask.png"]
        for name in candidate_names:
            p = os.path.join(mask_dir, name)
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(f"No mask found for image: {image_path}")

    def _build_index(self) -> List[Tuple[str, str, Optional[str]]]:
        root = self.source
        items: List[Tuple[str, str, Optional[str]]] = []

        if self.split == DatasetSplit.TRAIN:
            train_dir = os.path.join(root, "train", "good")
            imgs = _list_images(train_dir)
            if len(imgs) == 0:
                raise FileNotFoundError(f"No images found under {train_dir}")
            for p in imgs:
                items.append(("good", p, None))
            return items

        test_dir = os.path.join(root, "test")
        gt_dir   = os.path.join(root, "ground_truth")

        if not os.path.isdir(test_dir):
            raise FileNotFoundError(f"Missing directory: {test_dir}")

        for anomaly in sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]):
            img_dir = os.path.join(test_dir, anomaly)
            img_list = _list_images(img_dir)
            if anomaly.lower() == "good":
                for ip in img_list:
                    items.append(("good", ip, None))
                continue

            mask_dir = os.path.join(gt_dir, anomaly)
            if not os.path.isdir(mask_dir):
                raise FileNotFoundError(f"Missing mask directory for anomaly class: {mask_dir}")

            for ip in img_list:
                matched_mask = self._find_mask_path(mask_dir, ip)
                items.append((anomaly, ip, matched_mask))

        if len(items) == 0:
            raise FileNotFoundError(f"No test images found under {test_dir}")
        return items

    def __len__(self) -> int:
        return len(self.data_to_iterate)

    def __getitem__(self, idx: int):
        anomaly, image_path, mask_path = self.data_to_iterate[idx]

        img = PIL.Image.open(image_path).convert("RGB")
        img = self.transform_img(img)

        if anomaly.lower() != "good":
            m = PIL.Image.open(mask_path).convert("L")
            m = self.transform_mask(m)
            mask = (m > 0.5).float()
        else:
            mask = torch.zeros([1, *img.size()[1:]], dtype=torch.float32)

        sample = {
            "image": img,
            "mask": mask,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly.lower() != "good"),
            "image_name": os.path.relpath(image_path, self.source),
            "image_path": image_path,
        }
        return sample
