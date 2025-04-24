"""
공통 DataLoader 유틸

※ 모든 하이퍼파라미터는 main.ipynb 쪽 ex_dict 에서 넘겨받는다.
   ex_dict 구조(필수):
       ├─ 'Data Config' : {'train': str, 'val': str, 'test': str}
       ├─ 'Batch Size'  : int
       ├─ 'Image Size'  : int
       ├─ 'Hyp'        : dict  ← (lr0, momentum, weight_decay 등)
       └─ 'Num Workers' : int  (옵션, 기본 4)
"""

from __future__ import annotations
import os
from typing import Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import yaml

__all__ = [
    "YOLOTxtDataset",
    "collate_fn_yolo",
    "get_dataloader",
    "build_all_loaders",
]

# --------------------------------------------------
# Dataset
# --------------------------------------------------
class YOLOTxtDataset(Dataset):
    """Dataset for image paths listed in a .txt file (YOLO style)."""

    def __init__(self, txt_path: str, img_size: int, augment: bool = False):
        self.img_size = img_size
        self.augment = augment
        self._cached_labels = None

        with open(txt_path, "r", encoding="utf-8") as f:
            self.img_files = [ln.strip() for ln in f if ln.strip()]

        # basic transforms (you can plug Albumentations etc. if needed)
        self.tfms = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),  # converts to 0‑1 FloatTensor
        ])

    # --------------------------------------------------
    def _label_path(self, img_path: str) -> str:
        # images/aaa.jpg -> labels/aaa.txt  (robust to OS separator)
        base, _ = os.path.splitext(img_path)
        if "images" in base:
            label_path = base.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep) + ".txt"
        else:
            # fallback: sibling directory named labels/
            dir_, name = os.path.split(base)
            label_path = os.path.join(os.path.dirname(dir_), "labels", name + ".txt")
        return label_path

    # --------------------------------------------------
    def __len__(self) -> int:
        return len(self.img_files)

    # --------------------------------------------------
    def __getitem__(self, idx: int):
        img_path = self.img_files[idx]
        img = Image.open(img_path).convert("RGB")
        h0, w0 = img.height, img.width

        # (옵션) Data augmentation 자리 – 필요 시 self.augment 체크 후 적용
        img = self.tfms(img)  # → 3×S×S tensor (0‑1)

        # read label
        label_path = self._label_path(img_path)
        targets: List[List[float]] = []
        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                for ln in f:
                    if ln.strip():
                        cls, cx, cy, bw, bh = map(float, ln.strip().split())
                        targets.append([cls, cx, cy, bw, bh])
        targets = torch.tensor(targets, dtype=torch.float32)  # n×5
        return img, targets
    
    # --------------------------------------------------
    @property
    def labels(self):
        if self._cached_labels is None:
            print("[YOLOTxtDataset] Caching labels for anchor check...")
            label_list = []
            for i in range(len(self)):
                _, labels = self[i]
                label_list.append(labels.cpu().numpy() if torch.is_tensor(labels) else labels)
            self._cached_labels = label_list
        return self._cached_labels


# --------------------------------------------------
# Collate FN
# --------------------------------------------------

def collate_fn_yolo(batch):
    """Stack images (B×3×S×S) / concat targets → (N×6) with batch‑idx column."""
    imgs, labels = zip(*batch)  # tuple(dim=B)
    imgs = torch.stack(imgs, dim=0)

    batch_targets = []
    for bi, lab in enumerate(labels):
        if lab.numel():  # at least 1 box
            bi_col = torch.full((lab.shape[0], 1), float(bi))
            batch_targets.append(torch.cat([bi_col, lab], dim=1))
    if batch_targets:
        batch_targets = torch.cat(batch_targets, dim=0)  # N×6
    else:
        batch_targets = torch.zeros((0, 6), dtype=torch.float32)
    return imgs, batch_targets

# --------------------------------------------------
# Public helpers
# --------------------------------------------------

def get_dataloader(mode: str, ex_dict: dict, shuffle: bool | None = None):
    """mode ∈ {'train','val','test'} -> torch.utils.data.DataLoader"""
    assert mode in ("train", "val", "test"), "mode must be train|val|test"
    yaml_path = ex_dict["Data Config"]
    with open(yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    dataset_dir = os.path.dirname(yaml_path)
    txt_path = os.path.join(dataset_dir, data_cfg[mode])

    batch_size = ex_dict["Batch Size"]
    img_size = ex_dict["Image Size"]
    workers = ex_dict.get("Num Workers", 4)
    augment = bool(ex_dict.get("Augment", False) and mode == "train")

    ds = YOLOTxtDataset(txt_path, img_size, augment=augment)
    if shuffle is None:
        shuffle = (mode == "train")

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        drop_last=(mode == "train"),
        collate_fn=collate_fn_yolo,
    )
    return dl


def build_all_loaders(ex_dict: dict):
    """train_loader, val_loader, test_loader 3‑tuple"""
    return (
        get_dataloader("train", ex_dict),
        get_dataloader("val", ex_dict, shuffle=False),
        get_dataloader("test", ex_dict, shuffle=False),
    )

# --------------------------------------------------
# End of file
# --------------------------------------------------
