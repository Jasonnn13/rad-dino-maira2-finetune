from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

# ==============================
# constants
# ==============================
MEAN, STD = 0.5307, 0.2583  # from the model's preprocessor_config.json


# ==============================
# dataset
# ==============================
class CXR(Dataset):
    def __init__(self, df, img_dir, classes, res, train):
        self.ids = df.image_id.tolist()
        self.y = df[classes].values.astype(np.float32)
        self.img_dir, self.res = img_dir, res
        # no horizontal flip: laterality matters in chest X-rays
        self.aug = v2.Compose([
            v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
        ]) if train else None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = cv2.imread(f"{self.img_dir}/{self.ids[i]}", cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.res, self.res), interpolation=cv2.INTER_CUBIC)
        x = torch.from_numpy(img).float().div(255).unsqueeze(0)
        if self.aug:
            x = self.aug(x)
        return ((x - MEAN) / STD).expand(3, -1, -1), torch.from_numpy(self.y[i])


# ==============================
# loaders
# ==============================
def make_loaders(labels_csv, img_dir, res, bs, workers, pin_memory):
    """Returns (train_dl, val_dl, test_dl or None, class_names) from a labels.csv made by make_labels.py."""
    df = pd.read_csv(labels_csv)
    classes = [c for c in df.columns if c not in ("image_id", "split")]
    missing = [i for i in df.image_id if not Path(f"{img_dir}/{i}").exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(df)} images in {labels_csv} have no image in {img_dir} (e.g. {missing[:3]}). "
            f"Download them, or rerun make_labels.py with --img-dir to keep only downloaded images.")

    def loader(split, train):
        rows = df[df.split == split]
        if rows.empty:
            return None
        return DataLoader(CXR(rows, img_dir, classes, res, train), batch_size=bs, shuffle=train,
                          drop_last=train, num_workers=workers, pin_memory=pin_memory)

    return loader("train", True), loader("val", False), loader("test", False), classes
