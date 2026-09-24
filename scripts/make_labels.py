"""VinBigData train.csv (one row per box) -> image-level multi-label table with train/val/test split."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

NO_FINDING = 14

# ==============================
# arguments
# ==============================
p = argparse.ArgumentParser()
p.add_argument("--csv", default="data/train.csv")
p.add_argument("--img-dir", help="if set, keep only images that have a PNG here (for subset runs)")
p.add_argument("--out", default="data/labels.csv")
p.add_argument("--min-votes", type=int, default=2, help="radiologists (of 3) that must mark a class")
p.add_argument("--val-frac", type=float, default=0.1, help="fraction of all images, used to pick the best epoch")
p.add_argument("--test-frac", type=float, default=0.1, help="fraction of all images, evaluated once at the end")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

# ==============================
# image-level labels
# ==============================
df = pd.read_csv(a.csv)
names = df[df.class_id != NO_FINDING].drop_duplicates("class_id").set_index("class_id").class_name.sort_index()
ids = sorted(df.image_id.unique())
if a.img_dir:
    have = {f.stem for f in Path(a.img_dir).glob("*.png")}
    ids = [i for i in ids if i in have]

votes = df[df.class_id != NO_FINDING].groupby(["image_id", "class_id"]).rad_id.nunique().unstack(fill_value=0)
y = (votes >= a.min_votes).astype(int).reindex(index=ids, columns=names.index, fill_value=0)
y.columns = names.values

# ==============================
# train/val/test split
# ==============================
def split(idx, frac):
    """Multilabel-stratified split of row positions `idx`; returns (kept, held_out)."""
    s = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=frac, random_state=a.seed)
    kept, held = next(s.split(y.values[idx], y.values[idx]))
    return idx[kept], idx[held]


idx = np.arange(len(y))
idx, test_idx = split(idx, a.test_frac) if a.test_frac else (idx, idx[:0])
_, val_idx = split(idx, a.val_frac / (1 - a.test_frac))  # fraction of the rest, so val = val_frac of all
y.insert(0, "split", "train")
y.iloc[val_idx, 0] = "val"
y.iloc[test_idx, 0] = "test"

# ==============================
# write
# ==============================
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
y.rename_axis("image_id").reset_index().to_csv(a.out, index=False)
print(f"{len(y)} images, {(y.iloc[:, 1:].sum(axis=1) == 0).mean():.0%} with no positive label")
print(y.groupby("split").sum().T)
print(y.split.value_counts().to_dict())
