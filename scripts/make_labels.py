"""CheXpert train.csv -> image-level multi-label table with a patient-level train/val/test split."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# ==============================
# arguments
# ==============================
p = argparse.ArgumentParser()
p.add_argument("--csv", default="data/chexpert/train.csv")
p.add_argument("--img-dir", help="if set, keep only images that exist here (for subset runs)")
p.add_argument("--out", default="data/chexpert/labels.csv")
p.add_argument("--uncertain", type=int, choices=[0, 1], default=0, help="label given to CheXpert's uncertain (-1) findings")
p.add_argument("--val-frac", type=float, default=0.1, help="fraction of all patients, used to pick the best epoch")
p.add_argument("--test-frac", type=float, default=0.1, help="fraction of all patients, evaluated once at the end")
p.add_argument("--n", type=int, default=0, help="keep this many images in total (0 = all)")
p.add_argument("--balance", action="store_true", help="with --n: pick the train images to even out the class counts")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

# ==============================
# image-level labels
# ==============================
df = pd.read_csv(a.csv)
df = df[df["Frontal/Lateral"] == "Frontal"]  # RAD-DINO is a frontal-view model
df["image_id"] = df.Path.str.split("/", n=1).str[1]  # relative to the download folder, like the Kaggle file names
if a.img_dir:
    df = df[[(Path(a.img_dir) / i).exists() for i in df.image_id]]

classes = [c for c in df.columns[5:19] if c != "No Finding"]  # the 13 findings
y = df.set_index("image_id")[classes].replace(-1, a.uncertain).fillna(0).astype(int)  # blank = not mentioned

# ==============================
# train/val/test split
# ==============================
# split by patient so one patient's images never sit on both sides of a split
patient = y.index.str.extract(r"(patient\d+)", expand=False)
by_patient = y.groupby(patient.values).max()


def split(idx, frac):
    """Multilabel-stratified split of patient positions `idx`; returns (kept, held_out)."""
    s = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=frac, random_state=a.seed)
    kept, held = next(s.split(by_patient.values[idx], by_patient.values[idx]))
    return idx[kept], idx[held]


idx = np.arange(len(by_patient))
idx, test_idx = split(idx, a.test_frac) if a.test_frac else (idx, idx[:0])
_, val_idx = split(idx, a.val_frac / (1 - a.test_frac))  # fraction of the rest, so val = val_frac of all
by_patient["split"] = "train"
by_patient.iloc[val_idx, by_patient.columns.get_loc("split")] = "val"
by_patient.iloc[test_idx, by_patient.columns.get_loc("split")] = "test"
y.insert(0, "split", by_patient.split.reindex(patient.values).values)

# ==============================
# subsample
# ==============================
def balanced(Y, n, rng):
    """Positions of n rows of Y: repeatedly add a random unpicked row positive for the class with the fewest
    positives so far. Classes that run out of positives drop out, so the rarest stay below the rest."""
    queue = [list(rng.permutation(np.flatnonzero(Y[:, c]))) for c in range(Y.shape[1])]
    picked, count, live = np.zeros(len(Y), bool), np.zeros(Y.shape[1]), np.ones(Y.shape[1], bool)
    while picked.sum() < n and live.any():
        c = np.flatnonzero(live)[count[live].argmin()]
        while queue[c] and picked[queue[c][-1]]:
            queue[c].pop()
        if not queue[c]:
            live[c] = False
            continue
        i = queue[c].pop()
        picked[i] = True
        count += Y[i]
    return np.flatnonzero(picked)


if a.n:  # val and test stay at their natural class mix, so their metrics reflect the real data
    rng = np.random.default_rng(a.seed)
    size = {"val": round(a.n * a.val_frac), "test": round(a.n * a.test_frac)}
    size["train"] = a.n - size["val"] - size["test"]
    keep = []
    for name, k in size.items():
        rows = np.flatnonzero(y.split == name)
        k = min(k, len(rows))
        keep.append(rows[balanced(y.iloc[rows, 1:].values, k, rng)] if name == "train" and a.balance
                    else rng.choice(rows, k, replace=False))
    y = y.iloc[np.sort(np.concatenate(keep))]

# ==============================
# write
# ==============================
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
y.rename_axis("image_id").reset_index().to_csv(a.out, index=False)
print(f"{len(y)} images, {(y.iloc[:, 1:].sum(axis=1) == 0).mean():.0%} with no positive label")
print(y.groupby("split").sum().T)
print(y.split.value_counts().to_dict())
