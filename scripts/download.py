"""Download CheXpert train.csv and frontal train JPGs (CheXpert-v1.0-small) from Kaggle.
Needs ~/.kaggle/kaggle.json."""
import argparse
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from kaggle import api

DATASET = "ashery/chexpert"


# ==============================
# fetch one file
# ==============================
def fetch(name, dst):
    """Downloads dataset file `name` (e.g. train/patient00001/study1/view1_frontal.jpg) to dst/name.
    Kaggle saves it under its basename, so the folder is made first; small files may arrive zipped."""
    folder = dst / Path(name).parent
    folder.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):  # Kaggle answers 429 when requests come too fast
        try:
            api.dataset_download_file(DATASET, name, path=str(folder), quiet=True)
            break
        except Exception as e:
            if "429" not in str(e) or attempt == 4:
                raise
            time.sleep(2 ** attempt)
    zpath = folder / f"{Path(name).name}.zip"
    if zpath.exists():
        with zipfile.ZipFile(zpath) as z:
            z.extractall(folder)
        zpath.unlink()
    if not (dst / name).exists():
        raise FileNotFoundError(dst / name)


# ==============================
# command line
# ==============================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dst", default="data/chexpert")
    p.add_argument("--n", type=int, default=10, help="number of random frontal train images, one request each (0 = whole dataset as one zip, ~11 GB, much faster)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()

    api.authenticate()
    dst = Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)

    if not a.n:  # per-image requests get rate-limited, so take everything as one zip
        api.dataset_download_files(DATASET, path=str(dst), unzip=True, quiet=False)
        sys.exit(0)

    # ==============================
    # train.csv
    # ==============================
    if not (dst / "train.csv").exists():
        fetch("train.csv", dst)
    df = pd.read_csv(dst / "train.csv")
    # csv paths start with CheXpert-v1.0-small/, the Kaggle files do not
    ids = df[df["Frontal/Lateral"] == "Frontal"].Path.str.split("/", n=1).str[1]
    if a.n:
        ids = ids.sample(a.n, random_state=a.seed)
    todo = [i for i in ids if not (dst / i).exists()]
    print(f"{len(ids)} images selected, {len(todo)} to download")

    # ==============================
    # images
    # ==============================
    def get(image_id):
        try:
            fetch(image_id, dst)
        except Exception as e:  # keep going, report failures at the end
            return image_id, e

    failed = []
    with ThreadPoolExecutor(a.workers) as pool:
        for k, res in enumerate(pool.map(get, todo), 1):
            if res:
                failed.append(res)
            if k % 100 == 0 or k == len(todo):
                print(f"{k}/{len(todo)}", flush=True)
    for image_id, e in failed:
        print(f"FAILED {image_id}: {e}")
    print(f"done: {len(todo) - len(failed)} downloaded, {len(failed)} failed")
    sys.exit(1 if failed else 0)
