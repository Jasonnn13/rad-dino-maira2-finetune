"""Download VinBigData train.csv and train DICOMs from Kaggle.
Needs ~/.kaggle/kaggle.json and the competition rules accepted on kaggle.com."""
import argparse
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from kaggle import api

COMPETITION = "vinbigdata-chest-xray-abnormalities-detection"


# ==============================
# fetch one file
# ==============================
def fetch(name, dst):
    """Downloads a competition file into `dst` and unzips it (Kaggle serves single files zipped)."""
    api.competition_download_file(COMPETITION, name, path=str(dst), quiet=True)
    zpath = dst / f"{Path(name).name}.zip"
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dst)
    zpath.unlink()


# ==============================
# command line
# ==============================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dst", default="data")
    p.add_argument("--n", type=int, default=10, help="number of random train images (0 = all 15,000, ~190 GB)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()

    api.authenticate()
    dst = Path(a.dst)
    dicom = dst / "dicom"
    dicom.mkdir(parents=True, exist_ok=True)

    # ==============================
    # train.csv
    # ==============================
    if not (dst / "train.csv").exists():
        fetch("train.csv", dst)
    ids = pd.read_csv(dst / "train.csv").image_id.drop_duplicates()
    if a.n:
        ids = ids.sample(a.n, random_state=a.seed)
    # skip images already downloaded, or already converted (their DICOM may have been deleted)
    todo = [i for i in ids if not (dicom / f"{i}.dicom").exists() and not (dst / "png" / f"{i}.png").exists()]
    print(f"{len(ids)} images selected, {len(todo)} to download")

    # ==============================
    # dicoms
    # ==============================
    def get(image_id):
        try:
            fetch(f"train/{image_id}.dicom", dicom)
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
