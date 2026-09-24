"""DICOM -> 8-bit PNG, following the RAD-DINO model card: resize shorter side, min-max scale to [0, 255]."""
import argparse
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pydicom
from pydicom.pixels import apply_modality_lut


# ==============================
# convert one image
# ==============================
def convert(job):
    src, dst, size, delete = job
    ds = pydicom.dcmread(src)
    img = apply_modality_lut(ds.pixel_array, ds).astype(np.float32)
    if ds.PhotometricInterpretation == "MONOCHROME1":  # inverted polarity
        img = img.max() - img
    h, w = img.shape
    s = size / min(h, w)
    img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_CUBIC)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
    if not cv2.imwrite(str(dst), img.astype(np.uint8)):
        raise OSError(f"could not write {dst}")
    if delete:  # only reached once the PNG is on disk
        src.unlink()


# ==============================
# command line
# ==============================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="folder of .dicom/.dcm files")
    p.add_argument("--dst", default="data/png")
    p.add_argument("--size", type=int, default=518, help="shorter-side size")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--delete-dicom", action="store_true",
                   help="delete each DICOM once its PNG is written (irreversible: reprocessing needs a re-download)")
    a = p.parse_args()

    dst = Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted([*Path(a.src).glob("*.dicom"), *Path(a.src).glob("*.dcm")])
    png = lambda f: dst / f"{f.stem}.png"
    jobs = [(f, png(f), a.size, a.delete_dicom) for f in files if not png(f).exists()]
    print(f"{len(files)} DICOMs, {len(jobs)} to convert")
    if a.delete_dicom:  # leftovers whose PNG already exists from an earlier run
        stale = [f for f in files if png(f).exists() and png(f).stat().st_size > 0]
        for f in stale:
            f.unlink()
        print(f"deleted {len(stale)} DICOMs that were already converted")
    with Pool(a.workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(convert, jobs, chunksize=16), 1):
            if i % 500 == 0:
                print(f"{i}/{len(jobs)}")
