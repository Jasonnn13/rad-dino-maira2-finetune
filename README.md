# RAD-DINO-MAIRA-2 fine-tuning on VinBigData

Fine-tunes [`microsoft/rad-dino-maira-2`](https://huggingface.co/microsoft/rad-dino-maira-2) (a DINOv2 ViT-B chest
X-ray encoder) for **image-level multi-label classification** of 14 abnormalities on the Kaggle
[VinBigData](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection) dataset. The
model card lists research-only use and the MSRLA license, so check both licenses (and the dataset terms) before
using the weights for anything else.

The flow diagram is in [flow.excalidraw](flow.excalidraw) (open it at excalidraw.com).

## Status

| Verified                                                                                                                               | Not verified                                                                 |
| -------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Download, convert and label steps on real DICOMs (MONOCHROME1/2, 12/14-bit, JPEG 2000)                                                 | Real training runs: whether full fine-tuning beats a frozen probe is unknown |
| Full pipeline on CPU and on an RTX 3050 laptop GPU (4 GB), including validation, test report, checkpoints,`--pos-weight`, `--llrd` | Behavior on the GPU you will train on (RunPod)                               |
| Kill-and-resume gives the same LR schedule as an uninterrupted run (one kill scenario, on CPU)                                         | A kill during a checkpoint write                                             |

There are no automated tests in this repo yet.

## Setup

```bash
python -m venv .venv                      # on RunPod add --system-site-packages to reuse the template's torch
source .venv/Scripts/activate             # Windows Git Bash; PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch torchvision             # not in requirements.txt (RunPod's template already has them)
```

Downloading needs `~/.kaggle/kaggle.json` and the competition rules accepted on your Kaggle account.

## Pipeline

```bash
python scripts/download.py --n 2000                        # random subset; --n 0 = all 15,000 (~190 GB)
python scripts/convert.py --src data/dicom --delete-dicom  # DICOM -> data/png, then delete the DICOM
python scripts/make_labels.py --img-dir data/png           # data/labels.csv with train/val/test split
python train.py --bs 16 --grad-ckpt --epochs 20            # fine-tune
```

| Step               | What it does                                                                                                                                                                                                                                        |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `download.py`    | Fetches`train.csv` and DICOMs into `data/`. Skips images already downloaded or already converted.                                                                                                                                               |
| `convert.py`     | Fixes MONOCHROME1 polarity, resizes the shorter side to 518 px, min-max scales to 8-bit, writes PNG. Skips existing PNGs.`--delete-dicom` removes each DICOM after its PNG is written (**irreversible**: reprocessing needs a re-download). |
| `make_labels.py` | Boxes to 14 image-level labels (a class is positive if at least 2 of 3 radiologists marked it), then a multilabel-stratified 80/10/10 split.`--img-dir` keeps only converted images.                                                              |
| `train.py`       | Training on PNGs only, so deleting the DICOMs does not affect it.                                                                                                                                                                                   |

Preprocessing is split in two. The DICOM work happens **once**, in `convert.py`. Resize to a square, augmentation
(no horizontal flip), normalization and 3-channel repeat happen **per image, every epoch**, in `scripts/data.py`.
Changing the DICOM-level steps (or `--size`) means re-running `convert.py`.

## `train.py` options

| Flag                                                      | Default     | Meaning                                                                                              |
| --------------------------------------------------------- | ----------- | ---------------------------------------------------------------------------------------------------- |
| `--epochs`                                              | 1           | Number of epochs                                                                                     |
| `--bs`                                                  | 16          | Batch size                                                                                           |
| `--res`                                                 | 518         | Training resolution (PNGs are 518 px on the shorter side, so higher adds no detail)                  |
| `--lr-backbone` / `--lr-head`                         | 2e-5 / 1e-3 | Learning rates                                                                                       |
| `--llrd`                                                | 1.0 (off)   | Layer-wise LR decay per block, e.g. 0.75                                                             |
| `--pos-weight`                                          | off         | Weight positives by neg/pos per class (capped at 20).`val_loss` then uses the same weights.        |
| `--freeze-backbone`                                     | off         | Linear probe                                                                                         |
| `--grad-ckpt`                                           | off         | Trade speed for memory                                                                               |
| `--max-steps`                                           | 0           | Stop each epoch after N steps (timing runs)                                                          |
| `--project-images`                                      | 0           | Full train-set size (e.g. 13500), so the epoch-time estimate is right when`labels.csv` is a subset |
| `--resume RUN_DIR`                                      |             | Continue a run; reuses that run's`config.json` arguments                                           |
| `--out-dir`, `--workers`, `--seed`, `--drop-path` |             | Output folder, loader workers, seed, drop-path rate                                                  |

Weight decay is 0.05 and skips biases, norms and embedding tokens. The schedule is 5% linear warmup then cosine.

## Outputs (`runs/<timestamp>/`)

| File                  | Contents                                                                                                                                                             |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config.json`       | Arguments, classes, split sizes, GPU, library versions, start time                                                                                                   |
| `metrics.csv`       | Per epoch: losses, timing, throughput, peak GPU memory, val AUROC/AUPRC/balanced accuracy, mean LRs                                                                  |
| `per_class.csv`     | Per epoch and class: AUROC, AUPRC, prevalence, sensitivity, specificity, balanced accuracy                                                                           |
| `test_metrics.json` | Test set (evaluated once with`best.pt`): per-class and mean AUROC with 95% bootstrap CIs                                                                           |
| `results_table.csv` | Per-finding table printed at the end of training:`raddino_ft_val`, `raddino_ft_test` (`raddino_probe_*` with `--freeze-backbone`), CI, AUPRC, test positives |
| `summary.json`      | Epochs completed, total time, best epoch                                                                                                                             |
| `last.pt`           | Full training state for`--resume` (about 1 GB), overwritten each epoch                                                                                             |
| `best.pt`           | Weights of the best epoch by mean val AUROC (about 350 MB)                                                                                                           |

Sensitivity, specificity and balanced accuracy use a fixed 0.5 threshold, so rare classes show low sensitivity until
the model is confident enough. Judge ranking quality by AUROC and AUPRC.

## Measured speed (RTX 3050 Laptop, 4 GB, 518 px, bf16, full fine-tune)

Roughly +/-10%: each figure comes from 3 to 24 steps. The epoch column projects 13,500 training images.

| Config                        | Peak GPU mem | Speed      | Epoch   |
| ----------------------------- | ------------ | ---------- | ------- |
| bs 16,`--grad-ckpt`         | 2.8 GiB      | 6.9 img/s  | ~33 min |
| bs 8,`--grad-ckpt`          | 2.2 GiB      | 6.6 img/s  | ~34 min |
| bs 4,`--grad-ckpt`          | 1.8 GiB      | 6.3 img/s  | ~36 min |
| bs 4, no checkpointing        | 3.7 GiB      | 4.0 img/s  | ~57 min |
| bs 8, frozen backbone (probe) | 0.7 GiB      | 31.6 img/s | ~7 min  |

Data loading measured about 170 img/s per worker on a laptop CPU, so it is not the bottleneck at these speeds.
RunPod speeds are not measured yet: run `python train.py --max-steps 100 --bs 16 --project-images 13500` there.

## Dataset facts (full `train.csv`)

- 15,000 images, each read by exactly 3 radiologists; 71% have no positive label.
- Rarest classes have very few positives. In a 10% split, Atelectasis and Pneumothorax get about 6 each, so
  their per-class results are noisy (see the CIs and `n_pos_test`).
- Patient IDs are not provided, so patient-level leakage between splits cannot be ruled out.

## Known limitations

- Images are resized to a square without a center crop, so the aspect ratio is distorted (H/W ranged 0.89 to
  1.38 in a 60-image sample). Measured on 10 images, the CLS embedding stays close to the model card's
  cubic + center-crop version (cosine similarity 0.970 mean, 0.889 worst); padding instead was further away (0.947).
  Whether this affects downstream accuracy is untested.
- Resize filter: cubic vs a B-spline zoom changed embeddings very little (0.991), percentile-clipped min-max even less
  (0.996), while OpenCV's area filter changed them more (0.952 mean, 0.790 worst). Same 10 images, frozen model,
  CLS only, so this shows feature stability, not accuracy.
- One split only, no cross-validation. Requirements are not pinned.
- Runs made before the optimizer-group change cannot be resumed.
