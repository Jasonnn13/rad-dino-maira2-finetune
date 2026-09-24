# RAD-DINO-MAIRA-2 fine-tuning on CheXpert

Fine-tunes [`microsoft/rad-dino-maira-2`](https://huggingface.co/microsoft/rad-dino-maira-2) (a DINOv2 ViT-B chest
X-ray encoder) for **image-level multi-label classification** of 13 findings on the Kaggle
[CheXpert](https://www.kaggle.com/datasets/ashery/chexpert) dataset (the downsampled CheXpert-v1.0-small JPGs). The
model card lists research-only use and the MSRLA license, so check both licenses (and the dataset terms) before
using the weights for anything else.

The flow diagram is in [flow.excalidraw](flow.excalidraw) (open it at excalidraw.com).

## Status

| Verified                                                                                                                               | Not verified                                                                 |
| -------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Download and label steps on CheXpert (labels on the full `train.csv`, download on a 200-image subset)                                    | Real training runs: whether full fine-tuning beats a frozen probe is unknown |
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

Downloading needs `~/.kaggle/kaggle.json`.

## Pipeline

```bash
python scripts/download.py --n 0                           # whole dataset as one zip (~11 GB); --n 2000 = random subset, slower per image
python scripts/make_labels.py --n 50000 --balance          # data/chexpert/labels.csv: 50k images, patient-level train/val/test split
python train.py --bs 16 --grad-ckpt --epochs 20            # fine-tune
```

| Step               | What it does                                                                                                                                                                                                  |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `download.py`    | Fetches `train.csv` and the selected frontal JPGs into `data/chexpert/`, one Kaggle request per image (retries on 429). Skips images already downloaded. With `--n 0` it downloads the whole dataset zip instead.   |
| `make_labels.py` | Frontal views only, 13 findings (`No Finding` dropped), blank = 0, uncertain (-1) = 0 (`--uncertain 1` for 1), then a multilabel-stratified 80/10/10 split **by patient**. `--img-dir` keeps only downloaded images. `--n 50000` keeps 50k images in total (40k/5k/5k). `--balance` picks the train images to even out the class counts: it repeatedly adds an image positive for the class with the fewest positives so far. Val and test are random, so they keep the natural class mix. |
| `train.py`       | Reads the JPGs directly.                                                                                                                                                                                      |

There is no conversion step: the Kaggle JPGs are already 8-bit and downsampled (about 320 px). Resize to a square,
augmentation (no horizontal flip), normalization and 3-channel repeat happen **per image, every epoch**, in
`scripts/data.py`.

## `train.py` options

| Flag                                                      | Default     | Meaning                                                                                              |
| --------------------------------------------------------- | ----------- | ---------------------------------------------------------------------------------------------------- |
| `--epochs`                                              | 1           | Number of epochs                                                                                     |
| `--bs`                                                  | 16          | Batch size                                                                                           |
| `--res`                                                 | 518         | Training resolution (the JPGs are about 320 px, so 518 upsamples them)                  |
| `--lr-backbone` / `--lr-head`                         | 2e-5 / 1e-3 | Learning rates                                                                                       |
| `--llrd`                                                | 1.0 (off)   | Layer-wise LR decay per block, e.g. 0.75                                                             |
| `--pos-weight`                                          | off         | Weight positives by neg/pos per class (capped at 20).`val_loss` then uses the same weights.        |
| `--head`                                                | linear      | `mlp` = 768 -> 768 -> GELU -> classes instead of one linear layer; results columns get an `_mlp` suffix |
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

- 223,414 images of 64,540 patients; 191,027 are frontal (64,534 patients). Labels come from an automatic
  report labeler (positive / negative / uncertain / not mentioned), not from radiologists reading the image.
- 9% of frontal images have no positive finding among the 13 kept. Uncertain labels become negative by default,
  a modelling choice that shifts the results; try `--uncertain 1`.
- Rarest finding: Pleural Other (about 2,500 frontal positives), so per-class CIs are wider there.
- The Kaggle copy has no `valid.csv`, so all splits come from `train.csv`. Splits are by patient, so a patient's
  images (including several studies) stay on one side.

## Known limitations

- The two preprocessing measurements below were made on VinBigData DICOMs at 518 px, before the switch to CheXpert.
  CheXpert JPGs are about 320 px, so they are not verified for this data.
- Images are resized to a square without a center crop, so the aspect ratio is distorted (H/W ranged 0.89 to
  1.38 in a 60-image sample). Measured on 10 images, the CLS embedding stays close to the model card's
  cubic + center-crop version (cosine similarity 0.970 mean, 0.889 worst); padding instead was further away (0.947).
  Whether this affects downstream accuracy is untested.
- Resize filter: cubic vs a B-spline zoom changed embeddings very little (0.991), percentile-clipped min-max even less
  (0.996), while OpenCV's area filter changed them more (0.952 mean, 0.790 worst). Same 10 images, frozen model,
  CLS only, so this shows feature stability, not accuracy.
- One split only, no cross-validation. Requirements are not pinned.
- Runs made before the optimizer-group change cannot be resumed.
