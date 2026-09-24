"""Fine-tune RAD-DINO-MAIRA-2 for image-level multi-label classification.
Each run writes config.json, metrics.csv, per_class.csv, test_metrics.json, results_table.csv, summary.json
and checkpoints to runs/<timestamp>/."""
import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import transformers

from scripts.data import make_loaders
from scripts.engine import fit, results_table, test_report
from scripts.model import MODEL, Net, make_optimizer, make_scheduler


def main():
    # ==============================
    # arguments
    # ==============================
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default="data/labels.csv")
    p.add_argument("--img-dir", default="data/png")
    p.add_argument("--out-dir", help="default: runs/<timestamp>")
    p.add_argument("--res", type=int, default=518)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1, help="number of training epochs")
    p.add_argument("--lr-backbone", type=float, default=2e-5)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--llrd", type=float, default=1.0, help="layer-wise LR decay per block, e.g. 0.75 (1.0 = off)")
    p.add_argument("--pos-weight", action="store_true",
                   help="weight positives by neg/pos per class (capped at 20); val_loss then uses the same weights")
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--freeze-backbone", action="store_true", help="linear probe")
    p.add_argument("--grad-ckpt", action="store_true", help="trade speed for memory")
    p.add_argument("--max-steps", type=int, default=0, help="stop each epoch after N steps (timing runs)")
    p.add_argument("--project-images", type=int, default=0,
                   help="size of the full train set, to project epoch time when labels.csv is only a subset")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", help="run dir to continue from; reuses that run's config.json arguments")
    a = p.parse_args()

    # ==============================
    # resume config
    # ==============================
    if a.resume:  # same hyperparameters as the original run, so the LR schedule stays consistent
        saved = json.loads((Path(a.resume) / "config.json").read_text())["args"]
        a = argparse.Namespace(**{**vars(p.parse_args([])), **saved, "resume": a.resume, "out_dir": a.resume})

    # ==============================
    # setup + data
    # ==============================
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    out_dir = Path(a.out_dir or f"runs/{datetime.now():%Y%m%d-%H%M%S}")
    out_dir.mkdir(parents=True, exist_ok=True)
    a.out_dir = str(out_dir)
    train_dl, val_dl, test_dl, classes = make_loaders(a.labels, a.img_dir, a.res, a.bs, a.workers, pin_memory=amp)

    # ==============================
    # model + optimizer
    # ==============================
    net = Net(len(classes), a.drop_path).to(device)
    if a.grad_ckpt:
        net.backbone.gradient_checkpointing_enable()
    opt = make_optimizer(net, a.lr_head, a.lr_backbone, a.freeze_backbone, a.llrd)
    steps = min(len(train_dl), a.max_steps or len(train_dl))
    sched = make_scheduler(opt, steps * a.epochs)
    pos_weight = None
    if a.pos_weight:
        y = train_dl.dataset.y
        pos_weight = torch.tensor((len(y) - y.sum(0)) / y.sum(0).clip(min=1)).clamp(max=20).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ==============================
    # config snapshot
    # ==============================
    if not a.resume:  # a resumed run keeps its original config.json
        (out_dir / "config.json").write_text(json.dumps({
            "args": vars(a), "model": MODEL, "classes": classes,
            "n_train": len(train_dl.dataset), "n_val": len(val_dl.dataset),
            "n_test": len(test_dl.dataset) if test_dl else 0,
            "pos_weight": pos_weight.tolist() if pos_weight is not None else None,
            "steps_per_epoch_run": steps, "steps_per_epoch_full": len(train_dl),
            "device": device, "gpu": torch.cuda.get_device_name() if amp else None,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }, indent=2))

    # ==============================
    # train
    # ==============================
    history = fit(net, train_dl, val_dl, opt, sched, loss_fn, device, amp, classes,
                  epochs=a.epochs, steps_per_epoch=steps, out_dir=out_dir,
                  epoch_steps=a.project_images // a.bs if a.project_images else None,
                  freeze_backbone=a.freeze_backbone, resume=bool(a.resume))

    # ==============================
    # test
    # ==============================
    best = max(history, key=lambda r: r["mean_auroc"])
    if test_dl:  # best epoch by validation AUROC, evaluated once
        net.load_state_dict(torch.load(out_dir / "best.pt", map_location=device, weights_only=True)["model"])
        report = test_report(net, test_dl, loss_fn, device, amp, classes)
        (out_dir / "test_metrics.json").write_text(json.dumps(report, indent=2))
        tag = "raddino_probe" if a.freeze_backbone else "raddino_ft"
        table = results_table(out_dir / "per_class.csv", best["epoch"], report, tag)
        table.to_csv(out_dir / "results_table.csv", index=False)
        print(f"\nper-finding AUROC (validation = best epoch {best['epoch']}, test = {report['n_images']} images)")
        print(table.to_string(float_format="{:.3f}".format))

    # ==============================
    # summary
    # ==============================
    (out_dir / "summary.json").write_text(json.dumps({
        "epochs_completed": len(history), "total_time_s": sum(r["epoch_time_s"] for r in history),
        "best_epoch": best["epoch"], "best_mean_auroc": best["mean_auroc"],
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }, indent=2))
    print(f"run saved to {out_dir}")


if __name__ == "__main__":
    main()
