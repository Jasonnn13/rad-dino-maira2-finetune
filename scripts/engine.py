import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


# ==============================
# timing report
# ==============================
def report_timing(durations, bs, epoch_steps, amp):
    warm_steps = min(5, len(durations) - 1)  # skip cudnn/allocator warmup
    sec_per_step = float(np.median(durations[warm_steps:])) if len(durations) > 1 else float(durations[0])
    mem = torch.cuda.max_memory_allocated() / 2**30 if amp else 0
    est_min = epoch_steps * sec_per_step / 60
    print(f"train: {sum(durations):.0f}s for {len(durations)} steps | {sec_per_step:.3f}s/step "
          f"({bs / sec_per_step:.1f} img/s) | peak GPU mem {mem:.1f} GiB")
    print(f"estimated epoch ({epoch_steps} steps): {est_min:.1f} min")
    return {"steps": len(durations), "train_time_s": float(sum(durations)), "sec_per_step": sec_per_step,
            "img_per_s": bs / sec_per_step, "est_epoch_min": est_min, "peak_mem_gib": mem}


# ==============================
# training loop
# ==============================
def fit(net, train_dl, val_dl, opt, sched, loss_fn, device, amp, classes,
        epochs, steps_per_epoch, out_dir, epoch_steps=None, freeze_backbone=False, resume=False):
    """Trains for `epochs` epochs. After each epoch it validates, rewrites metrics.csv / per_class.csv in
    `out_dir`, saves last.pt (full training state) and best.pt (weights, when mean val AUROC improves).
    `epoch_steps` is the step count used to project epoch time (default: len(train_dl)).
    With resume=True it continues from out_dir/last.pt. Returns the history."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    epoch_steps = epoch_steps or len(train_dl)
    history, per_class, best, start = [], [], -1.0, 0

    # ==============================
    # resume
    # ==============================
    if resume:
        ckpt = torch.load(out_dir / "last.pt", map_location=device, weights_only=True)
        net.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        best, start = ckpt["best"], ckpt["epoch"] + 1
        # drop rows of an epoch that was logged but never checkpointed (killed between the two writes)
        history = [r for r in pd.read_csv(out_dir / "metrics.csv").to_dict("records") if r["epoch"] < start]
        per_class = [r for r in pd.read_csv(out_dir / "per_class.csv").to_dict("records") if r["epoch"] < start]
        print(f"resumed from {out_dir / 'last.pt'}: continuing at epoch {start}/{epochs}")

    for epoch in range(start, epochs):
        # ==============================
        # train epoch
        # ==============================
        t_epoch = time.time()
        if amp:
            torch.cuda.reset_peak_memory_stats()
        net.train()
        if freeze_backbone:
            net.backbone.eval()
        losses, lrs, ends, t0 = [], [], [], time.time()
        for step, (x, y) in enumerate(train_dl):
            if step >= steps_per_epoch:
                break
            with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
                loss = loss_fn(net(x.to(device, non_blocking=True)).float(), y.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            head = [g["lr"] for g in opt.param_groups if g["name"] == "head"]
            backbone = [g["lr"] for g in opt.param_groups if g["name"] == "backbone"]
            lrs.append([max(head), max(backbone) if backbone else np.nan])  # LR used for this step
            opt.step()
            sched.step()
            if amp:
                torch.cuda.synchronize()
            losses.append(loss.item())
            ends.append(time.time())
            if step % 20 == 0:
                print(f"epoch {epoch} step {step}/{steps_per_epoch} loss {losses[-1]:.4f}", flush=True)
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               **report_timing(np.diff([t0, *ends]), train_dl.batch_size, epoch_steps, amp)}

        # ==============================
        # validate
        # ==============================
        t1 = time.time()
        val = evaluate(net, val_dl, loss_fn, device, amp)
        lr_mean = np.mean(lrs, axis=0)  # max LR across the backbone groups = the top block's LR
        rec.update(val_time_s=time.time() - t1, val_loss=val["loss"],
                   mean_auroc=float(np.nanmean(val["auroc"])), mean_auprc=float(np.nanmean(val["auprc"])),
                   mean_bal_acc=float(np.nanmean(val["bal_acc"])),
                   lr_head_mean=float(lr_mean[0]), lr_backbone_mean=None if np.isnan(lr_mean[1]) else float(lr_mean[1]))
        rec["epoch_time_s"] = time.time() - t_epoch
        print(f"val: {rec['val_time_s']:.0f}s | loss {rec['val_loss']:.4f} | mean AUROC {rec['mean_auroc']:.4f} "
              f"| mean AUPRC {rec['mean_auprc']:.4f} | mean balanced acc@0.5 {rec['mean_bal_acc']:.4f}")
        print({c: round(v, 3) for c, v in zip(classes, val["auroc"])})

        # ==============================
        # log metrics
        # ==============================
        history.append(rec)
        per_class += [{"epoch": epoch, "class": c, **{k: val[k][i] for k in PER_CLASS_KEYS}}
                      for i, c in enumerate(classes)]
        pd.DataFrame(history).to_csv(out_dir / "metrics.csv", index=False)
        pd.DataFrame(per_class).to_csv(out_dir / "per_class.csv", index=False)

        # ==============================
        # checkpoint
        # ==============================
        state = {"model": net.state_dict(), "classes": classes, "epoch": epoch, "mean_auroc": rec["mean_auroc"]}
        improved = rec["mean_auroc"] > best
        best = rec["mean_auroc"] if improved else best
        torch.save({**state, "opt": opt.state_dict(), "sched": sched.state_dict(), "best": best},
                   out_dir / "last.pt.tmp")
        os.replace(out_dir / "last.pt.tmp", out_dir / "last.pt")  # atomic: a kill mid-save keeps the old last.pt
        if improved:
            torch.save(state, out_dir / "best.pt")
    return history


# ==============================
# evaluation
# ==============================
PER_CLASS_KEYS = ["auroc", "auprc", "prevalence", "sensitivity", "specificity", "bal_acc"]


@torch.no_grad()
def predict(net, dl, device, amp):
    """Returns (logits, labels) for every image in `dl`."""
    net.eval()
    logits, ys = [], []
    for x, y in dl:
        with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
            logits.append(net(x.to(device)).float().cpu())
        ys.append(y)
    return torch.cat(logits), torch.cat(ys)


def metrics(logits, ys, loss_fn):
    """Loss plus per-class metrics (see PER_CLASS_KEYS). AUROC is nan for classes without both labels
    present; AUPRC/sensitivity are nan without positives; specificity without negatives. Sensitivity,
    specificity and balanced accuracy use a fixed 0.5 threshold, so rare classes will show low sensitivity
    until the model is confident enough to cross it (use AUROC/AUPRC to judge ranking quality)."""
    probs, y = torch.sigmoid(logits).numpy(), ys.numpy()
    pos = y.sum(0)
    neg = len(y) - pos
    pred = probs > 0.5
    ratio = lambda num, den: np.divide(num, den, out=np.full(den.shape, np.nan), where=den > 0)
    sens = ratio((pred & (y == 1)).sum(0), pos)
    spec = ratio((~pred & (y == 0)).sum(0), neg)
    cols = range(y.shape[1])
    dev = next(loss_fn.buffers(), logits).device  # pos_weight is a buffer and may live on the GPU
    return {"loss": loss_fn(logits.to(dev), ys.to(dev)).item(),
            "auroc": [roc_auc_score(y[:, c], probs[:, c]) if 0 < pos[c] < len(y) else float("nan") for c in cols],
            "auprc": [average_precision_score(y[:, c], probs[:, c]) if pos[c] > 0 else float("nan") for c in cols],
            "prevalence": (pos / len(y)).tolist(),
            "sensitivity": sens.tolist(), "specificity": spec.tolist(),
            "bal_acc": ((sens + spec) / 2).tolist()}


def evaluate(net, dl, loss_fn, device, amp):
    return metrics(*predict(net, dl, device, amp), loss_fn)


# ==============================
# test report
# ==============================
def bootstrap_ci(logits, ys, n_boot=1000, seed=0):
    """95% CI of each class's AUROC and of their mean, from resampling images with replacement."""
    probs, y = torch.sigmoid(logits).numpy(), ys.numpy()
    rng = np.random.default_rng(seed)
    aucs = np.full((n_boot, y.shape[1]), np.nan)
    for b in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        for c in range(y.shape[1]):
            if 0 < y[i, c].sum() < len(i):
                aucs[b, c] = roc_auc_score(y[i, c], probs[i, c])
    with warnings.catch_warnings():  # classes without both labels are all-nan on purpose
        warnings.simplefilter("ignore", RuntimeWarning)
        per_class = np.nanpercentile(aucs, [2.5, 97.5], axis=0).T
        mean = np.nanpercentile(np.nanmean(aucs, axis=1), [2.5, 97.5])
    return per_class.tolist(), mean.tolist()


def _clean(o):
    """Makes nan/numpy values JSON-safe (nan -> None)."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return None if isinstance(o, (float, np.floating)) and np.isnan(o) else (float(o) if isinstance(o, np.floating) else o)


def test_report(net, dl, loss_fn, device, amp, classes, n_boot=1000):
    """Final evaluation: metrics on `dl` plus bootstrap 95% CIs on AUROC."""
    logits, ys = predict(net, dl, device, amp)
    m = metrics(logits, ys, loss_fn)
    ci, mean_ci = bootstrap_ci(logits, ys, n_boot)
    per_class = {c: {**{k: m[k][i] for k in PER_CLASS_KEYS}, "auroc_ci95": ci[i], "n_pos": int(ys[:, i].sum())}
                 for i, c in enumerate(classes)}
    return _clean({"n_images": len(ys), "loss": m["loss"], "mean_auroc": np.nanmean(m["auroc"]),
                   "mean_auroc_ci95": mean_ci, "mean_auprc": np.nanmean(m["auprc"]),
                   "mean_bal_acc": np.nanmean(m["bal_acc"]), "per_class": per_class})


# ==============================
# results table
# ==============================
def results_table(per_class_csv, best_epoch, report, tag):
    """One row per finding plus a MEAN row: best-epoch validation AUROC, test AUROC with its 95% bootstrap CI,
    test AUPRC and test positive count. `tag` prefixes the AUROC columns so tables from different runs
    (e.g. raddino_probe, raddino_ft) can be joined on `finding`."""
    val = pd.read_csv(per_class_csv).query("epoch == @best_epoch").set_index("class")["auroc"]
    rows = [{"finding": c, f"{tag}_val": val[c], f"{tag}_test": r["auroc"], "ci_lo": r["auroc_ci95"][0],
             "ci_hi": r["auroc_ci95"][1], "test_auprc": r["auprc"], "n_pos_test": r["n_pos"]}
            for c, r in report["per_class"].items()]
    rows.append({"finding": "MEAN", f"{tag}_val": val.mean(), f"{tag}_test": report["mean_auroc"],
                 "ci_lo": report["mean_auroc_ci95"][0], "ci_hi": report["mean_auroc_ci95"][1],
                 "test_auprc": report["mean_auprc"], "n_pos_test": None})
    table = pd.DataFrame(rows)
    table["n_pos_test"] = table["n_pos_test"].astype("Int64")
    return table
