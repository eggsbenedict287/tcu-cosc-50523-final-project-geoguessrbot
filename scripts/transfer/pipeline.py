"""Training pipeline for pretrained (transfer-learning) CNNs.

Called from ``run_pipeline.py --model transfer --backbone <name>``. One call = one
experiment ("run"). It trains, picks the best epoch on *validation* loss, evaluates,
and writes everything needed to reproduce and compare the run:

    results/<backbone>/<run_name>.json         full metrics, history, config
    results/<backbone>/experiment_log.csv      one summary row appended per run
    <weights-dir>/<backbone>/<run_name>/       checkpoint.pt (best weights) + config.json
                                               last.pt       (resume file, see below)

Colab survival: resume
----------------------
Colab disconnects (90 min idle, 12 h max). After **every epoch** the whole training
state (model, optimizer, LR schedule, AMP scaler, history, best weights) is written to
``last.pt``. Re-running the *same command* finds that file and continues from the
next epoch. Point ``--weights-dir`` at Google Drive so the file survives the VM.

Test-set discipline
-------------------
The test split is only evaluated when ``--eval-test`` is passed. Tuning runs should
not pass it: hyperparameters are chosen on validation, and the test set is looked at
only for the final reported runs.
"""
import csv
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from common.checkpoint import build_run_name, save_checkpoint
from common.dataset import make_cnn_loader
from common.transforms import get_eval_transform, get_train_transform
from cnn_1.trainer import run_epoch, topk_accuracy
from transfer.model import IMAGENET_MEAN, IMAGENET_STD, TransferNet, count_parameters, make_param_groups

TOPK = (1, 3, 5, 10)

LOG_COLUMNS = [
    "run_name", "date", "model", "pretrained", "freeze", "num_classes", "train_per_class",
    "n_train", "img_size", "lr", "head_lr_mult", "weight_decay", "batch_size",
    "epochs_planned", "epochs_run", "dropout", "label_smoothing", "hflip", "aug",
    "seed", "split_seed", "best_epoch", "train_loss", "val_loss",
    "val_top1", "val_top3", "val_top5", "val_top10", "val_macro_f1",
    "test_top1", "test_top3", "test_top5", "test_top10", "test_macro_f1",
    "sec_per_epoch", "gpu", "params_total", "params_trainable", "notes",
]


def evaluate(model, loader, device, criterion, countries: list[str]) -> dict:
    """Run *model* over *loader* and compute every metric we report.

    Top-k accuracy = fraction of images whose true country is among the model's k
    highest-probability guesses. For 111 classes top-5 is much more forgiving and more
    informative than top-1 (many countries look alike), which is why the professor asked
    for top-k metrics instead of a confusion matrix over 2 classes.
    """
    num_classes = len(countries)
    loss, probs, y = run_epoch(model, loader, device, criterion)
    preds = probs.argmax(1).numpy()
    report = classification_report(
        y.numpy(), preds, labels=range(num_classes), target_names=countries,
        output_dict=True, zero_division=0,
    )
    out = {"loss": loss, "n_images": len(y)}
    for k in TOPK:
        out[f"top{k}_acc"] = topk_accuracy(probs, y, k)
    out["topk_curve"] = [topk_accuracy(probs, y, k) for k in range(1, 11)]  # k = 1..10
    out["macro_precision"] = report["macro avg"]["precision"]
    out["macro_recall"] = report["macro avg"]["recall"]
    out["macro_f1"] = report["macro avg"]["f1-score"]
    out["per_class"] = {c: report[c] for c in countries}
    out["confusion_matrix"] = confusion_matrix(y.numpy(), preds, labels=range(num_classes)).tolist()
    return out


def _atomic_save(obj, path: Path) -> None:
    """Write to a temp file then rename, so a disconnect mid-write cannot corrupt *path*."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _append_log_row(path: Path, row: dict) -> None:
    """Add *row* to the experiment log; a row with the same run_name is replaced, not duplicated.

    (Re-running a finished command, e.g. to add ``--eval-test``, updates its row.)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.exists():
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["run_name"] != row["run_name"]]
    rows.append({k: row.get(k, "") for k in LOG_COLUMNS})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_transfer_pipeline(args, countries: list[str], splits: dict, device: torch.device,
                          results_dir: Path, weights_dir: Path) -> dict:
    """Train + evaluate one pretrained backbone. Returns the results dict (also saved to disk)."""
    num_classes = len(countries)
    pretrained = args.pretrained
    hp = dict(
        cls=num_classes, px=args.crop_size, lr=args.lr, bs=args.batch_size, wd=args.weight_decay,
        ep=args.epochs, drop=args.dropout, ls=args.label_smoothing, frz=args.freeze, aug=args.aug,
        hflip=args.use_hflip, seed=args.seed,
    )
    run_name = args.run_name or build_run_name(args.backbone, hp)
    run_dir = weights_dir / args.backbone / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run: {run_name}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # inputs are a fixed size, so let cuDNN autotune

    # ---- data: ImageNet normalisation, because the pretrained weights expect it ----
    train_tf = get_train_transform(IMAGENET_MEAN, IMAGENET_STD, crop_size=args.crop_size,
                                   use_hflip=args.use_hflip, aug=args.aug)
    eval_tf = get_eval_transform(IMAGENET_MEAN, IMAGENET_STD, resize_to=args.resize_to,
                                 crop_size=args.crop_size)
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.workers, seed=args.seed)
    loaders = {
        "train": make_cnn_loader(splits["train"], train_tf, shuffle=True, **loader_kw),
        "val": make_cnn_loader(splits["val"], eval_tf, **loader_kw),
        "test": make_cnn_loader(splits["test"], eval_tf, **loader_kw),
    }

    # ---- model ----
    model_kwargs = dict(backbone=args.backbone, num_classes=num_classes, pretrained=pretrained,
                        dropout=args.dropout, freeze=args.freeze)
    model = TransferNet(**model_kwargs).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    total_params, trainable_params = count_parameters(model)
    print(f"{args.backbone}: {total_params / 1e6:.2f}M params, {trainable_params / 1e6:.2f}M trainable"
          f" (freeze={args.freeze}, pretrained={pretrained})")

    # ---- optimiser + schedule ----
    optimizer = torch.optim.AdamW(make_param_groups(model, args.lr, args.weight_decay, args.head_lr_mult))
    steps_per_epoch = len(loaders["train"])
    # OneCycle: LR ramps up for the first 15% of steps (warmup), then decays along a cosine
    # to ~0. max_lr is a list so the head group can peak higher than the backbone group.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[g["lr"] for g in optimizer.param_groups],
        total_steps=args.epochs * steps_per_epoch, pct_start=0.15, anneal_strategy="cos",
    )
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None  # FP16 loss scaling

    # ---- optional Weights & Biases tracking ----
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    wandb_run = None
    if args.wandb:
        import wandb
        # id=run_name + resume="allow" => resuming after a disconnect continues the same W&B run.
        wandb_run = wandb.init(project=args.wandb_project, id=run_name, name=run_name,
                               group=args.backbone, config=config, resume="allow")

    # ---- resume if a previous session left last.pt ----
    state_path = run_dir / "last.pt"
    history, best_val, best_state, bad_epochs, best_epoch, finished = [], float("inf"), None, 0, 0, False
    start_epoch = 1
    if args.resume and state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        if scaler is not None and st.get("scaler"):
            scaler.load_state_dict(st["scaler"])
        history, best_val, best_state = st["history"], st["best_val"], st["best_state"]
        bad_epochs, best_epoch, finished = st["bad_epochs"], st["best_epoch"], st["finished"]
        start_epoch = st["epoch"] + 1
        print(f"Resumed from {state_path} (finished epoch {st['epoch']}"
              f"{', run already complete' if finished else ''})")

    def save_state(epoch: int) -> None:
        _atomic_save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict() if scaler else None,
            "history": history, "best_val": best_val, "best_state": best_state,
            "bad_epochs": bad_epochs, "best_epoch": best_epoch, "epoch": epoch, "finished": finished,
        }, state_path)

    # ---- training loop (best epoch chosen on validation loss) ----
    epoch_times = []
    for epoch in range(start_epoch, args.epochs + 1):
        if finished:
            break
        start = time.time()
        train_loss, _, _ = run_epoch(model, loaders["train"], device, criterion, optimizer, scaler, scheduler)
        val_loss, val_probs, val_y = run_epoch(model, loaders["val"], device, criterion)
        val_top1 = topk_accuracy(val_probs, val_y, 1)
        val_top5 = topk_accuracy(val_probs, val_y, 5)
        epoch_times.append(time.time() - start)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_top1": val_top1, "val_top5": val_top5,
                        "lr": optimizer.param_groups[0]["lr"], "seconds": epoch_times[-1]})
        print(f"epoch {epoch:3d}/{args.epochs} | train loss {train_loss:.4f} | val loss {val_loss:.4f}"
              f" | val top-1 {val_top1:.4f} | val top-5 {val_top5:.4f} | {epoch_times[-1]:.0f}s")
        if wandb_run:
            wandb_run.log({k: v for k, v in history[-1].items() if k != "epoch"}, step=epoch)

        if val_loss < best_val:
            best_val, bad_epochs, best_epoch = val_loss, 0, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping (no val improvement for {args.patience} epochs)")
                finished = True
        finished = finished or epoch == args.epochs
        save_state(epoch)  # <- the resume point: written after every epoch

    if best_state is None:
        raise SystemExit("No epochs were run, nothing to evaluate.")
    model.load_state_dict(best_state)  # evaluate the best-validation epoch, not the last one

    # ---- final evaluation ----
    results = {
        "model": f"transfer_{args.backbone}", "run_name": run_name, "config": config,
        "countries": countries, "chance": 1 / num_classes,
        "params": {"total": total_params, "trainable": trainable_params},
        "n_images": {k: len(v) for k, v in splits.items()},
        "best_epoch": best_epoch, "history": history,
        "val": evaluate(model, loaders["val"], device, criterion, countries),
        "test": None,
    }
    if args.eval_test:
        results["test"] = evaluate(model, loaders["test"], device, criterion, countries)
    else:
        print("Test set NOT evaluated (pass --eval-test for final runs).")

    val, test = results["val"], results["test"]
    print(f"\nVAL  ({val['n_images']} images) | top-1 {val['top1_acc']:.4f} | top-3 {val['top3_acc']:.4f}"
          f" | top-5 {val['top5_acc']:.4f} | top-10 {val['top10_acc']:.4f} | macro F1 {val['macro_f1']:.4f}"
          f" | chance {1 / num_classes:.4f}")
    if test:
        print(f"TEST ({test['n_images']} images) | top-1 {test['top1_acc']:.4f} | top-3 {test['top3_acc']:.4f}"
              f" | top-5 {test['top5_acc']:.4f} | top-10 {test['top10_acc']:.4f} | macro F1 {test['macro_f1']:.4f}")

    # ---- save results, checkpoint, and one summary row in the experiment log ----
    out_dir = results_dir / args.backbone
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{run_name}.json").write_text(json.dumps(results, indent=2))
    print(f"Saved results to {out_dir / run_name}.json")

    if args.save_weights:
        keys = ("loss", "top1_acc", "top5_acc", "macro_f1")
        ckpt = save_checkpoint(
            run_dir, model, model_name=results["model"], model_kwargs={**model_kwargs, "pretrained": False},
            epoch=best_epoch, countries=countries, config=config,
            metrics={s: {k: results[s][k] for k in keys} for s in ("val", "test") if results[s]},
            normalization={"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        )
        print(f"Saved weights to {ckpt}")

    best = history[best_epoch - 1] if 0 < best_epoch <= len(history) else history[-1]
    row = {
        "run_name": run_name, "date": date.today().isoformat(), "model": args.backbone,
        "pretrained": pretrained, "freeze": args.freeze, "num_classes": num_classes,
        "train_per_class": args.max_train_per_class or "all", "n_train": len(splits["train"]),
        "img_size": args.crop_size, "lr": args.lr, "head_lr_mult": args.head_lr_mult,
        "weight_decay": args.weight_decay, "batch_size": args.batch_size,
        "epochs_planned": args.epochs, "epochs_run": len(history), "dropout": args.dropout,
        "label_smoothing": args.label_smoothing, "hflip": args.use_hflip, "aug": args.aug,
        "seed": args.seed, "split_seed": args.split_seed, "best_epoch": best_epoch,
        "train_loss": round(best["train_loss"], 4), "val_loss": round(val["loss"], 4),
        **{f"val_top{k}": round(val[f"top{k}_acc"], 4) for k in TOPK},
        "val_macro_f1": round(val["macro_f1"], 4),
        "sec_per_epoch": round(float(np.mean(epoch_times)), 1) if epoch_times else "",
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "params_total": total_params, "params_trainable": trainable_params, "notes": args.notes,
    }
    if test:
        row.update({f"test_top{k}": round(test[f"top{k}_acc"], 4) for k in TOPK})
        row["test_macro_f1"] = round(test["macro_f1"], 4)
    _append_log_row(out_dir / "experiment_log.csv", row)

    if wandb_run:
        summary = {f"val_top{k}": val[f"top{k}_acc"] for k in TOPK} | {"val_macro_f1": val["macro_f1"]}
        if test:
            summary |= {f"test_top{k}": test[f"top{k}_acc"] for k in TOPK} | {"test_macro_f1": test["macro_f1"]}
        wandb_run.summary.update(summary)
        wandb_run.finish()
    return results
