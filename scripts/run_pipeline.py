"""General-purpose model pipeline runner.

Parses command-line arguments, indexes the dataset, builds train/val/test
splits, computes pixel statistics, and runs the selected model pipeline.

Currently supported models
--------------------------
  baseline  – single linear layer logistic regression (no hidden layers)
  cnn_1     – GWTNet, a ResNet-18 style CNN trained from scratch (mixed precision, OneCycleLR)

Usage examples (run from the repo root)
---------------------------------------
    python scripts/run_pipeline.py                           # baseline, 2 random countries
    python scripts/run_pipeline.py --countries FR JP
    python scripts/run_pipeline.py --num-classes 10
    python scripts/run_pipeline.py --list-countries
    python scripts/run_pipeline.py --model baseline --epochs 100
    python scripts/run_pipeline.py --model cnn_1 --countries FR JP --val-frac 0.1 --test-frac 0.1
    python scripts/run_pipeline.py --model cnn_1 --num-classes 111 --val-frac 0.1 --test-frac 0.1 --epochs 80
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# Put scripts/ on the path so sibling packages (common, baseline) are importable
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from common.checkpoint import build_run_name, save_checkpoint
from common.dataset import make_cnn_loader, make_loader
from common.preprocessing import channel_stats, check_leakage, feature_stats, index_dataset, split_per_country
from common.seed import set_seed
from common.transforms import get_eval_transform, get_train_transform
from baseline.model import LogisticRegression
from baseline.trainer import run_epoch, topk_accuracy
from cnn_1.model import GWTNet
from cnn_1.trainer import run_epoch as cnn_run_epoch
from cnn_1.trainer import topk_accuracy as cnn_topk_accuracy

REPO_ROOT = SCRIPTS_DIR.parent
DATA_DIR = REPO_ROOT / "data" / "raw" / "streetview-by-country"
RESULTS_DIR = REPO_ROOT / "results"
WEIGHTS_DIR = REPO_ROOT / "weights"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

SUPPORTED_MODELS = ("baseline", "cnn_1")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --- dataset ---
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--countries", nargs="+", help="Country folder codes, e.g. FR JP")
    p.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of random countries to sample when --countries is not given",
    )
    p.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Cap images per country (useful for quick smoke tests)",
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Images not already SIZE×SIZE are resized to it",
    )
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--list-countries", action="store_true",
                   help="Print available countries and image counts, then exit")
    # --- training ---
    p.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default="baseline",
        help="Which model pipeline to run",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="64 × 1.2 M floats ≈ 315 MB per batch on the GPU",
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularisation")
    p.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early-stopping patience on validation loss",
    )
    # --- runtime ---
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--workers",
        type=int,
        default=2,
        help="DataLoader worker processes (Colab has 2 CPUs)", # maybe remove?? more research...
    )
    p.add_argument(
        "--save-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the best (lowest val loss) weights as a PyTorch checkpoint",
    )
    p.add_argument(
        "--weights-dir",
        type=Path,
        default=WEIGHTS_DIR,
        help="Root folder for checkpoints; each run gets <model>/<run_name>/ inside it",
    )
    # --- cnn_1 specific ---
    p.add_argument(
        "--resize-to",
        type=int,
        default=360,
        help="[cnn_1] Resize shorter edge to this before CenterCrop in val/test transforms",
    )
    p.add_argument(
        "--crop-size",
        type=int,
        default=320,
        help="[cnn_1] Final square crop size for both train (RandomResizedCrop) and val/test (CenterCrop)",
    )
    p.add_argument(
        "--use-hflip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="[cnn_1] Apply random horizontal flip during training (ablation switch)",
    )
    p.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="[cnn_1] Label smoothing epsilon for cross-entropy loss", # Dont know, research this
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="[cnn_1] Dropout probability before the classification head",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pipeline implementations
# ---------------------------------------------------------------------------

def run_baseline_pipeline(
    args: argparse.Namespace,
    countries: list[str],
    splits: dict,
    device: torch.device,
) -> dict:
    """Train and evaluate the logistic regression baseline.

    Args:
        args: Parsed CLI arguments.
        countries: Ordered list of country codes (index == integer label).
        splits: ``{"train": [...], "val": [...], "test": [...]}`` from
            :func:`~src.data.preprocessing.split_per_country`.
        device: Torch device.

    Returns:
        ``(results, checkpoint_info)`` — the results dict ready to be serialised
        as JSON, and the model plus metadata needed to save a checkpoint.
    """
    num_classes = len(countries)

    loader_kwargs = dict(
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.workers,
        seed=args.seed,
    )
    loaders = {
        "train": make_loader(splits["train"], shuffle=True, **loader_kwargs),
        "val": make_loader(splits["val"], **loader_kwargs),
        "test": make_loader(splits["test"], **loader_kwargs),
    }

    # Per-feature standardisation using train statistics only (no leakage).
    start = time.time()
    mean, std = feature_stats(make_loader(splits["train"], **loader_kwargs), device)
    print(f"Computed train pixel statistics in {time.time() - start:.0f}s")

    model_kwargs = dict(in_features=mean.numel(), num_classes=num_classes)
    model = LogisticRegression(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    history, best_val, best_state, bad_epochs, best_epoch = [], float("inf"), None, 0, 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, _, _ = run_epoch(model, loaders["train"], mean, std, device, optimizer)
        val_loss, val_probs, val_y = run_epoch(model, loaders["val"], mean, std, device)
        val_acc = topk_accuracy(val_probs, val_y, 1)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}
        )
        print(
            f"epoch {epoch:3d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}"
            f" | val acc {val_acc:.4f} | {time.time() - start:.0f}s"
        )
        if val_loss < best_val:
            best_val, bad_epochs, best_epoch = val_loss, 0, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping (no val improvement for {args.patience} epochs)")
                break
    model.load_state_dict(best_state)

    # Final evaluation — val is re-run with best weights, test is run once.
    results = {
        "model": "baseline_logreg",
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "countries": countries,
        "history": history,
        "majority_class_baseline": 1 / num_classes,
    }
    for name in ("val", "test"):
        loss, probs, y = run_epoch(model, loaders[name], mean, std, device)
        preds = probs.argmax(1).numpy()
        report = classification_report(
            y.numpy(), preds,
            labels=range(num_classes),
            target_names=countries,
            output_dict=True,
            zero_division=0,
        )
        results[name] = {
            "loss": loss,
            "top1_acc": topk_accuracy(probs, y, 1),
            "top5_acc": topk_accuracy(probs, y, 5),
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "macro_f1": report["macro avg"]["f1-score"],
            "per_class": {c: report[c] for c in countries},
            "confusion_matrix": confusion_matrix(
                y.numpy(), preds, labels=range(num_classes)
            ).tolist(),
        }

    test = results["test"]
    print(
        f"\nTEST ({len(y)} images) | top-1 {test['top1_acc']:.4f}"
        f" | top-5 {test['top5_acc']:.4f}"
        f" | macro F1 {test['macro_f1']:.4f}"
        f" | chance {1 / num_classes:.4f}"
    )
    if num_classes <= 20:
        # y / preds still reference the test split from the last loop iteration.
        print(
            classification_report(
                y.numpy(), preds,
                labels=range(num_classes),
                target_names=countries,
                zero_division=0,
            )
        )
        print("Confusion matrix (rows = true, cols = predicted):")
        print(np.array(test["confusion_matrix"]))

    checkpoint_info = {
        "model": model,
        "model_kwargs": model_kwargs,
        "epoch": best_epoch,
        "normalization": {"mean": mean, "std": std},
    }
    return results, checkpoint_info


def run_cnn1_pipeline(
    args: argparse.Namespace,
    countries: list[str],
    splits: dict,
    device: torch.device,
) -> dict:
    """Train and evaluate the GWTNet CNN.

    Args:
        args:      Parsed CLI arguments.
        countries: Ordered list of country codes (index == integer label).
        splits:    ``{"train": [...], "val": [...], "test": [...]}`` from
                   :func:`~common.preprocessing.split_per_country`.
        device:    Torch device.

    Returns:
        ``(results, checkpoint_info)`` — the results dict ready to be serialised
        as JSON, and the model plus metadata needed to save a checkpoint.
    """
    import torchvision.transforms as T

    num_classes = len(countries)

    # ------------------------------------------------------------------
    # 1. Compute per-channel mean / std from the training split.
    #    Use a simple ToTensor (+ resize to crop_size) so the stat pass
    #    is fast and consistent with what the transforms will produce.
    # ------------------------------------------------------------------
    stat_transform = T.Compose([T.Resize(args.crop_size), T.CenterCrop(args.crop_size), T.ToTensor()])
    stat_loader = make_cnn_loader(
        splits["train"], stat_transform,
        batch_size=args.batch_size, num_workers=args.workers, seed=args.seed,
    )
    start = time.time()
    mean, std = channel_stats(stat_loader, device)
    print(f"Channel mean: {[f'{v:.4f}' for v in mean]}  std: {[f'{v:.4f}' for v in std]}  ({time.time() - start:.0f}s)")

    # ------------------------------------------------------------------
    # 2. Build augmented train and clean eval transforms.
    # ------------------------------------------------------------------
    train_transform = get_train_transform(mean, std, crop_size=args.crop_size, use_hflip=args.use_hflip)
    eval_transform = get_eval_transform(mean, std, resize_to=args.resize_to, crop_size=args.crop_size)

    loader_kw = dict(batch_size=args.batch_size, num_workers=args.workers, seed=args.seed)
    loaders = {
        "train": make_cnn_loader(splits["train"], train_transform, shuffle=True, **loader_kw),
        "val":   make_cnn_loader(splits["val"],   eval_transform,  **loader_kw),
        "test":  make_cnn_loader(splits["test"],  eval_transform,  **loader_kw),
    }

    # ------------------------------------------------------------------
    # 3. Model — move to channels-last for better GPU throughput.
    # ------------------------------------------------------------------
    model_kwargs = dict(num_classes=num_classes, dropout=args.dropout)
    model = GWTNet(**model_kwargs)
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    # ------------------------------------------------------------------
    # 4. Optimizer: AdamW with weight decay excluded from BN and biases.
    # ------------------------------------------------------------------
    no_decay_names = {"bn", "bias"}
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if any(nd in name for nd in no_decay_names):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
    )

    # ------------------------------------------------------------------
    # 5. OneCycleLR scheduler (per-step, 15 % warmup then cosine decay).
    # ------------------------------------------------------------------
    steps_per_epoch = len(loaders["train"])
    total_steps = args.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.15,
        anneal_strategy="cos",
    )

    # ------------------------------------------------------------------
    # 6. Loss: cross-entropy with label smoothing.
    # ------------------------------------------------------------------
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Mixed-precision scaler — None on CPU (autocast is a no-op there too).
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ------------------------------------------------------------------
    # 7. Training loop with early stopping on val loss.
    # ------------------------------------------------------------------
    history, best_val, best_state, bad_epochs, best_epoch = [], float("inf"), None, 0, 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, _, _ = cnn_run_epoch(
            model, loaders["train"], device, criterion, optimizer, scaler, scheduler
        )
        val_loss, val_probs, val_y = cnn_run_epoch(model, loaders["val"], device, criterion)
        val_top1 = cnn_topk_accuracy(val_probs, val_y, 1)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_top1": val_top1}
        )
        print(
            f"epoch {epoch:3d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}"
            f" | val top-1 {val_top1:.4f} | {time.time() - start:.0f}s"
        )
        if val_loss < best_val:
            best_val, bad_epochs, best_epoch = val_loss, 0, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping (no val improvement for {args.patience} epochs)")
                break
    model.load_state_dict(best_state)

    # ------------------------------------------------------------------
    # 8. Final evaluation — val with best weights, test evaluated once.
    # ------------------------------------------------------------------
    results = {
        "model": "cnn1_gwtnet",
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "countries": countries,
        "history": history,
        "majority_class_baseline": 1 / num_classes,
        "channel_mean": mean,
        "channel_std": std,
    }
    for name in ("val", "test"):
        loss, probs, y = cnn_run_epoch(model, loaders[name], device, criterion)
        preds = probs.argmax(1).numpy()
        report = classification_report(
            y.numpy(), preds,
            labels=range(num_classes),
            target_names=countries,
            output_dict=True,
            zero_division=0,
        )
        results[name] = {
            "loss": loss,
            "top1_acc": cnn_topk_accuracy(probs, y, 1),
            "top5_acc": cnn_topk_accuracy(probs, y, 5),
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall":    report["macro avg"]["recall"],
            "macro_f1":        report["macro avg"]["f1-score"],
            "per_class":       {c: report[c] for c in countries},
            "confusion_matrix": confusion_matrix(
                y.numpy(), preds, labels=range(num_classes)
            ).tolist(),
        }

    test = results["test"]
    print(
        f"\nTEST ({len(y)} images) | top-1 {test['top1_acc']:.4f}"
        f" | top-5 {test['top5_acc']:.4f}"
        f" | macro F1 {test['macro_f1']:.4f}"
        f" | chance {1 / num_classes:.4f}"
    )
    if num_classes <= 20:
        print(
            classification_report(
                y.numpy(), preds,
                labels=range(num_classes),
                target_names=countries,
                zero_division=0,
            )
        )
        print("Confusion matrix (rows = true, cols = predicted):")
        print(np.array(test["confusion_matrix"]))

    checkpoint_info = {
        "model": model,
        "model_kwargs": model_kwargs,
        "epoch": best_epoch,
        "normalization": {"mean": mean, "std": std},
    }
    return results, checkpoint_info


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Index dataset ---
    by_country = index_dataset(args.data_dir)
    if args.list_countries:
        for country, paths in by_country.items():
            print(f"{country:30s} {len(paths)}")
        print(f"{len(by_country)} countries, {sum(map(len, by_country.values()))} images")
        return

    # --- Resolve country list ---
    rng = random.Random(args.seed)
    if args.countries:
        missing = [c for c in args.countries if c not in by_country]
        if missing:
            raise SystemExit(
                f"Unknown countries {missing}. Use --list-countries to see valid names."
            )
        countries = list(args.countries)
    else:
        countries = sorted(rng.sample(list(by_country), args.num_classes))

    num_classes = len(countries)
    if num_classes < 2:
        raise SystemExit("Need at least 2 countries.")
    print(f"Device: {device} | {num_classes} countries: {countries}")

    # --- Split ---
    splits = split_per_country(
        by_country, countries,
        args.val_frac, args.test_frac,
        args.max_per_class, rng,
    )
    for name, items in splits.items():
        print(f"  {name:5s}: {len(items)} images")

    # --- Leakage check (aborts before training if any image is in two splits) ---
    leakage = check_leakage(splits)

    # --- Dispatch to the chosen model pipeline ---
    # Hyperparameters shared by every model; each branch adds its own to name
    # the weights folder.
    common_hparams = dict(
        lr=args.lr, bs=args.batch_size, wd=args.weight_decay,
        ep=args.epochs, seed=args.seed,
    )
    if args.model == "baseline":
        results, checkpoint_info = run_baseline_pipeline(args, countries, splits, device)
        out_dir = RESULTS_DIR / "baseline"
        out_name = f"logreg_{num_classes}cls_{args.image_size}px_seed{args.seed}.json"
        run_name = build_run_name(
            "logreg",
            dict(cls=num_classes, px=args.image_size, **common_hparams),
        )
    elif args.model == "cnn_1":
        results, checkpoint_info = run_cnn1_pipeline(args, countries, splits, device)
        out_dir = RESULTS_DIR / "cnn_1"
        out_name = f"gwtnet_{num_classes}cls_{args.crop_size}px_seed{args.seed}.json"
        run_name = build_run_name(
            "gwtnet",
            dict(
                cls=num_classes, px=args.crop_size, **common_hparams,
                drop=args.dropout, ls=args.label_smoothing, hflip=args.use_hflip,
            ),
        )
    else:
        raise SystemExit(f"Unknown model '{args.model}'. Choices: {SUPPORTED_MODELS}")
    results["leakage_check"] = leakage

    # --- Persist results ---
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / out_name
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {out}")

    # --- Persist weights ---
    if args.save_weights:
        metric_keys = ("loss", "top1_acc", "top5_acc", "macro_f1")
        ckpt_path = save_checkpoint(
            args.weights_dir / args.model / run_name,
            checkpoint_info["model"],
            model_name=results["model"],
            model_kwargs=checkpoint_info["model_kwargs"],
            epoch=checkpoint_info["epoch"],
            countries=countries,
            config=results["config"],
            metrics={s: {k: results[s][k] for k in metric_keys} for s in ("val", "test")},
            normalization=checkpoint_info["normalization"],
        )
        print(f"Saved weights to {ckpt_path}")


if __name__ == "__main__":
    main()

