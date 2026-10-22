"""General-purpose model pipeline runner.

Parses command-line arguments, indexes the dataset, builds train/val/test
splits, computes pixel statistics, and runs the selected model pipeline.

Currently supported models
--------------------------
  baseline  – single linear layer logistic regression (no hidden layers)

Usage examples (run from the repo root)
---------------------------------------
    python scripts/run_pipeline.py                          # baseline, 2 random countries
    python scripts/run_pipeline.py --countries FR JP
    python scripts/run_pipeline.py --num-classes 10
    python scripts/run_pipeline.py --list-countries
    python scripts/run_pipeline.py --model baseline --epochs 100
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

from common.dataset import make_loader
from common.preprocessing import check_leakage, feature_stats, index_dataset, split_per_country
from common.seed import set_seed
from baseline.model import LogisticRegression
from baseline.trainer import run_epoch, topk_accuracy

REPO_ROOT = SCRIPTS_DIR.parent
DATA_DIR = REPO_ROOT / "data" / "raw" / "streetview-by-country"
RESULTS_DIR = REPO_ROOT / "results"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

SUPPORTED_MODELS = ("baseline",)


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
        help="DataLoader worker processes (Colab has 2 CPUs)",
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
        Results dict ready to be serialised as JSON.
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

    model = LogisticRegression(mean.numel(), num_classes).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    history, best_val, best_state, bad_epochs = [], float("inf"), None, 0
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
            best_val, bad_epochs = val_loss, 0
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

    return results


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
    if args.model == "baseline":
        results = run_baseline_pipeline(args, countries, splits, device)
        out_dir = RESULTS_DIR / "baseline"
        out_name = f"logreg_{num_classes}cls_{args.image_size}px_seed{args.seed}.json"
    else:
        raise SystemExit(f"Unknown model '{args.model}'. Choices: {SUPPORTED_MODELS}")
    results["leakage_check"] = leakage

    # --- Persist results ---
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / out_name
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()

