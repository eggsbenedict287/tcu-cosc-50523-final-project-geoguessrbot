"""Logistic regression baseline for country classification (PyTorch).

Each image is kept at its native 640x640, flattened to raw RGB pixels
(1,228,800 features) and standardized per feature with train-set statistics.
Images are streamed from disk each epoch rather than held in RAM.
The model is a single linear layer:
  * 2 countries  -> binary logistic regression (sigmoid + BCE loss)
  * >2 countries -> multinomial logistic regression (softmax + cross-entropy)
No hidden layers, so this is the "skeleton" baseline the CNN has to beat.

Data protocol: images are split train/val/test *within each country*
(stratified, seeded), so every class appears in every split. The val split
picks the best epoch; the test split is evaluated exactly once at the end.

Expects the dataset from notebooks/init.ipynb at
data/raw/streetview-by-country/, with images in one folder per country named
by its two-letter code (FR = France, JP = Japan); the folder name is the label.
Run from the repo root, e.g. in Colab:

    !python scripts/baseline_model.py                          # 2 random countries
    !python scripts/baseline_model.py --countries FR JP
    !python scripts/baseline_model.py --num-classes 10         # multiclass
    !python scripts/baseline_model.py --list-countries
"""
import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "raw" / "streetview-by-country"
RESULTS_DIR = REPO_ROOT / "results" / "baseline"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--countries", nargs="+", help="Country folder codes to use, e.g. FR JP")
    p.add_argument("--num-classes", type=int, default=2, help="Random countries to sample if --countries not given")
    p.add_argument("--max-per-class", type=int, default=None, help="Cap images per country (for quick runs)")
    p.add_argument("--image-size", type=int, default=640, help="Images not already SIZE x SIZE are resized to it")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64, help="64 x 1.2M floats is ~315 MB per batch on the GPU")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularization")
    p.add_argument("--patience", type=int, default=5, help="Early stopping patience on val loss")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=2, help="DataLoader worker processes (Colab has 2 CPUs)")
    p.add_argument("--list-countries", action="store_true", help="Print countries and image counts, then exit")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def index_dataset(data_dir):
    """Map country name -> sorted list of image paths (label = parent folder name)."""
    by_country = defaultdict(list)
    for path in data_dir.rglob("*"):
        if path.suffix.lower() in IMAGE_EXTS:
            by_country[path.parent.name].append(path)
    if not by_country:
        raise SystemExit(f"No images found under {data_dir}. Run the dataset cell in notebooks/init.ipynb first.")
    return {c: sorted(paths) for c, paths in sorted(by_country.items())}


def split_per_country(by_country, countries, val_frac, test_frac, max_per_class, rng):
    """Stratified split: shuffle each country's images and cut into train/val/test."""
    splits = {"train": [], "val": [], "test": []}
    for label, country in enumerate(countries):
        paths = list(by_country[country])
        rng.shuffle(paths)
        if max_per_class:
            paths = paths[:max_per_class]
        n_test = int(round(len(paths) * test_frac))
        n_val = int(round(len(paths) * val_frac))
        splits["test"] += [(p, label) for p in paths[:n_test]]
        splits["val"] += [(p, label) for p in paths[n_test:n_test + n_val]]
        splits["train"] += [(p, label) for p in paths[n_test + n_val:]]
    return splits


class ImageDataset(Dataset):
    """Streams images from disk as flat uint8 pixel vectors (size*size*3).

    At 640x640 each image is 1.2M features, far too large to preload a whole
    split into RAM, so images are decoded per batch by DataLoader workers.
    """

    def __init__(self, items, size):
        self.items, self.size = items, size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        with Image.open(path) as img:
            img = img.convert("RGB")
            if img.size != (self.size, self.size):
                img = img.resize((self.size, self.size), Image.BILINEAR)
            return torch.from_numpy(np.array(img, dtype=np.uint8).reshape(-1)), label


def make_loader(items, args, shuffle=False):
    return DataLoader(ImageDataset(items, args.image_size), batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.workers, pin_memory=torch.cuda.is_available(),
                      generator=torch.Generator().manual_seed(args.seed) if shuffle else None)


def feature_stats(loader, device):
    """Per-feature mean/std over the train split, accumulated in one streaming pass."""
    total, total_sq, n = None, None, 0
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True).double() / 255.0
        total = xb.sum(0) if total is None else total + xb.sum(0)
        total_sq = (xb ** 2).sum(0) if total_sq is None else total_sq + (xb ** 2).sum(0)
        n += len(xb)
    mean = total / n
    std = (total_sq / n - mean ** 2).clamp_min(0).sqrt()
    return mean.float(), std.float() + 1e-6


class LogisticRegression(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.binary = num_classes == 2
        self.linear = nn.Linear(in_features, 1 if self.binary else num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out.squeeze(1) if self.binary else out

    def loss(self, logits, y):
        if self.binary:
            return nn.functional.binary_cross_entropy_with_logits(logits, y.float())
        return nn.functional.cross_entropy(logits, y)

    def probs(self, logits):
        """Per-class probabilities, shape (N, num_classes), for both cases."""
        if self.binary:
            p1 = torch.sigmoid(logits)
            return torch.stack([1 - p1, p1], dim=1)
        return torch.softmax(logits, dim=1)


def run_epoch(model, loader, mean, std, device, optimizer=None):
    """Train if optimizer is given, otherwise evaluate. Returns (mean loss, probs, labels)."""
    training = optimizer is not None
    model.train(training)
    total_loss, all_probs, all_labels = 0.0, [], []
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb = (xb.to(device, non_blocking=True).float() / 255.0 - mean) / std
            yb = yb.to(device)
            logits = model(xb)
            loss = model.loss(logits, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                all_probs.append(model.probs(logits).cpu())
                all_labels.append(yb.cpu())
            total_loss += loss.item() * len(yb)
    if training:
        return total_loss / len(loader.dataset), None, None
    return total_loss / len(loader.dataset), torch.cat(all_probs), torch.cat(all_labels)


def topk_accuracy(probs, y, k):
    k = min(k, probs.shape[1])
    return (probs.topk(k, dim=1).indices == y.unsqueeze(1)).any(dim=1).float().mean().item()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    by_country = index_dataset(args.data_dir)
    if args.list_countries:
        for country, paths in by_country.items():
            print(f"{country:30s} {len(paths)}")
        print(f"{len(by_country)} countries, {sum(map(len, by_country.values()))} images")
        return

    rng = random.Random(args.seed)
    if args.countries:
        missing = [c for c in args.countries if c not in by_country]
        if missing:
            raise SystemExit(f"Unknown countries {missing}. Use --list-countries to see valid names.")
        countries = list(args.countries)
    else:
        countries = sorted(rng.sample(list(by_country), args.num_classes))
    num_classes = len(countries)
    if num_classes < 2:
        raise SystemExit("Need at least 2 countries.")
    print(f"Device: {device} | {num_classes} countries: {countries}")

    splits = split_per_country(by_country, countries, args.val_frac, args.test_frac, args.max_per_class, rng)
    for name, items in splits.items():
        print(f"  {name:5s}: {len(items)} images")

    loaders = {"train": make_loader(splits["train"], args, shuffle=True),
               "val": make_loader(splits["val"], args),
               "test": make_loader(splits["test"], args)}

    # Per-feature standardization using train statistics only (no leakage from val/test).
    start = time.time()
    mean, std = feature_stats(make_loader(splits["train"], args), device)
    print(f"Computed train pixel statistics in {time.time() - start:.0f}s")

    model = LogisticRegression(mean.numel(), num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history, best_val, best_state, bad_epochs = [], float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, _, _ = run_epoch(model, loaders["train"], mean, std, device, optimizer)
        val_loss, val_probs, val_y = run_epoch(model, loaders["val"], mean, std, device)
        val_acc = topk_accuracy(val_probs, val_y, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})
        print(f"epoch {epoch:3d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | val acc {val_acc:.4f} "
              f"| {time.time() - start:.0f}s")
        if val_loss < best_val:
            best_val, bad_epochs = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping (no val improvement for {args.patience} epochs)")
                break
    model.load_state_dict(best_state)

    # Final evaluation on the held-out test split, once.
    results = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "countries": countries, "history": history,
               "majority_class_baseline": 1 / num_classes}
    for name in ("val", "test"):
        loss, probs, y = run_epoch(model, loaders[name], mean, std, device)
        preds = probs.argmax(1).numpy()
        report = classification_report(y.numpy(), preds, labels=range(num_classes), target_names=countries,
                                       output_dict=True, zero_division=0)
        results[name] = {
            "loss": loss,
            "top1_acc": topk_accuracy(probs, y, 1),
            "top5_acc": topk_accuracy(probs, y, 5),
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "macro_f1": report["macro avg"]["f1-score"],
            "per_class": {c: report[c] for c in countries},
            "confusion_matrix": confusion_matrix(y.numpy(), preds, labels=range(num_classes)).tolist(),
        }

    test = results["test"]
    print(f"\nTEST ({len(y)} images) | top-1 {test['top1_acc']:.4f} | top-5 {test['top5_acc']:.4f} "
          f"| macro F1 {test['macro_f1']:.4f} | chance {1 / num_classes:.4f}")
    if num_classes <= 20:
        # y / preds still hold the test split from the last loop iteration.
        print(classification_report(y.numpy(), preds, labels=range(num_classes), target_names=countries,
                                    zero_division=0))
        print("Confusion matrix (rows = true, cols = predicted):")
        print(np.array(test["confusion_matrix"]))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"logreg_{num_classes}cls_{args.image_size}px_seed{args.seed}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
