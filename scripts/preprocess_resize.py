"""Pre-resize the Streetview by Country images once so training is not CPU-bound.

Decoding 640x640 JPEGs on Colab's 2 CPUs is the training bottleneck, not the
GPU. This script converts every image to a SIZE x SIZE JPEG (default 256x256,
the standard "resize 256 -> crop 224" ImageNet recipe) and keeps the
one-folder-per-country layout, so the result is a drop-in replacement for
``data/raw/streetview-by-country`` (``index_dataset`` uses the parent folder
name as the label).

Properties
----------
* **Resumable**: images whose output file already exists are skipped, so a
  Colab disconnect only costs the file being written.
* **Non-square images** (the dataset is expected to be 640x640, but this is
  checked): the shorter edge is resized to SIZE and the longer edge is
  center-cropped, so nothing is stretched. The count is reported at the end.
* **Unreadable images** are logged and skipped, never fatal.
* ``--zip`` additionally writes an uncompressed zip (JPEGs do not compress)
  for caching on Google Drive. Unzip it into ``data/processed/`` to restore.

Usage (run from the repo root)
------------------------------
    python scripts/preprocess_resize.py                      # -> data/processed/streetview-256
    python scripts/preprocess_resize.py --size 320 --workers 4
    python scripts/preprocess_resize.py --zip /content/drive/MyDrive/tcu-cosc-50523-final-project-geoguessrbot/streetview-256.zip
    python scripts/preprocess_resize.py --limit 20           # quick smoke test (first 20 images per country)
"""
import argparse
import sys
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageOps

# Same extensions as common.preprocessing.IMAGE_EXTS, duplicated here so this
# script does not import torch (workers re-import it on macOS/spawn).
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = REPO_ROOT / "data" / "raw" / "streetview-by-country"


def resize_one(job: tuple[str, str, int, int]) -> tuple[str, str]:
    """Resize a single image. Returns ``(status, source_path)``.

    status is one of ``"ok"``, ``"skipped"`` (output already exists),
    ``"cropped"`` (input was not square) or ``"error"``.
    """
    src, dst, size, quality = job
    src_path, dst_path = Path(src), Path(dst)
    if dst_path.exists():
        return "skipped", src
    try:
        with Image.open(src_path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            status = "ok"
            if img.width != img.height:
                # Resize the short edge to `size`, center-crop the long edge.
                img = ImageOps.fit(img, (size, size), Image.LANCZOS)
                status = "cropped"
            else:
                img = img.resize((size, size), Image.LANCZOS)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            # Write to a temp name, then rename, so an interrupted write never
            # leaves a truncated file that a resumed run would skip.
            tmp = dst_path.with_suffix(".tmp")
            img.save(tmp, format="JPEG", quality=quality, optimize=True)
            tmp.replace(dst_path)
            return status, src
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
        print(f"ERROR {src}: {exc}", file=sys.stderr)
        return "error", src


def collect_jobs(src_dir: Path, dst_dir: Path, size: int, quality: int, limit: int | None) -> list:
    """Build one job per image, mirroring ``<country>/<file>`` under *dst_dir*."""
    per_country: dict[str, list[Path]] = {}
    for path in sorted(src_dir.rglob("*")):
        if path.suffix.lower() in IMAGE_EXTS:
            per_country.setdefault(path.parent.name, []).append(path)

    jobs, seen = [], {}
    for country, paths in per_country.items():
        for path in paths[:limit] if limit else paths:
            dst = dst_dir / country / (path.stem + ".jpg")
            if dst in seen:  # same stem with different extensions would overwrite
                raise SystemExit(f"Output collision: {path} and {seen[dst]} -> {dst}")
            seen[dst] = path
            jobs.append((str(path), str(dst), size, quality))
    return jobs


def write_zip(dst_dir: Path, zip_path: Path) -> None:
    """Store *dst_dir* in an uncompressed zip (JPEGs are already compressed)."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(dst_dir.rglob("*.jpg"))
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for path in files:
            zf.write(path, arcname=Path(dst_dir.name) / path.relative_to(dst_dir))
    print(f"Wrote {zip_path} ({zip_path.stat().st_size / 1e9:.2f} GB, {len(files)} images)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=DEFAULT_SRC, help="Raw dataset folder (one folder per country)")
    p.add_argument("--size", type=int, default=256, help="Output side length in pixels")
    p.add_argument("--dst", type=Path, default=None,
                   help="Output folder (default: data/processed/streetview-<size>)")
    p.add_argument("--quality", type=int, default=95, help="JPEG quality")
    p.add_argument("--workers", type=int, default=2, help="Worker processes (Colab has 2 CPUs)")
    p.add_argument("--limit", type=int, default=None, help="Only the first N images per country (smoke test)")
    p.add_argument("--zip", type=Path, default=None, help="Also write an uncompressed zip here (e.g. on Drive)")
    args = p.parse_args()

    dst_dir = args.dst or REPO_ROOT / "data" / "processed" / f"streetview-{args.size}"
    if not args.src.is_dir():
        raise SystemExit(f"Source folder {args.src} not found. Run the dataset cell in notebooks/init.ipynb first.")

    jobs = collect_jobs(args.src, dst_dir, args.size, args.quality, args.limit)
    if not jobs:
        raise SystemExit(f"No images found under {args.src}.")
    print(f"{len(jobs)} images | {args.src} -> {dst_dir} | {args.size}px q{args.quality} | {args.workers} workers")

    counts = {"ok": 0, "skipped": 0, "cropped": 0, "error": 0}
    start = time.time()
    with Pool(args.workers) as pool:
        for i, (status, _) in enumerate(pool.imap_unordered(resize_one, jobs, chunksize=64), 1):
            counts[status] += 1
            if i % 2000 == 0 or i == len(jobs):
                rate = i / (time.time() - start)
                print(f"  {i}/{len(jobs)} ({rate:.0f} img/s, ~{(len(jobs) - i) / rate / 60:.1f} min left)")

    print(
        f"Done in {(time.time() - start) / 60:.1f} min | converted {counts['ok']}"
        f" | skipped (already done) {counts['skipped']}"
        f" | non-square, center-cropped {counts['cropped']}"
        f" | errors {counts['error']}"
    )
    if counts["error"]:
        print("Some images failed; re-run the same command after checking the ERROR lines above.")

    if args.zip:
        write_zip(dst_dir, args.zip)


if __name__ == "__main__":
    main()
