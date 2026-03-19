"""Download SKU110K dataset and merge with NorgesGruppen for better detection.

SKU110K has 11,762 shelf images with 1.7M bounding boxes — all single-class.
Combined with our NorgesGruppen data, this dramatically improves detection.

Option 1 (easiest): Ultralytics auto-downloads with SKU-110K.yaml
Option 2 (manual): Download tarball and convert CSV to YOLO format

Usage:
    # Easiest — just train directly (ultralytics handles download):
    python train.py --data SKU-110K.yaml --model yolov8x.pt --epochs 50

    # Manual download + merge with NorgesGruppen:
    python download_sku110k.py
    python download_sku110k.py --skip-download  # if already downloaded
"""

import argparse
import csv
import random
import shutil
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path("data")
SKU_DIR = DATA_DIR / "SKU110K_fixed"
MERGED_DIR = DATA_DIR / "yolo_single_class_merged"

DOWNLOAD_URL = "http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz"


def download_sku110k():
    """Download SKU110K from the official S3 bucket (~13.6 GB)."""
    tarball = DATA_DIR / "SKU110K_fixed.tar.gz"

    if SKU_DIR.exists() and (SKU_DIR / "images").exists():
        img_count = len(list((SKU_DIR / "images").glob("*.jpg")))
        if img_count > 1000:
            print(f"SKU110K already downloaded ({img_count} images)")
            return True

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not tarball.exists():
        print(f"Downloading SKU110K (~13.6 GB)...")
        print(f"URL: {DOWNLOAD_URL}")
        subprocess.check_call(["wget", "-c", DOWNLOAD_URL, "-O", str(tarball)])

    print("Extracting...")
    subprocess.check_call(["tar", "-xzf", str(tarball), "-C", str(DATA_DIR)])
    print("Done extracting")
    return True


def convert_sku110k_to_yolo():
    """Convert SKU110K CSV annotations to YOLO single-class format."""
    annotations_dir = SKU_DIR / "annotations"
    images_dir = SKU_DIR / "images"
    labels_dir = SKU_DIR / "labels"

    if not annotations_dir.exists():
        print(f"ERROR: {annotations_dir} not found")
        return 0
    if not images_dir.exists():
        print(f"ERROR: {images_dir} not found")
        return 0

    total_images = 0
    total_boxes = 0

    for split in ["train", "val", "test"]:
        csv_path = annotations_dir / f"annotations_{split}.csv"
        if not csv_path.exists():
            print(f"  Skipping {split} — no annotations file")
            continue

        split_labels = labels_dir / split
        split_labels.mkdir(parents=True, exist_ok=True)

        # CSV columns: image, x1, y1, x2, y2, class, image_width, image_height
        annotations: dict[str, dict] = {}
        with open(csv_path) as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 8:
                    continue
                img_name = row[0].strip()
                x1, y1, x2, y2 = float(row[1]), float(row[2]), float(row[3]), float(row[4])
                img_w, img_h = float(row[6]), float(row[7])

                if img_name not in annotations:
                    annotations[img_name] = {"w": img_w, "h": img_h, "boxes": []}
                annotations[img_name]["boxes"].append((x1, y1, x2, y2))

        split_boxes = 0
        for img_name, data in annotations.items():
            w, h = data["w"], data["h"]
            lines = []
            for x1, y1, x2, y2 in data["boxes"]:
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                bw = max(0.001, min(1.0, bw))
                bh = max(0.001, min(1.0, bh))
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                split_boxes += 1

            label_name = Path(img_name).stem + ".txt"
            (split_labels / label_name).write_text("\n".join(lines) + "\n")

        total_boxes += split_boxes
        total_images += len(annotations)
        print(f"  {split}: {len(annotations)} images, {split_boxes} boxes")

    print(f"Total: {total_images} images, {total_boxes} boxes")
    return total_images


def merge_datasets():
    """Merge NorgesGruppen single-class + SKU110K into one training set."""
    ng_dir = DATA_DIR / "yolo_single_class"
    sku_images = SKU_DIR / "images"
    sku_labels = SKU_DIR / "labels"

    if not ng_dir.exists():
        print(f"ERROR: {ng_dir} not found. Run convert_single_class.py first.")
        return

    for split in ["train", "val"]:
        (MERGED_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (MERGED_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Copy NorgesGruppen data (symlink)
    for split in ["train", "val"]:
        for subdir in ["images", "labels"]:
            src = ng_dir / subdir / split
            if not src.exists():
                continue
            for f in src.iterdir():
                link = MERGED_DIR / subdir / split / f.name
                if not link.exists():
                    target = f.resolve() if f.is_symlink() else f
                    link.symlink_to(target)

    # Add SKU110K train images
    sku_train_labels = sku_labels / "train"
    sku_count = 0
    if sku_train_labels.exists():
        for lbl_file in sorted(sku_train_labels.iterdir()):
            img_name = lbl_file.stem + ".jpg"
            img_file = sku_images / img_name
            if not img_file.exists():
                continue

            # Prefix to avoid name conflicts
            new_img = MERGED_DIR / "images" / "train" / f"sku_{img_name}"
            new_lbl = MERGED_DIR / "labels" / "train" / f"sku_{lbl_file.name}"

            if not new_img.exists():
                new_img.symlink_to(img_file.resolve())
            if not new_lbl.exists():
                new_lbl.symlink_to(lbl_file.resolve())
            sku_count += 1

    ng_train = len(list((MERGED_DIR / "images" / "train").iterdir())) - sku_count
    print(f"Merged: {ng_train} NorgesGruppen + {sku_count} SKU110K train images")

    # data.yaml
    yaml_content = f"""\
path: {MERGED_DIR.resolve()}
train: images/train
val: images/val

nc: 1
names:
  0: product
"""
    (MERGED_DIR / "data.yaml").write_text(yaml_content)
    print(f"Wrote {MERGED_DIR / 'data.yaml'}")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare SKU110K")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    if not args.skip_download:
        print("=== Downloading SKU110K ===")
        download_sku110k()

    print("\n=== Converting to YOLO format ===")
    count = convert_sku110k_to_yolo()

    if count > 0:
        print("\n=== Merging with NorgesGruppen data ===")
        merge_datasets()

    print("\nDone!")
    print("\nTraining options:")
    print("  # Option A — Ultralytics auto-download (easiest, SKU110K only):")
    print("  python train.py --data SKU-110K.yaml --model yolov8x.pt --epochs 50")
    print("")
    print("  # Option B — Merged NorgesGruppen + SKU110K:")
    print(f"  python train.py --data {MERGED_DIR / 'data.yaml'} --model yolov8x.pt")


if __name__ == "__main__":
    main()
