"""Prepare SKU-110K dataset for YOLO pre-training.

Converts SKU-110K CSV annotations to YOLO single-class format,
creates train/val split, generates data.yaml, and optionally
merges with competition data for combined training.

SKU-110K format (CSV): image_name, x1, y1, x2, y2, class, image_width, image_height

Usage:
    # Convert SKU-110K only:
    python prepare_sku110k.py

    # Also merge with competition data:
    python prepare_sku110k.py --merge

    # Custom paths:
    python prepare_sku110k.py --sku-dir data/SKU110K_fixed --out-dir data/sku110k_yolo
"""

import argparse
import csv
import random
from pathlib import Path


def convert_csv_to_yolo(csv_path: Path, images_dir: Path, output_labels_dir: Path) -> dict:
    """Convert a SKU-110K CSV annotation file to YOLO format labels.

    Returns dict with stats: {image_count, box_count, skipped}.
    """
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    annotations: dict[str, dict] = {}
    with open(csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 8:
                continue
            img_name = row[0].strip()
            try:
                x1, y1, x2, y2 = float(row[1]), float(row[2]), float(row[3]), float(row[4])
                img_w, img_h = float(row[6]), float(row[7])
            except (ValueError, IndexError):
                continue

            if img_w <= 0 or img_h <= 0:
                continue

            if img_name not in annotations:
                annotations[img_name] = {"w": img_w, "h": img_h, "boxes": []}
            annotations[img_name]["boxes"].append((x1, y1, x2, y2))

    box_count = 0
    skipped = 0
    for img_name, data in annotations.items():
        # Verify image exists
        img_path = images_dir / img_name
        if not img_path.exists():
            skipped += 1
            continue

        w, h = data["w"], data["h"]
        lines = []
        for x1, y1, x2, y2 in data["boxes"]:
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            # Clamp to valid range
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            bw = max(0.001, min(1.0, bw))
            bh = max(0.001, min(1.0, bh))

            # Filter out degenerate boxes
            if bw < 0.005 or bh < 0.005:
                continue

            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            box_count += 1

        if lines:
            label_name = Path(img_name).stem + ".txt"
            (output_labels_dir / label_name).write_text("\n".join(lines) + "\n")

    return {
        "image_count": len(annotations) - skipped,
        "box_count": box_count,
        "skipped": skipped,
    }


def prepare_sku110k(sku_dir: Path, out_dir: Path) -> Path:
    """Convert full SKU-110K dataset to YOLO format with proper splits.

    Returns path to the generated data.yaml.
    """
    annotations_dir = sku_dir / "annotations"
    images_dir = sku_dir / "images"

    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_boxes = 0

    for split in ["train", "val", "test"]:
        csv_path = annotations_dir / f"annotations_{split}.csv"
        if not csv_path.exists():
            print(f"  Skipping {split} — {csv_path.name} not found")
            continue

        # Map test -> val for YOLO (we want train + val only)
        yolo_split = "val" if split == "test" else split

        # Create output directories
        split_images_dir = out_dir / "images" / yolo_split
        split_labels_dir = out_dir / "labels" / yolo_split
        split_images_dir.mkdir(parents=True, exist_ok=True)

        # Convert annotations
        label_dir = out_dir / "labels" / yolo_split
        stats = convert_csv_to_yolo(csv_path, images_dir, label_dir)

        # Symlink images that have labels
        linked = 0
        for label_file in label_dir.iterdir():
            if label_file.suffix != ".txt":
                continue
            img_name = label_file.stem + ".jpg"
            src_img = images_dir / img_name
            dst_img = split_images_dir / f"sku_{img_name}"
            if src_img.exists() and not dst_img.exists():
                dst_img.symlink_to(src_img.resolve())
                linked += 1

        print(f"  {split} -> {yolo_split}: {stats['image_count']} images, "
              f"{stats['box_count']} boxes, {linked} linked")

        total_images += stats["image_count"]
        total_boxes += stats["box_count"]

    # Generate data.yaml
    yaml_path = out_dir / "data.yaml"
    yaml_content = f"""\
path: {out_dir.resolve()}
train: images/train
val: images/val

nc: 1
names:
  0: product
"""
    yaml_path.write_text(yaml_content)

    print(f"\nTotal: {total_images} images, {total_boxes} boxes")
    print(f"data.yaml: {yaml_path}")
    return yaml_path


def merge_with_competition(sku_dir: Path, ng_dir: Path, merged_dir: Path) -> Path:
    """Merge SKU-110K YOLO data with NorgesGruppen competition data.

    Both datasets must already be in YOLO single-class format.
    Returns path to merged data.yaml.
    """
    merged_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        (merged_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (merged_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"ng": 0, "sku": 0}

    # Copy NorgesGruppen data (symlinks)
    for split in ["train", "val"]:
        for subdir in ["images", "labels"]:
            src = ng_dir / subdir / split
            if not src.exists():
                continue
            for f in src.iterdir():
                link = merged_dir / subdir / split / f"ng_{f.name}"
                if not link.exists():
                    target = f.resolve() if f.is_symlink() else f
                    link.symlink_to(target)
                    if subdir == "images" and split == "train":
                        counts["ng"] += 1

    # Copy SKU-110K data (symlinks)
    for split in ["train", "val"]:
        for subdir in ["images", "labels"]:
            src = sku_dir / subdir / split
            if not src.exists():
                continue
            for f in src.iterdir():
                # Already prefixed with sku_ from prepare step
                link = merged_dir / subdir / split / f.name
                if not link.exists():
                    target = f.resolve() if f.is_symlink() else f
                    link.symlink_to(target)
                    if subdir == "images" and split == "train":
                        counts["sku"] += 1

    print(f"Merged train: {counts['ng']} NorgesGruppen + {counts['sku']} SKU-110K images")

    # Generate data.yaml
    yaml_path = merged_dir / "data.yaml"
    yaml_content = f"""\
path: {merged_dir.resolve()}
train: images/train
val: images/val

nc: 1
names:
  0: product
"""
    yaml_path.write_text(yaml_content)
    print(f"data.yaml: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Prepare SKU-110K for YOLO pre-training")
    parser.add_argument("--sku-dir", type=Path, default=Path("data/SKU110K_fixed"),
                        help="Path to extracted SKU110K_fixed directory")
    parser.add_argument("--out-dir", type=Path, default=Path("data/sku110k_yolo"),
                        help="Output directory for YOLO-formatted SKU-110K")
    parser.add_argument("--merge", action="store_true",
                        help="Also merge with NorgesGruppen competition data")
    parser.add_argument("--ng-dir", type=Path, default=Path("data/yolo_single_class"),
                        help="NorgesGruppen single-class YOLO directory")
    parser.add_argument("--merged-dir", type=Path, default=Path("data/yolo_single_class_merged"),
                        help="Output directory for merged dataset")
    args = parser.parse_args()

    print("=== Converting SKU-110K to YOLO format ===")
    sku_yaml = prepare_sku110k(args.sku_dir, args.out_dir)

    if args.merge:
        if not args.ng_dir.exists():
            print(f"\nERROR: NorgesGruppen data not found at {args.ng_dir}")
            print("Run convert_single_class.py first.")
            return

        print("\n=== Merging with NorgesGruppen data ===")
        merged_yaml = merge_with_competition(args.out_dir, args.ng_dir, args.merged_dir)
        print(f"\nMerged dataset ready: {merged_yaml}")

    print("\nDone! Next steps:")
    print(f"  # Pre-train on SKU-110K only:")
    print(f"  python train_pretrain.py --sku-data {sku_yaml}")
    if args.merge:
        print(f"\n  # Or train on merged data:")
        print(f"  python train.py --data {args.merged_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
