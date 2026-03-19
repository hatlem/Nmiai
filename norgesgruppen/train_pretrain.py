"""Two-phase pre-training pipeline for NorgesGruppen object detection.

Phase 1: Pre-train YOLOv8x on SKU-110K (~11K shelf images, single-class)
Phase 2: Fine-tune on NorgesGruppen competition data (248 images, single-class)

Progressive resolution: 640 -> 1280 for better feature learning.
CLAHE applied to match inference preprocessing.

Requires: ultralytics==8.1.0

Usage:
    # Full pipeline (SKU-110K pre-train + NorgesGruppen fine-tune):
    python train_pretrain.py

    # Skip phase 1 if already pre-trained:
    python train_pretrain.py --skip-pretrain --pretrained-weights runs/pretrain/weights/best.pt

    # Custom paths:
    python train_pretrain.py --sku-data data/sku110k_yolo/data.yaml --ng-data data/yolo_single_class/data.yaml

    # Resume phase 1:
    python train_pretrain.py --resume-pretrain
"""

import argparse
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO


def apply_clahe_to_dataset(data_yaml: str) -> None:
    """Apply CLAHE to all images in a YOLO dataset (in-place).

    Matches the CLAHE preprocessing used at inference time.
    Skips if already applied (checks marker file).
    """
    data_path = Path(data_yaml)
    dataset_dir = data_path.parent

    clahe_marker = dataset_dir / ".clahe_done"
    if clahe_marker.exists():
        print("  CLAHE already applied, skipping")
        return

    print("  Applying CLAHE to images...")
    total = 0

    for split in ["train", "val"]:
        img_dir = dataset_dir / "images" / split
        if not img_dir.exists():
            continue

        count = 0
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            # Skip symlinks — we need to copy first, then enhance
            if img_path.is_symlink():
                real_path = img_path.resolve()
                img_path.unlink()
                shutil.copy2(real_path, img_path)

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l_ch)
            lab_enhanced = cv2.merge([l_enhanced, a_ch, b_ch])
            result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

            cv2.imwrite(str(img_path), result)
            count += 1

        print(f"    {split}: {count} images")
        total += count

    clahe_marker.write_text("done")
    print(f"  Total: {total} images enhanced")


def phase1_pretrain(
    model_name: str,
    sku_data: str,
    epochs: int,
    batch: int,
    device: str,
    resume: bool,
    apply_clahe: bool,
) -> Path:
    """Phase 1: Pre-train on SKU-110K at 640px resolution.

    Returns path to best.pt weights.
    """
    print("\n" + "=" * 60)
    print("PHASE 1: Pre-train on SKU-110K")
    print("=" * 60)

    if apply_clahe:
        apply_clahe_to_dataset(sku_data)

    model = YOLO(model_name)

    results = model.train(
        data=sku_data,
        epochs=epochs,
        batch=batch,
        imgsz=640,  # Start at lower resolution for pre-training
        device=device,
        workers=8,
        patience=20,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        project="runs",
        name="pretrain",
        exist_ok=True,
        # Optimizer — standard LR for pre-training
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        # Augmentation — moderate for large dataset
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=3.0,
        translate=0.1,
        scale=0.4,
        shear=1.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.0,
        close_mosaic=10,
        resume=resume,
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = Path(results.save_dir) / "weights" / "last.pt"

    print(f"\nPhase 1 complete. Best weights: {best_pt}")
    return best_pt


def phase2_finetune(
    pretrained_weights: Path,
    ng_data: str,
    epochs: int,
    batch: int,
    device: str,
    apply_clahe: bool,
) -> Path:
    """Phase 2: Fine-tune on NorgesGruppen competition data at 1280px.

    Uses lower learning rate and stronger augmentation for small dataset.
    Returns path to best.pt weights.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tune on NorgesGruppen")
    print("=" * 60)

    if apply_clahe:
        apply_clahe_to_dataset(ng_data)

    print(f"  Loading pre-trained weights: {pretrained_weights}")
    model = YOLO(str(pretrained_weights))

    results = model.train(
        data=ng_data,
        epochs=epochs,
        batch=batch,
        imgsz=1280,  # High resolution for fine-tuning
        device=device,
        workers=8,
        patience=80,
        save=True,
        save_period=25,
        val=True,
        plots=True,
        project="runs",
        name="finetune",
        exist_ok=True,
        # Optimizer — lower LR for transfer learning
        optimizer="AdamW",
        lr0=0.0002,  # 5x lower than pre-training
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,
        # Augmentation — strong for small dataset (248 images)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.15,
        scale=0.5,
        shear=2.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        close_mosaic=30,
        resume=False,
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = Path(results.save_dir) / "weights" / "last.pt"

    print(f"\nPhase 2 complete. Best weights: {best_pt}")
    return best_pt


def main():
    parser = argparse.ArgumentParser(
        description="Two-phase pre-training: SKU-110K -> NorgesGruppen"
    )
    # Data paths
    parser.add_argument("--sku-data", default="data/sku110k_yolo/data.yaml",
                        help="SKU-110K YOLO data.yaml (from prepare_sku110k.py)")
    parser.add_argument("--ng-data", default="data/yolo_single_class/data.yaml",
                        help="NorgesGruppen single-class YOLO data.yaml")

    # Model
    parser.add_argument("--model", default="yolov8x.pt",
                        help="Base YOLO model for phase 1")

    # Phase 1 settings
    parser.add_argument("--pretrain-epochs", type=int, default=50,
                        help="Epochs for phase 1 (SKU-110K pre-training)")
    parser.add_argument("--pretrain-batch", type=int, default=16,
                        help="Batch size for phase 1 (640px, can be larger)")
    parser.add_argument("--skip-pretrain", action="store_true",
                        help="Skip phase 1, go directly to fine-tuning")
    parser.add_argument("--resume-pretrain", action="store_true",
                        help="Resume interrupted phase 1 training")
    parser.add_argument("--pretrained-weights", type=Path, default=None,
                        help="Path to pre-trained weights (skips phase 1)")

    # Phase 2 settings
    parser.add_argument("--finetune-epochs", type=int, default=300,
                        help="Epochs for phase 2 (NorgesGruppen fine-tuning)")
    parser.add_argument("--finetune-batch", type=int, default=4,
                        help="Batch size for phase 2 (1280px, needs more VRAM)")

    # General
    parser.add_argument("--device", default="0", help="GPU device")
    parser.add_argument("--no-clahe", action="store_true",
                        help="Skip CLAHE preprocessing")
    parser.add_argument("--output", type=Path, default=Path("best.pt"),
                        help="Where to copy final best.pt for submission")

    args = parser.parse_args()

    apply_clahe = not args.no_clahe

    # Phase 1: Pre-train on SKU-110K
    if args.skip_pretrain or args.pretrained_weights:
        if args.pretrained_weights:
            pretrained = args.pretrained_weights
        else:
            # Look for existing pre-trained weights
            pretrained = Path("runs/pretrain/weights/best.pt")
            if not pretrained.exists():
                pretrained = Path("runs/pretrain/weights/last.pt")
        if not pretrained.exists():
            print(f"ERROR: Pre-trained weights not found at {pretrained}")
            print("Run without --skip-pretrain first, or specify --pretrained-weights")
            return
        print(f"Using existing pre-trained weights: {pretrained}")
    else:
        sku_data_path = Path(args.sku_data)
        if not sku_data_path.exists():
            print(f"ERROR: SKU-110K data not found at {args.sku_data}")
            print("Run prepare_sku110k.py first:")
            print("  python prepare_sku110k.py")
            return
        pretrained = phase1_pretrain(
            model_name=args.model,
            sku_data=args.sku_data,
            epochs=args.pretrain_epochs,
            batch=args.pretrain_batch,
            device=args.device,
            resume=args.resume_pretrain,
            apply_clahe=apply_clahe,
        )

    # Phase 2: Fine-tune on NorgesGruppen
    ng_data_path = Path(args.ng_data)
    if not ng_data_path.exists():
        print(f"ERROR: NorgesGruppen data not found at {args.ng_data}")
        print("Run convert_single_class.py first.")
        return

    final_weights = phase2_finetune(
        pretrained_weights=pretrained,
        ng_data=args.ng_data,
        epochs=args.finetune_epochs,
        batch=args.finetune_batch,
        device=args.device,
        apply_clahe=apply_clahe,
    )

    # Copy final weights for submission
    if final_weights.exists():
        shutil.copy2(final_weights, args.output)
        size_mb = args.output.stat().st_size / 1024 / 1024
        print(f"\n{'=' * 60}")
        print(f"DONE! Final model: {args.output} ({size_mb:.1f} MB)")
        print(f"{'=' * 60}")

        if size_mb > 420:
            print(f"\nWARNING: Model is {size_mb:.1f} MB — exceeds 420 MB sandbox limit!")
            print("Consider using yolov8l.pt or yolov8m.pt instead of yolov8x.pt")
    else:
        print(f"\nERROR: Final weights not found at {final_weights}")


if __name__ == "__main__":
    main()
