"""Treningsskript for Vertex AI Custom Training Job.

Laster datasett fra GCS, trener YOLO26 med augmentations,
og eksporterer optimalisert .onnx-modell tilbake til GCS.

Kjøres på GCP, IKKE lokalt ved innlevering.

Bruk:
    # Lokalt (test)
    python train_vertex.py --data ./data --epochs 5 --export-only

    # På Vertex AI (settes opp via gcloud CLI eller Vertex UI)
    python train_vertex.py --data gs://bucket/dataset --epochs 300
"""

import argparse
import os
import subprocess
import sys


def install_training_deps():
    """Installer treningsavhengigheter (kun på Vertex AI)."""
    deps = ["ultralytics>=8.3.0", "albumentations>=1.4.0"]
    for dep in deps:
        subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "-q"])


def setup_augmentations():
    """Konfigurer albumentations for butikkhylledata."""
    import albumentations as A

    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=30, p=0.4),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.MotionBlur(blur_limit=5, p=0.2),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 3), p=0.3),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        A.RandomScale(scale_limit=0.2, p=0.3),
        A.PadIfNeeded(min_height=640, min_width=640, border_mode=0),
        A.RandomCrop(height=640, width=640, p=1.0),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))


def download_from_gcs(gcs_path: str, local_path: str) -> None:
    """Last ned datasett fra GCS."""
    os.makedirs(local_path, exist_ok=True)
    subprocess.check_call(["gsutil", "-m", "cp", "-r", f"{gcs_path}/*", local_path])
    print(f"Datasett lastet ned til {local_path}")


def train(data_path: str, epochs: int, batch_size: int, img_size: int) -> str:
    """Tren YOLO26-modellen."""
    from ultralytics import YOLO

    model = YOLO("yolo26n.pt")  # Nano-variant for rask inferens

    results = model.train(
        data=os.path.join(data_path, "data.yaml"),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device="0",  # GPU
        workers=8,
        patience=50,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        # Augmentation-innstillinger
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
    )

    best_model_path = str(results.save_dir / "weights" / "best.pt")
    print(f"Beste modell: {best_model_path}")
    return best_model_path


def export_to_onnx(model_path: str, output_dir: str) -> str:
    """Eksporter til ONNX for offline inferens."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    onnx_path = model.export(format="onnx", imgsz=640, simplify=True, opset=17)
    print(f"ONNX eksportert: {onnx_path}")

    os.makedirs(output_dir, exist_ok=True)
    final_path = os.path.join(output_dir, "yolo26.onnx")
    os.rename(onnx_path, final_path)
    return final_path


def upload_to_gcs(local_path: str, gcs_path: str) -> None:
    """Last opp modell til GCS."""
    subprocess.check_call(["gsutil", "cp", local_path, gcs_path])
    print(f"Modell lastet opp til {gcs_path}")


def main():
    parser = argparse.ArgumentParser(description="YOLO26 Trening for NorgesGruppen")
    parser.add_argument("--data", required=True, help="Sti til datasett (lokal eller gs://)")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--output-gcs", default="gs://nmiai-2026/models/", help="GCS-sti for modell-output")
    parser.add_argument("--export-only", action="store_true", help="Bare eksporter eksisterende modell")
    args = parser.parse_args()

    install_training_deps()

    data_path = args.data
    if data_path.startswith("gs://"):
        local_data = "/tmp/dataset"
        download_from_gcs(data_path, local_data)
        data_path = local_data

    if args.export_only:
        best_pt = os.path.join(data_path, "best.pt")
        if not os.path.exists(best_pt):
            print(f"FEIL: Fant ikke {best_pt}")
            return
    else:
        best_pt = train(data_path, args.epochs, args.batch_size, args.img_size)

    onnx_path = export_to_onnx(best_pt, "models")

    if args.output_gcs.startswith("gs://"):
        upload_to_gcs(onnx_path, f"{args.output_gcs}yolo26.onnx")

    print("Ferdig!")


if __name__ == "__main__":
    main()
