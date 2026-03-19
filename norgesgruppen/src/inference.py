"""Offline evalueringsskript for NM i AI 2026 - Task 3 (NorgesGruppen).

Laster YOLO26 .onnx-modell, kjører inferens på testbilder,
og genererer detections.json med resultater.

Bruk:
    python src/inference.py --input <bilder-mappe> --output detections.json
"""

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from utils import enhance_retail_image, preprocess_for_model


# Konfigurasjon
DEFAULT_MODEL_PATH = "models/yolo26.onnx"
DEFAULT_INPUT_SIZE = (640, 640)
CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45


def load_model(model_path: str) -> ort.InferenceSession:
    """Last inn ONNX-modellen med optimal provider."""
    providers = []
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(model_path, providers=providers)
    print(f"Modell lastet: {model_path}")
    print(f"Provider: {session.get_providers()}")
    return session


def run_inference(session: ort.InferenceSession, img: np.ndarray) -> list[dict]:
    """Kjør inferens på ett bilde og returner deteksjoner.

    Args:
        session: ONNX InferenceSession.
        img: BGR-bilde (original størrelse).

    Returns:
        Liste med deteksjoner: [{"bbox": [x1,y1,x2,y2], "class_id": int,
                                  "class_name": str, "confidence": float}]
    """
    orig_h, orig_w = img.shape[:2]
    input_tensor = preprocess_for_model(img, DEFAULT_INPUT_SIZE)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})

    # Parse YOLO output: (1, num_detections, 6) -> [x1, y1, x2, y2, conf, class_id]
    raw_output = outputs[0]
    if raw_output.ndim == 3:
        raw_output = raw_output[0]

    detections = []
    boxes = []
    confidences = []
    class_ids = []

    for det in raw_output:
        conf = float(det[4])
        if conf < CONF_THRESHOLD:
            continue

        x1 = float(det[0]) * orig_w / DEFAULT_INPUT_SIZE[0]
        y1 = float(det[1]) * orig_h / DEFAULT_INPUT_SIZE[1]
        x2 = float(det[2]) * orig_w / DEFAULT_INPUT_SIZE[0]
        y2 = float(det[3]) * orig_h / DEFAULT_INPUT_SIZE[1]
        class_id = int(det[5])

        boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
        confidences.append(conf)
        class_ids.append(class_id)

    # NMS via OpenCV
    if boxes:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                detections.append({
                    "bbox": [x, y, x + w, y + h],
                    "class_id": class_ids[i],
                    "confidence": round(confidences[i], 4),
                })

    return detections


def process_folder(model_path: str, input_dir: str, output_path: str) -> None:
    """Prosesser alle bilder i en mappe og skriv resultater til JSON."""
    session = load_model(model_path)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = sorted(
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() in image_extensions
    )

    if not image_files:
        print(f"Ingen bilder funnet i {input_dir}")
        return

    print(f"Fant {len(image_files)} bilder i {input_dir}")

    all_results = {}
    total_time = 0.0

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Kunne ikke lese: {img_path.name}")
            continue

        start = time.perf_counter()
        detections = run_inference(session, img)
        elapsed = time.perf_counter() - start
        total_time += elapsed

        all_results[img_path.name] = detections
        print(f"  {img_path.name}: {len(detections)} deteksjoner ({elapsed*1000:.1f}ms)")

    # Skriv resultater
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    avg_ms = (total_time / len(image_files)) * 1000 if image_files else 0
    print(f"\nFerdig! {len(image_files)} bilder, snitt {avg_ms:.1f}ms/bilde")
    print(f"Resultater skrevet til {output_path}")


def main():
    parser = argparse.ArgumentParser(description="NM i AI 2026 - Task 3 Offline Inferens")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Sti til .onnx modell")
    parser.add_argument("--input", required=True, help="Mappe med testbilder")
    parser.add_argument("--output", default="detections.json", help="Output JSON-fil")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"FEIL: Modell ikke funnet: {args.model}")
        print("Kjør train_vertex.py først, eller legg .onnx-filen i models/")
        return

    process_folder(args.model, args.input, args.output)


if __name__ == "__main__":
    main()
