"""Lokal verifikasjon av .onnx-modellen.

Kjører inferens på et testbilde, måler FPS, og tegner bounding-bokser
for manuell kvalitetskontroll.

Bruk:
    python test_local.py --model models/yolo26.onnx --image test_images/sample.jpg
"""

import argparse
import time

import cv2
import numpy as np
import onnxruntime as ort

from src.utils import enhance_retail_image, preprocess_for_model


# Farger for visualisering (BGR)
COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
]


def benchmark(session: ort.InferenceSession, img: np.ndarray, runs: int = 50) -> float:
    """Kjør benchmark og returner gjennomsnittlig ms per inferens."""
    input_tensor = preprocess_for_model(img)
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(5):
        session.run(None, {input_name: input_tensor})

    # Benchmark
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, {input_name: input_tensor})
        times.append(time.perf_counter() - start)

    avg_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    fps = 1000 / avg_ms

    print(f"\nBenchmark ({runs} kjøringer):")
    print(f"  Snitt:  {avg_ms:.1f}ms")
    print(f"  Std:    {std_ms:.1f}ms")
    print(f"  FPS:    {fps:.1f}")
    print(f"  Min:    {min(times)*1000:.1f}ms")
    print(f"  Max:    {max(times)*1000:.1f}ms")

    return avg_ms


def draw_detections(img: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Tegn bounding-bokser på bildet."""
    result = img.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        class_id = det["class_id"]
        conf = det["confidence"]
        color = COLORS[class_id % len(COLORS)]

        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        label = f"cls{class_id}: {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
        cv2.putText(result, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return result


def main():
    parser = argparse.ArgumentParser(description="Lokal test av YOLO26 ONNX-modell")
    parser.add_argument("--model", default="models/yolo26.onnx", help="Sti til .onnx modell")
    parser.add_argument("--image", required=True, help="Testbilde")
    parser.add_argument("--output", default="test_output.jpg", help="Output-bilde med bokser")
    parser.add_argument("--benchmark", action="store_true", help="Kjør hastighetsbenchmark")
    args = parser.parse_args()

    # Last modell
    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")

    session = ort.InferenceSession(args.model, providers=providers)
    print(f"Provider: {session.get_providers()}")

    # Last bilde
    img = cv2.imread(args.image)
    if img is None:
        print(f"Kunne ikke lese bilde: {args.image}")
        return

    print(f"Bildestørrelse: {img.shape[1]}x{img.shape[0]}")

    # Test CLAHE
    enhanced = enhance_retail_image(img)
    cv2.imwrite("test_enhanced.jpg", enhanced)
    print("CLAHE-forbedret bilde: test_enhanced.jpg")

    # Kjør inferens
    from src.inference import run_inference
    detections = run_inference(session, img)
    print(f"\nDeteksjoner: {len(detections)}")
    for det in detections:
        print(f"  Klasse {det['class_id']}: {det['confidence']:.3f} @ {det['bbox']}")

    # Tegn og lagre
    result_img = draw_detections(img, detections)
    cv2.imwrite(args.output, result_img)
    print(f"\nVisualisering lagret: {args.output}")

    # Benchmark
    if args.benchmark:
        benchmark(session, img)


if __name__ == "__main__":
    main()
