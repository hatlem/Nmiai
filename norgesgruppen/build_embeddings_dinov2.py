#!/usr/bin/env python3.11
"""
Pre-compute product reference embeddings using fine-tuned DINOv2.

Outputs:
  models/dinov2_product_embeddings.npy  — shape (356, embed_dim), indexed by category_id
  models/dinov2_embedding_config.json   — model metadata

DINOv2 ViT-B/14 produces 768-dim embeddings (vs 1536 for EfficientNet-B3).
DINOv2 ViT-S/14 produces 384-dim embeddings.

Mapping chain:
  product_code (folder name) -> product_name (metadata.json) -> category_id (annotations.json)

Usage:
    python build_embeddings_dinov2.py
    python build_embeddings_dinov2.py --weights models/dinov2_embeddings_weights.pt
"""

import argparse
import json

import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_CATEGORIES = 356
UNKNOWN_CATEGORY_ID = 355
IMAGE_SIZE = 518  # DINOv2 ViT-B/14: 37 * 14
ANGLES = ["main", "front", "back", "left", "right", "top", "bottom"]

DATA_DIR = Path("data")
PRODUCT_IMAGES_DIR = DATA_DIR / "NM_NGD_product_images"
ANNOTATIONS_FILE = DATA_DIR / "train" / "annotations.json"
METADATA_FILE = PRODUCT_IMAGES_DIR / "metadata.json"
MODELS_DIR = Path("models")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# DINOv2 model candidates (in order of preference)
DINOV2_MODEL_CANDIDATES = [
    "vit_base_patch14_dinov2.lvd142m",
    "vit_base_patch14_dinov2",
    "vit_small_patch14_dinov2.lvd142m",
    "vit_small_patch14_dinov2",
]


def resolve_dinov2_model_name() -> str:
    """Find the best available DINOv2 model in timm."""
    import timm

    available = timm.list_models("*dinov2*")
    print(f"Available DINOv2 models in timm: {available}")

    for candidate in DINOV2_MODEL_CANDIDATES:
        if candidate in available:
            print(f"Selected model: {candidate}")
            return candidate

    for candidate in DINOV2_MODEL_CANDIDATES:
        base_name = candidate.split(".")[0]
        matches = [m for m in available if base_name in m]
        if matches:
            selected = matches[0]
            print(f"Selected model (partial match): {selected}")
            return selected

    raise RuntimeError(
        f"No DINOv2 model found in timm. Available: {available}. "
        f"Tried: {DINOV2_MODEL_CANDIDATES}"
    )


def build_product_code_to_category_id() -> dict[str, int]:
    """Build mapping: product_code -> category_id via product_name."""
    with open(ANNOTATIONS_FILE) as f:
        annotations = json.load(f)
    with open(METADATA_FILE) as f:
        metadata = json.load(f)

    name_to_cat_id: dict[str, int] = {
        cat["name"]: cat["id"] for cat in annotations["categories"]
    }

    product_code_to_cat_id: dict[str, int] = {}
    unmatched: list[str] = []

    for product in metadata["products"]:
        product_code = product["product_code"]
        product_name = product["product_name"]

        if product_name in name_to_cat_id:
            product_code_to_cat_id[product_code] = name_to_cat_id[product_name]
        else:
            name_lower = product_name.lower()
            matched = False
            for cat_name, cat_id in name_to_cat_id.items():
                if cat_name.lower() == name_lower:
                    product_code_to_cat_id[product_code] = cat_id
                    matched = True
                    break
            if not matched:
                unmatched.append(f"{product_code}: {product_name}")

    if unmatched:
        print(f"WARNING: {len(unmatched)} products could not be mapped to a category_id:")
        for u in unmatched:
            print(f"  {u}")

    print(f"Mapped {len(product_code_to_cat_id)} products to category IDs")
    return product_code_to_cat_id


def load_model_and_transform(weights_path: Path | None = None):
    """Load DINOv2 model in feature extraction mode.

    Args:
        weights_path: Optional path to fine-tuned weights (.pt state_dict).
                      If provided, loads these instead of pretrained weights.
    """
    import timm
    import torch
    from torchvision import transforms

    model_name = resolve_dinov2_model_name()

    use_pretrained = weights_path is None
    print(f"Loading model: {model_name} (pretrained={use_pretrained})")
    model = timm.create_model(model_name, pretrained=use_pretrained, num_classes=0)

    if weights_path is not None:
        print(f"Loading fine-tuned weights from: {weights_path}")
        state_dict = torch.load(str(weights_path), map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    model.eval()

    # Standard transform for DINOv2
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    embed_dim = model.num_features
    print(f"Model loaded. embed_dim={embed_dim}, image_size={IMAGE_SIZE}")

    return model, transform, model_name, embed_dim


def extract_embedding(model, transform, image_path: Path):
    """Extract embedding from a single image. Returns numpy array or None."""
    import torch
    from PIL import Image

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  WARNING: Could not load {image_path}: {e}")
        return None

    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        embedding = model(tensor)

    return embedding.squeeze(0).cpu().numpy()


def compute_product_embedding(model, transform, product_dir: Path) -> np.ndarray | None:
    """Load all available angle images and return averaged embedding."""
    embeddings = []

    for angle in ANGLES:
        img_path = product_dir / f"{angle}.jpg"
        if img_path.exists():
            emb = extract_embedding(model, transform, img_path)
            if emb is not None:
                embeddings.append(emb)

    if not embeddings:
        return None

    return np.mean(embeddings, axis=0)


def main():
    import torch

    parser = argparse.ArgumentParser(description="Build DINOv2 product reference embeddings")
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Path to fine-tuned DINOv2 weights (.pt state_dict). "
             "Default: models/dinov2_embeddings_weights.pt if it exists, else pretrained.",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Auto-detect fine-tuned weights if not specified
    weights_path = None
    if args.weights:
        weights_path = Path(args.weights)
    else:
        default_weights = MODELS_DIR / "dinov2_embeddings_weights.pt"
        if default_weights.exists():
            print(f"Auto-detected fine-tuned weights: {default_weights}")
            weights_path = default_weights

    # Build product_code -> category_id mapping
    print("Building product code -> category ID mapping...")
    product_code_to_cat_id = build_product_code_to_category_id()

    # Load model
    model, transform, model_name, embed_dim = load_model_and_transform(weights_path)

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Using device: {device}")

    # Verify embedding dim
    with torch.no_grad():
        dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
        dummy_out = model(dummy)
        embed_dim = dummy_out.shape[1]
    print(f"Embedding dimension: {embed_dim}")

    # Initialize embeddings
    embeddings = np.zeros((NUM_CATEGORIES, embed_dim), dtype=np.float32)

    # Process each product directory
    product_dirs = [d for d in PRODUCT_IMAGES_DIR.iterdir() if d.is_dir()]
    print(f"\nProcessing {len(product_dirs)} product directories...")

    # Move model to CPU for per-image processing (simpler, works with extract_embedding)
    # Or patch extract_embedding to use GPU
    model = model.cpu()

    processed = 0
    skipped_no_mapping = 0
    skipped_no_images = 0

    for i, product_dir in enumerate(sorted(product_dirs)):
        product_code = product_dir.name

        if product_code not in product_code_to_cat_id:
            skipped_no_mapping += 1
            if i < 5 or skipped_no_mapping <= 3:
                print(f"  SKIP (no category mapping): {product_code}")
            continue

        cat_id = product_code_to_cat_id[product_code]

        if cat_id == UNKNOWN_CATEGORY_ID:
            continue

        print(f"  [{i + 1}/{len(product_dirs)}] {product_code} -> category {cat_id}")
        embedding = compute_product_embedding(model, transform, product_dir)

        if embedding is None:
            skipped_no_images += 1
            print(f"    WARNING: No valid images found")
            continue

        # L2-normalize for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        embeddings[cat_id] = embedding
        processed += 1

    print(f"\nDone!")
    print(f"  Processed:          {processed}")
    print(f"  Skipped (no map):   {skipped_no_mapping}")
    print(f"  Skipped (no imgs):  {skipped_no_images}")
    print(f"  Zero embeddings:    {NUM_CATEGORIES - processed} (includes unknown_product at id={UNKNOWN_CATEGORY_ID})")

    # Save embeddings
    embeddings_path = MODELS_DIR / "dinov2_product_embeddings.npy"
    np.save(embeddings_path, embeddings)
    print(f"\nSaved embeddings: {embeddings_path}  shape={embeddings.shape}")

    # Save config
    config = {
        "model_name": model_name,
        "embedding_dim": embed_dim,
        "num_categories": NUM_CATEGORIES,
        "unknown_category_id": UNKNOWN_CATEGORY_ID,
        "image_size": IMAGE_SIZE,
        "l2_normalized": True,
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
        "angles_used": ANGLES,
        "fine_tuned": weights_path is not None,
        "backbone": "dinov2",
    }
    config_path = MODELS_DIR / "dinov2_embedding_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config:     {config_path}")

    # Report file size
    size_mb = embeddings_path.stat().st_size / (1024 * 1024)
    print(f"\nEmbedding file size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
