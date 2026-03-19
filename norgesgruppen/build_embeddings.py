#!/usr/bin/env python3.11
"""
Pre-compute product reference embeddings for classification stage.

Outputs:
  models/product_embeddings.npy  — shape (356, embedding_dim), indexed by category_id
  models/embedding_config.json   — model metadata

Mapping chain:
  product_code (folder name) → product_name (metadata.json) → category_id (categories in annotations.json)
"""

import json
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "efficientnet_b3"
NUM_CATEGORIES = 356
UNKNOWN_CATEGORY_ID = 355
IMAGE_SIZE = 300  # efficientnet_b3 native resolution
ANGLES = ["main", "front", "back", "left", "right", "top", "bottom"]

DATA_DIR = Path("data")
PRODUCT_IMAGES_DIR = DATA_DIR / "NM_NGD_product_images"
ANNOTATIONS_FILE = DATA_DIR / "train" / "annotations.json"
METADATA_FILE = PRODUCT_IMAGES_DIR / "metadata.json"
MODELS_DIR = Path("models")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_product_code_to_category_id() -> dict[str, int]:
    """Build mapping: product_code -> category_id via product_name."""
    with open(ANNOTATIONS_FILE) as f:
        annotations = json.load(f)
    with open(METADATA_FILE) as f:
        metadata = json.load(f)

    # category name -> category_id
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
            # Try case-insensitive match as fallback
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


def load_model_and_transforms(weights_path: Path | None = None):
    """Load timm model in feature extraction mode.

    Args:
        weights_path: Optional path to fine-tuned weights (.pt state_dict).
                      If provided, loads these instead of pretrained ImageNet weights.
    """
    import timm
    from timm.data import resolve_data_config, create_transform

    use_pretrained = weights_path is None
    print(f"Loading model: {MODEL_NAME} (pretrained={use_pretrained})")
    model = timm.create_model(MODEL_NAME, pretrained=use_pretrained, num_classes=0)

    if weights_path is not None:
        import torch
        print(f"Loading fine-tuned weights from: {weights_path}")
        state_dict = torch.load(str(weights_path), map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    model.eval()

    data_config = resolve_data_config({}, model=model)
    # Override with ImageNet standard normalization to be explicit
    data_config["mean"] = IMAGENET_MEAN
    data_config["std"] = IMAGENET_STD
    transform = create_transform(**data_config)

    print(f"Model loaded. Input size: {data_config.get('input_size')}")
    return model, transform


def extract_embedding(model, transform, image_path: Path):
    """Extract embedding from a single image. Returns numpy array or None."""
    import torch
    from PIL import Image

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  WARNING: Could not load {image_path}: {e}")
        return None

    tensor = transform(img).unsqueeze(0)  # (1, C, H, W)

    with torch.no_grad():
        embedding = model(tensor)  # (1, embedding_dim) — global avg pool output

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

    # Average embeddings across angles
    return np.mean(embeddings, axis=0)


def main():
    import argparse
    import torch

    parser = argparse.ArgumentParser(description="Build product reference embeddings")
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Path to fine-tuned weights (.pt state_dict). If omitted, uses pretrained ImageNet weights.",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Build product_code → category_id mapping
    print("Building product code → category ID mapping...")
    product_code_to_cat_id = build_product_code_to_category_id()

    # Load model
    weights_path = Path(args.weights) if args.weights else None
    model, transform = load_model_and_transforms(weights_path)

    # Determine embedding dimension from a dummy forward pass
    with torch.no_grad():
        dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        dummy_out = model(dummy)
        embedding_dim = dummy_out.shape[1]
    print(f"Embedding dimension: {embedding_dim}")

    # Initialize embeddings array with zeros (covers unknown and missing products)
    embeddings = np.zeros((NUM_CATEGORIES, embedding_dim), dtype=np.float32)

    # Process each product directory
    product_dirs = [d for d in PRODUCT_IMAGES_DIR.iterdir() if d.is_dir()]
    print(f"\nProcessing {len(product_dirs)} product directories...")

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
            # unknown_product — leave as zero embedding
            continue

        print(f"  [{i+1}/{len(product_dirs)}] {product_code} → category {cat_id}")
        embedding = compute_product_embedding(model, transform, product_dir)

        if embedding is None:
            skipped_no_images += 1
            print(f"    WARNING: No valid images found")
            continue

        # L2-normalize the embedding for cosine similarity
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
    embeddings_path = MODELS_DIR / "product_embeddings.npy"
    np.save(embeddings_path, embeddings)
    print(f"\nSaved embeddings: {embeddings_path}  shape={embeddings.shape}")

    # Save config
    config = {
        "model_name": MODEL_NAME,
        "embedding_dim": embedding_dim,
        "num_categories": NUM_CATEGORIES,
        "unknown_category_id": UNKNOWN_CATEGORY_ID,
        "l2_normalized": True,
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
        "angles_used": ANGLES,
        "fine_tuned": weights_path is not None,
    }
    config_path = MODELS_DIR / "embedding_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config:     {config_path}")

    # Report file size
    size_mb = embeddings_path.stat().st_size / (1024 * 1024)
    print(f"\nEmbedding file size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
