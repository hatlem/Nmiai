#!/usr/bin/env python3
"""
Synthetic data generation for NorgesGruppen object detection.

Generates augmented training images by pasting clean product reference images
onto shelf images for rare categories (<=10 annotations).
"""

import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# --- Configuration ---
BASE_DIR = Path("/Users/andreashatlem/Projects/nmiai/norgesgruppen")
ANNOTATIONS_PATH = BASE_DIR / "data/train/annotations.json"
PRODUCT_IMAGES_DIR = BASE_DIR / "data/NM_NGD_product_images"
METADATA_PATH = PRODUCT_IMAGES_DIR / "metadata.json"
TRAIN_IMAGES_DIR = BASE_DIR / "data/train/images"
YOLO_DATASET_DIR = BASE_DIR / "data/yolo_dataset"
OUTPUT_DIR = BASE_DIR / "data/yolo_synthetic"

RARE_THRESHOLD = 10  # categories with <= this many annotations
TARGET_ANNOTATIONS = 50  # target annotations per rare category
MAX_PASTE_ATTEMPTS = 50  # max attempts to find non-overlapping location
IOU_THRESHOLD = 0.05  # max IoU with existing boxes for placement
IMAGES_PER_CATEGORY = 15  # number of synthetic images to create per rare category

random.seed(42)
np.random.seed(42)


def load_coco_annotations(path):
    """Load COCO format annotations."""
    with open(path) as f:
        data = json.load(f)
    return data


def load_metadata(path):
    """Load product metadata."""
    with open(path) as f:
        return json.load(f)


def get_rare_categories(annotations, threshold=RARE_THRESHOLD):
    """Find categories with <= threshold annotations."""
    cat_counts = Counter(a["category_id"] for a in annotations["annotations"])
    cat_names = {c["id"]: c["name"] for c in annotations["categories"]}
    rare = {}
    for cid, name in cat_names.items():
        count = cat_counts.get(cid, 0)
        if count <= threshold:
            rare[cid] = {"name": name, "count": count}
    return rare, cat_counts


def build_category_product_map(annotations, metadata):
    """Map category names to product codes with images."""
    name_to_catid = {c["name"]: c["id"] for c in annotations["categories"]}
    catid_to_product = {}

    for product in metadata["products"]:
        pname = product["product_name"]
        if pname in name_to_catid and product["has_images"]:
            catid = name_to_catid[pname]
            catid_to_product[catid] = {
                "product_code": product["product_code"],
                "image_types": product["image_types"],
            }

    return catid_to_product


def get_existing_boxes_for_image(annotations, image_id):
    """Get all bounding boxes for a given image (in pixel coords)."""
    boxes = []
    for ann in annotations["annotations"]:
        if ann["image_id"] == image_id:
            x, y, w, h = ann["bbox"]
            boxes.append((x, y, x + w, y + h))
    return boxes


def compute_iou(box1, box2):
    """Compute IoU between two boxes (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def remove_background(img):
    """Remove white/light background from product image using alpha mask."""
    img_array = np.array(img.convert("RGBA"))
    # Convert to grayscale for thresholding
    gray = np.mean(img_array[:, :, :3], axis=2)

    # White background detection - pixels close to white
    # Use a generous threshold since product images have clean backgrounds
    white_mask = gray > 220

    # Also check if all channels are similar (grayish/white)
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    channel_diff = np.max(np.stack([r, g, b]), axis=0) - np.min(
        np.stack([r, g, b]), axis=0
    )
    neutral_mask = channel_diff < 40

    # Combined: white AND neutral colored = background
    bg_mask = white_mask & neutral_mask

    # Create alpha channel (0 for background, 255 for foreground)
    alpha = np.where(bg_mask, 0, 255).astype(np.uint8)

    # Slight erosion to clean edges
    kernel = np.ones((3, 3), np.uint8)
    alpha = cv2.erode(alpha, kernel, iterations=1)

    # Gaussian blur on alpha for smoother edges
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)

    img_array[:, :, 3] = alpha
    return Image.fromarray(img_array)


def load_product_image(product_code, image_types):
    """Load a random product image and remove background."""
    product_dir = PRODUCT_IMAGES_DIR / product_code

    # Prefer front/main views for shelf placement
    preferred = ["front", "main"]
    available = [t for t in preferred if t in image_types]
    if not available:
        available = [t for t in image_types if t not in ["top", "bottom"]]
    if not available:
        available = image_types

    img_type = random.choice(available)
    img_path = product_dir / f"{img_type}.jpg"

    if not img_path.exists():
        return None

    img = Image.open(img_path).convert("RGBA")
    img = remove_background(img)
    return img


def get_bbox_size_stats(annotations, cat_counts):
    """Get median bbox sizes from annotations for realistic product sizing."""
    widths = []
    heights = []
    for ann in annotations["annotations"]:
        w, h = ann["bbox"][2], ann["bbox"][3]
        widths.append(w)
        heights.append(h)

    return {
        "median_w": int(np.median(widths)),
        "median_h": int(np.median(heights)),
        "p25_w": int(np.percentile(widths, 25)),
        "p75_w": int(np.percentile(widths, 75)),
        "p25_h": int(np.percentile(heights, 25)),
        "p75_h": int(np.percentile(heights, 75)),
    }


def get_category_specific_size(annotations, cat_id, size_stats):
    """Get size for a specific category, falling back to global stats."""
    cat_bboxes = [
        a["bbox"] for a in annotations["annotations"] if a["category_id"] == cat_id
    ]
    if len(cat_bboxes) >= 3:
        widths = [b[2] for b in cat_bboxes]
        heights = [b[3] for b in cat_bboxes]
        return int(np.median(widths)), int(np.median(heights))
    # Fall back to global median with some variation
    return size_stats["median_w"], size_stats["median_h"]


def find_placement(img_w, img_h, product_w, product_h, existing_boxes, max_attempts=MAX_PASTE_ATTEMPTS):
    """Find a valid placement for a product that doesn't overlap existing boxes."""
    for _ in range(max_attempts):
        x = random.randint(0, max(0, img_w - product_w))
        y = random.randint(0, max(0, img_h - product_h))

        new_box = (x, y, x + product_w, y + product_h)

        # Check IoU with all existing boxes
        overlap = False
        for ebox in existing_boxes:
            if compute_iou(new_box, ebox) > IOU_THRESHOLD:
                overlap = True
                break

        if not overlap:
            return x, y

    return None


def apply_augmentations(product_img):
    """Apply random augmentations to product image."""
    # Random rotation (-5 to +5 degrees)
    angle = random.uniform(-5, 5)
    product_img = product_img.rotate(angle, expand=True, resample=Image.BICUBIC)

    # Random brightness variation (0.8 to 1.2)
    enhancer = ImageEnhance.Brightness(product_img)
    product_img = enhancer.enhance(random.uniform(0.8, 1.2))

    # Random contrast (0.9 to 1.1)
    enhancer = ImageEnhance.Contrast(product_img)
    product_img = enhancer.enhance(random.uniform(0.9, 1.1))

    # Slight color jitter
    enhancer = ImageEnhance.Color(product_img)
    product_img = enhancer.enhance(random.uniform(0.9, 1.1))

    return product_img


def paste_product(shelf_img, product_img, x, y, product_w, product_h):
    """Paste product onto shelf image with alpha blending."""
    # Resize product to target size
    resized = product_img.resize((product_w, product_h), Image.LANCZOS)

    # Apply augmentations
    resized = apply_augmentations(resized)

    # Ensure RGBA
    if resized.mode != "RGBA":
        resized = resized.convert("RGBA")

    # Get actual size after rotation may have changed it
    actual_w, actual_h = resized.size

    # Create a copy of shelf image as RGBA
    shelf_rgba = shelf_img.convert("RGBA")

    # Paste with alpha compositing
    # Create a transparent layer
    overlay = Image.new("RGBA", shelf_rgba.size, (0, 0, 0, 0))

    # Clip if needed
    paste_x = min(x, shelf_rgba.width - actual_w)
    paste_y = min(y, shelf_rgba.height - actual_h)
    paste_x = max(0, paste_x)
    paste_y = max(0, paste_y)

    overlay.paste(resized, (paste_x, paste_y))
    result = Image.alpha_composite(shelf_rgba, overlay)

    return result.convert("RGB"), paste_x, paste_y, actual_w, actual_h


def build_image_annotations_map(annotations):
    """Build map of image_id -> list of annotations."""
    img_anns = defaultdict(list)
    for ann in annotations["annotations"]:
        img_anns[ann["image_id"]].append(ann)
    return img_anns


def main():
    print("=" * 60)
    print("Synthetic Data Generation for NorgesGruppen")
    print("=" * 60)

    # Load data
    print("\n[1/6] Loading annotations and metadata...")
    annotations = load_coco_annotations(ANNOTATIONS_PATH)
    metadata = load_metadata(METADATA_PATH)

    # Find rare categories
    print("[2/6] Identifying rare categories...")
    rare_cats, cat_counts = get_rare_categories(annotations)
    print(f"  Found {len(rare_cats)} categories with <= {RARE_THRESHOLD} annotations")

    # Map categories to product images
    print("[3/6] Mapping categories to product images...")
    cat_product_map = build_category_product_map(annotations, metadata)
    mappable_rare = {
        cid: info for cid, info in rare_cats.items() if cid in cat_product_map
    }
    print(
        f"  {len(mappable_rare)}/{len(rare_cats)} rare categories have product images"
    )

    # Get size statistics
    size_stats = get_bbox_size_stats(annotations, cat_counts)
    print(
        f"  Median bbox size: {size_stats['median_w']}x{size_stats['median_h']} pixels"
    )

    # Prepare output directory
    print("[4/6] Preparing output directory...")
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    # Copy existing YOLO dataset
    print("  Copying existing YOLO dataset...")
    shutil.copytree(YOLO_DATASET_DIR, OUTPUT_DIR)

    # Build image annotations map for efficient lookup
    img_anns_map = build_image_annotations_map(annotations)
    img_info = {img["id"]: img for img in annotations["images"]}

    # Get list of training images
    train_images_dir = OUTPUT_DIR / "images" / "train"
    train_labels_dir = OUTPUT_DIR / "labels" / "train"
    train_image_files = sorted(train_images_dir.glob("*.jpg"))
    print(f"  Found {len(train_image_files)} training images")

    # Generate synthetic data
    print("[5/6] Generating synthetic annotations...")
    total_new_annotations = 0
    total_new_images = 0
    cats_augmented = 0
    skipped_cats = 0

    # Build file_name -> image_id map
    fname_to_imgid = {img["file_name"]: img["id"] for img in annotations["images"]}

    for cat_id, info in sorted(mappable_rare.items(), key=lambda x: x[1]["count"]):
        cat_name = info["name"]
        existing_count = info["count"]
        needed = max(0, TARGET_ANNOTATIONS - existing_count)

        if needed == 0:
            continue

        product_info = cat_product_map[cat_id]
        product_code = product_info["product_code"]
        image_types = product_info["image_types"]

        # Get target size for this category
        target_w, target_h = get_category_specific_size(
            annotations, cat_id, size_stats
        )

        # Load product image
        product_img = load_product_image(product_code, image_types)
        if product_img is None:
            skipped_cats += 1
            continue

        cat_new_annotations = 0
        images_used = set()

        # Randomly sample shelf images and paste products
        shelf_images = random.sample(
            train_image_files, min(IMAGES_PER_CATEGORY, len(train_image_files))
        )

        for shelf_path in shelf_images:
            if cat_new_annotations >= needed:
                break

            # Determine how many products to paste on this image (1-4)
            num_paste = random.randint(1, min(4, needed - cat_new_annotations))

            # Get existing boxes for this image
            fname = shelf_path.name
            img_id = fname_to_imgid.get(fname)
            if img_id:
                existing_boxes = get_existing_boxes_for_image(annotations, img_id)
            else:
                existing_boxes = []

            # Load shelf image
            shelf_img = Image.open(shelf_path)
            img_w, img_h = shelf_img.size

            # Read existing YOLO labels for this image
            label_path = train_labels_dir / fname.replace(".jpg", ".txt")
            existing_labels = []
            if label_path.exists():
                with open(label_path) as f:
                    existing_labels = f.read().strip().split("\n")
                    if existing_labels == [""]:
                        existing_labels = []

            new_labels = list(existing_labels)
            new_boxes = list(existing_boxes)
            modified = False

            for _ in range(num_paste):
                # Random scale variation (0.7 to 1.3 of target size)
                scale = random.uniform(0.7, 1.3)
                pw = int(target_w * scale)
                ph = int(target_h * scale)

                # Ensure product fits in image
                pw = min(pw, img_w - 10)
                ph = min(ph, img_h - 10)

                if pw <= 10 or ph <= 10:
                    continue

                # Find valid placement
                placement = find_placement(img_w, img_h, pw, ph, new_boxes)
                if placement is None:
                    continue

                px, py = placement

                # Reload product image each time for variety
                prod_img = load_product_image(product_code, image_types)
                if prod_img is None:
                    continue

                # Paste product onto shelf
                shelf_img, actual_x, actual_y, actual_w, actual_h = paste_product(
                    shelf_img, prod_img, px, py, pw, ph
                )

                # Add YOLO format label (class_id cx cy w h, normalized)
                cx = (actual_x + actual_w / 2) / img_w
                cy = (actual_y + actual_h / 2) / img_h
                nw = actual_w / img_w
                nh = actual_h / img_h

                # Clamp to [0, 1]
                cx = max(0, min(1, cx))
                cy = max(0, min(1, cy))
                nw = max(0, min(1, nw))
                nh = max(0, min(1, nh))

                new_labels.append(f"{cat_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                new_boxes.append(
                    (actual_x, actual_y, actual_x + actual_w, actual_y + actual_h)
                )
                cat_new_annotations += 1
                modified = True

            if modified:
                # Save modified shelf image (overwrite in synthetic dir)
                # Create a new filename to avoid overwriting originals
                synth_fname = f"synth_{cat_id}_{fname}"
                synth_img_path = train_images_dir / synth_fname
                synth_label_path = train_labels_dir / synth_fname.replace(
                    ".jpg", ".txt"
                )

                shelf_img.save(synth_img_path, quality=95)

                with open(synth_label_path, "w") as f:
                    f.write("\n".join(new_labels) + "\n")

                total_new_images += 1

        total_new_annotations += cat_new_annotations
        if cat_new_annotations > 0:
            cats_augmented += 1
            if cats_augmented <= 10 or cats_augmented % 20 == 0:
                print(
                    f"  [{cats_augmented}] {cat_name}: +{cat_new_annotations} annotations (was {existing_count})"
                )

    print(f"\n  Total categories augmented: {cats_augmented}")
    print(f"  Categories skipped (no product images): {skipped_cats}")

    # Update data.yaml
    print("[6/6] Updating data.yaml...")
    data_yaml_path = OUTPUT_DIR / "data.yaml"
    with open(YOLO_DATASET_DIR / "data.yaml") as f:
        yaml_content = f.read()

    yaml_content = yaml_content.replace(
        "path: data/yolo_dataset", "path: data/yolo_synthetic"
    )

    with open(data_yaml_path, "w") as f:
        f.write(yaml_content)

    # Final report
    print("\n" + "=" * 60)
    print("SYNTHETIC DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"New synthetic images created: {total_new_images}")
    print(f"New annotations added: {total_new_annotations}")
    print(f"Categories augmented: {cats_augmented}/{len(mappable_rare)}")

    # Count total files
    total_train_images = len(list(train_images_dir.glob("*.jpg")))
    total_train_labels = len(list(train_labels_dir.glob("*.txt")))
    total_label_lines = 0
    for lf in train_labels_dir.glob("*.txt"):
        with open(lf) as f:
            total_label_lines += sum(1 for line in f if line.strip())

    print(f"\nFinal dataset:")
    print(f"  Training images: {total_train_images}")
    print(f"  Training label files: {total_train_labels}")
    print(f"  Total annotations: {total_label_lines}")

    # GCS upload command
    print(f"\n--- To upload to GCS ---")
    print(
        f"gsutil -q -m cp -r {OUTPUT_DIR} gs://ainm26osl-710-data/norgesgruppen/"
    )


if __name__ == "__main__":
    main()
