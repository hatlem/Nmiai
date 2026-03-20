"""Precrop all annotations to disk for fast training.

Running DINOv2 training with on-the-fly cropping from full images is
extremely slow due to disk I/O. This script pre-extracts all crops once,
so the training dataloader only reads small JPGs.

Usage: python precrop.py
Output: data/crops/*.jpg + data/crop_manifest.json
"""

import json
from pathlib import Path
from PIL import Image

DATA_DIR = Path("data")
ANNOTATIONS = DATA_DIR / "train" / "annotations.json"
IMAGES_DIR = DATA_DIR / "train" / "images"
CROPS_DIR = DATA_DIR / "crops"
CROPS_DIR.mkdir(parents=True, exist_ok=True)

with open(ANNOTATIONS) as f:
    coco = json.load(f)

img_info = {img["id"]: img for img in coco["images"]}

# Cache opened images to avoid re-reading
img_cache = {}
manifest = []
skipped = 0

for i, ann in enumerate(coco["annotations"]):
    if ann.get("iscrowd", 0):
        continue
    cat_id = ann["category_id"]
    if cat_id == 355:  # Unknown category
        continue

    bx, by, bw, bh = ann["bbox"]
    if bw < 10 or bh < 10:
        skipped += 1
        continue

    img_id = ann["image_id"]
    img_meta = img_info[img_id]
    img_path = IMAGES_DIR / img_meta["file_name"]

    try:
        if img_id not in img_cache:
            img_cache[img_id] = Image.open(str(img_path)).convert("RGB")
        pil = img_cache[img_id]
        w, h = pil.size

        pad_x = bw * 0.05
        pad_y = bh * 0.05
        crop = pil.crop((
            max(0, int(bx - pad_x)), max(0, int(by - pad_y)),
            min(w, int(bx + bw + pad_x)), min(h, int(by + bh + pad_y)),
        ))
        crop_path = CROPS_DIR / f"crop_{i:06d}.jpg"
        crop.save(str(crop_path), quality=95)
        manifest.append({"path": str(crop_path), "category_id": cat_id})
    except Exception as e:
        print(f"  Skip {i}: {e}")
        skipped += 1

    if (i + 1) % 5000 == 0:
        print(f"  Processed {i+1}/{len(coco['annotations'])} annotations...")

with open(str(DATA_DIR / "crop_manifest.json"), "w") as f:
    json.dump(manifest, f)

print(f"Done: {len(manifest)} crops saved to {CROPS_DIR}, {skipped} skipped")
