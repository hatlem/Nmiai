"""Convert COCO annotations to YOLO format for training.

Creates the directory structure expected by ultralytics:
  dataset/
    images/train/
    images/val/
    labels/train/
    labels/val/
    data.yaml
"""

import json
import shutil
import random
from pathlib import Path


def convert_coco_to_yolo(
    annotations_path: str = "data/train/annotations.json",
    images_dir: str = "data/train/images",
    output_dir: str = "data/yolo_dataset",
    val_split: float = 0.15,
    seed: int = 42,
):
    random.seed(seed)

    with open(annotations_path) as f:
        coco = json.load(f)

    output = Path(output_dir)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Build image info lookup
    img_info = {img["id"]: img for img in coco["images"]}

    # Group annotations by image_id
    ann_by_img: dict[int, list] = {}
    for ann in coco["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    # Split images into train/val
    image_ids = list(img_info.keys())
    random.shuffle(image_ids)
    val_count = max(1, int(len(image_ids) * val_split))
    val_ids = set(image_ids[:val_count])

    num_categories = len(coco["categories"])
    print(f"Categories: {num_categories}")
    print(f"Images: {len(image_ids)} (train: {len(image_ids) - val_count}, val: {val_count})")

    total_annotations = 0

    for img_id in image_ids:
        info = img_info[img_id]
        split = "val" if img_id in val_ids else "train"
        w, h = info["width"], info["height"]
        fname = info["file_name"]

        # Copy image
        src = Path(images_dir) / fname
        dst = output / "images" / split / fname
        if not dst.exists():
            shutil.copy2(src, dst)

        # Convert annotations to YOLO format
        label_path = output / "labels" / split / (Path(fname).stem + ".txt")
        anns = ann_by_img.get(img_id, [])
        total_annotations += len(anns)

        lines = []
        for ann in anns:
            if ann.get("iscrowd", 0):
                continue
            bx, by, bw, bh = ann["bbox"]  # COCO: [x, y, width, height]
            # Convert to YOLO: [class_id, x_center, y_center, width, height] (normalized)
            x_center = (bx + bw / 2) / w
            y_center = (by + bh / 2) / h
            nw = bw / w
            nh = bh / h
            # Clamp to [0, 1]
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            nw = max(0.001, min(1.0, nw))
            nh = max(0.001, min(1.0, nh))
            lines.append(f"{ann['category_id']} {x_center:.6f} {y_center:.6f} {nw:.6f} {nh:.6f}")

        label_path.write_text("\n".join(lines) + "\n" if lines else "")

    # Create data.yaml
    # Build category names list
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    names_dict = {i: cat_names.get(i, f"class_{i}") for i in range(num_categories)}

    yaml_content = f"""path: {output.resolve()}
train: images/train
val: images/val

nc: {num_categories}
names:
"""
    for i in range(num_categories):
        yaml_content += f"  {i}: '{names_dict[i]}'\n"

    (output / "data.yaml").write_text(yaml_content)

    print(f"Total annotations converted: {total_annotations}")
    print(f"Dataset written to: {output}")
    print(f"data.yaml: {output / 'data.yaml'}")


if __name__ == "__main__":
    convert_coco_to_yolo()
