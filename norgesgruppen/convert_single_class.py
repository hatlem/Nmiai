"""
Convert existing YOLO dataset to single-class (class 0 = "product").
Images are symlinked; labels are rewritten with class_id 0.
"""

from pathlib import Path

SRC = Path("data/yolo_dataset")
DST = Path("data/yolo_single_class")

splits = ["train", "val"]

for split in splits:
    src_images = SRC / "images" / split
    src_labels = SRC / "labels" / split

    dst_images = DST / "images" / split
    dst_labels = DST / "labels" / split

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    # Symlink images
    image_count = 0
    for img_path in src_images.iterdir():
        link = dst_images / img_path.name
        if not link.exists():
            link.symlink_to(img_path.resolve())
        image_count += 1
    print(f"[{split}] symlinked {image_count} images")

    # Rewrite labels with class_id = 0
    label_count = 0
    for lbl_path in src_labels.iterdir():
        lines = lbl_path.read_text().splitlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                # Replace class_id with 0, keep rest unchanged
                new_lines.append("0 " + " ".join(parts[1:]))
        dst_label = dst_labels / lbl_path.name
        dst_label.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))
        label_count += 1
    print(f"[{split}] converted {label_count} label files")

# Write data.yaml
yaml_content = """\
path: data/yolo_single_class
train: images/train
val: images/val

nc: 1
names:
  0: product
"""
(DST / "data.yaml").write_text(yaml_content)
print(f"\nWrote {DST / 'data.yaml'}")
print("Done.")
