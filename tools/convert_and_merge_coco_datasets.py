"""
Dataset Conversion & Merger Tool for YOLOv8 Training
Converts Roboflow COCO format datasets (Cardboard Box & Hand) to YOLO format,
integrates existing mission keyframes (box_manipulation_dataset), and creates
a unified balanced dataset in dataset/unified_detector_dataset/.
"""

import os
import sys
import json
import shutil
import random
from typing import Dict, List, Tuple
from pathlib import Path

# Fix random seed for reproducible sampling
random.seed(42)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

# Class ontology mapping
CLASS_MAPPING = {
    "container_box": 0,
    "container_lid": 1,
    "component_box": 2,
    "operator_hand": 3,
}

CLASS_NAMES = ["container_box", "container_lid", "component_box", "operator_hand"]


def coco_to_yolo_bbox(
    bbox: List[float], img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """Convert COCO [x_min, y_min, w, h] to YOLO [cx, cy, w, h] normalized."""
    x, y, w, h = bbox
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h

    # Clamp bounds to [0.0, 1.0]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))

    return cx, cy, nw, nh


def process_coco_dataset(
    dataset_dir: Path,
    splits: List[Tuple[str, str]],  # (source_subfolder, dest_split)
    target_class_id: int,
    output_dir: Path,
    prefix: str,
    max_samples: int = -1,
) -> Tuple[int, int]:
    """
    Parses COCO JSON annotations and copies images + writes YOLO txt labels.
    Returns (num_train, num_val) images added.
    """
    counts = {"train": 0, "val": 0}

    for src_folder, dest_split in splits:
        ann_path = dataset_dir / src_folder / "_annotations.coco.json"
        if not ann_path.exists():
            print(f"[WARN] Annotation file not found: {ann_path}")
            continue

        with open(ann_path, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        images = coco_data.get("images", [])
        annotations = coco_data.get("annotations", [])

        # Group annotations by image_id
        ann_by_image: Dict[int, List[dict]] = {}
        for ann in annotations:
            img_id = ann["image_id"]
            if img_id not in ann_by_image:
                ann_by_image[img_id] = []
            ann_by_image[img_id].append(ann)

        # Shuffle or sample if max_samples is set
        if max_samples > 0 and len(images) > max_samples:
            sampled_images = random.sample(images, max_samples)
        else:
            sampled_images = images

        for img_info in sampled_images:
            img_id = img_info["id"]
            file_name = img_info["file_name"]
            img_w = img_info["width"]
            img_h = img_info["height"]

            src_img_file = dataset_dir / src_folder / file_name
            if not src_img_file.exists():
                continue

            # Target split can be determined by dest_split or dynamically
            actual_split = dest_split
            if dest_split == "auto_split":
                # 85% train / 15% val
                actual_split = "train" if random.random() < 0.85 else "val"

            # Create unique destination filename
            ext = src_img_file.suffix
            base_name = f"{prefix}_{img_id}_{Path(file_name).stem}"
            dest_img_name = f"{base_name}{ext}"
            dest_lbl_name = f"{base_name}.txt"

            dest_img_path = output_dir / "images" / actual_split / dest_img_name
            dest_lbl_path = output_dir / "labels" / actual_split / dest_lbl_name

            # Copy image
            shutil.copyfile(src_img_file, dest_img_path)

            # Write YOLO labels
            img_anns = ann_by_image.get(img_id, [])
            label_lines = []
            for ann in img_anns:
                bbox = ann.get("bbox", [])
                if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
                    continue
                cx, cy, nw, nh = coco_to_yolo_bbox(bbox, img_w, img_h)
                label_lines.append(f"{target_class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

            with open(dest_lbl_path, "w", encoding="utf-8") as lf:
                lf.writelines(label_lines)

            counts[actual_split] += 1

    return counts["train"], counts["val"]


def copy_existing_mission_dataset(
    mission_dir: Path, output_dir: Path
) -> Tuple[int, int]:
    """
    Copies existing 178 keyframes from box_manipulation_dataset
    (maintaining all 4 classes: container_box, container_lid, component_box, operator_hand).
    """
    counts = {"train": 0, "val": 0}

    for split in ["train", "val"]:
        src_img_dir = mission_dir / "images" / split
        src_lbl_dir = mission_dir / "labels" / split

        if not src_img_dir.exists() or not src_lbl_dir.exists():
            continue

        for img_file in src_img_dir.iterdir():
            if not img_file.is_file():
                continue
            lbl_file = src_lbl_dir / f"{img_file.stem}.txt"
            if not lbl_file.exists():
                continue

            dest_img_name = f"mission_{img_file.name}"
            dest_lbl_name = f"mission_{lbl_file.name}"

            dest_img_path = output_dir / "images" / split / dest_img_name
            dest_lbl_path = output_dir / "labels" / split / dest_lbl_name

            shutil.copyfile(img_file, dest_img_path)
            shutil.copyfile(lbl_file, dest_lbl_path)
            counts[split] += 1

    return counts["train"], counts["val"]


def build_unified_dataset(
    output_dir: Path = WORKSPACE_ROOT / "dataset" / "unified_detector_dataset",
    max_hand_samples: int = 600,
):
    """
    Builds the unified dataset from:
    1. Cardboard Box COCO (class 0: container_box)
    2. Hand COCO (class 3: operator_hand)
    3. Existing box manipulation dataset (classes 0, 1, 2, 3)
    """
    print("=" * 70)
    print("   BHARATIYA ANTARIKSH STATION (BAS) - DATASET UNIFICATION & CONVERTER")
    print(f"   Target Directory : {output_dir}")
    print(f"   Max Hand Samples : {max_hand_samples}")
    print("=" * 70)

    # Prepare directories
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 1. Convert Cardboard Box COCO dataset -> Class 0 (container_box)
    cb_dir = WORKSPACE_ROOT / "dataset" / "Cardboard Box.v1-carboard-dataset1.coco"
    cb_splits = [
        ("train", "train"),
        ("valid", "val"),
        ("test", "val"),
    ]
    print("\n[1/3] Converting Cardboard Box COCO dataset -> class 0 (container_box)...")
    cb_train, cb_val = process_coco_dataset(
        dataset_dir=cb_dir,
        splits=cb_splits,
        target_class_id=CLASS_MAPPING["container_box"],
        output_dir=output_dir,
        prefix="cbox",
        max_samples=-1,  # include all cardboard boxes
    )
    print(f"      Added Cardboard Box: {cb_train} train, {cb_val} val images.")

    # 2. Convert Hand COCO dataset -> Class 3 (operator_hand)
    hand_dir = WORKSPACE_ROOT / "dataset" / "Hand.v8i.coco"
    hand_splits = [
        ("train", "auto_split"),
        ("valid", "val"),
    ]
    print(f"\n[2/3] Converting Hand COCO dataset -> class 3 (operator_hand) (sampling {max_hand_samples})...")
    h_train, h_val = process_coco_dataset(
        dataset_dir=hand_dir,
        splits=hand_splits,
        target_class_id=CLASS_MAPPING["operator_hand"],
        output_dir=output_dir,
        prefix="hand",
        max_samples=max_hand_samples,
    )
    print(f"      Added Hand: {h_train} train, {h_val} val images.")

    # 3. Merge existing mission keyframes (classes 0, 1, 2, 3)
    mission_dir = WORKSPACE_ROOT / "dataset" / "box_manipulation_dataset"
    print("\n[3/3] Merging existing mission dataset (lids, components, boxes, hands)...")
    m_train, m_val = copy_existing_mission_dataset(mission_dir, output_dir)
    print(f"      Added Mission Frames: {m_train} train, {m_val} val images.")

    # 4. Generate data.yaml
    total_train = cb_train + h_train + m_train
    total_val = cb_val + h_val + m_val
    yaml_content = f"""# Bharatiya Antariksh Station (BAS) - Unified Multi-Class Detection Dataset
path: {output_dir.as_posix()}
train: images/train
val: images/val

# Classes
nc: 4
names:
  0: container_box
  1: container_lid
  2: component_box
  3: operator_hand
"""
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yf.write(yaml_content)

    print("\n" + "=" * 70)
    print(f"[SUCCESS] Unified Dataset Built Successfully!")
    print(f"   Train Images : {total_train}")
    print(f"   Val Images   : {total_val}")
    print(f"   Total Images : {total_train + total_val}")
    print(f"   Config YAML  : {yaml_path}")
    print("=" * 70)
    return str(yaml_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-hands", type=int, default=600, help="Max hand images to sample")
    args = parser.parse_args()
    build_unified_dataset(max_hand_samples=args.max_hands)
