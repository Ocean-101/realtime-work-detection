"""
BAS Autonomous HAR System - Human Body Auto-Annotator for Box Manipulation Dataset
Extracts human body (person) bounding boxes using YOLOv8-pose from all frames in
dataset/box_manipulation_dataset and appends class 4 (human_body) to YOLO label files.
Updates data.yaml to 5 classes and clears cache files.
"""

import os
import sys
import glob
import cv2
import yaml
from ultralytics import YOLO

DATASET_ROOT = os.path.abspath("dataset/box_manipulation_dataset")
POSE_MODEL_PATH = "models/yolov8n-pose.pt"

CLASSES = {
    0: "container_box",
    1: "container_lid",
    2: "component_box",
    3: "operator_hand",
    4: "human_body"
}


def augment_dataset_with_human_body(verify_only: bool = False):
    print("=" * 70)
    print("   BHARATIYA ANTARIKSH STATION (BAS) - HUMAN BODY DATASET AUGMENTER")
    print(f"   Target Dataset : {DATASET_ROOT}")
    print(f"   Pose Model     : {POSE_MODEL_PATH}")
    print(f"   Mode           : {'VERIFY ONLY' if verify_only else 'AUGMENT & UPDATE'}")
    print("=" * 70)

    if not os.path.exists(DATASET_ROOT):
        raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")

    model = YOLO(POSE_MODEL_PATH)

    total_images_processed = 0
    total_human_added = 0

    for split in ["train", "val"]:
        img_dir = os.path.join(DATASET_ROOT, "images", split)
        lbl_dir = os.path.join(DATASET_ROOT, "labels", split)

        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            continue

        img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
        print(f"\nProcessing {split.upper()} split: {len(img_files)} images found.")

        for img_path in img_files:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, f"{base_name}.txt")

            total_images_processed += 1

            # Read existing label lines
            existing_lines = []
            has_human_body = False
            if os.path.exists(lbl_path):
                with open(lbl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            if parts[0] == "4":
                                has_human_body = True
                            existing_lines.append(line.strip())

            if has_human_body and not verify_only:
                continue

            # Run pose detection on the frame
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            res = model(img, verbose=False, conf=0.30)
            if not res or len(res) == 0 or len(res[0].boxes) == 0:
                continue

            # Find largest person bounding box
            boxes = res[0].boxes
            best_box = None
            max_area = 0.0
            for b in boxes:
                xyxy = b.xyxy[0].cpu().numpy()
                area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                if area > max_area and float(b.conf[0]) >= 0.30:
                    max_area = area
                    best_box = xyxy

            if best_box is not None:
                bx1, by1, bx2, by2 = best_box
                cx = float(((bx1 + bx2) / 2.0) / w)
                cy = float(((by1 + by2) / 2.0) / h)
                bw = float((bx2 - bx1) / w)
                bh = float((by2 - by1) / h)

                # Clamp values
                cx = max(0.001, min(0.999, cx))
                cy = max(0.001, min(0.999, cy))
                bw = max(0.001, min(0.999, bw))
                bh = max(0.001, min(0.999, bh))

                human_line = f"4 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

                if not verify_only:
                    # Clean out any old human body lines before appending to ensure no duplicate
                    filtered_lines = [l for l in existing_lines if not l.startswith("4 ")]
                    filtered_lines.append(human_line)
                    with open(lbl_path, "w", encoding="utf-8") as f:
                        for l in filtered_lines:
                            f.write(l + "\n")
                total_human_added += 1

    print(f"\n[Summary] Total frames inspected: {total_images_processed}")
    print(f"[Summary] Total human body annotations confirmed/added: {total_human_added}")

    if not verify_only:
        # Update data.yaml
        yaml_path = os.path.join(DATASET_ROOT, "data.yaml")
        yaml_content = f"""# Bharatiya Antariksh Station (BAS) - Box Manipulation Dataset
path: {DATASET_ROOT.replace('\\', '/')}
train: images/train
val: images/val

# Classes
nc: 5
names:
  0: container_box
  1: container_lid
  2: component_box
  3: operator_hand
  4: human_body
"""
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        print(f"[Config] Updated 5-class config in: {yaml_path}")

        # Clear YOLO label cache files
        cache_files = [
            os.path.join(DATASET_ROOT, "labels", "train.cache"),
            os.path.join(DATASET_ROOT, "labels", "val.cache")
        ]
        for c in cache_files:
            if os.path.exists(c):
                os.remove(c)
                print(f"[Cache] Removed stale cache: {c}")

    print("=" * 70)


if __name__ == "__main__":
    verify_flag = "--verify-only" in sys.argv
    augment_dataset_with_human_body(verify_only=verify_flag)
