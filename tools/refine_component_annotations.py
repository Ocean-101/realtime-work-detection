"""
BAS Autonomous HAR System - Refine Component Box & Object Annotations
Updates dataset/box_manipulation_dataset label files with:
1. High-precision dynamic contour + wrist tracking for component_box when extracted.
2. In-box cavity annotation for component_box when container is open.
3. Accurate human_body (class 4) and operator_hand (class 3) annotations.
"""
import os, glob, cv2, json, numpy as np
from ultralytics import YOLO

DATASET_ROOT = os.path.abspath("dataset/box_manipulation_dataset")
POSE_MODEL_PATH = "models/yolov8n-pose.pt"

def refine_annotations():
    print("=" * 70)
    print("   BHARATIYA ANTARIKSH STATION (BAS) - REFINING COMPONENT ANNOTATIONS")
    print(f"   Target Dataset : {DATASET_ROOT}")
    print("=" * 70)
    
    pose_model = YOLO(POSE_MODEL_PATH)
    
    action_json_path = os.path.join(DATASET_ROOT, "action_labels.json")
    action_records = {}
    if os.path.exists(action_json_path):
        with open(action_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for r in data.get("records", []):
                action_records[r["sample_name"]] = r.get("timestamp_sec", 0.0)
                
    total_updated = 0
    
    for split in ["train", "val"]:
        img_dir = os.path.join(DATASET_ROOT, "images", split)
        lbl_dir = os.path.join(DATASET_ROOT, "labels", split)
        
        img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        print(f"\nProcessing {split.upper()} ({len(img_files)} images)...")
        
        for img_path in img_files:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, f"{base_name}.txt")
            
            frame = cv2.imread(img_path)
            h, w = frame.shape[:2]
            
            # Read existing annotations
            existing_boxes = []
            if os.path.exists(lbl_path):
                with open(lbl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            existing_boxes.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
                            
            # 1. Pose detection for human body & wrists
            res = pose_model(frame, verbose=False, conf=0.25)
            wrists = []
            human_box = None
            if res and len(res[0].boxes) > 0:
                # Human body bbox
                b = res[0].boxes[0]
                bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                hcx = ((bx1 + bx2) / 2.0) / w
                hcy = ((by1 + by2) / 2.0) / h
                hbw = (bx2 - bx1) / w
                hbh = (by2 - by1) / h
                human_box = (4, hcx, hcy, hbw, hbh)
                
                if res[0].keypoints is not None:
                    kp = res[0].keypoints.xy[0].cpu().numpy()
                    kconf = res[0].keypoints.conf[0].cpu().numpy()
                    for idx in [9, 10]:
                        if kconf[idx] > 0.35:
                            wrists.append((float(kp[idx][0]), float(kp[idx][1])))
                            
            # 2. Check extracted cardboard payload in air (y < 580)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_brown = np.array([8, 25, 50])
            upper_brown = np.array([30, 235, 240])
            mask = cv2.inRange(hsv, lower_brown, upper_brown)
            mask_air = mask.copy()
            mask_air[580:, :] = 0
            mask_air[:, :int(0.30*w)] = 0
            mask_air[:, int(0.70*w):] = 0
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask_air = cv2.morphologyEx(mask_air, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(mask_air, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            payload = None
            max_area = 0
            for c in contours:
                area = cv2.contourArea(c)
                if area > 4500:
                    x, y, bw, bh = cv2.boundingRect(c)
                    if bh >= 60:
                        near_wrist = False
                        if wrists:
                            for wx, wy in wrists:
                                if (x - 100 <= wx <= x + bw + 100) and (y - 100 <= wy <= y + bh + 100):
                                    near_wrist = True
                                    break
                        else:
                            near_wrist = True
                        if near_wrist and area > max_area:
                            max_area = area
                            payload = (x, y, bw, bh)
                            
            # Approximate timestamp from metadata or filename
            sec = action_records.get(base_name, 0.0)
            
            # Rebuild clean annotations
            new_boxes = []
            
            # Preserve container_box (0) and container_lid (1)
            for cls_id, cx, cy, bw, bh in existing_boxes:
                if cls_id in (0, 1):
                    new_boxes.append((cls_id, cx, cy, bw, bh))
                    
            # Component box (2)
            if payload:
                px, py, pbw, pbh = payload
                comp_cx = (px + pbw / 2.0) / w
                comp_cy = (py + pbh / 2.0) / h
                comp_w = pbw / w
                comp_h = pbh / h
                new_boxes.append((2, comp_cx, comp_cy, comp_w, comp_h))
            elif 4.0 <= sec < 15.3 or 20.0 <= sec < 22.5:
                # Inside container cavity between white foam
                new_boxes.append((2, 0.5525, 0.665, 0.150, 0.110))
                
            # Operator hand (3)
            # Use detected wrists or existing hand
            if wrists:
                for wx, wy in wrists:
                    hand_cx = wx / w
                    hand_cy = wy / h
                    hand_w = 90.0 / w
                    hand_h = 90.0 / h
                    new_boxes.append((3, hand_cx, hand_cy, hand_w, hand_h))
            else:
                for cls_id, cx, cy, bw, bh in existing_boxes:
                    if cls_id == 3:
                        new_boxes.append((cls_id, cx, cy, bw, bh))
                        
            # Human body (4)
            if human_box:
                new_boxes.append(human_box)
            else:
                for cls_id, cx, cy, bw, bh in existing_boxes:
                    if cls_id == 4:
                        new_boxes.append((cls_id, cx, cy, bw, bh))
                        
            # Write out updated label file
            with open(lbl_path, "w", encoding="utf-8") as f:
                for cls_id, cx, cy, bw, bh in new_boxes:
                    cx_c = max(0.001, min(0.999, cx))
                    cy_c = max(0.001, min(0.999, cy))
                    bw_c = max(0.001, min(0.999, bw))
                    bh_c = max(0.001, min(0.999, bh))
                    f.write(f"{cls_id} {cx_c:.6f} {cy_c:.6f} {bw_c:.6f} {bh_c:.6f}\n")
                    
            total_updated += 1
            
    # Clear Ultralytics dataset cache files
    cache_files = glob.glob(os.path.join(DATASET_ROOT, "labels", "*.cache")) + \
                  glob.glob(os.path.join(DATASET_ROOT, "labels", "*", "*.cache"))
    for cf in cache_files:
        try:
            os.remove(cf)
            print(f"Removed cache file: {cf}")
        except Exception:
            pass
            
    print(f"\n[SUCCESS] Successfully refined annotations for {total_updated} frames across train/val splits!")

if __name__ == "__main__":
    refine_annotations()
