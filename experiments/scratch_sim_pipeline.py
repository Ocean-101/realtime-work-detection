import cv2, numpy as np
from ultralytics import YOLO
from typing import Any
import sys, os

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.core.types import BBox2D, ExperimentObject, EntityState, FSMStep, AnomalyType
from src.agents.validation_agent import ValidationAgent
from src.agents.har_agent import HARAgent

cap = cv2.VideoCapture("clip1.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

detector = YOLO("models/detector_offline.pt")
pose_model = YOLO("models/yolov8n-pose.pt")
validation = ValidationAgent(config_path="configs/box_return_fsm.json")
har = HARAgent()

history = []
last_lid_angle = 0.0
payload_was_extracted = False

for frame_idx in range(0, total_frames, 5):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret: break
    sec = frame_idx / fps
    h, w = frame.shape[:2]
    
    # 1. Neural detections
    res_det = list(detector(frame, verbose=False, conf=0.25))
    objects = {}
    cont_bbox = None
    if res_det:
        det0: Any = res_det[0]
        if det0.boxes is not None and len(det0.boxes) > 0:
            names = getattr(detector, "names", {})
            for b in det0.boxes:
                cid = int(b.cls[0].item())
                cname = names[cid]
                conf = float(b.conf[0].item())
                bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                bbox = BBox2D(bx1, by1, bx2, by2, conf, cid, cname)
                if cname == "container_box" and not cont_bbox:
                    cont_bbox = bbox
                    objects["container_box"] = ExperimentObject(name="container_box", class_name="container_box", bbox=bbox)
                elif cname == "human_body" and "human_body" not in objects:
                    objects["human_body"] = ExperimentObject(name="human_body", class_name="human_body", bbox=bbox)
                elif cname == "container_lid" and "container_lid" not in objects:
                    objects["container_lid"] = ExperimentObject(name="container_lid", class_name="container_lid", bbox=bbox)
                elif cname == "component_box" and "component_box" not in objects:
                    objects["component_box"] = ExperimentObject(name="component_box", class_name="component_box", bbox=bbox)
                
    # 2. Pose estimation
    res_pose = list(pose_model(frame, verbose=False, conf=0.25))
    wrists = []
    if res_pose:
        pose0: Any = res_pose[0]
        if pose0.boxes is not None and len(pose0.boxes) > 0 and pose0.keypoints is not None:
            kp = pose0.keypoints.xy[0].cpu().numpy()
            kconf = pose0.keypoints.conf[0].cpu().numpy()
            for idx in [9, 10]:
                if kconf[idx] > 0.35:
                    wrists.append((float(kp[idx][0]), float(kp[idx][1])))
                
    # 3. Lid elevation angle via brown density & container_lid
    target_angle = 0.0
    if cont_bbox:
        y_top = int(cont_bbox.ymin)
        y_search_top = max(0, y_top - 160)
        crop = frame[y_search_top:y_top, int(cont_bbox.xmin):int(cont_bbox.xmax)]
        if crop.size > 0:
            hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            mask_brown = cv2.inRange(hsv_crop, np.array([8, 25, 50]), np.array([30, 235, 240]))
            brown_density = np.sum(mask_brown > 0) / mask_brown.size
            if brown_density > 0.05 or "container_lid" in objects:
                target_angle = min(75.0, max(25.0, (brown_density / 0.25) * 60.0))
            else:
                target_angle = 0.0
                
    last_lid_angle = 0.70 * last_lid_angle + 0.30 * target_angle
    lid_angle = last_lid_angle
    
    # 4. Check extracted component box in air
    found_payload = None
    if cont_bbox:
        cont_ymin = cont_bbox.ymin
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([8, 25, 50]), np.array([30, 235, 240]))
        mask_air = mask.copy()
        mask_air[max(0, int(cont_ymin - 25)):, :] = 0
        mask_air[:, :max(0, int(0.25 * w))] = 0
        mask_air[:, min(w, int(0.75 * w)):] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_air = cv2.morphologyEx(mask_air, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask_air, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
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
                        found_payload = (x, y, bw, bh)
                        
    # 5. Resolve component_box
    if found_payload:
        px, py, pbw, pbh = found_payload
        item_bbox = BBox2D(px, py, px + pbw, py + pbh, 0.90, 2, "component_box")
        objects["component_box"] = ExperimentObject(name="component_box", class_name="component_box", bbox=item_bbox, state=EntityState.EXTRACTED, is_inside_container=False)
        payload_was_extracted = True
    elif cont_bbox:
        if payload_was_extracted:
            # Lowered back inside
            in_bbox = BBox2D(cont_bbox.xmin + 40, cont_bbox.ymin + 40, cont_bbox.xmax - 40, cont_bbox.ymax - 40, 0.85, 2, "component_box")
            objects["component_box"] = ExperimentObject(name="component_box", class_name="component_box", bbox=in_bbox, state=EntityState.DOCKED, is_inside_container=True)
        elif lid_angle >= 18.0 or "container_lid" in objects:
            # Open, waiting in cavity
            in_bbox = BBox2D(cont_bbox.xmin + 50, cont_bbox.ymin + 50, cont_bbox.xmax - 50, cont_bbox.ymax - 50, 0.80, 2, "component_box")
            objects["component_box"] = ExperimentObject(name="component_box", class_name="component_box", bbox=in_bbox, state=EntityState.DOCKED, is_inside_container=True)
            
    active_hoi = []
    step, deb, anom, msg, evt = validation.evaluate_step(objects, lid_angle, active_hoi, frame_idx)
    comp = objects.get("component_box")
    comp_str = f"inside={comp.is_inside_container}, state={comp.state.value}" if comp else "None"
    history.append((frame_idx, round(sec, 2), step.name, comp_str, lid_angle, validation.step_verdict))

cap.release()

print("\n--- STEP TRANSITIONS ACROSS CLIP1.MP4 ---")
current_s = None
for f_idx, sec, step_name, comp_str, lid_ang, verd in history:
    if step_name != current_s:
        print(f"t={sec:04.1f}s (F{f_idx:04d}): STEP -> {step_name:<16} | lid={lid_ang:4.1f}° | comp: {comp_str:<28} | verdict: {verd}")
        current_s = step_name

print("\nFinal verdict at end of video:")
print("Last step:", history[-1][2], "Verdict:", history[-1][5])
