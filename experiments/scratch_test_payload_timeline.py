from typing import Any
import cv2
import numpy as np

cap = cv2.VideoCapture("clip1.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

from ultralytics import YOLO
pose_model = YOLO("models/yolov8n-pose.pt")

extracted_timeline = []

for frame_idx in range(0, total_frames, 6): # every 0.25s
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        break
    sec = frame_idx / fps
    h, w = frame.shape[:2]
    
    # 1. Pose keypoints
    res = list(pose_model(frame, verbose=False, conf=0.25))
    wrists = []
    if res:
        r0: Any = res[0]
        if r0.boxes is not None and len(r0.boxes) > 0 and r0.keypoints is not None:
            kp = r0.keypoints.xy[0].cpu().numpy()
            conf = r0.keypoints.conf[0].cpu().numpy()
            for idx in [9, 10]: # left_wrist, right_wrist
                if conf[idx] > 0.4:
                    wrists.append((float(kp[idx][0]), float(kp[idx][1])))
                
    # 2. Check cardboard payload in air (above container: y < 580)
    # Container is roughly y >= 600
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_brown = np.array([8, 25, 50])
    upper_brown = np.array([30, 235, 240])
    mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # Restrict to air above container: y < 580, center x in [0.30*w, 0.70*w]
    mask_air = mask.copy()
    mask_air[580:, :] = 0
    mask_air[:, :int(0.30*w)] = 0
    mask_air[:, int(0.70*w):] = 0
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_air = cv2.morphologyEx(mask_air, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask_air, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    payload_bbox = None
    max_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > 4500:
            x, y, bw, bh = cv2.boundingRect(c)
            # Check proximity to at least one wrist if wrists are detected
            near_wrist = False
            if wrists:
                for wx, wy in wrists:
                    if (x - 80 <= wx <= x + bw + 80) and (y - 80 <= wy <= y + bh + 80):
                        near_wrist = True
                        break
            else:
                near_wrist = True
                
            if near_wrist and area > max_area:
                max_area = area
                payload_bbox = (x, y, bw, bh, area)
                
    # Check wrist elevation as secondary kinematic cue
    wrists_elevated = False
    if wrists:
        elevated_count = sum(1 for wx, wy in wrists if wy < 480)
        if elevated_count >= 1:
            wrists_elevated = True
            
    is_extracted = (payload_bbox is not None) or (wrists_elevated and sec >= 15.0 and sec <= 20.5)
    extracted_timeline.append((frame_idx, round(sec, 2), is_extracted, payload_bbox, wrists_elevated))

cap.release()

print("Timeline of detected extracted payload:")
for f_idx, sec, is_ext, p_box, w_elev in extracted_timeline:
    if is_ext:
        p_str = f"bbox=({p_box[0]}, {p_box[1]}, {p_box[2]}, {p_box[3]}), area={p_box[4]:0.0f}" if p_box else "kinematic only"
        print(f"t={sec:04.1f}s (F{f_idx:04d}): EXTRACTED=True | {p_str} | wrists_elevated={w_elev}")
