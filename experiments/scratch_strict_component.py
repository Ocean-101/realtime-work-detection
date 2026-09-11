import cv2, numpy as np
from ultralytics import YOLO

cap = cv2.VideoCapture("clip1.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

pose_model = YOLO("models/yolov8n-pose.pt")

for frame_idx in range(0, total_frames, 10):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret: break
    sec = frame_idx / fps
    h, w = frame.shape[:2]
    
    # 1. Pose keypoints
    res = pose_model(frame, verbose=False, conf=0.25)
    wrists = []
    if res and len(res[0].boxes) > 0 and res[0].keypoints is not None:
        kp = res[0].keypoints.xy[0].cpu().numpy()
        conf = res[0].keypoints.conf[0].cpu().numpy()
        for idx in [9, 10]:
            if conf[idx] > 0.35:
                wrists.append((float(kp[idx][0]), float(kp[idx][1])))
                
    # 2. Check for real extracted component box in upper air (y < 500)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_brown = np.array([8, 25, 50])
    upper_brown = np.array([30, 235, 240])
    mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # Restrict to air strictly above container flaps: y < 500
    mask_air = mask.copy()
    mask_air[500:, :] = 0
    mask_air[:, :int(0.30*w)] = 0
    mask_air[:, int(0.70*w):] = 0
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_air = cv2.morphologyEx(mask_air, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask_air, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    found_box = None
    for c in contours:
        area = cv2.contourArea(c)
        if area > 6000:
            x, y, bw, bh = cv2.boundingRect(c)
            aspect = bh / float(bw)
            if bh >= 110 and aspect >= 0.70: # Tall or square package, NOT a horizontal flap
                near_wrist = False
                if wrists:
                    for wx, wy in wrists:
                        if (x - 120 <= wx <= x + bw + 120) and (y - 120 <= wy <= y + bh + 120):
                            near_wrist = True
                            break
                else:
                    near_wrist = True
                if near_wrist:
                    found_box = (x, y, bw, bh, area, aspect)
                    break
                    
    if found_box:
        x, y, bw, bh, area, aspect = found_box
        print(f"t={sec:4.1f}s (F{frame_idx:03d}): [REAL COMPONENT IN AIR] x={x}, y={y}, w={bw}, h={bh}, aspect={aspect:.2f}, area={area:.0f}")

cap.release()
