import os, glob, cv2, numpy as np
from typing import Any
from ultralytics import YOLO

dataset_dir = "dataset/box_manipulation_dataset"
pose_model = YOLO("models/yolov8n-pose.pt")

train_imgs = glob.glob(os.path.join(dataset_dir, "images", "*", "*.jpg"))
print(f"Total dataset images: {len(train_imgs)}")

detected_comp_boxes = []

for img_p in sorted(train_imgs):
    frame = cv2.imread(img_p)
    if frame is None:
        continue
    h, w = frame.shape[:2]
    
    # 1. Pose keypoints
    res = list(pose_model(frame, verbose=False, conf=0.25))
    wrists = []
    if res:
        r0: Any = res[0]
        if r0.boxes is not None and len(r0.boxes) > 0 and r0.keypoints is not None:
            kp = r0.keypoints.xy[0].cpu().numpy()
            conf = r0.keypoints.conf[0].cpu().numpy()
            for idx in [9, 10]:
                if conf[idx] > 0.35:
                    wrists.append((float(kp[idx][0]), float(kp[idx][1])))
                
    # 2. Check for cardboard payload in air (above container: y < 580)
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
                # Check proximity to at least one wrist
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
                    payload = (x, y, bw, bh, area)
                    
    if payload:
        x, y, bw, bh, area = payload
        cx = (x + bw / 2.0) / w
        cy = (y + bh / 2.0) / h
        norm_w = bw / w
        norm_h = bh / h
        detected_comp_boxes.append((os.path.basename(img_p), cx, cy, norm_w, norm_h, area))

print(f"Detected extracted component box in {len(detected_comp_boxes)} frames.")
for item in detected_comp_boxes:
    print(f"   {item[0]}: cx={item[1]:.3f}, cy={item[2]:.3f}, w={item[3]:.3f}, h={item[4]:.3f}, area={item[5]:.0f}")
