from typing import Any
import cv2
from ultralytics import YOLO

model = YOLO("models/detector_offline.pt")
cap = cv2.VideoCapture("clip1.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

for frame_idx in range(0, 650, 24):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret: break
    sec = frame_idx / fps
    res = list(model(frame, verbose=False, conf=0.35))
    classes_detected = []
    if res:
        r0: Any = res[0]
        if r0.boxes is not None and len(r0.boxes) > 0:
            names = getattr(model, "names", {})
            for b in r0.boxes:
                cid = int(b.cls[0].item())
                cname = names[cid]
                conf = float(b.conf[0].item())
                classes_detected.append(f"{cname} ({conf:.2f})")
    print(f"t={sec:.1f}s (F{frame_idx}): {classes_detected}")

cap.release()
