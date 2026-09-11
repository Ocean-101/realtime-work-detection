import cv2, numpy as np

cap = cv2.VideoCapture("clip1.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

for f_idx in range(96, 240, 20):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap.read()
    if not ret: break
    sec = f_idx / fps
    
    # Check brown density in flap region: y in [520, 680]
    crop = frame[520:680, 800:1300]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([8, 25, 50]), np.array([30, 235, 240]))
    brown_d = np.sum(mask > 0) / mask.size
    print(f"t={sec:.1f}s (F{f_idx}): brown_density={brown_d:.4f}")

cap.release()
