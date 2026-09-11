import cv2, numpy as np

cap = cv2.VideoCapture("clip1.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

for f_idx in [0, 24, 48, 96, 192, 288, 384, 480, 576, 624]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap.read()
    if not ret: break
    h, w = frame.shape[:2]
    # Container box is approx x: 840-1280, y: 650-980
    y_top = 650
    crop = frame[y_top - 160:y_top, 840:1280]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    density = np.sum(edges > 0) / edges.size
    
    # Also check cardboard color above container (when flaps are up, brown cardboard flaps stick up)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask_brown = cv2.inRange(hsv, np.array([8, 25, 50]), np.array([30, 235, 240]))
    brown_density = np.sum(mask_brown > 0) / mask_brown.size
    
    sec = f_idx / fps
    print(f"t={sec:4.1f}s (F{f_idx:03d}): edge_density={density:.4f} | brown_density={brown_density:.4f}")

cap.release()
