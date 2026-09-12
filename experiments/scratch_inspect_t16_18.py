import cv2, numpy as np, glob, os

images = sorted(glob.glob("experiments/clip1_analysis_frames/frame_t1*.jpg"))
for img_path in images:
    frame = cv2.imread(img_path)
    if frame is None:
        continue
    h, w = frame.shape[:2]
    cont_ymin = int(0.55 * h)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_brown = np.array([8, 25, 50])
    upper_brown = np.array([30, 230, 240])
    mask = cv2.inRange(hsv, lower_brown, upper_brown)
    mask_above = mask.copy()
    mask_above[cont_ymin - 20:, :] = 0
    mask_above[:, :int(0.25*w)] = 0
    mask_above[:, int(0.75*w):] = 0
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_clean = cv2.morphologyEx(mask_above, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    base = os.path.basename(img_path)
    print(f"--- {base} ---")
    for c in contours:
        area = cv2.contourArea(c)
        if area > 3000:
            x, y, bw, bh = cv2.boundingRect(c)
            aspect = bh / float(bw)
            print(f"   area={area:0.0f}, x={x}, y={y}, w={bw}, h={bh}, aspect={aspect:0.2f}")
