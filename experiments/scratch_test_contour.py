import cv2
import numpy as np
import glob
import os

images = sorted(glob.glob("experiments/clip1_analysis_frames/*.jpg"))

# Let's inspect each image with HSV thresholding for the cardboard component box
# And check its relationship with the container box
for img_path in images:
    frame = cv2.imread(img_path)
    h, w = frame.shape[:2]
    
    # Container box is at x: [0.40*w, 0.70*w], y: [0.55*h, 0.95*h]
    cont_ymin = int(0.55 * h)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Cardboard / brown box range
    lower_brown = np.array([8, 30, 60])
    upper_brown = np.array([28, 220, 230])
    mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # Exclude container box area on floor (y >= cont_ymin)
    mask_above = mask.copy()
    mask_above[cont_ymin - 20:, :] = 0
    # Also exclude the dark computer desk sides if needed (focus on center corridor x in [0.25*w, 0.75*w])
    mask_above[:, :int(0.30*w)] = 0
    mask_above[:, int(0.70*w):] = 0
    
    # Morph open to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask_clean = cv2.morphologyEx(mask_above, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    found_box = None
    for c in contours:
        area = cv2.contourArea(c)
        if area > 8000: # large cardboard area
            x, y, bw, bh = cv2.boundingRect(c)
            aspect = bh / float(bw)
            if 1.0 < aspect < 3.0: # vertical cardboard box
                found_box = (x, y, bw, bh, area)
                break
                
    base = os.path.basename(img_path)
    if found_box:
        x, y, bw, bh, area = found_box
        print(f"[{base}] DETECTED COMPONENT BOX: x={x}, y={y}, w={bw}, h={bh}, area={area}")
    else:
        print(f"[{base}] No component box in air")
