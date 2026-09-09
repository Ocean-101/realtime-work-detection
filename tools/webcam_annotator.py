"""
BAS Autonomous HAR System - Interactive Webcam Dataset Collector & Annotator
Enables recording real webcam footage and auto-labeling red/yellow experimental boxes
with color-space polygon extraction for rapid SIH dataset curation.
"""

import os
import time
import argparse
import cv2
import numpy as np


def run_webcam_collector(camera_idx=0, output_dir="dataset/real_webcam"):
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"[Error] Cannot open webcam index {camera_idx}")
        return

    print("=" * 60)
    print("   BAS WEBCAM DATASET COLLECTOR & AUTO-ANNOTATOR")
    print("=" * 60)
    print("Controls:")
    print("  [SPACE] : Capture & Auto-Annotate Current Frame")
    print("  [R]     : Toggle Recording Mode (captures 2 FPS)")
    print("  [Q]     : Quit")
    print("=" * 60)

    recording = False
    captured_count = 0
    last_record_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        display = frame.copy()

        # HSV Color Pre-segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Red mask
        mr1 = cv2.inRange(hsv, np.array([0, 100, 80]), np.array([10, 255, 255]))
        mr2 = cv2.inRange(hsv, np.array([170, 100, 80]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(mr1, mr2)
        
        # Yellow mask
        mask_yel = cv2.inRange(hsv, np.array([18, 100, 100]), np.array([35, 255, 255]))

        annotations = []

        # Find Red boxes
        cnts_r, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_r:
            if cv2.contourArea(c) > 600:
                rx, ry, rw, rh = cv2.boundingRect(c)
                cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)
                cv2.putText(display, "RED BOX", (rx, ry - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                annotations.append((2, (rx + rw/2)/w, (ry + rh/2)/h, rw/w, rh/h))

        # Find Yellow boxes
        cnts_y, _ = cv2.findContours(mask_yel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_y:
            if cv2.contourArea(c) > 600:
                yx, yy, yw, yh = cv2.boundingRect(c)
                cv2.rectangle(display, (yx, yy), (yx + yw, yy + yh), (0, 255, 255), 2)
                cv2.putText(display, "YELLOW BOX", (yx, yy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                annotations.append((3, (yx + yw/2)/w, (yy + yh/2)/h, yw/w, yh/h))

        # HUD Info
        rec_txt = "● REC ACTIVE" if recording else "○ STANDBY"
        rec_col = (0, 0, 255) if recording else (180, 180, 180)
        cv2.putText(display, f"STATUS: {rec_txt} | CAPTURED: {captured_count}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, rec_col, 2)

        now = time.time()
        should_save = False

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 32: # SPACE
            should_save = True
        elif key == ord('r'):
            recording = not recording
            print(f"[Webcam Collector] Recording set to: {recording}")

        if recording and (now - last_record_time) >= 0.5:
            should_save = True
            last_record_time = now

        if should_save:
            captured_count += 1
            img_file = os.path.join(output_dir, "images", f"real_{captured_count:04d}.jpg")
            lbl_file = os.path.join(output_dir, "labels", f"real_{captured_count:04d}.txt")
            cv2.imwrite(img_file, frame)
            with open(lbl_file, "w") as f:
                for cls_id, cx, cy, bw, bh in annotations:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            print(f"[Saved #{captured_count}] {len(annotations)} objects annotated -> {img_file}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    args = parser.parse_args()
    run_webcam_collector(args.camera)
