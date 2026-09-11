from ultralytics import YOLO
import cv2, glob, os

model = YOLO("models/yolov8n-pose.pt")
images = sorted(glob.glob("experiments/clip1_analysis_frames/frame_t*.jpg"))

for img_path in images:
    frame = cv2.imread(img_path)
    h, w = frame.shape[:2]
    res = model(frame, verbose=False, conf=0.25)
    base = os.path.basename(img_path)
    if res and len(res[0].boxes) > 0 and res[0].keypoints is not None:
        kp = res[0].keypoints.xy[0].cpu().numpy()
        conf = res[0].keypoints.conf[0].cpu().numpy()
        # Keypoints: 9: left_wrist, 10: right_wrist, 7: left_elbow, 8: right_elbow, 5: left_shoulder, 6: right_shoulder
        lw = (int(kp[9][0]), int(kp[9][1]), float(conf[9]))
        rw = (int(kp[10][0]), int(kp[10][1]), float(conf[10]))
        print(f"[{base}] Left Wrist: {lw}, Right Wrist: {rw}")
    else:
        print(f"[{base}] No pose detected")
