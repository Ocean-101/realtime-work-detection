import cv2
import numpy as np
from src.agents.perception_agent import PerceptionAgent

cap = cv2.VideoCapture("clip1.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

perception = PerceptionAgent(model_path="models/detector_offline.pt")

for frame_idx in range(0, 100, 10):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret: break
    sec = frame_idx / fps
    objs, pose, lid_angle = perception.process_frame(frame)
    comp = objs.get("component_box")
    cont = objs.get("container_box")
    lid = objs.get("container_lid")
    print(f"t={sec:.2f}s (F{frame_idx}): lid_angle={lid_angle:.1f} | cont={cont is not None} | lid={lid is not None} | comp={comp.state if comp else None} | comp_inside={comp.is_inside_container if comp else None}")
    if comp and comp.bbox:
        print(f"   comp bbox: {comp.bbox.xmin:.1f}, {comp.bbox.ymin:.1f}, {comp.bbox.xmax:.1f}, {comp.bbox.ymax:.1f}")

cap.release()
