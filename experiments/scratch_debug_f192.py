import cv2, sys
sys.path.insert(0, ".")

from src.agents.perception_agent import PerceptionAgent

cap = cv2.VideoCapture("clip1.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
perception = PerceptionAgent(model_path="models/detector_offline.pt")

for f in range(180, 240, 5):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ret, frame = cap.read()
    if not ret: break
    sec = f / fps
    objs, pose, lid = perception.process_frame(frame)
    comp = objs.get("component_box")
    b_str = f"({comp.bbox.xmin:.0f},{comp.bbox.ymin:.0f},{comp.bbox.xmax:.0f},{comp.bbox.ymax:.0f})" if comp and comp.bbox else "None"
    c_str = f"inside={comp.is_inside_container}, state={comp.state.value}, bbox={b_str}" if comp else "None"
    print(f"F{f:03d} (t={sec:.2f}s): lid={lid:.1f} | comp={c_str}")

cap.release()
