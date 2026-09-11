"""
Test payload detection & FSM step transitions across clip1.mp4
"""
import os, sys, cv2, numpy as np

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.core.types import BBox2D, ExperimentObject, EntityState, FSMStep, AnomalyType
from src.agents.perception_agent import PerceptionAgent
from src.agents.validation_agent import ValidationAgent
from src.agents.har_agent import HARAgent

cap = cv2.VideoCapture("clip1.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

perception = PerceptionAgent(model_path="models/detector_offline.pt")
validation = ValidationAgent(config_path="configs/box_return_fsm.json")
har = HARAgent()

history = []

for frame_idx in range(0, total_frames, 5):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret: break
    sec = frame_idx / fps
    
    objs, pose, lid_angle = perception.process_frame(frame)
    active_hoi, objs, activity = har.evaluate_interactions(pose, objs, lid_angle)
    step, deb, anom, msg, evt = validation.evaluate_step(objs, lid_angle, active_hoi, frame_idx)
    
    comp = objs.get("component_box")
    comp_str = f"inside={comp.is_inside_container}, state={comp.state.value}" if comp else "None"
    history.append((frame_idx, round(sec, 2), step.name, comp_str, activity, validation.step_verdict))

cap.release()

print(f"Processed {len(history)} sampled frames.")
# Print transitions
current_s = None
for f_idx, sec, step_name, comp_str, act, verd in history:
    if step_name != current_s:
        print(f"t={sec:04.1f}s (F{f_idx:04d}): STEP -> {step_name} | comp: {comp_str} | act: {act} | verdict: {verd}")
        current_s = step_name

print("\nFinal verdict at end of video:")
print("Last step:", history[-1][2], "Verdict:", history[-1][5])
