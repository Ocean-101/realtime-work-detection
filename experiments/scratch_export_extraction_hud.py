import cv2, os, sys
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.agents.perception_agent import PerceptionAgent
from src.agents.har_agent import HARAgent
from src.agents.validation_agent import ValidationAgent
from src.agents.monitoring_agent import MonitoringAgent

cap = cv2.VideoCapture("clip1.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

perception = PerceptionAgent(model_path="models/detector_offline.pt")
har = HARAgent()
validation = ValidationAgent(config_path="configs/box_return_fsm.json")
monitoring = MonitoringAgent(enable_tts=False, enable_streaming=False)

# Seek to t=17.0s (Frame 408) where the object is extracted in air
target_f = int(17.0 * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
ret, frame = cap.read()
cap.release()

if ret:
    objs, pose, lid_angle = perception.process_frame(frame)
    active_hoi, objs, activity = har.evaluate_interactions(pose, objs, lid_angle)
    step, deb, anom, msg, evt = validation.evaluate_step(objs, lid_angle, active_hoi, target_f)
    
    # Render full HUD frame
    hud_frame = monitoring.process_egress(
        raw_frame=frame,
        fused_pose=pose,
        objects=objs,
        lid_angle=lid_angle,
        current_step=step,
        debounce_count=deb,
        anomaly=anom,
        instruction="Hold component outside container for inspection",
        voice_alert="",
        transition_event=evt,
        frame_id=target_f,
        fps=fps,
        latency_ms=15.0,
        current_activity=activity,
        is_step_correct=True,
        step_verdict=validation.step_verdict,
        source_type="CLIP1_VIDEO"
    )
    
    out_path = "experiments/test_verification/hud_object_extracted_clip1.jpg"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, hud_frame)
    
    art_path = "C:/Users/AGEMC6/.gemini/antigravity-ide/brain/6cdbb8b7-ab4f-4db4-af1d-855526f87729/hud_object_extracted_clip1.jpg"
    cv2.imwrite(art_path, hud_frame)
    print(f"Exported extraction frame: {out_path} and {art_path}")
    print(f"Objects detected: {list(objs.keys())}")
    if "component_box" in objs:
        cb = objs["component_box"].bbox
        print(f"Component box: [{cb.xmin:.0f}, {cb.ymin:.0f}, {cb.xmax:.0f}, {cb.ymax:.0f}], inside={objs['component_box'].is_inside_container}")
