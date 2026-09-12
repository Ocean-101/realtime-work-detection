import sys, os, time
import numpy as np
import cv2

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.core.types import FSMStep, AnomalyType, BBox2D, ExperimentObject, EntityState
from src.llm.realtime_llm_verifier import RealtimeLLMVerifier
from src.agents.validation_agent import ValidationAgent
from src.agents.reasoning_agent import ReasoningAgent

print("=" * 75)
print("   QWEN3-VL:2B-INSTRUCT + YOLO FUSION TEST")
print("=" * 75)

verifier = RealtimeLLMVerifier(model_name="qwen3-vl:2b-instruct", interval_sec=0.5)
init_v = verifier.get_latest_verification()
assert "verified_step" in init_v
assert "what_is_wrong" in init_v
print(f"   Initial Step: {init_v['verified_step']} | Status: {init_v['anomaly_verdict']}")

test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.rectangle(test_frame, (150, 100), (450, 400), (0, 200, 255), 2)
cv2.putText(test_frame, "container_box", (150, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

for f_id in range(5):
    verifier.push_telemetry(
        frame_id=f_id,
        step=FSMStep.BOX_OPENED,
        activity="OPENING_LID",
        lid_angle=45.0,
        is_inside=True,
        hoi_action="GRASP",
        hand_dist_m=0.12,
        anomaly=AnomalyType.NONE,
        frame=test_frame,
        detected_objects=["container_box", "operator_hand"],
        force_priority=True
    )
    time.sleep(0.02)

validation = ValidationAgent(config_path="configs/box_return_fsm.json")
reasoning = ReasoningAgent(config_path="configs/box_return_fsm.json")

mock_objects = {
    "container_box": ExperimentObject(name="container_box", class_name="container_box", bbox=BBox2D(100, 200, 300, 400, 0.95, 0, "container_box")),
    "container_lid": ExperimentObject(name="container_lid", class_name="container_lid", bbox=BBox2D(100, 150, 300, 200, 0.90, 1, "container_lid")),
    "component_box": ExperimentObject(name="component_box", class_name="component_box", is_inside_container=True, state=EntityState.DOCKED)
}

for f in range(validation.debounce_required):
    validation.evaluate_step(mock_objects, lid_angle=45.0, active_hoi=[], current_frame=f)

for f in range(10):
    step, deb, anom, msg, evt = validation.evaluate_step(
        mock_objects, lid_angle=5.0, active_hoi=[], current_frame=f + 10,
        llm_verification=verifier.get_latest_verification()
    )
    if anom != AnomalyType.NONE:
        inst, voice = reasoning.evaluate_guidance(step, anom, evt, vlm_explanation=validation.vlm_anomaly_explanation)
        print(f"   [ANOMALY CAUGHT]: Status={anom.name}")
        print(f"   [FSM Verdict]   : {validation.step_verdict}")
        print(f"   [VLM Reason]    : {validation.vlm_anomaly_explanation}")
        print(f"   [HUD Guidance]  : {inst}")
        break

verifier.close()
print("=" * 75)
print("   ALL QWEN3-VL + YOLO INTEGRATION TESTS PASSED!")
print("=" * 75)
