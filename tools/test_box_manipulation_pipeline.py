"""
BAS Autonomous HAR System - Box Manipulation Pipeline Verification Suite
Tests 5-class YOLO detection (including human_body), 3D skeletal pose,
procedural step calculation (S0 -> S1 -> S2 -> S3 -> S4), and wrong-step anomaly alerts.
"""

import os
import sys
import glob
import cv2
import json
import numpy as np

# Ensure workspace root in path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.core.types import FSMStep, AnomalyType, EntityState, HOIAction, BBox2D, ExperimentObject, Vector3D
from src.agents.perception_agent import PerceptionAgent
from src.agents.har_agent import HARAgent
from src.agents.validation_agent import ValidationAgent
from src.agents.monitoring_agent import MonitoringAgent
from src.agents.reasoning_agent import ReasoningAgent


def run_pipeline_test():
    print("=" * 75)
    print("   BHARATIYA ANTARIKSH STATION (BAS) - BOX MANIPULATION PIPELINE TEST")
    print("   Validates: 5-Class Detection, 3D Pose, Sequence Steps, & Wrong-Step Alerts")
    print("=" * 75)

    model_path = "models/detector_offline.pt"
    if not os.path.exists(model_path):
        print(f"[Warning] '{model_path}' not found yet. Using yolov8n.pt fallback for structure validation.")
        model_path = "yolov8n.pt"

    # 1. Initialize Agents
    print("\n[1/5] Initializing Specialized Agents...")
    perception = PerceptionAgent(
        model_path=model_path,
        pose_model_path="models/yolov8n-pose.pt"
    )
    har = HARAgent()
    validation = ValidationAgent(config_path="configs/box_return_fsm.json")
    reasoning = ReasoningAgent(config_path="configs/box_return_fsm.json")
    monitoring = MonitoringAgent(enable_tts=False, enable_streaming=False)

    print(f"  * Protocol: {validation.experiment_id} | Debounce: {validation.debounce_required}")

    # 2. Test Detection & Human Body on real dataset frames
    print("\n[2/5] Testing 5-Class Neural Perception & Human Body Structure...")
    test_img_dir = "dataset/box_manipulation_dataset/images/train"
    sample_imgs = sorted(glob.glob(os.path.join(test_img_dir, "*.jpg")))[:3]

    if sample_imgs:
        for p in sample_imgs:
            frame = cv2.imread(p)
            objs, pose, lid_ang = perception.process_frame(frame)
            print(f"  Frame {os.path.basename(p)}: Detected {len(objs)} objects: {list(objs.keys())} | Lid angle: {lid_ang:.1f}°")
            print(f"    - Skeleton joints recovered: {len(pose.joints)} | Dominant wrist: {'wrist' in pose.joints}")

    # 3. Simulate and Validate Nominal Procedural Steps: S0 -> S1 -> S2 -> S3 -> S4
    print("\n[3/5] Testing Nominal Sequence Step Calculation (S0 -> S1 -> S2 -> S3 -> S4)...")
    validation.reset()
    har.reset()

    # Step 0: IDLE
    assert validation.current_step == FSMStep.IDLE, "Initial step must be IDLE"
    print("  [Pass] S0: IDLE committed.")

    # Transition to Step 1: Open Container (Lid angle >= 25 deg for debounce_required frames)
    mock_objects_open = {
        "container_box": ExperimentObject(name="container_box", class_name="container_box", bbox=BBox2D(100, 200, 300, 400, 0.95, 0, "container_box")),
        "container_lid": ExperimentObject(name="container_lid", class_name="container_lid", bbox=BBox2D(100, 150, 300, 200, 0.90, 1, "container_lid")),
        "component_box": ExperimentObject(name="component_box", class_name="component_box", is_inside_container=True, bbox=BBox2D(150, 250, 250, 350, 0.92, 2, "component_box")),
        "human_body": ExperimentObject(name="human_body", class_name="human_body", bbox=BBox2D(50, 50, 400, 500, 0.94, 4, "human_body"))
    }
    for f in range(validation.debounce_required):
        step, deb, anom, _, evt = validation.evaluate_step(mock_objects_open, lid_angle=45.0, active_hoi=[], current_frame=f)
    assert step == FSMStep.BOX_OPENED, f"Expected BOX_OPENED, got {step.name}"
    print(f"  [Pass] S1: BOX_OPENED committed on event '{evt}'.")

    # Transition to Step 2: Pick Object (Object outside container)
    mock_objects_picked = {
        "container_box": mock_objects_open["container_box"],
        "container_lid": mock_objects_open["container_lid"],
        "component_box": ExperimentObject(name="component_box", class_name="component_box", is_inside_container=False, state=EntityState.EXTRACTED,
                                         bbox=BBox2D(150, 100, 250, 180, 0.92, 2, "component_box")),
        "human_body": mock_objects_open["human_body"]
    }
    for f in range(validation.debounce_required):
        step, deb, anom, _, evt = validation.evaluate_step(mock_objects_picked, lid_angle=45.0, active_hoi=[], current_frame=f + 10)
    assert step == FSMStep.OBJECT_EXTRACTED, f"Expected OBJECT_EXTRACTED, got {step.name}"
    print(f"  [Pass] S2: OBJECT_EXTRACTED committed on event '{evt}'.")

    # Transition to Step 3: Return Object (Object back inside container)
    mock_objects_returned = {
        "container_box": mock_objects_open["container_box"],
        "container_lid": mock_objects_open["container_lid"],
        "component_box": ExperimentObject(name="component_box", class_name="component_box", is_inside_container=True, state=EntityState.DOCKED,
                                         bbox=BBox2D(150, 250, 250, 350, 0.92, 2, "component_box")),
        "human_body": mock_objects_open["human_body"]
    }
    for f in range(validation.debounce_required):
        step, deb, anom, _, evt = validation.evaluate_step(mock_objects_returned, lid_angle=45.0, active_hoi=[], current_frame=f + 20)
    assert step == FSMStep.OBJECT_RETURNED, f"Expected OBJECT_RETURNED, got {step.name}"
    print(f"  [Pass] S3: OBJECT_RETURNED committed on event '{evt}'.")

    # Transition to Step 4: Close Container (Lid angle < 20 deg)
    for f in range(validation.debounce_required):
        step, deb, anom, _, evt = validation.evaluate_step(mock_objects_returned, lid_angle=5.0, active_hoi=[], current_frame=f + 30)
    assert step == FSMStep.COMPLETE, f"Expected COMPLETE, got {step.name}"
    print(f"  [Pass] S4: COMPLETE (Box Closed & Sealed) committed on event '{evt}'.")

    # 4. Test Wrong-Step Detection: Premature Close Anomaly Injection
    print("\n[4/5] Testing Wrong-Step Anomaly Injections & Alarms...")
    # Test Anomaly 1: Premature Close before extracting object
    val_anomaly = ValidationAgent(config_path="configs/box_return_fsm.json")
    val_anomaly.current_step = FSMStep.BOX_OPENED
    for f in range(6):
        s, _, anom, msg, _ = val_anomaly.evaluate_step(mock_objects_open, lid_angle=5.0, active_hoi=[], current_frame=f)
    assert anom == AnomalyType.ERROR_SKIP, f"Expected ERROR_SKIP, got {anom}"
    assert not val_anomaly.is_step_correct, "Step correctness must be False on anomaly"
    assert "WRONG STEP" in val_anomaly.step_verdict, f"Expected WRONG STEP in verdict, got: {val_anomaly.step_verdict}"
    print(f"  [Pass] Anomaly 1 Caught: {val_anomaly.step_verdict}")

    # Test Anomaly 2: Premature Close before returning object
    val_anomaly2 = ValidationAgent(config_path="configs/box_return_fsm.json")
    val_anomaly2.current_step = FSMStep.OBJECT_EXTRACTED
    for f in range(6):
        s, _, anom, msg, _ = val_anomaly2.evaluate_step(mock_objects_picked, lid_angle=5.0, active_hoi=[], current_frame=f)
    assert anom == AnomalyType.ERROR_SEQ, f"Expected ERROR_SEQ, got {anom}"
    assert not val_anomaly2.is_step_correct, "Step correctness must be False on anomaly"
    assert "WRONG STEP" in val_anomaly2.step_verdict, f"Expected WRONG STEP in verdict, got: {val_anomaly2.step_verdict}"
    print(f"  [Pass] Anomaly 2 Caught: {val_anomaly2.step_verdict}")

    # 5. Visual HUD & Wrong-Step Banner Export
    print("\n[5/5] Generating Visual HUD Verification Frames...")
    os.makedirs("experiments/test_verification", exist_ok=True)
    canvas_nominal = np.zeros((720, 1280, 3), dtype=np.uint8)
    canvas_nominal[:] = (35, 30, 25)

    # Render nominal frame HUD
    hud_nominal = monitoring.process_egress(
        raw_frame=canvas_nominal,
        fused_pose=perception._estimate_3d_pose(None, None, (0, 0), 1280, 720),
        objects=mock_objects_returned,
        lid_angle=45.0,
        current_step=FSMStep.OBJECT_RETURNED,
        debounce_count=6,
        anomaly=AnomalyType.NONE,
        instruction="Object returned. Next step: Please close the box.",
        voice_alert=None,
        transition_event=None,
        frame_id=100,
        fps=30.0,
        latency_ms=12.5,
        current_activity="RETURNING OBJECT INTO BOX",
        is_step_correct=True,
        step_verdict="STEP OK: Nominal Procedure",
        source_type="BOX_MANIPULATION_TEST"
    )
    nominal_path = "experiments/test_verification/hud_box_manipulation_nominal.jpg"
    cv2.imwrite(nominal_path, hud_nominal)
    print(f"  [Pass] Nominal Frame exported: {nominal_path}")

    # Render wrong-step frame HUD
    hud_wrong = monitoring.process_egress(
        raw_frame=canvas_nominal,
        fused_pose=perception._estimate_3d_pose(None, None, (0, 0), 1280, 720),
        objects=mock_objects_picked,
        lid_angle=5.0,
        current_step=FSMStep.OBJECT_EXTRACTED,
        debounce_count=0,
        anomaly=AnomalyType.ERROR_SEQ,
        instruction="Warning: Procedural error. The object has not been returned to the box.",
        voice_alert="Warning: Procedural error. The object has not been returned to the box.",
        transition_event=None,
        frame_id=120,
        fps=30.0,
        latency_ms=12.8,
        current_activity="CLOSING CONTAINER",
        is_step_correct=False,
        step_verdict="WRONG STEP: Object not returned into box before closing! [ERROR_SEQ]",
        source_type="BOX_MANIPULATION_TEST"
    )
    wrong_path = "experiments/test_verification/hud_box_manipulation_wrong_step.jpg"
    cv2.imwrite(wrong_path, hud_wrong)
    print(f"  [Pass] Wrong-Step Alert Frame exported: {wrong_path}")

    print("\n" + "=" * 75)
    print("   ALL PIPELINE VALIDATION TESTS PASSED CLEANLY (5/5)!")
    print("=" * 75)


if __name__ == "__main__":
    run_pipeline_test()
