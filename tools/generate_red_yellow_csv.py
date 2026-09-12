"""
BAS Autonomous HAR System - Dedicated Red-Yellow Experiment Telemetry Generator
Ingests e:\\SIH\\red_yellow.mp4, runs the full 8-agent procedural validation pipeline
under configs/red_yellow_fsm.json, and exports telemetry strictly into:
  - experiments_red_yellow/red_yellow_telemetry.csv
  - experiments_red_yellow/latest_telemetry.csv
  - experiments_red_yellow/red_yellow_action_stages.csv
  - realtime_feed_red_yellow/current_feed_telemetry.csv

Leaves previous experiment telemetry (experiments/latest_telemetry.csv) completely untouched.
"""

import os
import sys
import csv
import json
import time
import shutil
import argparse
import cv2
import numpy as np

# Ensure workspace root in sys.path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.core.types import FSMStep, AnomalyType
from src.agents.perception_agent import PerceptionAgent
from src.agents.imu_agent import IMUAgent
from src.agents.fusion_agent import FusionAgent
from src.agents.har_agent import HARAgent
from src.agents.validation_agent import ValidationAgent
from src.agents.reasoning_agent import ReasoningAgent
from src.telemetry.csv_logger import RealtimeCSVTelemetryLogger


def run_red_yellow_telemetry_generation(
    video_path: str = "red_yellow.mp4",
    config_path: str = "configs/red_yellow_fsm.json",
    output_dir: str = "experiments_red_yellow",
    realtime_dir: str = "realtime_feed_red_yellow"
):
    """
    Ingests red_yellow.mp4, executes the complete 8-agent validation pipeline with
    isolated output folders, and records frame-by-frame telemetry.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Target video file not found: {video_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Target FSM configuration not found: {config_path}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(realtime_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / video_fps

    output_telemetry_csv = os.path.join(output_dir, "red_yellow_telemetry.csv")
    output_stages_csv = os.path.join(output_dir, "red_yellow_action_stages.csv")

    print("=" * 80)
    print("   BHARATIYA ANTARIKSH STATION (BAS) - RED-YELLOW EXPERIMENT TELEMETRY")
    print(f"   Input Video File     : {video_path}")
    print(f"   Resolution & FPS     : {width}x{height} @ {video_fps:.2f} FPS ({total_frames} frames, {duration_sec:.2f}s)")
    print(f"   FSM Procedure Config : {config_path}")
    print(f"   Target Directory     : {output_dir}/")
    print(f"   Real-Time Feed Dir   : {realtime_dir}/")
    print("=" * 80)

    # Initialize specialized agents
    agent_perception = PerceptionAgent()
    agent_imu = IMUAgent()
    agent_fusion = FusionAgent()
    agent_har = HARAgent()
    agent_validation = ValidationAgent(config_path=config_path)
    agent_reasoning = ReasoningAgent(config_path=config_path)

    # Realtime CSV logger targeting isolated experiments_red_yellow directory
    csv_logger = RealtimeCSVTelemetryLogger(
        output_dir=output_dir,
        output_csv_path=output_telemetry_csv,
        realtime_feed_dir=realtime_dir
    )

    dt = 1.0 / video_fps
    frame_idx = 0
    stage_transitions = []
    last_step = FSMStep.IDLE
    step_start_frame = 0

    t_start = time.time()

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Agent 1: Perception (Masking, Contours, Box & Lid Tracking)
        objects_cam, pose_cam, lid_angle = agent_perception.process_frame(raw_frame)

        # Agent 2: IMU (Synthetic / Optical Flow Gyro & Accelerometer)
        imu_telemetry = agent_imu.update_from_vision(pose_cam)

        # Agent 3: Kinematic Fusion (Camera -> Payload Rack Coordinate Frame)
        fused_pose, objects_rack = agent_fusion.fuse_kinematics(
            pose_cam, imu_telemetry, objects_cam, dt=dt
        )

        # Agent 4: HAR (Human-Object Interaction & Spatial Status)
        active_hoi, objects_state, current_activity = agent_har.evaluate_interactions(
            fused_pose, objects_rack, lid_angle
        )

        # Agent 6: Validation (FSM Sequence & Precondition Enforcement)
        step, deb_count, anomaly, anomaly_msg, trans_event = agent_validation.evaluate_step(
            objects_state, lid_angle, active_hoi, frame_idx
        )

        # Track milestone transitions for stage summary
        if trans_event is not None or step != last_step:
            timestamp_sec = round(frame_idx / video_fps, 3)
            stage_transitions.append({
                "from_step": int(last_step),
                "from_step_name": last_step.name,
                "to_step": int(step),
                "to_step_name": step.name,
                "event": trans_event or "TRANSITION",
                "frame_start": step_start_frame,
                "frame_end": frame_idx,
                "timestamp_sec": timestamp_sec,
                "lid_angle": round(lid_angle, 1),
                "activity": current_activity
            })
            print(f"   [MILESTONE @ t={timestamp_sec:6.2f}s | Frame {frame_idx:4d}] Step {int(last_step)} -> Step {int(step)} ({step.name}) via '{trans_event}'")
            last_step = step
            step_start_frame = frame_idx

        # Agent 7: Reasoning (Contextual Guidance & Root Cause Analysis)
        instruction, voice_alert = agent_reasoning.evaluate_guidance(
            current_step=step,
            anomaly=anomaly,
            transition_event=trans_event,
            vlm_explanation=agent_validation.vlm_anomaly_explanation,
            is_step_correct=agent_validation.is_step_correct,
            step_verdict=agent_validation.step_verdict,
            anomaly_message=anomaly_msg
        )

        # Log frame telemetry row
        csv_logger.log_frame(
            frame_id=frame_idx,
            fps=video_fps,
            step=step,
            activity=current_activity,
            anomaly=anomaly,
            instruction=instruction,
            lid_angle=lid_angle,
            pose=fused_pose,
            objects=objects_state,
            llm_verification=None
        )

        if frame_idx % 100 == 0 or frame_idx == total_frames:
            elapsed = time.time() - t_start
            fps_proc = frame_idx / max(0.001, elapsed)
            print(f"   Processing: {frame_idx:4d}/{total_frames} frames ({frame_idx/total_frames*100:5.1f}%) @ {fps_proc:5.1f} FPS | Active: Step {int(step)} ({step.name})")

    cap.release()
    csv_logger.close()

    # Also make a clean symlink or copy to latest_telemetry.csv inside experiments_red_yellow
    latest_csv_path = os.path.join(output_dir, "latest_telemetry.csv")
    try:
        shutil.copyfile(output_telemetry_csv, latest_csv_path)
    except Exception as e:
        print(f"   [Notice] Could not copy to latest_telemetry.csv: {e}")

    # Write stage transitions CSV
    with open(output_stages_csv, "w", newline="", encoding="utf-8") as sf:
        fieldnames = [
            "from_step", "from_step_name", "to_step", "to_step_name",
            "event", "frame_start", "frame_end", "timestamp_sec",
            "lid_angle", "activity"
        ]
        writer = csv.DictWriter(sf, fieldnames=fieldnames)
        writer.writeheader()
        for st in stage_transitions:
            writer.writerow(st)

    elapsed_total = time.time() - t_start
    print("=" * 80)
    print("   PROCESSING COMPLETED SUCCESSFULLY")
    print(f"   Total Frames Logged : {frame_idx} frames in {elapsed_total:.2f}s ({frame_idx/max(0.001, elapsed_total):.1f} FPS)")
    print(f"   Telemetry CSV       : {output_telemetry_csv}")
    print(f"   Latest Telemetry    : {latest_csv_path}")
    print(f"   Action Stages CSV   : {output_stages_csv}")
    print(f"   Live Feed Snapshot  : {os.path.join(realtime_dir, 'current_feed_telemetry.csv')}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Red-Yellow Experiment Telemetry CSV")
    parser.add_argument("--video", default="red_yellow.mp4", help="Path to red_yellow.mp4")
    parser.add_argument("--config", default="configs/red_yellow_fsm.json", help="Path to red_yellow_fsm.json")
    parser.add_argument("--outdir", default="experiments_red_yellow", help="Output directory for telemetry")
    parser.add_argument("--feeddir", default="realtime_feed_red_yellow", help="Directory for real-time feed CSV")
    args = parser.parse_args()

    run_red_yellow_telemetry_generation(
        video_path=args.video,
        config_path=args.config,
        output_dir=args.outdir,
        realtime_dir=args.feeddir
    )
