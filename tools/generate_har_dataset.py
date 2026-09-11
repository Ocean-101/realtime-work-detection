"""
BAS Autonomous HAR System - Multi-Modal 3D Pose & Telemetry Dataset Generator
Processes recorded video or camera streams to generate a standardized dataset:
1. har_dataset_3d_joints.json: Full 3D joint and pose metadata (compatible with HMR2/YOLO11 schema)
2. har_dataset_3d_joints.npz: Compressed NumPy arrays of 3D skeleton joints, bboxes, and activities
3. har_dataset_telemetry.csv: 42-channel telemetry table for training offline HAR neural networks
4. har_dataset_summary.json: Dataset breakdown, class distribution, and procedural metrics
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime, timezone
import cv2
import numpy as np

# Ensure workspace root in path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.core.types import FSMStep, AnomalyType, Vector3D
from src.agents.perception_agent import PerceptionAgent
from src.agents.imu_agent import IMUAgent
from src.agents.fusion_agent import FusionAgent
from src.agents.har_agent import HARAgent
from src.agents.digital_twin_agent import DigitalTwinAgent
from src.agents.validation_agent import ValidationAgent
from src.agents.reasoning_agent import ReasoningAgent
from src.telemetry.csv_logger import RealtimeCSVTelemetryLogger


COCO_JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


def generate_har_dataset(
    video_source: str = "clip1.mp4",
    output_dir: str = "dataset/har_dataset",
    config_path: str = "configs/box_return_fsm.json",
    max_frames: int = None,
    stride: int = 1
):
    """
    Ingests video, executes the complete 8-agent HAR pipeline, and exports
    the synchronized multi-modal 3D pose, telemetry, and action dataset.
    """
    if str(video_source).isdigit():
        source_val = int(video_source)
    else:
        source_val = video_source
        if not os.path.exists(source_val):
            raise FileNotFoundError(f"Video file not found: {source_val}")

    os.makedirs(output_dir, exist_ok=True)
    csv_output_path = os.path.join(output_dir, "har_dataset_telemetry.csv")
    json_output_path = os.path.join(output_dir, "har_dataset_3d_joints.json")
    npz_output_path = os.path.join(output_dir, "har_dataset_3d_joints.npz")
    summary_output_path = os.path.join(output_dir, "har_dataset_summary.json")

    cap = cv2.VideoCapture(source_val)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video stream: {source_val}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("=" * 75)
    print("   BHARATIYA ANTARIKSH STATION (BAS) - MULTI-MODAL HAR DATASET GENERATOR")
    print(f"   Input Stream     : {video_source} ({width}x{height}, {total_frames} frames @ {video_fps:.2f} FPS)")
    print(f"   Procedure Config : {config_path}")
    print(f"   Target Directory : {output_dir}")
    print("=" * 75)

    # Initialize specialized agents
    agent_perception = PerceptionAgent()
    agent_imu = IMUAgent()
    agent_fusion = FusionAgent()
    agent_har = HARAgent()
    agent_twin = DigitalTwinAgent()
    agent_validation = ValidationAgent(config_path=config_path)
    agent_reasoning = ReasoningAgent(config_path=config_path)
    csv_logger = RealtimeCSVTelemetryLogger(
        output_csv_path=csv_output_path,
        realtime_feed_dir=os.path.join(output_dir, "realtime_feed")
    )

    dt = 1.0 / video_fps
    frame_idx = 0
    exported_count = 0
    t0 = time.time()

    # Collectors for NPZ and JSON datasets
    all_joints_3d = []
    all_boxes = []
    all_confidences = []
    all_camera_translations = []
    all_activities = []
    all_step_ids = []
    all_timestamps = []

    json_records = []
    activity_counts = {}
    step_counts = {}

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if stride > 1 and (frame_idx % stride != 0):
            continue

        if max_frames and exported_count >= max_frames:
            break

        # 1. Perception
        objects_cam, pose_cam, lid_angle = agent_perception.process_frame(raw_frame)

        # 2. IMU
        imu_telemetry = agent_imu.update_from_vision(pose_cam)

        # 3. Kinematic Fusion
        fused_pose, objects_rack = agent_fusion.fuse_kinematics(
            pose_cam, imu_telemetry, objects_cam, dt=dt
        )

        # 4. HAR Agent
        active_hoi, objects_state, current_activity = agent_har.evaluate_interactions(
            fused_pose, objects_rack, lid_angle
        )

        # 5. Validation FSM
        step, deb_count, anomaly, anomaly_msg, trans_event = agent_validation.evaluate_step(
            objects_state, lid_angle, active_hoi, frame_idx
        )

        # 6. Reasoning
        instruction, voice_alert = agent_reasoning.evaluate_guidance(
            step, anomaly, trans_event
        )

        # 7. Write to CSV Telemetry
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

        # 8. Collect 3D Joint and Skeleton Metadata
        timestamp_sec = round((frame_idx - 1) * dt, 3)

        # Extract 17 3D joints in camera coordinates
        joints_arr = np.zeros((17, 3), dtype=np.float32)
        person_box = [0.0, 0.0, float(width), float(height)]
        person_conf = 0.90
        cam_trans = [0.0, 0.0, 1.5]

        # Use 2D keypoints projected into 3D camera frame if available
        if fused_pose.keypoints_2d:
            # Estimate bounding box around all keypoints
            xs = [pt[0] for pt in fused_pose.keypoints_2d.values()]
            ys = [pt[1] for pt in fused_pose.keypoints_2d.values()]
            confs = [pt[2] for pt in fused_pose.keypoints_2d.values()]
            if xs and ys:
                min_x = max(0.0, min(xs) - 20)
                min_y = max(0.0, min(ys) - 20)
                max_x = min(float(width), max(xs) + 20)
                max_y = min(float(height), max(ys) + 20)
                person_box = [min_x, min_y, max_x, max_y]
                person_conf = float(np.mean(confs)) if confs else 0.90

            for j_idx, j_name in enumerate(COCO_JOINT_NAMES):
                if j_name in fused_pose.keypoints_2d:
                    px, py, c = fused_pose.keypoints_2d[j_name]
                    # Back-project using camera intrinsics at nominal depth 1.5m
                    depth = 1.5
                    cam_x = (px - agent_perception.cx) * depth / agent_perception.fx
                    cam_y = (py - agent_perception.cy) * depth / agent_perception.fy
                    joints_arr[j_idx] = [cam_x, cam_y, depth]
        else:
            # Fallback to key joints from fused_pose.joints
            if "wrist" in fused_pose.joints:
                w_cam = fused_pose.joints["wrist"].pos_camera
                joints_arr[10] = [w_cam.x, w_cam.y, w_cam.z]
            if "elbow" in fused_pose.joints:
                e_cam = fused_pose.joints["elbow"].pos_camera
                joints_arr[8] = [e_cam.x, e_cam.y, e_cam.z]
            if "shoulder" in fused_pose.joints:
                s_cam = fused_pose.joints["shoulder"].pos_camera
                joints_arr[6] = [s_cam.x, s_cam.y, s_cam.z]

        # Camera translation relative to body center
        hip_pt = joints_arr[11] if np.any(joints_arr[11]) else joints_arr[6]
        cam_trans = [float(hip_pt[0]), float(hip_pt[1]), float(hip_pt[2] or 1.5)]

        all_joints_3d.append(joints_arr)
        all_boxes.append(person_box)
        all_confidences.append(person_conf)
        all_camera_translations.append(cam_trans)
        all_activities.append(current_activity)
        all_step_ids.append(int(step))
        all_timestamps.append(timestamp_sec)

        # JSON record structure matching extended HMR2 schema
        frame_json = {
            "frame_id": frame_idx,
            "timestamp_sec": timestamp_sec,
            "fps": round(video_fps, 1),
            "image": {
                "width": width,
                "height": height,
                "channels": 3
            },
            "persons": [
                {
                    "person_id": 0,
                    "detection": {
                        "confidence": round(person_conf, 3),
                        "bbox": {
                            "x1": round(person_box[0], 1),
                            "y1": round(person_box[1], 1),
                            "x2": round(person_box[2], 1),
                            "y2": round(person_box[3], 1)
                        }
                    },
                    "camera_translation": {
                        "x": round(cam_trans[0], 3),
                        "y": round(cam_trans[1], 3),
                        "z": round(cam_trans[2], 3)
                    },
                    "pose": {
                        "joint_count": 17,
                        "joints_3d": [
                            {
                                "joint_id": i,
                                "name": COCO_JOINT_NAMES[i],
                                "x": round(float(joints_arr[i, 0]), 3),
                                "y": round(float(joints_arr[i, 1]), 3),
                                "z": round(float(joints_arr[i, 2]), 3)
                            }
                            for i in range(17)
                        ],
                        "biomechanics": {
                            "elbow_angle_deg": round(fused_pose.elbow_angle_deg, 1),
                            "shoulder_angle_deg": round(fused_pose.shoulder_angle_deg, 1),
                            "body_orientation_deg": round(fused_pose.body_orientation_deg, 1),
                            "rom_violated": fused_pose.rom_limits_violated
                        }
                    },
                    "activity": current_activity,
                    "fsm_step": {
                        "step_id": int(step),
                        "step_name": step.name
                    }
                }
            ],
            "objects": {
                k: {
                    "state": v.state.value if hasattr(v, "state") else "UNKNOWN",
                    "bbox": [v.bbox.xmin, v.bbox.ymin, v.bbox.xmax, v.bbox.ymax] if v.bbox else None,
                    "is_inside": getattr(v, "is_inside_container", True)
                }
                for k, v in objects_state.items()
            },
            "digital_twin_scene_graph": agent_twin.sync_scene_state(
                fused_pose, objects_state, lid_angle, step
            )
        }
        frame_json["digital_twin_scene_graph"]["activity"] = current_activity
        json_records.append(frame_json)

        # Accumulate metrics
        activity_counts[current_activity] = activity_counts.get(current_activity, 0) + 1
        step_counts[step.name] = step_counts.get(step.name, 0) + 1
        exported_count += 1

        if exported_count % 100 == 0:
            print(f"   [Processing] Extracted {exported_count}/{total_frames} frames (Current Activity: {current_activity})")

    cap.release()
    csv_logger.close()

    # Save Compressed NPZ Dataset
    np.savez_compressed(
        npz_output_path,
        joints_3d=np.array(all_joints_3d, dtype=np.float32),
        boxes=np.array(all_boxes, dtype=np.float32),
        confidences=np.array(all_confidences, dtype=np.float32),
        camera_translation=np.array(all_camera_translations, dtype=np.float32),
        activities=np.array(all_activities),
        step_ids=np.array(all_step_ids, dtype=np.int32),
        timestamps=np.array(all_timestamps, dtype=np.float32),
        joint_names=np.array(COCO_JOINT_NAMES)
    )

    # Save JSON Dataset
    master_json = {
        "system": {
            "name": "BAS Multi-Modal 3D HAR Pipeline",
            "detector": "YOLO11n / YOLOv8n-pose",
            "date_created": datetime.now(timezone.utc).isoformat(),
            "source_stream": video_source
        },
        "total_frames": exported_count,
        "records": json_records
    }

    def to_json_serializable(val):
        if isinstance(val, (np.bool_, bool)):
            return bool(val)
        if isinstance(val, (np.integer, int)):
            return int(val)
        if isinstance(val, (np.floating, float)):
            return float(val)
        if isinstance(val, np.ndarray):
            return val.tolist()
        if hasattr(val, "to_list"):
            return val.to_list()
        if hasattr(val, "value"):
            return val.value
        return str(val)

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(master_json, f, indent=2, default=to_json_serializable)

    # Save Summary JSON
    elapsed_time = time.time() - t0
    summary_data = {
        "source_video": video_source,
        "total_frames_exported": exported_count,
        "elapsed_seconds": round(elapsed_time, 2),
        "fps_processing": round(exported_count / max(1e-3, elapsed_time), 1),
        "activity_distribution": activity_counts,
        "step_distribution": step_counts,
        "output_files": {
            "telemetry_csv": csv_output_path,
            "joints_json": json_output_path,
            "joints_npz": npz_output_path
        }
    }
    with open(summary_output_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, default=to_json_serializable)

    print("\n" + "=" * 75)
    print("   [SUCCESS] MULTI-MODAL HAR DATASET GENERATION COMPLETE")
    print("=" * 75)
    print(f"   Total Frames Exported: {exported_count}")
    print(f"   Processing Time      : {elapsed_time:.2f}s ({exported_count/elapsed_time:.1f} FPS)")
    print("\n   [Activity Distribution]:")
    for act, count in sorted(activity_counts.items(), key=lambda x: -x[1]):
        print(f"     - {act:30s}: {count:5d} frames ({count/exported_count*100:.1f}%)")
    print(f"\n   1. Telemetry CSV : {csv_output_path}")
    print(f"   2. 3D Joints JSON: {json_output_path}")
    print(f"   3. 3D Joints NPZ : {npz_output_path}")
    print(f"   4. Dataset Stats : {summary_output_path}")
    print("=" * 75)
    return summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Multi-Modal 3D Pose & Telemetry HAR Dataset")
    parser.add_argument("--video", default="clip1.mp4", help="Path to input video file or camera index")
    parser.add_argument("--output-dir", default="dataset/har_dataset", help="Target output directory")
    parser.add_argument("--config", default="configs/box_return_fsm.json", help="FSM config protocol")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--stride", type=int, default=1, help="Frame subsampling stride")
    args = parser.parse_args()

    generate_har_dataset(
        video_source=args.video,
        output_dir=args.output_dir,
        config_path=args.config,
        max_frames=args.max_frames,
        stride=args.stride
    )
