"""
BAS Autonomous HAR System - Real-Time Frame Telemetry CSV Logger
Records frame-by-frame 3D kinematics, bounding boxes, object states, lid elevation,
procedural sequence progress, and Local LLM verification consensus into a high-precision
CSV format for 3D animation and Digital Twin model ingestion.
Maintains continuous real-time feeds in dedicated 'realtime_feed/' directory and 'experiments/'.
"""

import os
import csv
import time
import shutil
from datetime import datetime, timezone
from typing import Dict, Optional, Any
from src.core.types import (
    AstronautPose3D,
    ExperimentObject,
    FSMStep,
    AnomalyType,
    EntityState
)


class RealtimeCSVTelemetryLogger:
    """Logs per-frame 3D kinematics, Digital Twin state variables, and Local LLM verification to CSV."""

    CSV_HEADER = [
        "frame_id",
        "timestamp_sec",
        "timestamp_iso",
        "fps",
        "step_id",
        "step_name",
        "activity",
        "anomaly",
        "instruction",
        "lid_angle_deg",
        "lid_state",
        "wrist_cam_x",
        "wrist_cam_y",
        "wrist_cam_z",
        "wrist_rack_x",
        "wrist_rack_y",
        "wrist_rack_z",
        "elbow_angle_deg",
        "shoulder_angle_deg",
        "body_orientation_deg",
        "rom_violated",
        "container_rack_x",
        "container_rack_y",
        "container_rack_z",
        "container_xmin",
        "container_ymin",
        "container_xmax",
        "container_ymax",
        "component_rack_x",
        "component_rack_y",
        "component_rack_z",
        "component_xmin",
        "component_ymin",
        "component_xmax",
        "component_ymax",
        "component_state",
        "component_is_inside",
        "hand_to_object_dist_m",
        "llm_verified_step",
        "llm_verified_name",
        "llm_confidence",
        "llm_reason"
    ]

    def __init__(
        self,
        output_csv_path: Optional[str] = None,
        realtime_feed_dir: str = "realtime_feed"
    ):
        os.makedirs("experiments", exist_ok=True)
        if output_csv_path is None:
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.output_path = f"experiments/telemetry_{ts}.csv"
        else:
            self.output_path = output_csv_path
            os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)

        self.latest_symlink_path = "experiments/latest_telemetry.csv"

        # 1. Primary archival file
        self._file = open(self.output_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.CSV_HEADER)
        self._file.flush()

        # 2. Dedicated real-time video feed folder and files
        self.realtime_feed_dir = realtime_feed_dir
        os.makedirs(self.realtime_feed_dir, exist_ok=True)
        self.realtime_current_path = os.path.join(self.realtime_feed_dir, "current_feed_telemetry.csv")
        self.realtime_stream_path = os.path.join(self.realtime_feed_dir, "live_stream.csv")

        self._realtime_file = open(self.realtime_current_path, "w", newline="", encoding="utf-8")
        self._realtime_writer = csv.writer(self._realtime_file)
        self._realtime_writer.writerow(self.CSV_HEADER)
        self._realtime_file.flush()

        self.start_time = time.time()
        self.rows_written = 0

    def log_frame(
        self,
        frame_id: int,
        fps: float,
        step: FSMStep,
        activity: str,
        anomaly: AnomalyType,
        instruction: str,
        lid_angle: float,
        pose: AstronautPose3D,
        objects: Dict[str, ExperimentObject],
        llm_verification: Optional[Dict[str, Any]] = None
    ):
        """Writes a telemetry row capturing the complete spatial, physical, and LLM state of the frame."""
        now_time = time.time()
        elapsed_sec = round(now_time - self.start_time, 3)
        now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        # 1. Lid State
        lid_state = "OPEN" if lid_angle >= 20.0 else "CLOSED"

        # 2. Kinematics (Wrist / Elbow / Shoulder / ROM)
        wrist_cam = [0.0, 0.0, 0.0]
        wrist_rack = [0.0, 0.0, 0.0]
        if "wrist" in pose.joints:
            w_joint = pose.joints["wrist"]
            wrist_cam = [round(w_joint.pos_camera.x, 3), round(w_joint.pos_camera.y, 3), round(w_joint.pos_camera.z, 3)]
            wrist_rack = [round(w_joint.pos_rack.x, 3), round(w_joint.pos_rack.y, 3), round(w_joint.pos_rack.z, 3)]

        elbow_deg = round(pose.elbow_angle_deg, 1)
        shoulder_deg = round(pose.shoulder_angle_deg, 1)
        body_orient = round(pose.body_orientation_deg, 1)
        rom_flag = 1 if pose.rom_limits_violated else 0

        # 3. Container Box Coordinates & Bounding Box
        cont_x, cont_y, cont_z = 0.0, 0.0, 0.0
        cont_x1, cont_y1, cont_x2, cont_y2 = 0.0, 0.0, 0.0, 0.0
        if "container_box" in objects:
            c_obj = objects["container_box"]
            cont_x = round(c_obj.pos_rack.x, 3)
            cont_y = round(c_obj.pos_rack.y, 3)
            cont_z = round(c_obj.pos_rack.z, 3)
            if c_obj.bbox:
                cont_x1 = round(c_obj.bbox.xmin, 1)
                cont_y1 = round(c_obj.bbox.ymin, 1)
                cont_x2 = round(c_obj.bbox.xmax, 1)
                cont_y2 = round(c_obj.bbox.ymax, 1)

        # 4. Component Object Coordinates, State & Bounding Box
        comp_x, comp_y, comp_z = 0.0, 0.0, 0.0
        comp_x1, comp_y1, comp_x2, comp_y2 = 0.0, 0.0, 0.0, 0.0
        comp_state = "DOCKED"
        comp_is_inside = 1
        comp_dist = 0.50

        if "component_box" in objects:
            item = objects["component_box"]
            comp_x = round(item.pos_rack.x, 3)
            comp_y = round(item.pos_rack.y, 3)
            comp_z = round(item.pos_rack.z, 3)
            if item.bbox:
                comp_x1 = round(item.bbox.xmin, 1)
                comp_y1 = round(item.bbox.ymin, 1)
                comp_x2 = round(item.bbox.xmax, 1)
                comp_y2 = round(item.bbox.ymax, 1)
            comp_state = item.state.value if isinstance(item.state, EntityState) else str(item.state)
            comp_is_inside = 1 if item.is_inside_container else 0

            # Distance from wrist to component object
            if "wrist" in pose.joints:
                comp_dist = round(pose.joints["wrist"].pos_rack.distance_to(item.pos_rack), 3)

        # 5. Local LLM Step Verification Channels
        llm_step = int(step)
        llm_name = step.name
        llm_conf = 1.0
        llm_reason = "Deterministic validation consensus."
        if llm_verification:
            llm_step = int(llm_verification.get("verified_step", llm_step))
            llm_name = str(llm_verification.get("step_name", llm_name))
            llm_conf = round(float(llm_verification.get("confidence", 0.9)), 2)
            llm_reason = str(llm_verification.get("reason", llm_reason))

        row = [
            frame_id,
            elapsed_sec,
            now_iso,
            round(fps, 1),
            int(step),
            step.name,
            activity,
            anomaly.value,
            instruction,
            round(lid_angle, 1),
            lid_state,
            wrist_cam[0], wrist_cam[1], wrist_cam[2],
            wrist_rack[0], wrist_rack[1], wrist_rack[2],
            elbow_deg,
            shoulder_deg,
            body_orient,
            rom_flag,
            cont_x, cont_y, cont_z,
            cont_x1, cont_y1, cont_x2, cont_y2,
            comp_x, comp_y, comp_z,
            comp_x1, comp_y1, comp_x2, comp_y2,
            comp_state,
            comp_is_inside,
            comp_dist,
            llm_step,
            llm_name,
            llm_conf,
            llm_reason
        ]

        # Write to primary archival CSV
        self._writer.writerow(row)

        # Write to dedicated realtime_feed CSV and FLUSH IMMEDIATELY for live 3D stream
        self._realtime_writer.writerow(row)
        self._realtime_file.flush()

        self.rows_written += 1

        # Real-time line flush every 5 frames for archival file
        if self.rows_written % 5 == 0:
            self._file.flush()

        # Update latest_telemetry.csv and live_stream.csv every 15 frames (~0.5s)
        if self.rows_written % 15 == 0:
            try:
                self._file.flush()
                shutil.copyfile(self.output_path, self.latest_symlink_path)
                shutil.copyfile(self.realtime_current_path, self.realtime_stream_path)
            except Exception:
                pass

    def close(self):
        """Flushes and finalizes all telemetry CSV files."""
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()

            try:
                shutil.copyfile(self.output_path, self.latest_symlink_path)
            except Exception:
                pass

        if self._realtime_file and not self._realtime_file.closed:
            self._realtime_file.flush()
            self._realtime_file.close()

            try:
                shutil.copyfile(self.realtime_current_path, self.realtime_stream_path)
            except Exception:
                pass
