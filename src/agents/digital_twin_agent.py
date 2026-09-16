"""
BAS Autonomous HAR System - Agent 5: Digital Twin Agent
Maintains the real-time 3D virtual environment synchronizing physical rack entities,
astronaut kinematic rig, container lid angle, and dynamic object states.
"""

import math
import cv2
import numpy as np
from typing import Dict, Any, Tuple
from src.core.types import (
    AstronautPose3D,
    ExperimentObject,
    EntityState,
    Vector3D,
    FSMStep
)


class DigitalTwinAgent:
    """Master 3D Scene Synchronizer and Digital Twin Engine."""

    def __init__(self, is_dual: bool = False):
        # Virtual Payload Rack Static Model
        self.rack_dimensions_m = (0.80, 0.60, 0.50) # Width, Height, Depth
        self.container_origin_rack = Vector3D(0.0, 0.0, 0.0)
        self.is_dual = is_dual
        self.persisted_entities = {}
        
        if self.is_dual:
            self.persisted_entities["red_box"] = {
                "state": "DOCKED",
                "pos_rack": [0.0, 0.0, 0.0],
                "is_inside_container": True
            }
            self.persisted_entities["yellow_box"] = {
                "state": "DOCKED",
                "pos_rack": [0.0, 0.0, 0.0],
                "is_inside_container": True
            }
        else:
            self.persisted_entities["component_box"] = {
                "state": "DOCKED",
                "pos_rack": [0.0, 0.0, 0.0],
                "is_inside_container": True
            }

    def sync_scene_state(
        self,
        pose: AstronautPose3D,
        objects: Dict[str, ExperimentObject],
        lid_angle: float,
        step: FSMStep
    ) -> Dict[str, Any]:
        """
        Builds the current 3D Scene Graph representation for WebGL/Three.js renderers.
        """
        scene_graph = {
            "version": "1.0",
            "step": step.name,
            "rack": {
                "label": "BAS-03 Payload Experiment Rack",
                "bounds": self.rack_dimensions_m
            },
            "container": {
                "lid_angle_deg": round(float(lid_angle), 1),
                "is_open": bool(lid_angle >= 28.0),
                "origin": self.container_origin_rack.to_list()
            },
            "entities": {},
            "astronaut": {
                "joints": {},
                "elbow_angle_deg": round(float(pose.elbow_angle_deg), 1),
                "rom_violated": bool(pose.rom_limits_violated)
            }
        }

        # Update persisted entities with latest vision detections
        for name, obj in objects.items():
            if "box" in name:
                self.persisted_entities[name] = {
                    "state": obj.state.value if isinstance(obj.state, EntityState) else str(obj.state),
                    "pos_rack": [float(x) for x in obj.pos_rack.to_list()],
                    "is_inside_container": bool(obj.is_inside_container)
                }

        # Write all persisted entities to the scene graph
        for name, entity_data in self.persisted_entities.items():
            scene_graph["entities"][name] = entity_data

        # Astronaut Joints
        for j_name, joint in pose.joints.items():
            scene_graph["astronaut"]["joints"][j_name] = {
                "pos_rack": joint.pos_rack.to_list(),
                "confidence": joint.confidence
            }

        # Full 17-Keypoint 3D Skeleton in Rack Frame
        if pose.keypoints_2d:
            scene_graph["astronaut"]["keypoints_2d"] = pose.keypoints_2d

        scene_graph["activity"] = getattr(pose, "activity", "IDLE")
        return scene_graph

    def sync_from_dataset_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ingests a frame record from har_dataset_3d_joints.json or telemetry CSV,
        returning a standardized 3D Scene Graph for the Digital Twin.
        """
        # 1. If pre-computed scene graph exists, return it directly
        if "digital_twin_scene_graph" in record:
            sg = dict(record["digital_twin_scene_graph"])
            if "activity" not in sg and "persons" in record and record["persons"]:
                sg["activity"] = record["persons"][0].get("activity", "IDLE")
            return sg

        # 2. Reconstruct from JSON record format
        if "persons" in record:
            person = record["persons"][0] if record["persons"] else {}
            pose_data = person.get("pose", {})
            biomech = pose_data.get("biomechanics", {})
            lid_ang = 0.0
            step_name = person.get("fsm_step", {}).get("step_name", "IDLE")
            activity = person.get("activity", "IDLE")

            # Extract joints
            joints_map = {}
            for j in pose_data.get("joints_3d", []):
                j_name = j.get("name", str(j.get("joint_id")))
                joints_map[j_name] = {
                    "pos_rack": [j.get("x", 0.0), j.get("y", 0.0), j.get("z", 0.0)],
                    "confidence": 0.90
                }

            # Extract entities
            entities = {}
            for obj_k, obj_v in record.get("objects", {}).items():
                entities[obj_k] = {
                    "state": obj_v.get("state", "DOCKED"),
                    "pos_rack": [0.0, 0.0, 0.0],
                    "is_inside_container": obj_v.get("is_inside", True)
                }

            return {
                "version": "1.0",
                "step": step_name,
                "activity": activity,
                "rack": {"label": "BAS-03 Payload Experiment Rack", "bounds": self.rack_dimensions_m},
                "container": {"lid_angle_deg": lid_ang, "is_open": lid_ang >= 28.0, "origin": self.container_origin_rack.to_list()},
                "entities": entities,
                "astronaut": {
                    "joints": joints_map,
                    "elbow_angle_deg": biomech.get("elbow_angle_deg", 90.0),
                    "shoulder_angle_deg": biomech.get("shoulder_angle_deg", 45.0),
                    "rom_violated": biomech.get("rom_violated", False)
                }
            }

        # 3. Reconstruct from CSV row format
        step_name = str(record.get("step_name", "IDLE"))
        activity = str(record.get("activity", "IDLE"))
        lid_ang = float(record.get("lid_angle_deg", 0.0))
        elbow = float(record.get("elbow_angle_deg", 90.0))
        shoulder = float(record.get("shoulder_angle_deg", 45.0))
        rom_flag = bool(int(record.get("rom_violated", 0)))

        wx = float(record.get("wrist_rack_x", 0.0))
        wy = float(record.get("wrist_rack_y", 0.0))
        wz = float(record.get("wrist_rack_z", 0.0))

        comp_inside = bool(int(record.get("component_is_inside", 1)))
        comp_state = str(record.get("component_state", "DOCKED"))

        return {
            "version": "1.0",
            "step": step_name,
            "activity": activity,
            "rack": {"label": "BAS-03 Payload Experiment Rack", "bounds": self.rack_dimensions_m},
            "container": {"lid_angle_deg": round(lid_ang, 1), "is_open": lid_ang >= 28.0, "origin": self.container_origin_rack.to_list()},
            "entities": {
                "component_box": {
                    "state": comp_state,
                    "pos_rack": [float(record.get("component_rack_x", 0.0)), float(record.get("component_rack_y", 0.0)), float(record.get("component_rack_z", 0.0))],
                    "is_inside_container": comp_inside
                }
            },
            "astronaut": {
                "joints": {
                    "wrist": {"pos_rack": [wx, wy, wz], "confidence": 0.95},
                    "elbow": {"pos_rack": [wx * 0.6, wy * 0.7 - 0.15, wz], "confidence": 0.95},
                    "shoulder": {"pos_rack": [0.25, -0.35, wz], "confidence": 0.95}
                },
                "elbow_angle_deg": round(elbow, 1),
                "shoulder_angle_deg": round(shoulder, 1),
                "rom_violated": rom_flag
            }
        }

    def render_digital_twin_canvas(
        self,
        scene_graph: Dict[str, Any],
        width: int = 480,
        height: int = 360
    ) -> np.ndarray:
        """
        Renders a 2.5D isometric view of the Digital Twin for desktop HUD / Mission GUI.
        """
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        # Deep space dark theme
        canvas[:] = (20, 22, 28)

        # Rack Border
        cv2.rectangle(canvas, (15, 15), (width - 15, height - 15), (60, 65, 75), 2)
        cv2.putText(canvas, "DIGITAL TWIN (RACK FRAME R)", (25, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)

        # Draw Coordinate Axes in top-right
        ax_ox, ax_oy = width - 45, 45
        cv2.arrowedLine(canvas, (ax_ox, ax_oy), (ax_ox + 25, ax_oy), (0, 0, 255), 2, tipLength=0.3)
        cv2.arrowedLine(canvas, (ax_ox, ax_oy), (ax_ox, ax_oy - 25), (0, 255, 0), 2, tipLength=0.3)
        cv2.putText(canvas, "+X_R", (ax_ox + 10, ax_oy + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        cv2.putText(canvas, "+Y_R", (ax_ox - 22, ax_oy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        # Step and Activity Banner
        step_text = scene_graph.get("step", "IDLE")
        act_text = scene_graph.get("activity", "IDLE")
        cv2.putText(canvas, f"FSM: {step_text}", (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 230, 255), 1)
        cv2.putText(canvas, f"HAR: {act_text}", (25, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 180, 0), 1)

        # Virtual Workbench
        wb_cy = int(height * 0.65)
        wb_cx = int(width * 0.5)
        cv2.ellipse(canvas, (wb_cx, wb_cy), (180, 50), 0, 0, 360, (40, 45, 55), -1)
        cv2.ellipse(canvas, (wb_cx, wb_cy), (180, 50), 0, 0, 360, (80, 85, 95), 1)

        # Draw Virtual Container Box
        cb_w, cb_h = 130, 60
        cb_x1 = wb_cx - cb_w // 2
        cb_y1 = wb_cy - cb_h // 2
        cv2.rectangle(canvas, (cb_x1, cb_y1), (cb_x1 + cb_w, cb_y1 + cb_h), (90, 95, 105), -1)
        cv2.rectangle(canvas, (cb_x1, cb_y1), (cb_x1 + cb_w, cb_y1 + cb_h), (140, 145, 160), 2)
        cv2.putText(canvas, "CONTAINER", (cb_x1 + 18, cb_y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)

        # Draw Virtual Lid
        lid_ang = float(scene_graph.get("container", {}).get("lid_angle_deg", 0.0))
        lid_elev = int(45 * (min(90.0, lid_ang) / 90.0))
        lid_col = (0, 255, 200) if lid_ang >= 28.0 else (140, 145, 160)
        cv2.line(canvas, (cb_x1, cb_y1), (cb_x1 + cb_w, cb_y1 - lid_elev), lid_col, 3)

        # Draw Component Box (Box Manipulation Experiment)
        entities = scene_graph.get("entities", {})
        if "component_box" in entities:
            c_info = entities["component_box"]
            cx = int(wb_cx if c_info["is_inside_container"] else wb_cx - 110)
            cy = int(cb_y1 + 15 if c_info["is_inside_container"] else cb_y1 - 40)
            cv2.rectangle(canvas, (cx - 20, cy - 14), (cx + 20, cy + 14), (255, 140, 0), -1)
            cv2.rectangle(canvas, (cx - 20, cy - 14), (cx + 20, cy + 14), (255, 255, 255), 1)
            cv2.putText(canvas, "COMP", (cx - 16, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        if "red_box" in entities:
            c_info = entities["red_box"]
            cx = int(wb_cx - 25 if c_info["is_inside_container"] else wb_cx - 110)
            cy = int(cb_y1 + 15 if c_info["is_inside_container"] else cb_y1 - 40)
            cv2.rectangle(canvas, (cx - 20, cy - 14), (cx + 20, cy + 14), (0, 0, 200), -1)
            cv2.rectangle(canvas, (cx - 20, cy - 14), (cx + 20, cy + 14), (255, 255, 255), 1)
            cv2.putText(canvas, "RED", (cx - 14, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        if "yellow_box" in entities:
            c_info = entities["yellow_box"]
            cx = int(wb_cx + 25 if c_info["is_inside_container"] else wb_cx + 110)
            cy = int(cb_y1 + 15 if c_info["is_inside_container"] else cb_y1 - 40)
            cv2.rectangle(canvas, (cx - 20, cy - 14), (cx + 20, cy + 14), (0, 220, 220), -1)
            cv2.rectangle(canvas, (cx - 20, cy - 14), (cx + 20, cy + 14), (0, 0, 0), 1)
            cv2.putText(canvas, "YEL", (cx - 14, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

        # Draw Astronaut Rig Skeleton
        joints = scene_graph.get("astronaut", {}).get("joints", {})

        # 17-Keypoint Connections if available
        coco_twin_links = [
            ("left_shoulder", "right_shoulder"),
            ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
            ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
            ("left_hip", "right_hip"),
            ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
            ("right_hip", "right_knee"), ("right_knee", "right_ankle")
        ]
        has_full_skeleton = any(j in joints for j in ("left_shoulder", "right_shoulder"))

        if has_full_skeleton:
            for j1, j2 in coco_twin_links:
                if j1 in joints and j2 in joints:
                    p1 = joints[j1]["pos_rack"]
                    p2 = joints[j2]["pos_rack"]
                    x1 = max(20, min(width - 20, int(wb_cx + p1[0] * 300)))
                    y1 = max(20, min(height - 20, int(wb_cy + p1[1] * 300)))
                    x2 = max(20, min(width - 20, int(wb_cx + p2[0] * 300)))
                    y2 = max(20, min(height - 20, int(wb_cy + p2[1] * 300)))
                    col = (0, 240, 255) if "wrist" in (j1, j2) or "elbow" in (j1, j2) else (80, 200, 120)
                    cv2.line(canvas, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)

            for j_name, j_data in joints.items():
                p = j_data["pos_rack"]
                px = max(20, min(width - 20, int(wb_cx + p[0] * 300)))
                py = max(20, min(height - 20, int(wb_cy + p[1] * 300)))
                cv2.circle(canvas, (px, py), 5, (255, 200, 50), -1)
        else:
            # Fallback to shoulder -> elbow -> wrist rig
            rig_links = [("shoulder", "elbow"), ("elbow", "wrist")]
            for j1, j2 in rig_links:
                if j1 in joints and j2 in joints:
                    p1 = joints[j1]["pos_rack"]
                    p2 = joints[j2]["pos_rack"]
                    x1 = max(20, min(width - 20, int(wb_cx + p1[0] * 320)))
                    y1 = max(20, min(height - 20, int(wb_cy + p1[1] * 320)))
                    x2 = max(20, min(width - 20, int(wb_cx + p2[0] * 320)))
                    y2 = max(20, min(height - 20, int(wb_cy + p2[1] * 320)))
                    cv2.line(canvas, (x1, y1), (x2, y2), (0, 240, 255), 3, cv2.LINE_AA)

            for j_name, j_col in [("shoulder", (0, 255, 128)), ("elbow", (0, 230, 255)), ("wrist", (255, 180, 0))]:
                if j_name in joints:
                    p = joints[j_name]["pos_rack"]
                    px = max(20, min(width - 20, int(wb_cx + p[0] * 320)))
                    py = max(20, min(height - 20, int(wb_cy + p[1] * 320)))
                    cv2.circle(canvas, (px, py), 8, j_col, -1)
                    cv2.circle(canvas, (px, py), 10, (255, 255, 255), 1, cv2.LINE_AA)
                    if j_name == "wrist":
                        cv2.putText(canvas, "ASTRONAUT HAND", (px - 45, py + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 50), 1)

        # Status text footer
        rom_flag = scene_graph.get("astronaut", {}).get("rom_violated", False)
        rom_str = " | [ROM LIMIT]" if rom_flag else ""
        cv2.putText(canvas, f"LID: {lid_ang:.1f} DEG | ELBOW: {scene_graph.get('astronaut', {}).get('elbow_angle_deg', 0)} DEG{rom_str}",
                    (25, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 170, 180), 1)

        return canvas

