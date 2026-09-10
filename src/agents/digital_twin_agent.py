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

    def __init__(self):
        # Virtual Payload Rack Static Model
        self.rack_dimensions_m = (0.80, 0.60, 0.50) # Width, Height, Depth
        self.container_origin_rack = Vector3D(0.0, 0.0, 0.0)

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
                "lid_angle_deg": round(lid_angle, 1),
                "is_open": lid_angle >= 40.0,
                "origin": self.container_origin_rack.to_list()
            },
            "entities": {},
            "astronaut": {
                "joints": {},
                "elbow_angle_deg": round(pose.elbow_angle_deg, 1),
                "rom_violated": pose.rom_limits_violated
            }
        }

        # Entities (Red Box, Yellow Box)
        for name, obj in objects.items():
            scene_graph["entities"][name] = {
                "state": obj.state.value if isinstance(obj.state, EntityState) else str(obj.state),
                "pos_rack": obj.pos_rack.to_list(),
                "is_inside_container": obj.is_inside_container
            }

        # Astronaut Joints
        for j_name, joint in pose.joints.items():
            scene_graph["astronaut"]["joints"][j_name] = {
                "pos_rack": joint.pos_rack.to_list(),
                "confidence": joint.confidence
            }

        return scene_graph

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
        lid_ang = scene_graph["container"]["lid_angle_deg"]
        lid_elev = int(45 * (lid_ang / 90.0))
        cv2.line(canvas, (cb_x1, cb_y1), (cb_x1 + cb_w, cb_y1 - lid_elev), (0, 255, 200) if lid_ang > 35 else (140, 145, 160), 3)

        # Draw Component Box (Box Manipulation Experiment)
        entities = scene_graph["entities"]
        if "component_box" in entities:
            c_info = entities["component_box"]
            cx = int(wb_cx if c_info["is_inside_container"] else wb_cx - 110)
            cy = int(cb_y1 + 15 if c_info["is_inside_container"] else cb_y1 - 40)
            cv2.rectangle(canvas, (cx - 20, cy - 14), (cx + 20, cy + 14), (255, 140, 0), -1)
            cv2.rectangle(canvas, (cx - 20, cy - 14), (cx + 20, cy + 14), (255, 255, 255), 1)
            cv2.putText(canvas, "COMP", (cx - 16, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # Draw Astronaut Rig Skeleton
        joints = scene_graph["astronaut"]["joints"]
        if "wrist" in joints:
            w_rack = joints["wrist"]["pos_rack"]
            # Map Rack coordinates [x, y, z] to canvas pixel space
            px = int(wb_cx + w_rack[0] * 320)
            py = int(wb_cy + w_rack[1] * 320)
            px = max(20, min(width - 20, px))
            py = max(20, min(height - 20, py))

            # Draw Hand node
            cv2.circle(canvas, (px, py), 10, (255, 180, 0), -1)
            cv2.circle(canvas, (px, py), 12, (255, 255, 255), 2)
            cv2.putText(canvas, "ASTRONAUT HAND", (px - 45, py + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 50), 1)

        # Status text footer
        cv2.putText(canvas, f"LID: {lid_ang:.1f} DEG | ELBOW: {scene_graph['astronaut']['elbow_angle_deg']} DEG",
                    (25, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 170, 180), 1)

        return canvas
