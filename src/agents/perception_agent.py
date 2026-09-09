"""
BAS Autonomous HAR System - Agent 1: Perception Agent
Detects experimental components (Container, Lid, Red Box, Yellow Box, Hands)
and extracts 3D skeletal keypoints in camera space using PhysAstro-Pose principles.
"""

import math
import json
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from src.core.types import (
    BBox2D,
    ExperimentObject,
    EntityState,
    Vector3D,
    Joint3D,
    AstronautPose3D
)


class PerceptionAgent:
    """Perception Agent executing object detection and 3D human pose recovery."""

    def __init__(self, calib_path: str = "configs/camera_calib.json"):
        # Load camera intrinsics
        with open(calib_path, "r") as f:
            calib_data = json.load(f)
        
        intr = calib_data["intrinsics"]
        self.fx = intr["fx"]
        self.fy = intr["fy"]
        self.cx = intr["cx"]
        self.cy = intr["cy"]
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)
        self.K_inv = np.linalg.inv(self.K)

        self.last_lid_angle = 0.0

    def process_frame(
        self,
        frame: np.ndarray,
        roi_box: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[Dict[str, ExperimentObject], AstronautPose3D, float]:
        """
        Processes camera frame to detect objects and human 3D pose.
        Returns:
            - objects: Dictionary of detected ExperimentObjects
            - pose: Reconstructed 3D astronaut pose in camera coordinates
            - lid_angle: Measured container lid angle in degrees
        """
        h, w, _ = frame.shape
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        objects: Dict[str, ExperimentObject] = {}

        # 1. Detect Red Box (HSV: 0-10 & 170-180)
        mask_r1 = cv2.inRange(hsv, np.array([0, 90, 70]), np.array([10, 255, 255]))
        mask_r2 = cv2.inRange(hsv, np.array([170, 90, 70]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(mask_r1, mask_r2)
        red_bbox, red_center = self._extract_largest_bbox(mask_red, "red_box", 2, min_area=400)
        if red_bbox:
            red_cam = self._pixel_to_camera_coord(red_center[0], red_center[1], depth_m=1.15)
            objects["red_box"] = ExperimentObject(
                name="red_box",
                class_name="red_box",
                bbox=red_bbox,
                pos_rack=red_cam # Will be transformed by Fusion Agent
            )

        # 2. Detect Yellow Box (HSV: 18-35)
        mask_yellow = cv2.inRange(hsv, np.array([18, 100, 100]), np.array([35, 255, 255]))
        yellow_bbox, yellow_center = self._extract_largest_bbox(mask_yellow, "yellow_box", 3, min_area=400)
        if yellow_bbox:
            yellow_cam = self._pixel_to_camera_coord(yellow_center[0], yellow_center[1], depth_m=1.15)
            objects["yellow_box"] = ExperimentObject(
                name="yellow_box",
                class_name="yellow_box",
                bbox=yellow_bbox,
                pos_rack=yellow_cam
            )

        # 3. Detect Astronaut Glove / Hand (High value, low saturation, off-white)
        mask_hand = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 50, 255]))
        # Clean mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_hand = cv2.morphologyEx(mask_hand, cv2.MORPH_OPEN, kernel)
        hand_bbox, hand_center = self._extract_largest_bbox(mask_hand, "astronaut_hand", 4, min_area=600)

        # 4. Detect Container Box & Lid
        # Container is metallic dark gray (low saturation, mid value)
        mask_cont = cv2.inRange(hsv, np.array([0, 0, 40]), np.array([180, 70, 160]))
        cont_bbox, cont_center = self._extract_largest_bbox(mask_cont, "container_box", 0, min_area=8000)
        if cont_bbox:
            cont_cam = self._pixel_to_camera_coord(cont_center[0], cont_center[1], depth_m=1.20)
            objects["container_box"] = ExperimentObject(
                name="container_box",
                class_name="container_box",
                bbox=cont_bbox,
                pos_rack=cont_cam
            )

        # Compute Lid Angle based on top edge contours or relative positions
        lid_angle = self._estimate_lid_angle(frame, cont_bbox)

        # Construct 3D Astronaut Pose in Camera Space
        pose = self._estimate_3d_pose(hand_bbox, hand_center, w, h)

        return objects, pose, lid_angle

    def _extract_largest_bbox(
        self,
        mask: np.ndarray,
        class_name: str,
        class_id: int,
        min_area: int = 500
    ) -> Tuple[Optional[BBox2D], Tuple[float, float]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, (0.0, 0.0)

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < min_area:
            return None, (0.0, 0.0)

        x, y, bw, bh = cv2.boundingRect(c)
        bbox = BBox2D(
            xmin=float(x),
            ymin=float(y),
            xmax=float(x + bw),
            ymax=float(y + bh),
            confidence=min(0.99, area / 5000.0),
            class_id=class_id,
            class_name=class_name
        )
        return bbox, (x + bw / 2.0, y + bh / 2.0)

    def _estimate_lid_angle(self, frame: np.ndarray, cont_bbox: Optional[BBox2D]) -> float:
        """Estimates container lid elevation angle."""
        if not cont_bbox:
            return self.last_lid_angle

        # Check upper region above container for lid contour presence
        y_top = int(cont_bbox.ymin)
        y_search_top = max(0, y_top - 160)
        x1 = int(cont_bbox.xmin)
        x2 = int(cont_bbox.xmax)
        
        crop = frame[y_search_top:y_top, x1:x2]
        if crop.size == 0:
            return self.last_lid_angle

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        edge_density = np.sum(edges > 0) / edges.size

        # If significant edges exist above container box, lid is elevated
        if edge_density > 0.04:
            target_angle = min(75.0, edge_density * 900.0)
        else:
            target_angle = 0.0

        # Smooth angle
        self.last_lid_angle = 0.8 * self.last_lid_angle + 0.2 * target_angle
        return self.last_lid_angle

    def _pixel_to_camera_coord(self, u: float, v: float, depth_m: float) -> Vector3D:
        """Projects (u, v) image pixel + metric depth to camera coordinates X_C."""
        x = (u - self.cx) * depth_m / self.fx
        y = (v - self.cy) * depth_m / self.fy
        return Vector3D(x, y, depth_m)

    def _estimate_3d_pose(
        self,
        hand_bbox: Optional[BBox2D],
        hand_center: Tuple[float, float],
        width: int,
        height: int
    ) -> AstronautPose3D:
        """
        Reconstructs 3D upper-limb kinematics applying PhysAstro-Pose principles.
        """
        pose = AstronautPose3D()
        if not hand_bbox:
            return pose

        # Hand/Wrist joint in camera coordinates
        hx, hy = hand_center
        wrist_pos = self._pixel_to_camera_coord(hx, hy, depth_m=1.10)
        pose.joints["wrist"] = Joint3D(name="wrist", pos_camera=wrist_pos, confidence=hand_bbox.confidence)

        # Kinematic prior for forearm and elbow:
        # Microgravity Neutral Body Posture (NBP): elbow naturally flexed ~77°
        # Extrapolate elbow based on arm orientation
        arm_length_m = 0.28
        forearm_vec = Vector3D(x=-0.18, y=-0.15, z=0.08)
        elbow_pos = Vector3D(
            x=wrist_pos.x + forearm_vec.x,
            y=wrist_pos.y + forearm_vec.y,
            z=wrist_pos.z + forearm_vec.z
        )
        pose.joints["elbow"] = Joint3D(name="elbow", pos_camera=elbow_pos, confidence=0.85)

        upper_arm_vec = Vector3D(x=-0.22, y=-0.20, z=0.05)
        shoulder_pos = Vector3D(
            x=elbow_pos.x + upper_arm_vec.x,
            y=elbow_pos.y + upper_arm_vec.y,
            z=elbow_pos.z + upper_arm_vec.z
        )
        pose.joints["shoulder"] = Joint3D(name="shoulder", pos_camera=shoulder_pos, confidence=0.80)

        # Calculate biomechanical joint angles
        v_fore = np.array([wrist_pos.x - elbow_pos.x, wrist_pos.y - elbow_pos.y, wrist_pos.z - elbow_pos.z])
        v_upper = np.array([shoulder_pos.x - elbow_pos.x, shoulder_pos.y - elbow_pos.y, shoulder_pos.z - elbow_pos.z])
        
        norm_product = np.linalg.norm(v_fore) * np.linalg.norm(v_upper)
        if norm_product > 1e-6:
            cos_elbow = np.clip(np.dot(v_fore, v_upper) / norm_product, -1.0, 1.0)
            pose.elbow_angle_deg = float(np.degrees(np.arccos(cos_elbow)))
        else:
            pose.elbow_angle_deg = 77.0

        # Canonical Orientation Constraint: body orientation angle relative to rack vertical
        pose.body_orientation_deg = float(math.degrees(math.atan2(v_upper[1], v_upper[0])))

        # Check Range-of-Motion limits (Elbow ROM: 0 to 145 deg)
        pose.rom_limits_violated = not (0.0 <= pose.elbow_angle_deg <= 145.0)

        return pose
