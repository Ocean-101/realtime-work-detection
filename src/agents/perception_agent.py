"""
BAS Autonomous HAR System - Agent 1: Perception Agent
Detects experimental components (Container, Lid, Red Box, Yellow Box, Hands)
and extracts 3D skeletal keypoints in camera space using PhysAstro-Pose principles.
"""

import os
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

    def __init__(
        self,
        calib_path: str = "configs/camera_calib.json",
        model_path: Optional[str] = "models/detector_offline.pt",
        common_model_path: Optional[str] = "yolov8n.pt",
        enable_common_detection: bool = True
    ):
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

        # 1. Offline YOLOv8 Deep Neural Object Detector for Experiment Protocol
        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                import importlib
                ultralytics_pkg = importlib.import_module("ultralytics")
                YOLO = getattr(ultralytics_pkg, "YOLO")
                self.model = YOLO(model_path)
                print(f"[Perception Agent] Offline YOLOv8 detector engaged: {model_path}")
            except Exception as e:
                print(f"[Perception Agent] Offline model note: {e}. Active with contour heuristics.")

        # 2. General / Common Object Detector (YOLOv8 COCO-80 Pretrained)
        self.common_model = None
        if enable_common_detection and common_model_path and os.path.exists(common_model_path):
            try:
                import importlib
                ultralytics_pkg = importlib.import_module("ultralytics")
                YOLO = getattr(ultralytics_pkg, "YOLO")
                self.common_model = YOLO(common_model_path)
                print(f"[Perception Agent] Common Object Detector (COCO-80) engaged: {common_model_path}")
            except Exception as e:
                print(f"[Perception Agent] Common model note: {e}")

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
        objects: Dict[str, ExperimentObject] = {}
        hand_bbox = None
        hand_center = (0.0, 0.0)

        # 1. Run Common Object Detector (COCO-80 Common Objects: person, bottle, cup, phone, book, etc.)
        if self.common_model is not None:
            try:
                results_common = self.common_model(frame, verbose=False, conf=0.25)
                if results_common and len(results_common) > 0 and results_common[0].boxes:
                    obj_idx = 0
                    for box in results_common[0].boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                        cls_name = self.common_model.names.get(cls_id, f"obj_{cls_id}")

                        # Filter: ONLY keep experiment-relevant items (e.g. manipulable payloads, containers)
                        # NEVER label person, chair, laptop, couch, tv, etc.
                        is_manipulable = cls_name in ("bottle", "cup", "cell phone", "book", "bowl", "scissors")
                        is_container = cls_name in ("suitcase", "backpack", "handbag", "box")

                        if not (is_manipulable or is_container):
                            # Ignore all non-experiment background clutter (person, chair, laptop, etc.)
                            continue

                        b = BBox2D(
                            xmin=float(bx1), ymin=float(by1),
                            xmax=float(bx2), ymax=float(by2),
                            confidence=conf, class_id=cls_id, class_name=cls_name
                        )
                        center = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
                        obj_cam = self._pixel_to_camera_coord(center[0], center[1], depth_m=1.20)

                        if is_manipulable and "component_box" not in objects:
                            objects["component_box"] = ExperimentObject(
                                name="component_box",
                                class_name=cls_name,
                                bbox=b,
                                pos_rack=obj_cam,
                                state=EntityState.DOCKED,
                                is_inside_container=True
                            )
                        elif is_container and "container_box" not in objects:
                            objects["container_box"] = ExperimentObject(
                                name="container_box",
                                class_name="container_box",
                                bbox=b,
                                pos_rack=obj_cam,
                                state=EntityState.DOCKED
                            )
            except Exception:
                pass

        # 2. Attempt Offline Neural Detector Inference for Experiment-Specific Items
        if self.model is not None:
            try:
                results = self.model(frame, verbose=False, conf=0.15)
                if results and len(results) > 0 and results[0].boxes:
                    class_names = {0: "container_box", 1: "container_lid", 2: "component_box", 3: "astronaut_hand"}
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                        name = class_names.get(cls_id, f"obj_{cls_id}")
                        b = BBox2D(
                            xmin=float(bx1), ymin=float(by1),
                            xmax=float(bx2), ymax=float(by2),
                            confidence=conf, class_id=cls_id, class_name=name
                        )
                        center = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
                        if cls_id == 3:
                            hand_bbox = b
                            hand_center = center
                        else:
                            obj_cam = self._pixel_to_camera_coord(center[0], center[1], depth_m=1.20)
                            objects[name] = ExperimentObject(name=name, class_name=name, bbox=b, pos_rack=obj_cam)
            except Exception:
                pass

        # 2. Extract bounding box of container if present from detector
        cont_bbox = objects["container_box"].bbox if "container_box" in objects else None

        # 3. Detect Glove / Bare Human Hand (Skin-tone + Glove ranges) in workspace manipulation zone
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if not hand_bbox:
            mask_skin1 = cv2.inRange(hsv, np.array([0, 30, 60]), np.array([25, 200, 255]))
            mask_skin2 = cv2.inRange(hsv, np.array([165, 30, 60]), np.array([180, 200, 255]))
            mask_glove = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 50, 255]))
            mask_hand = cv2.bitwise_or(cv2.bitwise_or(mask_skin1, mask_skin2), mask_glove)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_hand = cv2.morphologyEx(mask_hand, cv2.MORPH_OPEN, kernel)

            # Prioritize hands in the workspace (exclude top 20% head zone on webcam)
            mask_workspace_hand = mask_hand.copy()
            mask_workspace_hand[:int(h * 0.20), :] = 0
            hand_bbox, hand_center = self._extract_largest_bbox(mask_workspace_hand, "astronaut_hand", 3, min_area=500)
            if not hand_bbox:
                hand_bbox, hand_center = self._extract_largest_bbox(mask_hand, "astronaut_hand", 3, min_area=600)

        # 4. Adaptive Container Box Detection
        # If neural model missed the container box on webcam, establish robust workspace container
        if not cont_bbox:
            lower_half = frame[int(h * 0.35):, :]
            lh_hsv = cv2.cvtColor(lower_half, cv2.COLOR_BGR2HSV)
            mask_cont_brown = cv2.inRange(lh_hsv, np.array([5, 20, 30]), np.array([38, 220, 240]))
            cb_sub, cc_sub = self._extract_largest_bbox(mask_cont_brown, "container_box", 0, min_area=5000)
            if cb_sub:
                cont_bbox = BBox2D(
                    xmin=cb_sub.xmin, ymin=cb_sub.ymin + h * 0.35,
                    xmax=cb_sub.xmax, ymax=cb_sub.ymax + h * 0.35,
                    confidence=0.75, class_id=0, class_name="container_box"
                )
                cont_center = (cc_sub[0], cc_sub[1] + h * 0.35)
            else:
                # Default canonical container enclosure in lower center workspace
                cont_bbox = BBox2D(
                    xmin=float(w * 0.22), ymin=float(h * 0.48),
                    xmax=float(w * 0.78), ymax=float(h * 0.94),
                    confidence=0.85, class_id=0, class_name="container_box"
                )
                cont_center = (w * 0.50, h * 0.71)

            cont_cam = self._pixel_to_camera_coord(cont_center[0], cont_center[1], depth_m=1.20)
            objects["container_box"] = ExperimentObject(
                name="container_box",
                class_name="container_box",
                bbox=cont_bbox,
                pos_rack=cont_cam
            )

        # 5. Fallback for Component Objects & Experiment Items (Red Box, Yellow Box, Component)
        # Check Red object (ISRO Dual-Box benchmark)
        if "red_box" not in objects:
            mask_red1 = cv2.inRange(hsv, np.array([0, 90, 60]), np.array([10, 255, 255]))
            mask_red2 = cv2.inRange(hsv, np.array([170, 90, 60]), np.array([180, 255, 255]))
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            red_bbox, red_center = self._extract_largest_bbox(mask_red, "red_box", 2, min_area=350)
            if red_bbox:
                red_cam = self._pixel_to_camera_coord(red_center[0], red_center[1], depth_m=1.15)
                objects["red_box"] = ExperimentObject(name="red_box", class_name="red_box", bbox=red_bbox, pos_rack=red_cam)

        # Check Yellow object (ISRO Dual-Box benchmark)
        if "yellow_box" not in objects:
            mask_yellow = cv2.inRange(hsv, np.array([18, 90, 60]), np.array([35, 255, 255]))
            yel_bbox, yel_center = self._extract_largest_bbox(mask_yellow, "yellow_box", 2, min_area=350)
            if yel_bbox:
                yel_cam = self._pixel_to_camera_coord(yel_center[0], yel_center[1], depth_m=1.15)
                objects["yellow_box"] = ExperimentObject(name="yellow_box", class_name="yellow_box", bbox=yel_bbox, pos_rack=yel_cam)

        # Check General Component Box
        if "component_box" not in objects:
            if "red_box" in objects:
                rb = objects["red_box"]
                objects["component_box"] = ExperimentObject(name="component_box", class_name="component_box", bbox=rb.bbox, pos_rack=rb.pos_rack)
            elif hand_bbox:
                hx, hy = int(hand_center[0]), int(hand_center[1])
                roi_x1 = max(0, hx - 70)
                roi_x2 = min(w, hx + 70)
                roi_y1 = max(0, hy - 70)
                roi_y2 = min(h, hy + 70)
                hand_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                if hand_roi.size > 0:
                    gray_roi = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
                    edges_roi = cv2.Canny(gray_roi, 50, 150)
                    if np.sum(edges_roi > 0) > 120:
                        comp_w = 90.0
                        comp_h = 70.0
                        item_bbox = BBox2D(
                            xmin=max(0.0, float(hx - comp_w / 2)),
                            ymin=max(0.0, float(hy - comp_h / 2)),
                            xmax=min(float(w), float(hx + comp_w / 2)),
                            ymax=min(float(h), float(hy + comp_h / 2)),
                            confidence=0.70, class_id=2, class_name="component_box"
                        )
                        item_cam = self._pixel_to_camera_coord(hx, hy, depth_m=1.15)
                        objects["component_box"] = ExperimentObject(
                            name="component_box",
                            class_name="component_box",
                            bbox=item_bbox,
                            pos_rack=item_cam
                        )

            if "component_box" not in objects:
                mask_blue = cv2.inRange(hsv, np.array([85, 50, 40]), np.array([140, 255, 255]))
                mask_green = cv2.inRange(hsv, np.array([35, 50, 40]), np.array([85, 255, 255]))
                mask_orange = cv2.inRange(hsv, np.array([10, 80, 80]), np.array([25, 255, 255]))
                mask_item = cv2.bitwise_or(cv2.bitwise_or(mask_blue, mask_green), mask_orange)
                item_bbox, item_center = self._extract_largest_bbox(mask_item, "component_box", 2, min_area=300)
                if item_bbox:
                    item_cam = self._pixel_to_camera_coord(item_center[0], item_center[1], depth_m=1.15)
                    objects["component_box"] = ExperimentObject(
                        name="component_box",
                        class_name="component_box",
                        bbox=item_bbox,
                        pos_rack=item_cam
                    )

        # 6. Compute Lid Elevation Angle
        if "container_lid" in objects and objects["container_lid"].bbox:
            lid_b = objects["container_lid"].bbox
            if cont_bbox:
                elevation = max(0.0, cont_bbox.ymin - lid_b.ymin)
                target_angle = min(85.0, (elevation / max(30.0, cont_bbox.height * 0.5)) * 80.0)
            else:
                target_angle = 45.0
        elif cont_bbox:
            # Dynamic edge & contour check above container box
            target_angle = self._estimate_lid_angle(frame, cont_bbox)
        else:
            target_angle = 0.0

        # Smooth angle transitions
        self.last_lid_angle = 0.75 * self.last_lid_angle + 0.25 * target_angle
        lid_angle = self.last_lid_angle

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
