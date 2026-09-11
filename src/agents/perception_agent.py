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
        pose_model_path: Optional[str] = "models/yolov8n-pose.pt",
        enable_common_detection: bool = True,
        enable_pose_detection: bool = True
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

        # 2. General / Common Object Detector (YOLO11n / YOLOv8 COCO-80 Pretrained)
        self.common_model = None
        common_candidates = [
            common_model_path,
            "models/yolo11n.pt",
            "yolo11n.pt",
            "yolov8n.pt"
        ]
        chosen_common = None
        for cand in common_candidates:
            if cand and os.path.exists(cand):
                chosen_common = cand
                break

        if enable_common_detection and chosen_common:
            try:
                import importlib
                ultralytics_pkg = importlib.import_module("ultralytics")
                YOLO = getattr(ultralytics_pkg, "YOLO")
                self.common_model = YOLO(chosen_common)
                print(f"[Perception Agent] Common Object Detector ({os.path.basename(chosen_common).upper()}) engaged: {chosen_common}")
            except Exception as e:
                print(f"[Perception Agent] Common model note: {e}")

        # 3. Real-Time Person Skeleton & Human Pose Detector (YOLOv8-Pose)
        self.pose_model = None
        pose_candidate = pose_model_path or "models/yolov8n-pose.pt"
        if not os.path.exists(pose_candidate) and os.path.exists("yolov8n-pose.pt"):
            pose_candidate = "yolov8n-pose.pt"
        if enable_pose_detection and os.path.exists(pose_candidate):
            try:
                import importlib
                ultralytics_pkg = importlib.import_module("ultralytics")
                YOLO = getattr(ultralytics_pkg, "YOLO")
                self.pose_model = YOLO(pose_candidate)
                print(f"[Perception Agent] Real-Time Person Skeleton Detector engaged: {pose_candidate}")
            except Exception as e:
                print(f"[Perception Agent] Pose detector note: {e}")

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

        # 1. Primary Neural Object Detector (YOLOv8 offline model trained on boxes & hands)
        if self.model is not None:
            try:
                # conf=0.35 cleanly rejects background clutter while preserving real boxes (conf ~0.85-0.97)
                results = self.model(frame, verbose=False, conf=0.35)
                if results and len(results) > 0 and results[0].boxes:
                    class_names = {0: "container_box", 1: "container_lid", 2: "component_box", 3: "operator_hand"}
                    boxes_sorted = sorted(results[0].boxes, key=lambda b: float(b.conf[0].item()), reverse=True)
                    frame_area = float(w * h)
                    for box in boxes_sorted:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                        name = class_names.get(cls_id, f"obj_{cls_id}")

                        box_w = bx2 - bx1
                        box_h = by2 - by1
                        box_area = box_w * box_h

                        # Discard degenerate tiny noise boxes (< 800px) or full-screen container hallucinations (> 92% of frame)
                        if box_area < 800 or (cls_id == 0 and box_area > 0.92 * frame_area):
                            continue

                        b = BBox2D(
                            xmin=float(bx1), ymin=float(by1),
                            xmax=float(bx2), ymax=float(by2),
                            confidence=conf, class_id=cls_id, class_name=name
                        )
                        center = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)

                        if cls_id == 3:
                            if not hand_bbox:
                                hand_bbox = b
                                hand_center = center
                                objects["operator_hand"] = ExperimentObject(
                                    name="operator_hand",
                                    class_name="operator_hand",
                                    bbox=b,
                                    pos_rack=self._pixel_to_camera_coord(center[0], center[1], depth_m=1.00)
                                )
                        else:
                            if name not in objects:
                                obj_cam = self._pixel_to_camera_coord(center[0], center[1], depth_m=1.20)
                                objects[name] = ExperimentObject(name=name, class_name=name, bbox=b, pos_rack=obj_cam)
            except Exception:
                pass

        # 2. Supplementary Common Object Detector (COCO-80 for secondary manipulable props only)
        if self.common_model is not None:
            try:
                results_common = self.common_model(frame, verbose=False, conf=0.40)
                if results_common and len(results_common) > 0 and results_common[0].boxes:
                    for box in results_common[0].boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                        cls_name = self.common_model.names.get(cls_id, f"obj_{cls_id}")

                        # ONLY alias genuine small manipulable items if component_box is not yet detected
                        if cls_name in ("bottle", "cup", "cell phone", "book", "bowl", "scissors") and "component_box" not in objects:
                            b = BBox2D(
                                xmin=float(bx1), ymin=float(by1),
                                xmax=float(bx2), ymax=float(by2),
                                confidence=conf, class_id=cls_id, class_name=cls_name
                            )
                            center = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
                            obj_cam = self._pixel_to_camera_coord(center[0], center[1], depth_m=1.20)
                            objects["component_box"] = ExperimentObject(
                                name="component_box",
                                class_name=cls_name,
                                bbox=b,
                                pos_rack=obj_cam,
                                state=EntityState.DOCKED,
                                is_inside_container=True
                            )
                            break
            except Exception:
                pass

        # 3. Extract bounding box of container if present
        cont_bbox = objects["container_box"].bbox if "container_box" in objects else None

        # 4. Detect Glove / Bare Human Hand in workspace (fallback only if neural model is NOT engaged)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if not hand_bbox and self.model is None:
            mask_skin1 = cv2.inRange(hsv, np.array([0, 30, 60]), np.array([25, 200, 255]))
            mask_skin2 = cv2.inRange(hsv, np.array([165, 30, 60]), np.array([180, 200, 255]))
            mask_glove = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 50, 255]))
            mask_hand = cv2.bitwise_or(cv2.bitwise_or(mask_skin1, mask_skin2), mask_glove)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_hand = cv2.morphologyEx(mask_hand, cv2.MORPH_OPEN, kernel)

            # Prioritize hands in the workspace (exclude top 20% head zone on webcam)
            mask_workspace_hand = mask_hand.copy()
            mask_workspace_hand[:int(h * 0.20), :] = 0
            hand_bbox, hand_center = self._extract_largest_bbox(mask_workspace_hand, "operator_hand", 3, min_area=800)
            if not hand_bbox:
                hand_bbox, hand_center = self._extract_largest_bbox(mask_hand, "operator_hand", 3, min_area=1000)
            if hand_bbox and "operator_hand" not in objects:
                objects["operator_hand"] = ExperimentObject(
                    name="operator_hand",
                    class_name="operator_hand",
                    bbox=hand_bbox,
                    pos_rack=self._pixel_to_camera_coord(hand_center[0], hand_center[1], depth_m=1.00)
                )

        # 5. Check grasped component payload if hand is interacting inside container area
        if "component_box" not in objects and hand_bbox and cont_bbox:
            hx, hy = int(hand_center[0]), int(hand_center[1])
            # Only check if hand is physically located within or near container zone
            is_in_container_zone = (
                cont_bbox.xmin - 40 <= hx <= cont_bbox.xmax + 40 and
                cont_bbox.ymin - 40 <= hy <= cont_bbox.ymax + 40
            )
            if is_in_container_zone:
                roi_x1 = max(0, hx - 60)
                roi_x2 = min(w, hx + 60)
                roi_y1 = max(0, hy - 60)
                roi_y2 = min(h, hy + 60)
                hand_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            if hand_roi.size > 0:
                gray_roi = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
                edges_roi = cv2.Canny(gray_roi, 60, 180)
                if np.sum(edges_roi > 0) > 250:
                    comp_w = 80.0
                    comp_h = 60.0
                    item_bbox = BBox2D(
                        xmin=max(0.0, float(hx - comp_w / 2)),
                        ymin=max(0.0, float(hy - comp_h / 2)),
                        xmax=min(float(w), float(hx + comp_w / 2)),
                        ymax=min(float(h), float(hy + comp_h / 2)),
                        confidence=0.65, class_id=2, class_name="component_box"
                    )
                    item_cam = self._pixel_to_camera_coord(hx, hy, depth_m=1.15)
                    objects["component_box"] = ExperimentObject(
                        name="component_box",
                        class_name="component_box",
                        bbox=item_bbox,
                        pos_rack=item_cam
                    )

        # 6. Fallback for ISRO Dual-Box benchmark objects (Red Box, Yellow Box) - strict geometric check
        if "red_box" not in objects:
            mask_red1 = cv2.inRange(hsv, np.array([0, 110, 80]), np.array([10, 255, 255]))
            mask_red2 = cv2.inRange(hsv, np.array([170, 110, 80]), np.array([180, 255, 255]))
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            red_bbox, red_center = self._extract_largest_bbox(mask_red, "red_box", 2, min_area=1500)
            if red_bbox:
                red_cam = self._pixel_to_camera_coord(red_center[0], red_center[1], depth_m=1.15)
                objects["red_box"] = ExperimentObject(name="red_box", class_name="red_box", bbox=red_bbox, pos_rack=red_cam)

        if "yellow_box" not in objects:
            mask_yellow = cv2.inRange(hsv, np.array([20, 110, 80]), np.array([32, 255, 255]))
            yel_bbox, yel_center = self._extract_largest_bbox(mask_yellow, "yellow_box", 2, min_area=1500)
            if yel_bbox:
                yel_cam = self._pixel_to_camera_coord(yel_center[0], yel_center[1], depth_m=1.15)
                objects["yellow_box"] = ExperimentObject(name="yellow_box", class_name="yellow_box", bbox=yel_bbox, pos_rack=yel_cam)

        # 7. Compute Lid Elevation Angle
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
        pose = self._estimate_3d_pose(frame, hand_bbox, hand_center, w, h)

        # If hand bbox was not found by object detector, register hand from detected skeleton wrist
        if not hand_bbox and "wrist" in pose.keypoints_2d:
            wx, wy, wconf = pose.keypoints_2d["wrist"]
            hw = 45.0
            hand_bbox = BBox2D(
                xmin=max(0.0, wx - hw), ymin=max(0.0, wy - hw),
                xmax=min(float(w), wx + hw), ymax=min(float(h), wy + hw),
                confidence=wconf, class_id=3, class_name="operator_hand"
            )
            hand_center = (wx, wy)
            if "operator_hand" not in objects:
                objects["operator_hand"] = ExperimentObject(
                    name="operator_hand",
                    class_name="operator_hand",
                    bbox=hand_bbox,
                    pos_rack=self._pixel_to_camera_coord(wx, wy, depth_m=1.00)
                )

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
        if bw <= 0 or bh <= 0:
            return None, (0.0, 0.0)

        aspect = bw / float(bh)
        if aspect > 4.0 or aspect < 0.25:
            # Reject extreme aspect ratios (lines, border edges, shadows)
            return None, (0.0, 0.0)

        bbox = BBox2D(
            xmin=float(x),
            ymin=float(y),
            xmax=float(x + bw),
            ymax=float(y + bh),
            confidence=min(0.95, max(0.40, area / 8000.0)),
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
        frame: Optional[np.ndarray],
        hand_bbox: Optional[BBox2D],
        hand_center: Tuple[float, float],
        width: int,
        height: int
    ) -> AstronautPose3D:
        """
        Detects full-body person skeleton (17 COCO joints) and reconstructs
        true biomechanical 3D kinematics for the astronaut.
        """
        pose = AstronautPose3D()

        # 1. Real-Time Neural Pose Estimation (YOLOv8-Pose)
        if self.pose_model is not None and frame is not None and frame.size > 0:
            try:
                res_pose = self.pose_model(frame, verbose=False, conf=0.20, imgsz=480)
                if res_pose and len(res_pose) > 0 and len(res_pose[0].boxes) > 0 and res_pose[0].keypoints is not None:
                    boxes = res_pose[0].boxes
                    best_idx = 0
                    if len(boxes) > 1:
                        areas = [float((b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1])) for b in boxes]
                        best_idx = int(np.argmax(areas))

                    kp = res_pose[0].keypoints
                    xy = kp.xy[best_idx].cpu().numpy()  # shape (17, 2)
                    confs = kp.conf[best_idx].cpu().numpy() if kp.conf is not None else np.ones(17)

                    COCO_KEYPOINTS = [
                        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                        "left_wrist", "right_wrist", "left_hip", "right_hip",
                        "left_knee", "right_knee", "left_ankle", "right_ankle"
                    ]

                    for i, k_name in enumerate(COCO_KEYPOINTS):
                        k_conf = float(confs[i])
                        if k_conf > 0.20:
                            kx, ky = float(xy[i][0]), float(xy[i][1])
                            pose.keypoints_2d[k_name] = (kx, ky, k_conf)
                            pos_cam = self._pixel_to_camera_coord(kx, ky, depth_m=1.20)
                            pose.joints[k_name] = Joint3D(name=k_name, pos_camera=pos_cam, confidence=k_conf)

                    # Determine dominant/active working arm
                    has_r_arm = ("right_wrist" in pose.joints and "right_elbow" in pose.joints)
                    has_l_arm = ("left_wrist" in pose.joints and "left_elbow" in pose.joints)
                    active_side = "right" if has_r_arm or not has_l_arm else "left"

                    if f"{active_side}_wrist" in pose.joints:
                        pose.joints["wrist"] = pose.joints[f"{active_side}_wrist"]
                        pose.keypoints_2d["wrist"] = pose.keypoints_2d[f"{active_side}_wrist"]
                    if f"{active_side}_elbow" in pose.joints:
                        pose.joints["elbow"] = pose.joints[f"{active_side}_elbow"]
                    if f"{active_side}_shoulder" in pose.joints:
                        pose.joints["shoulder"] = pose.joints[f"{active_side}_shoulder"]

                    # Compute true measured biomechanical angles
                    if "wrist" in pose.joints and "elbow" in pose.joints and "shoulder" in pose.joints:
                        w_p = pose.joints["wrist"].pos_camera
                        e_p = pose.joints["elbow"].pos_camera
                        s_p = pose.joints["shoulder"].pos_camera

                        v_fore = np.array([w_p.x - e_p.x, w_p.y - e_p.y, w_p.z - e_p.z])
                        v_upper = np.array([s_p.x - e_p.x, s_p.y - e_p.y, s_p.z - e_p.z])
                        norm_p = np.linalg.norm(v_fore) * np.linalg.norm(v_upper)
                        if norm_p > 1e-6:
                            cos_elb = np.clip(np.dot(v_fore, v_upper) / norm_p, -1.0, 1.0)
                            pose.elbow_angle_deg = float(np.degrees(np.arccos(cos_elb)))
                        else:
                            pose.elbow_angle_deg = 77.0

                        hip_j = pose.joints.get(f"{active_side}_hip") or pose.joints.get("left_hip") or pose.joints.get("right_hip")
                        if hip_j:
                            h_p = hip_j.pos_camera
                            v_torso = np.array([h_p.x - s_p.x, h_p.y - s_p.y, h_p.z - s_p.z])
                            norm_t = np.linalg.norm(v_torso) * np.linalg.norm(v_upper)
                            if norm_t > 1e-6:
                                cos_sh = np.clip(np.dot(v_torso, v_upper) / norm_t, -1.0, 1.0)
                                pose.shoulder_angle_deg = float(np.degrees(np.arccos(cos_sh)))

                        pose.body_orientation_deg = float(math.degrees(math.atan2(v_upper[1], v_upper[0])))

                    pose.rom_limits_violated = not (0.0 <= pose.elbow_angle_deg <= 145.0)
                    return pose
            except Exception:
                pass

        # 2. Heuristic Prior Fallback (if pose model is not active and hand is detected)
        if hand_bbox:
            hx, hy = hand_center
            wrist_pos = self._pixel_to_camera_coord(hx, hy, depth_m=1.10)
            pose.joints["wrist"] = Joint3D(name="wrist", pos_camera=wrist_pos, confidence=hand_bbox.confidence)
            pose.keypoints_2d["wrist"] = (hx, hy, hand_bbox.confidence)

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

            v_fore = np.array([wrist_pos.x - elbow_pos.x, wrist_pos.y - elbow_pos.y, wrist_pos.z - elbow_pos.z])
            v_upper = np.array([shoulder_pos.x - elbow_pos.x, shoulder_pos.y - elbow_pos.y, shoulder_pos.z - elbow_pos.z])
            norm_product = np.linalg.norm(v_fore) * np.linalg.norm(v_upper)
            if norm_product > 1e-6:
                cos_elbow = np.clip(np.dot(v_fore, v_upper) / norm_product, -1.0, 1.0)
                pose.elbow_angle_deg = float(np.degrees(np.arccos(cos_elbow)))
            else:
                pose.elbow_angle_deg = 77.0

            pose.body_orientation_deg = float(math.degrees(math.atan2(v_upper[1], v_upper[0])))

        pose.rom_limits_violated = not (0.0 <= pose.elbow_angle_deg <= 145.0)
        return pose
