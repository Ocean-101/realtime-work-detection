"""
BAS Autonomous HAR System - Agent 3: Fusion Agent
Fuses monocular camera 3D pose with 128 Hz IMU telemetry using a Constrained Unscented Kalman Filter (UKF).
Transforms all coordinates into the stationary Payload Rack Frame (R) and enforces human Range of Motion (ROM).
"""

import json
import math
import numpy as np
from typing import Dict, Tuple
from src.core.types import (
    AstronautPose3D,
    ExperimentObject,
    IMUReading,
    Vector3D,
    Joint3D
)


class FusionAgent:
    """Opto-Inertial Fusion Agent executing Constrained UKF in Payload Rack coordinates."""

    def __init__(self, calib_path: str = "configs/camera_calib.json"):
        with open(calib_path, "r") as f:
            calib = json.load(f)

        ext = calib["extrinsics_to_rack_frame"]
        self.R_cam_to_rack = np.array(ext["rotation_matrix"], dtype=np.float32)
        self.T_cam_to_rack = np.array(ext["translation_meters"], dtype=np.float32)

        # UKF State: [x, y, z, vx, vy, vz] for wrist in Rack frame
        self.state_mean = np.zeros(6, dtype=np.float32)
        self.state_cov = np.eye(6, dtype=np.float32) * 0.05

        # Process and Measurement Noise Covariances
        self.Q = np.eye(6, dtype=np.float32) * 0.01
        self.R_meas = np.eye(3, dtype=np.float32) * 0.02

        # Biomechanical limits
        self.elbow_min_deg = 0.0
        self.elbow_max_deg = 145.0

    def reset(self):
        """Resets UKF state and covariance matrices."""
        self.state_mean = np.zeros(6, dtype=np.float32)
        self.state_cov = np.eye(6, dtype=np.float32) * 0.05

    def transform_camera_to_rack(self, p_cam: Vector3D) -> Vector3D:
        """Applies homogeneous transformation: X_R = R * X_C + T."""
        v_cam = np.array([p_cam.x, p_cam.y, p_cam.z], dtype=np.float32)
        v_rack = np.dot(self.R_cam_to_rack, v_cam) + self.T_cam_to_rack
        return Vector3D(float(v_rack[0]), float(v_rack[1]), float(v_rack[2]))

    def fuse_kinematics(
        self,
        pose_cam: AstronautPose3D,
        imu_readings: Dict[str, IMUReading],
        objects: Dict[str, ExperimentObject],
        dt: float = 0.033
    ) -> Tuple[AstronautPose3D, Dict[str, ExperimentObject]]:
        """
        Executes Constrained UKF step, transforms all entities into Rack Frame R,
        and projects kinematics onto anatomical Range-of-Motion bounds.
        """
        fused_pose = AstronautPose3D()
        
        # 1. Transform all object centroids into stationary Rack Frame R
        fused_objects: Dict[str, ExperimentObject] = {}
        for name, obj in objects.items():
            pos_rack = self.transform_camera_to_rack(obj.pos_rack)
            fused_objects[name] = ExperimentObject(
                name=obj.name,
                class_name=obj.class_name,
                bbox=obj.bbox,
                pos_rack=pos_rack,
                state=obj.state,
                is_inside_container=obj.is_inside_container
            )

        # 2. Transform joint positions into Rack Frame
        wrist_joint_cam = pose_cam.joints.get("wrist")
        if wrist_joint_cam:
            wrist_rack = self.transform_camera_to_rack(wrist_joint_cam.pos_camera)
            
            # UKF Predict Step with IMU acceleration
            wrist_imu = imu_readings.get("wrist")
            if wrist_imu and not wrist_imu.is_stationary:
                # Transform acceleration to Rack Frame
                acc_cam = np.array([wrist_imu.acc.x, wrist_imu.acc.y, wrist_imu.acc.z], dtype=np.float32)
                acc_rack = np.dot(self.R_cam_to_rack, acc_cam)
            else:
                acc_rack = np.zeros(3, dtype=np.float32)

            # State transition F
            F = np.eye(6, dtype=np.float32)
            F[0:3, 3:6] = np.eye(3, dtype=np.float32) * dt

            # Predict mean and covariance
            self.state_mean = np.dot(F, self.state_mean)
            self.state_mean[3:6] += acc_rack * dt
            self.state_cov = np.dot(np.dot(F, self.state_cov), F.T) + self.Q

            # ZUPT: If stationary, clamp velocity to zero
            if wrist_imu and wrist_imu.is_stationary:
                self.state_mean[3:6] = 0.0
                self.state_cov[3:6, 3:6] *= 0.1

            # UKF Update Step with camera observation
            z = np.array([wrist_rack.x, wrist_rack.y, wrist_rack.z], dtype=np.float32)
            H = np.zeros((3, 6), dtype=np.float32)
            H[0:3, 0:3] = np.eye(3, dtype=np.float32)

            y = z - np.dot(H, self.state_mean)
            S = np.dot(np.dot(H, self.state_cov), H.T) + self.R_meas
            K = np.dot(np.dot(self.state_cov, H.T), np.linalg.inv(S))

            self.state_mean = self.state_mean + np.dot(K, y)
            self.state_cov = np.dot(np.eye(6) - np.dot(K, H), self.state_cov)

            # Store fused wrist position
            fused_wrist = Vector3D(
                float(self.state_mean[0]),
                float(self.state_mean[1]),
                float(self.state_mean[2])
            )
            fused_pose.joints["wrist"] = Joint3D(
                name="wrist",
                pos_camera=wrist_joint_cam.pos_camera,
                pos_rack=fused_wrist,
                confidence=wrist_joint_cam.confidence
            )

        # 3. Transform other joints (elbow, shoulder)
        for j_name in ("elbow", "shoulder"):
            if j_name in pose_cam.joints:
                j_cam = pose_cam.joints[j_name]
                j_rack = self.transform_camera_to_rack(j_cam.pos_camera)
                fused_pose.joints[j_name] = Joint3D(
                    name=j_name,
                    pos_camera=j_cam.pos_camera,
                    pos_rack=j_rack,
                    confidence=j_cam.confidence
                )

        # 4. Biomechanical ROM Projection (Elbow 0 to 145 degrees)
        fused_pose.elbow_angle_deg = pose_cam.elbow_angle_deg
        fused_pose.shoulder_angle_deg = pose_cam.shoulder_angle_deg
        fused_pose.body_orientation_deg = pose_cam.body_orientation_deg

        if fused_pose.elbow_angle_deg < self.elbow_min_deg:
            fused_pose.elbow_angle_deg = self.elbow_min_deg
            fused_pose.rom_limits_violated = True
        elif fused_pose.elbow_angle_deg > self.elbow_max_deg:
            fused_pose.elbow_angle_deg = self.elbow_max_deg
            fused_pose.rom_limits_violated = True
        else:
            fused_pose.rom_limits_violated = False

        # Forward 2D skeleton keypoints for visualization and rendering
        fused_pose.keypoints_2d = pose_cam.keypoints_2d

        return fused_pose, fused_objects
