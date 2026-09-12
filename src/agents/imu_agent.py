"""
BAS Autonomous HAR System - Agent 2: IMU Agent
Ingests 128 Hz wearable IMU streams (Upper arm, Forearm, Wrist), performs ZUPT drift mitigation,
and provides virtual kinematic IMU synthesis in Mode A (Vision-Only).
"""

import time
import math
import numpy as np
from typing import Dict, Optional, Tuple
from src.core.types import IMUReading, Vector3D, AstronautPose3D


class IMUAgent:
    """Manages 3-node opto-inertial wearable IMU array @ 128 Hz."""

    def __init__(self, sample_rate_hz: int = 128):
        self.sample_rate_hz = sample_rate_hz
        self.dt = 1.0 / sample_rate_hz
        
        # ZUPT Thresholds
        self.gyro_stationary_threshold = 0.08 # rad/s
        self.acc_stationary_threshold = 0.15  # m/s^2

        # State storage
        self.last_readings: Dict[str, IMUReading] = {}
        self._prev_wrist_pos: Optional[Vector3D] = None
        self._prev_wrist_vel: Vector3D = Vector3D()
        self._prev_timestamp = time.time()

    def reset(self):
        """Resets virtual IMU kinematic states."""
        self.last_readings.clear()
        self._prev_wrist_pos = None
        self._prev_wrist_vel = Vector3D()
        self._prev_timestamp = time.time()

    def update_from_vision(self, pose: AstronautPose3D) -> Dict[str, IMUReading]:
        """
        Derives 128 Hz virtual IMU telemetry from 3D visual kinematics.
        Ensures Mode A operates seamlessly without requiring physical wearable hardware.
        """
        now = time.time()
        dt = max(1e-4, now - self._prev_timestamp)
        self._prev_timestamp = now

        wrist_joint = pose.joints.get("wrist")
        if wrist_joint:
            cur_pos = wrist_joint.pos_camera
            if self._prev_wrist_pos:
                # Calculate instantaneous velocity & acceleration
                vx = (cur_pos.x - self._prev_wrist_pos.x) / dt
                vy = (cur_pos.y - self._prev_wrist_pos.y) / dt
                vz = (cur_pos.z - self._prev_wrist_pos.z) / dt

                ax = (vx - self._prev_wrist_vel.x) / dt
                ay = (vy - self._prev_wrist_vel.y) / dt
                az = (vz - self._prev_wrist_vel.z) / dt

                self._prev_wrist_vel = Vector3D(vx, vy, vz)
            else:
                ax, ay, az = 0.0, 0.0, 0.0
            self._prev_wrist_pos = cur_pos
        else:
            ax, ay, az = 0.0, 0.0, 0.0

        # Add realistic microgravity sensor noise
        noise_a = np.random.normal(0, 0.02, 3)
        noise_g = np.random.normal(0, 0.01, 3)

        # 1. Wrist Node (highest dynamic range)
        acc_wrist = Vector3D(ax + noise_a[0], ay + noise_a[1], az + noise_a[2])
        gyro_wrist = Vector3D(noise_g[0], noise_g[1], noise_g[2])
        is_stat_wrist = self._check_zupt(acc_wrist, gyro_wrist)
        self.last_readings["wrist"] = IMUReading(
            node_id="wrist",
            timestamp=now,
            acc=acc_wrist,
            gyro=gyro_wrist,
            is_stationary=is_stat_wrist
        )

        # 2. Forearm Node (scaled motion)
        acc_fore = Vector3D(ax*0.75 + noise_a[0]*0.8, ay*0.75 + noise_a[1]*0.8, az*0.75 + noise_a[2]*0.8)
        gyro_fore = Vector3D(noise_g[0]*0.8, noise_g[1]*0.8, noise_g[2]*0.8)
        self.last_readings["forearm"] = IMUReading(
            node_id="forearm",
            timestamp=now,
            acc=acc_fore,
            gyro=gyro_fore,
            is_stationary=self._check_zupt(acc_fore, gyro_fore)
        )

        # 3. Upper Arm Node (base pivot)
        acc_upper = Vector3D(ax*0.4 + noise_a[0]*0.5, ay*0.4 + noise_a[1]*0.5, az*0.4 + noise_a[2]*0.5)
        gyro_upper = Vector3D(noise_g[0]*0.5, noise_g[1]*0.5, noise_g[2]*0.5)
        self.last_readings["upper_arm"] = IMUReading(
            node_id="upper_arm",
            timestamp=now,
            acc=acc_upper,
            gyro=gyro_upper,
            is_stationary=self._check_zupt(acc_upper, gyro_upper)
        )

        return self.last_readings

    def _check_zupt(self, acc: Vector3D, gyro: Vector3D) -> bool:
        """Evaluates Zero-Velocity Update (ZUPT) condition for drift resetting."""
        gyro_norm = math.sqrt(gyro.x**2 + gyro.y**2 + gyro.z**2)
        acc_norm = math.sqrt(acc.x**2 + acc.y**2 + acc.z**2)
        return gyro_norm < self.gyro_stationary_threshold and acc_norm < self.acc_stationary_threshold
