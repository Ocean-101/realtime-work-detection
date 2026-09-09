"""
BAS Autonomous HAR System - Shared State (Digital Twin Memory Blackboard)
Provides a thread-safe central memory hub for asynchronous agent collaboration.
"""

import threading
import time
from typing import Dict, List, Optional, Any
from src.core.types import (
    AstronautPose3D,
    ExperimentObject,
    HOIInteraction,
    HOIAction,
    FSMStep,
    AnomalyType,
    IMUReading,
    TelemetryEvent,
    EntityState,
    Vector3D
)


class DigitalTwinBlackboard:
    """Thread-safe Shared Memory Blackboard bridging all 8 specialized agents."""

    def __init__(self):
        self._lock = threading.RLock()
        self.frame_id: int = 0
        self.timestamp: float = time.time()
        
        # Mode A (Vision Only) vs Mode B (Opto-Inertial Fused)
        self.operating_mode: str = "MODE_A_VISION"

        # Entity States in Payload Rack Frame
        self.objects: Dict[str, ExperimentObject] = {
            "container_box": ExperimentObject(name="container_box", class_name="container_box"),
            "container_lid": ExperimentObject(name="container_lid", class_name="container_lid"),
            "red_box": ExperimentObject(name="red_box", class_name="red_box"),
            "yellow_box": ExperimentObject(name="yellow_box", class_name="yellow_box")
        }
        self.lid_angle_deg: float = 0.0

        # Kinematic State
        self.astronaut_pose: AstronautPose3D = AstronautPose3D()
        self.imu_telemetry: Dict[str, IMUReading] = {}

        # Semantic Interaction State
        self.active_hoi: List[HOIInteraction] = []
        self.current_activity: str = "IDLE"

        # Deterministic Validation State
        self.fsm_step: FSMStep = FSMStep.IDLE
        self.debounce_counter: int = 0
        self.anomaly_status: AnomalyType = AnomalyType.NONE
        self.anomaly_message: str = ""

        # Guidance & Speech
        self.active_instruction: str = "System initialized. Please open the primary container box."
        self._voice_queue: List[str] = []

        # Diagnostics & Telemetry
        self.fps: float = 0.0
        self.inference_latency_ms: float = 0.0
        self.telemetry_history: List[Dict[str, Any]] = []

    def set_mode(self, mode: str):
        with self._lock:
            self.operating_mode = mode

    def update_frame_metadata(self, frame_id: int, fps: float, latency_ms: float):
        with self._lock:
            self.frame_id = frame_id
            self.timestamp = time.time()
            self.fps = fps
            self.inference_latency_ms = latency_ms

    def update_objects(self, objects: Dict[str, ExperimentObject], lid_angle: float):
        with self._lock:
            self.objects.update(objects)
            self.lid_angle_deg = lid_angle

    def update_pose(self, pose: AstronautPose3D):
        with self._lock:
            self.astronaut_pose = pose

    def update_imu(self, node_id: str, reading: IMUReading):
        with self._lock:
            self.imu_telemetry[node_id] = reading

    def update_hoi(self, hoi_list: List[HOIInteraction], activity: str):
        with self._lock:
            self.active_hoi = hoi_list
            self.current_activity = activity

    def update_fsm_state(self, step: FSMStep, debounce: int, anomaly: AnomalyType, 
                         anomaly_msg: str, instruction: str):
        with self._lock:
            self.fsm_step = step
            self.debounce_counter = debounce
            self.anomaly_status = anomaly
            self.anomaly_message = anomaly_msg
            self.active_instruction = instruction

    def queue_voice_alert(self, text: str):
        with self._lock:
            self._voice_queue.append(text)

    def pop_voice_alert(self) -> Optional[str]:
        with self._lock:
            if self._voice_queue:
                return self._voice_queue.pop(0)
            return None

    def record_telemetry_event(self, event: Dict[str, Any]):
        with self._lock:
            self.telemetry_history.append(event)
            if len(self.telemetry_history) > 200:
                self.telemetry_history.pop(0)

    def get_snapshot(self) -> Dict[str, Any]:
        """Returns a complete JSON-serializable snapshot of the Digital Twin state."""
        with self._lock:
            objects_dict = {}
            for name, obj in self.objects.items():
                objects_dict[name] = {
                    "name": obj.name,
                    "state": obj.state.value if isinstance(obj.state, EntityState) else str(obj.state),
                    "pos_rack": obj.pos_rack.to_list(),
                    "is_inside": obj.is_inside_container,
                    "bbox": [obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax] if obj.bbox else None
                }

            joints_dict = {}
            for j_name, joint in self.astronaut_pose.joints.items():
                joints_dict[j_name] = {
                    "rack": joint.pos_rack.to_list(),
                    "camera": joint.pos_camera.to_list(),
                    "confidence": joint.confidence
                }

            hoi_list = []
            for h in self.active_hoi:
                hoi_list.append({
                    "object": h.object_name,
                    "hand": h.hand_name,
                    "distance_m": round(h.distance_m, 3),
                    "action": h.action.value if isinstance(h.action, HOIAction) else str(h.action)
                })

            return {
                "timestamp": self.timestamp,
                "frame_id": self.frame_id,
                "mode": self.operating_mode,
                "fps": round(self.fps, 1),
                "latency_ms": round(self.inference_latency_ms, 1),
                "fsm_step": int(self.fsm_step),
                "fsm_step_name": self.fsm_step.name,
                "debounce_counter": self.debounce_counter,
                "anomaly_status": self.anomaly_status.value,
                "anomaly_message": self.anomaly_message,
                "active_instruction": self.active_instruction,
                "lid_angle_deg": round(self.lid_angle_deg, 1),
                "current_activity": self.current_activity,
                "objects": objects_dict,
                "joints": joints_dict,
                "hoi": hoi_list,
                "elbow_angle_deg": round(self.astronaut_pose.elbow_angle_deg, 1),
                "shoulder_angle_deg": round(self.astronaut_pose.shoulder_angle_deg, 1),
                "rom_limits_violated": self.astronaut_pose.rom_limits_violated
            }
