"""
BAS Autonomous HAR System - Core Data Types & Data Contracts
Defines standardized data classes, enums, and structures shared across all 8 agents.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
import time


class EntityState(str, Enum):
    DOCKED = "DOCKED"
    APPROACHED = "APPROACHED"
    GRASPED = "GRASPED"
    EXTRACTED = "EXTRACTED"
    RELEASED = "RELEASED"


class HOIAction(str, Enum):
    IDLE = "IDLE"
    APPROACH = "APPROACH"
    CONTACT = "CONTACT"
    GRASP = "GRASP"
    EXTRACT = "EXTRACT"
    RELEASE = "RELEASE"


class FSMStep(int, Enum):
    IDLE = 0
    BOX_OPENED = 1
    OBJECT_EXTRACTED = 2
    OBJECT_RETURNED = 3
    COMPLETE = 4

    # Aliases
    CONTAINER_OPEN = 1
    RED_EXTRACTED = 2


class AnomalyType(str, Enum):
    NONE = "NONE"
    ERROR_SEQ = "ERROR_SEQ"
    ERROR_SKIP = "ERROR_SKIP"
    STALL_TIMEOUT = "STALL_TIMEOUT"


@dataclass
class Vector3D:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z]

    def distance_to(self, other: 'Vector3D') -> float:
        return ((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)**0.5


@dataclass
class Quaternion:
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class BBox2D:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def centroid(self) -> Tuple[float, float]:
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    def iou(self, other: 'BBox2D') -> float:
        ix1 = max(self.xmin, other.xmin)
        iy1 = max(self.ymin, other.ymin)
        ix2 = min(self.xmax, other.xmax)
        iy2 = min(self.ymax, other.ymax)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = (self.width * self.height) + (other.width * other.height) - inter
        return inter / union if union > 0 else 0.0


@dataclass
class Joint3D:
    name: str
    pos_camera: Vector3D = field(default_factory=Vector3D)
    pos_rack: Vector3D = field(default_factory=Vector3D)
    confidence: float = 0.0


@dataclass
class AstronautPose3D:
    joints: Dict[str, Joint3D] = field(default_factory=dict)
    elbow_angle_deg: float = 0.0
    shoulder_angle_deg: float = 0.0
    body_orientation_deg: float = 0.0
    rom_limits_violated: bool = False


@dataclass
class IMUReading:
    node_id: str  # "upper_arm", "forearm", "wrist"
    timestamp: float
    acc: Vector3D
    gyro: Vector3D
    is_stationary: bool = False


@dataclass
class ExperimentObject:
    name: str
    class_name: str
    bbox: Optional[BBox2D] = None
    pos_rack: Vector3D = field(default_factory=Vector3D)
    state: EntityState = EntityState.DOCKED
    is_inside_container: bool = True


@dataclass
class HOIInteraction:
    object_name: str
    hand_name: str
    distance_m: float
    action: HOIAction
    duration_frames: int = 0


@dataclass
class TelemetryEvent:
    timestamp: str
    frame_id: int
    step_id: int
    step_name: str
    event: str
    status: str
    instruction: str
    anomaly: str = "NONE"
    metrics: Dict[str, float] = field(default_factory=dict)
