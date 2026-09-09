"""
BAS Autonomous HAR System - Agent 6: Validation Agent (Safety-Critical FSM)
Authoritative, deterministic finite state machine validating procedural compliance
with 15-frame temporal debouncing and strict anomaly detection.
"""

import json
import time
from typing import Dict, Tuple, Optional
from src.core.types import (
    FSMStep,
    AnomalyType,
    ExperimentObject,
    EntityState,
    HOIInteraction,
    HOIAction
)


class ValidationAgent:
    """Deterministic Sequence Validator and Safety Gatekeeper."""

    def __init__(self, config_path: str = "configs/experiment_fsm.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.debounce_required: int = self.config.get("debounce_frames", 15)
        self.current_step: FSMStep = FSMStep.IDLE
        self.debounce_counter: int = 0
        self.candidate_step: Optional[FSMStep] = None

        self.anomaly_status: AnomalyType = AnomalyType.NONE
        self.anomaly_message: str = ""
        self.step_start_time: float = time.time()
        self.stall_timeout_sec: float = 60.0

    def reset(self):
        """Resets the state machine back to IDLE (Step 0) for a fresh real-time test run."""
        self.current_step = FSMStep.IDLE
        self.debounce_counter = 0
        self.candidate_step = None
        self.anomaly_status = AnomalyType.NONE
        self.anomaly_message = ""
        self.step_start_time = time.time()

    def evaluate_step(
        self,
        objects: Dict[str, ExperimentObject],
        lid_angle: float,
        active_hoi: list,
        current_frame: int
    ) -> Tuple[FSMStep, int, AnomalyType, str, Optional[str]]:
        """
        Evaluates current physical state against procedural state machine.
        Returns:
            - current_step: committed FSMStep
            - debounce_count: frames accumulated towards candidate transition
            - anomaly: AnomalyType
            - anomaly_msg: diagnostic string
            - transition_event: string name if a new state was committed on this frame
        """
        now = time.time()
        transition_committed: Optional[str] = None

        # Reset non-persistent anomalies if conditions clear
        if self.anomaly_status == AnomalyType.ERROR_SEQ:
            # Clear if yellow box is released
            yellow_obj = objects.get("yellow_box")
            if yellow_obj and yellow_obj.state in (EntityState.DOCKED, EntityState.APPROACHED):
                self.anomaly_status = AnomalyType.NONE
                self.anomaly_message = ""

        # Check Stall Timeout
        if self.current_step not in (FSMStep.IDLE, FSMStep.COMPLETE):
            if (now - self.step_start_time) > self.stall_timeout_sec:
                self.anomaly_status = AnomalyType.STALL_TIMEOUT
                self.anomaly_message = "Activity paused. Awaiting required procedural action."

        # Detect experiment mode
        is_box_return_mode = "BOX-RETURN" in self.config.get("experiment_id", "")

        if is_box_return_mode:
            # -----------------------------------------------------------------
            # 4-STEP EXPERIMENT: OPEN -> EXTRACT -> RETURN -> CLOSE
            # -----------------------------------------------------------------
            # Step 0: IDLE -> Awaiting Box Open
            if self.current_step == FSMStep.IDLE:
                if lid_angle >= 28.0:
                    self._accumulate_debounce(FSMStep.BOX_OPENED)
                    if self.debounce_counter >= self.debounce_required:
                        self.current_step = FSMStep.BOX_OPENED
                        self.candidate_step = None
                        self.debounce_counter = 0
                        self.step_start_time = now
                        transition_committed = "BOX_OPENED"
                else:
                    self._reset_debounce()

            # Step 1: BOX_OPENED -> Awaiting Object Extraction
            elif self.current_step == FSMStep.BOX_OPENED:
                # Check premature close anomaly
                if lid_angle < 15.0:
                    self.anomaly_status = AnomalyType.ERROR_SKIP
                    self.anomaly_message = "Warning: Step skipped. Please extract the object before closing the box."
                    self._reset_debounce(soft=False)
                    return self.current_step, 0, self.anomaly_status, self.anomaly_message, None

                # Check if any manipulable object is extracted outside the box
                extracted = False
                for name, obj in objects.items():
                    if name not in ("container_box", "container_lid") and (not obj.is_inside_container or obj.state == EntityState.EXTRACTED):
                        extracted = True
                        break

                if extracted:
                    self._accumulate_debounce(FSMStep.OBJECT_EXTRACTED)
                    if self.debounce_counter >= self.debounce_required:
                        self.current_step = FSMStep.OBJECT_EXTRACTED
                        self.candidate_step = None
                        self.debounce_counter = 0
                        self.step_start_time = now
                        transition_committed = "OBJECT_EXTRACTED"
                else:
                    self._reset_debounce()

            # Step 2: OBJECT_EXTRACTED -> Awaiting Object Return into Box
            elif self.current_step == FSMStep.OBJECT_EXTRACTED:
                # Check if object has returned back inside container
                all_inside = True
                found_target = False
                for name, obj in objects.items():
                    if name not in ("container_box", "container_lid"):
                        found_target = True
                        if not obj.is_inside_container:
                            all_inside = False
                            break

                if (found_target and all_inside and lid_angle >= 25.0) or (all_inside and lid_angle >= 35.0):
                    self._accumulate_debounce(FSMStep.OBJECT_RETURNED)
                    if self.debounce_counter >= self.debounce_required:
                        self.current_step = FSMStep.OBJECT_RETURNED
                        self.candidate_step = None
                        self.debounce_counter = 0
                        self.step_start_time = now
                        transition_committed = "OBJECT_RETURNED"
                else:
                    self._reset_debounce()

            # Step 3: OBJECT_RETURNED -> Awaiting Box Close
            elif self.current_step == FSMStep.OBJECT_RETURNED:
                lid_docked = ("container_lid" not in objects) or (lid_angle < 28.0)
                if lid_docked:
                    self._accumulate_debounce(FSMStep.COMPLETE)
                    if self.debounce_counter >= self.debounce_required:
                        self.current_step = FSMStep.COMPLETE
                        self.candidate_step = None
                        self.debounce_counter = self.debounce_required
                        self.step_start_time = now
                        transition_committed = "BOX_CLOSED"
                else:
                    self._reset_debounce()

            # Step 4: COMPLETE
            elif self.current_step == FSMStep.COMPLETE:
                self.debounce_counter = self.debounce_required

        else:
            # -----------------------------------------------------------------
            # 2-BOX EXTRACTION EXPERIMENT (RED THEN YELLOW)
            # -----------------------------------------------------------------
            # State 0: IDLE -> Awaiting Container Lid Open
            if self.current_step == FSMStep.IDLE:
                if lid_angle >= 40.0:
                    self._accumulate_debounce(FSMStep.CONTAINER_OPEN)
                    if self.debounce_counter >= self.debounce_required:
                        self.current_step = FSMStep.CONTAINER_OPEN
                        self.candidate_step = None
                        self.debounce_counter = 0
                        self.step_start_time = now
                        transition_committed = "CONTAINER_OPENED"
                else:
                    self._reset_debounce()

            # State 1: CONTAINER_OPEN -> Awaiting Red Box Extraction
            elif self.current_step == FSMStep.CONTAINER_OPEN:
                red_obj = objects.get("red_box")
                yellow_obj = objects.get("yellow_box")

                # Check for OUT-OF-SEQUENCE ANOMALY
                if yellow_obj and (yellow_obj.state in (EntityState.GRASPED, EntityState.EXTRACTED) 
                                   or not yellow_obj.is_inside_container):
                    self.anomaly_status = AnomalyType.ERROR_SEQ
                    self.anomaly_message = self.config["anomalies"]["ERROR_SEQ"]["alert_tts"]
                    self._reset_debounce()
                    return self.current_step, 0, self.anomaly_status, self.anomaly_message, None

                # Nominal transition: Red Box extracted
                if red_obj and (red_obj.state == EntityState.EXTRACTED or not red_obj.is_inside_container):
                    self._accumulate_debounce(FSMStep.RED_EXTRACTED)
                    if self.debounce_counter >= self.debounce_required:
                        self.current_step = FSMStep.RED_EXTRACTED
                        self.candidate_step = None
                        self.debounce_counter = 0
                        self.step_start_time = now
                        transition_committed = "RED_BOX_EXTRACTED"
                else:
                    self._reset_debounce()

            # State 2: RED_EXTRACTED -> Awaiting Yellow Box Extraction
            elif self.current_step == FSMStep.RED_EXTRACTED:
                yellow_obj = objects.get("yellow_box")

                # Check for SEQUENCE SKIP ANOMALY
                if lid_angle < 15.0 and yellow_obj and yellow_obj.is_inside_container:
                    self.anomaly_status = AnomalyType.ERROR_SKIP
                    self.anomaly_message = self.config["anomalies"]["ERROR_SKIP"]["alert_tts"]
                    self._reset_debounce()
                    return self.current_step, 0, self.anomaly_status, self.anomaly_message, None

                # Nominal transition: Yellow Box extracted
                if yellow_obj and (yellow_obj.state == EntityState.EXTRACTED or not yellow_obj.is_inside_container):
                    self._accumulate_debounce(FSMStep.COMPLETE)
                    if self.debounce_counter >= self.debounce_required:
                        self.current_step = FSMStep.COMPLETE
                        self.candidate_step = None
                        self.debounce_counter = 0
                        self.step_start_time = now
                        transition_committed = "YELLOW_BOX_EXTRACTED"
                else:
                    self._reset_debounce()

            # State 3: COMPLETE
            elif self.current_step == FSMStep.COMPLETE:
                self.debounce_counter = self.debounce_required

        return self.current_step, self.debounce_counter, self.anomaly_status, self.anomaly_message, transition_committed

    def _accumulate_debounce(self, target: FSMStep):
        if self.candidate_step == target:
            self.debounce_counter += 1
        else:
            self.candidate_step = target
            self.debounce_counter = 1

    def _reset_debounce(self, soft: bool = True):
        if soft and self.debounce_counter > 0:
            self.debounce_counter -= 1
            if self.debounce_counter == 0:
                self.candidate_step = None
        else:
            self.candidate_step = None
            self.debounce_counter = 0
