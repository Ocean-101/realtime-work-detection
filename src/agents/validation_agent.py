"""
BAS Autonomous HAR System - Agent 6: Validation Agent (Safety-Critical FSM)
Authoritative, deterministic finite state machine validating procedural compliance
with adaptive temporal debouncing, Local LLM consensus verification, and robust anomaly protection.
"""

import json
import time
from typing import Dict, Tuple, Optional, Any
from src.core.types import (
    FSMStep,
    AnomalyType,
    ExperimentObject,
    EntityState,
    HOIInteraction,
    HOIAction
)


class ValidationAgent:
    """Deterministic Sequence Validator and Safety Gatekeeper with Local LLM Consensus."""

    def __init__(self, config_path: str = "configs/box_return_fsm.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.debounce_required: int = self.config.get("debounce_frames", 6)
        self.current_step: FSMStep = FSMStep.IDLE
        self.debounce_counter: int = 0
        self.candidate_step: Optional[FSMStep] = None

        self.anomaly_status: AnomalyType = AnomalyType.NONE
        self.anomaly_message: str = ""
        self.anomaly_debounce_counter: int = 0
        self.step_start_time: float = time.time()
        self.stall_timeout_sec: float = 60.0

    def reset(self):
        """Resets the state machine back to IDLE (Step 0) for a fresh real-time test run."""
        self.current_step = FSMStep.IDLE
        self.debounce_counter = 0
        self.candidate_step = None
        self.anomaly_status = AnomalyType.NONE
        self.anomaly_message = ""
        self.anomaly_debounce_counter = 0
        self.step_start_time = time.time()

    def evaluate_step(
        self,
        objects: Dict[str, ExperimentObject],
        lid_angle: float,
        active_hoi: list,
        current_frame: int,
        llm_verification: Optional[Dict[str, Any]] = None
    ) -> Tuple[FSMStep, int, AnomalyType, str, Optional[str]]:
        """
        Evaluates current physical state against procedural state machine with Local LLM consensus.
        Returns:
            - current_step: committed FSMStep
            - debounce_count: frames accumulated towards candidate transition
            - anomaly: AnomalyType
            - anomaly_msg: diagnostic string
            - transition_event: string name if a new state was committed on this frame
        """
        now = time.time()
        transition_committed: Optional[str] = None

        # Check Stall Timeout
        if self.current_step not in (FSMStep.IDLE, FSMStep.COMPLETE):
            if (now - self.step_start_time) > self.stall_timeout_sec:
                self.anomaly_status = AnomalyType.STALL_TIMEOUT
                self.anomaly_message = "Activity paused. Awaiting required procedural action."

        # Extract LLM verified step if available
        llm_step_val = None
        llm_confidence = 0.0
        if llm_verification:
            llm_step_val = llm_verification.get("verified_step")
            llm_confidence = float(llm_verification.get("confidence", 0.0))

        # -----------------------------------------------------------------
        # 4-STEP PROCEDURE: OPEN -> EXTRACT OBJECT -> RETURN OBJECT -> CLOSE BOX
        # -----------------------------------------------------------------
        # Step 0: IDLE -> Awaiting Box Open
        if self.current_step == FSMStep.IDLE:
            # Box opening condition: lid angle elevated OR lid detected open OR LLM consensus
            is_opening = (
                lid_angle >= 18.0
                or ("container_lid" in objects and objects["container_lid"].bbox is not None)
                or (llm_step_val is not None and llm_step_val >= 1 and llm_confidence >= 0.70)
            )
            if is_opening:
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
            # Check premature close anomaly with 5-frame debouncing to filter sensor flicker
            if lid_angle <= 10.0:
                self.anomaly_debounce_counter += 1
                if self.anomaly_debounce_counter >= 5:
                    if not (llm_verification and llm_verification.get("anomaly_verdict") == "NOMINAL"):
                        self.anomaly_status = AnomalyType.ERROR_SKIP
                        self.anomaly_message = "Warning: Step skipped. Please extract the object before closing the box."
                        return self.current_step, 0, self.anomaly_status, self.anomaly_message, None
            else:
                self.anomaly_debounce_counter = 0

            # Check if manipulable object (component_box) is extracted outside the box
            extracted = False
            for name, obj in objects.items():
                if name not in ("container_box", "container_lid"):
                    if not obj.is_inside_container or obj.state == EntityState.EXTRACTED:
                        extracted = True
                        break

            # Also check if any HOI interaction registered an EXTRACT action
            if not extracted and any(h.action == HOIAction.EXTRACT for h in active_hoi):
                extracted = True

            # Also check Local LLM consensus verification
            if not extracted and (llm_step_val == 2 and llm_confidence >= 0.75):
                extracted = True

            if extracted:
                self._accumulate_debounce(FSMStep.OBJECT_EXTRACTED)
                if self.debounce_counter >= self.debounce_required:
                    self.current_step = FSMStep.OBJECT_EXTRACTED
                    self.candidate_step = None
                    self.debounce_counter = 0
                    self.step_start_time = now
                    self.anomaly_status = AnomalyType.NONE
                    self.anomaly_message = ""
                    self.anomaly_debounce_counter = 0
                    transition_committed = "OBJECT_EXTRACTED"
            else:
                self._reset_debounce()

        # Step 2: OBJECT_EXTRACTED -> Awaiting Object Return into Box
        elif self.current_step == FSMStep.OBJECT_EXTRACTED:
            # Check premature close anomaly with 5-frame debouncing
            if lid_angle <= 10.0:
                self.anomaly_debounce_counter += 1
                if self.anomaly_debounce_counter >= 5:
                    if not (llm_verification and llm_verification.get("anomaly_verdict") == "NOMINAL"):
                        self.anomaly_status = AnomalyType.ERROR_SEQ
                        self.anomaly_message = "Warning: Procedural error. The object has not been returned to the box."
                        return self.current_step, 0, self.anomaly_status, self.anomaly_message, None
            else:
                self.anomaly_debounce_counter = 0

            # Check if object has returned back inside container
            all_inside = True
            found_target = False
            for name, obj in objects.items():
                if name not in ("container_box", "container_lid"):
                    found_target = True
                    if not obj.is_inside_container:
                        all_inside = False
                        break

            # If component object returned inside container OR verified by Local LLM consensus
            returned = (found_target and all_inside)
            if not returned and (llm_step_val == 3 and llm_confidence >= 0.75):
                returned = True

            if returned:
                self._accumulate_debounce(FSMStep.OBJECT_RETURNED)
                if self.debounce_counter >= self.debounce_required:
                    self.current_step = FSMStep.OBJECT_RETURNED
                    self.candidate_step = None
                    self.debounce_counter = 0
                    self.step_start_time = now
                    self.anomaly_status = AnomalyType.NONE
                    self.anomaly_message = ""
                    self.anomaly_debounce_counter = 0
                    transition_committed = "OBJECT_RETURNED"
            else:
                self._reset_debounce()

        # Step 3: OBJECT_RETURNED -> Awaiting Box Close
        elif self.current_step == FSMStep.OBJECT_RETURNED:
            # Box closed condition: lid lowered below threshold or flaps closed or LLM verified COMPLETE
            is_closed = (
                lid_angle < 22.0
                or (llm_step_val == 4 and llm_confidence >= 0.75)
            )
            if is_closed:
                self._accumulate_debounce(FSMStep.COMPLETE)
                if self.debounce_counter >= self.debounce_required:
                    self.current_step = FSMStep.COMPLETE
                    self.candidate_step = None
                    self.debounce_counter = self.debounce_required
                    self.step_start_time = now
                    self.anomaly_status = AnomalyType.NONE
                    self.anomaly_message = ""
                    transition_committed = "BOX_CLOSED"
            else:
                self._reset_debounce()

        # Step 4: COMPLETE
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
