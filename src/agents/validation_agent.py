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
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.experiment_id: str = self.config.get("experiment_id", "BAS-EXP-BOX-RETURN")
        self.debounce_required: int = self.config.get("debounce_frames", 6)
        self.current_step: FSMStep = FSMStep.IDLE
        self.debounce_counter: int = 0
        self.candidate_step: Optional[FSMStep] = None

        self.anomaly_status: AnomalyType = AnomalyType.NONE
        self.anomaly_message: str = ""
        self.anomaly_debounce_counter: int = 0
        self.step_start_time: float = time.time()
        self.stall_timeout_sec: float = 60.0

        # Step Correctness Tracking (Nominal vs Anomaly)
        self.is_step_correct: bool = True
        self.step_verdict: str = "CORRECT (NOMINAL)"

    def load_protocol(self, config_path: str):
        """Dynamically switches the active procedural FSM protocol."""
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.config_path = config_path
        self.experiment_id = self.config.get("experiment_id", "BAS-EXP-BOX-RETURN")
        self.debounce_required = self.config.get("debounce_frames", 6)
        self.reset()

    def reset(self):
        """Resets the state machine back to IDLE (Step 0) for a fresh real-time test run."""
        self.current_step = FSMStep.IDLE
        self.debounce_counter = 0
        self.candidate_step = None
        self.anomaly_status = AnomalyType.NONE
        self.anomaly_message = ""
        self.anomaly_debounce_counter = 0
        self.step_start_time = time.time()
        self.is_step_correct = True
        self.step_verdict = "CORRECT (NOMINAL)"

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

        # =================================================================
        # BRANCH A: ISRO BENCHMARK DUAL-BOX EXPERIMENT (PS #26174)
        # Sequence: IDLE -> OPEN CONTAINER -> EXTRACT RED -> EXTRACT YELLOW -> COMPLETE
        # Forbidden: Yellow touched/extracted before Red (ERROR_SEQ)
        # =================================================================
        if self.experiment_id == "BAS-EXP-26174":
            if self.current_step == FSMStep.IDLE:
                is_opening = (
                    lid_angle >= 20.0
                    or ("container_lid" in objects and objects["container_lid"].bbox is not None)
                    or (llm_step_val is not None and llm_step_val >= 1 and llm_confidence >= 0.70)
                )
                if is_opening:
                    self._accumulate_debounce(FSMStep.CONTAINER_OPEN)
                    if self.debounce_counter >= self.debounce_required:
                        self.current_step = FSMStep.CONTAINER_OPEN
                        self.candidate_step = None
                        self.debounce_counter = 0
                        self.step_start_time = now
                        transition_committed = "LID_OPENED"
                else:
                    self._reset_debounce()

            elif self.current_step == FSMStep.CONTAINER_OPEN:
                # Check for out-of-order sequence error (Yellow touched/extracted before Red)
                yellow_obj = objects.get("yellow_box")
                red_obj = objects.get("red_box")
                
                yellow_violation = False
                if yellow_obj and (not yellow_obj.is_inside_container or yellow_obj.state == EntityState.EXTRACTED):
                    if not red_obj or red_obj.is_inside_container:
                        yellow_violation = True
                
                if not yellow_violation:
                    for h in active_hoi:
                        if h.object_name == "yellow_box" and h.action in (HOIAction.CONTACT, HOIAction.GRASP, HOIAction.EXTRACT):
                            if not red_obj or red_obj.is_inside_container:
                                yellow_violation = True
                                break

                if yellow_violation:
                    self.anomaly_status = AnomalyType.ERROR_SEQ
                    self.anomaly_message = "Warning: Procedural error. Red box must be extracted before yellow box."
                    self.is_step_correct = False
                    self.step_verdict = "PROCEDURAL ERROR [ERROR_SEQ]"
                    return self.current_step, 0, self.anomaly_status, self.anomaly_message, None

                # Check premature close anomaly
                if lid_angle <= 10.0:
                    self.anomaly_debounce_counter += 1
                    if self.anomaly_debounce_counter >= 5:
                        self.anomaly_status = AnomalyType.ERROR_SKIP
                        self.anomaly_message = "Warning: Step skipped. Please extract the red box before closing."
                        self.is_step_correct = False
                        self.step_verdict = "PROCEDURAL ERROR [ERROR_SKIP]"
                        return self.current_step, 0, self.anomaly_status, self.anomaly_message, None
                else:
                    self.anomaly_debounce_counter = 0

                # Check Red Box Extracted
                red_extracted = False
                if red_obj and (not red_obj.is_inside_container or red_obj.state == EntityState.EXTRACTED):
                    red_extracted = True
                elif any(h.object_name == "red_box" and h.action == HOIAction.EXTRACT for h in active_hoi):
                    red_extracted = True
                elif "component_box" in objects and not objects["component_box"].is_inside_container:
                    red_extracted = True

                if red_extracted:
                    self._accumulate_debounce(FSMStep.RED_EXTRACTED)
                    if self.debounce_counter >= self.debounce_required:
                        self.current_step = FSMStep.RED_EXTRACTED
                        self.candidate_step = None
                        self.debounce_counter = 0
                        self.step_start_time = now
                        self.anomaly_status = AnomalyType.NONE
                        self.anomaly_message = ""
                        transition_committed = "RED_BOX_EXTRACTED"
                else:
                    self._reset_debounce()

            elif self.current_step == FSMStep.RED_EXTRACTED:
                # Check premature close anomaly
                if lid_angle <= 10.0:
                    self.anomaly_debounce_counter += 1
                    if self.anomaly_debounce_counter >= 5:
                        self.anomaly_status = AnomalyType.ERROR_SKIP
                        self.anomaly_message = "Warning: Step skipped. Please extract the yellow box to complete the procedure."
                        self.is_step_correct = False
                        self.step_verdict = "PROCEDURAL ERROR [ERROR_SKIP]"
                        return self.current_step, 0, self.anomaly_status, self.anomaly_message, None
                else:
                    self.anomaly_debounce_counter = 0

                # Check Yellow Box Extracted
                yellow_extracted = False
                yellow_obj = objects.get("yellow_box")
                if yellow_obj and (not yellow_obj.is_inside_container or yellow_obj.state == EntityState.EXTRACTED):
                    yellow_extracted = True
                elif any(h.object_name == "yellow_box" and h.action == HOIAction.EXTRACT for h in active_hoi):
                    yellow_extracted = True

                if yellow_extracted:
                    self._accumulate_debounce(FSMStep.COMPLETE)
                    if self.debounce_counter >= self.debounce_required:
                        self.current_step = FSMStep.COMPLETE
                        self.candidate_step = None
                        self.debounce_counter = self.debounce_required
                        self.step_start_time = now
                        self.anomaly_status = AnomalyType.NONE
                        self.anomaly_message = ""
                        transition_committed = "YELLOW_BOX_EXTRACTED"
                else:
                    self._reset_debounce()

            elif self.current_step == FSMStep.COMPLETE:
                self.debounce_counter = self.debounce_required
                if (now - self.step_start_time) >= 3.0 and lid_angle >= 28.0:
                    self.current_step = FSMStep.BOX_OPENED
                    self.candidate_step = None
                    self.step_start_time = now
                    self.anomaly_status = AnomalyType.NONE
                    self.anomaly_message = ""
                    transition_committed = "CONTAINER_REOPENED"

        # =================================================================
        # BRANCH B: BOX OBJECT EXTRACTION & RETURN PROCEDURE (BAS-EXP-BOX-RETURN)
        # Sequence: IDLE -> OPEN BOX -> EXTRACT OBJECT -> RETURN OBJECT -> CLOSE BOX
        # =================================================================
        else:
            # Step 0: IDLE -> Awaiting Box Open
            if self.current_step == FSMStep.IDLE:
                is_opening = (
                    lid_angle >= 22.0
                    or ("container_lid" in objects and objects["container_lid"].bbox is not None)
                )
                if is_opening:
                    self._accumulate_debounce(FSMStep.BOX_OPENED)
                    if self.debounce_counter >= self.debounce_required:
                        self.current_step = FSMStep.BOX_OPENED
                        self.candidate_step = None
                        self.debounce_counter = 0
                        self.step_start_time = now
                        self.is_step_correct = True
                        self.step_verdict = "STEP OK: Nominal Procedure"
                        transition_committed = "BOX_OPENED"
                else:
                    self._reset_debounce()

            # Step 1: BOX_OPENED -> Awaiting Object Extraction
            elif self.current_step == FSMStep.BOX_OPENED:
                # Check premature close anomaly
                if lid_angle <= 10.0:
                    self.anomaly_debounce_counter += 1
                    if self.anomaly_debounce_counter >= 8:
                        if not (llm_verification and llm_verification.get("anomaly_verdict") == "NOMINAL"):
                            self.anomaly_status = AnomalyType.ERROR_SKIP
                            self.anomaly_message = "Warning: Step skipped. Please extract the object before closing the box."
                            self.is_step_correct = False
                            self.step_verdict = "WRONG STEP: Box closed before extracting object! [ERROR_SKIP]"
                            return self.current_step, 0, self.anomaly_status, self.anomaly_message, None
                else:
                    self.anomaly_debounce_counter = 0
                    if self.anomaly_status == AnomalyType.ERROR_SKIP:
                        self.anomaly_status = AnomalyType.NONE
                        self.anomaly_message = ""
                        self.is_step_correct = True
                        self.step_verdict = "STEP OK: Nominal Procedure"

                # Check if manipulable object is physically extracted outside the box
                extracted = False
                EXCLUDED_NON_PAYLOAD = {"container_box", "container_lid", "operator_hand", "human_body", "person"}
                for name, obj in objects.items():
                    if name in ("component_box", "red_box", "yellow_box") or (name not in EXCLUDED_NON_PAYLOAD):
                        if not obj.is_inside_container or obj.state == EntityState.EXTRACTED:
                            extracted = True
                            break

                if not extracted and any(h.action == HOIAction.EXTRACT for h in active_hoi):
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
                        self.is_step_correct = True
                        self.step_verdict = "STEP OK: Nominal Procedure"
                        transition_committed = "OBJECT_EXTRACTED"
                else:
                    self._reset_debounce()

            # Step 2: OBJECT_EXTRACTED -> Awaiting Object Return into Box
            elif self.current_step == FSMStep.OBJECT_EXTRACTED:
                # Check premature close anomaly
                if lid_angle <= 10.0:
                    self.anomaly_debounce_counter += 1
                    if self.anomaly_debounce_counter >= 8:
                        if not (llm_verification and llm_verification.get("anomaly_verdict") == "NOMINAL"):
                            self.anomaly_status = AnomalyType.ERROR_SEQ
                            self.anomaly_message = "Warning: Procedural error. The object has not been returned to the box."
                            self.is_step_correct = False
                            self.step_verdict = "WRONG STEP: Object not returned into box before closing! [ERROR_SEQ]"
                            return self.current_step, 0, self.anomaly_status, self.anomaly_message, None
                else:
                    self.anomaly_debounce_counter = 0
                    if self.anomaly_status == AnomalyType.ERROR_SEQ:
                        self.anomaly_status = AnomalyType.NONE
                        self.anomaly_message = ""
                        self.is_step_correct = True
                        self.step_verdict = "STEP OK: Nominal Procedure"

                # Check if object has physically returned back inside container cavity
                all_inside = True
                found_target = False
                EXCLUDED_NON_PAYLOAD = {"container_box", "container_lid", "operator_hand", "human_body", "person"}
                for name, obj in objects.items():
                    if name in ("component_box", "red_box", "yellow_box") or (name not in EXCLUDED_NON_PAYLOAD):
                        found_target = True
                        if not obj.is_inside_container or obj.state == EntityState.EXTRACTED:
                            all_inside = False
                            break

                returned = (found_target and all_inside)
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
                        self.is_step_correct = True
                        self.step_verdict = "STEP OK: Nominal Procedure"
                        transition_committed = "OBJECT_RETURNED"
                else:
                    self._reset_debounce()

            # Step 3: OBJECT_RETURNED -> Awaiting Box Close
            elif self.current_step == FSMStep.OBJECT_RETURNED:
                is_closed = (
                    lid_angle <= 28.0
                    and ("container_lid" not in objects or objects["container_lid"].bbox is None)
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
                        self.is_step_correct = True
                        self.step_verdict = "VERIFIED COMPLETE (NOMINAL)"
                        transition_committed = "BOX_CLOSED"
                else:
                    self._reset_debounce()

            # Step 4: COMPLETE
            elif self.current_step == FSMStep.COMPLETE:
                self.debounce_counter = self.debounce_required
                self.is_step_correct = True
                self.step_verdict = "VERIFIED COMPLETE (NOMINAL)"
                if (now - self.step_start_time) >= 3.0:
                    is_reopening = (
                        lid_angle >= 25.0
                        or ("container_lid" in objects and objects["container_lid"].bbox is not None)
                    )
                    if is_reopening:
                        self.current_step = FSMStep.BOX_OPENED
                        self.candidate_step = None
                        self.step_start_time = now
                        self.anomaly_status = AnomalyType.NONE
                        self.anomaly_message = ""
                        transition_committed = "BOX_REOPENED"

        # Real-time Step Correctness Verdict
        if self.anomaly_status != AnomalyType.NONE:
            self.is_step_correct = False
            self.step_verdict = f"PROCEDURAL ERROR [{self.anomaly_status.value}]"
        else:
            self.is_step_correct = True
            if self.current_step == FSMStep.COMPLETE:
                self.step_verdict = "VERIFIED COMPLETE (NOMINAL)"
            else:
                self.step_verdict = "CORRECT (NOMINAL)"

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
