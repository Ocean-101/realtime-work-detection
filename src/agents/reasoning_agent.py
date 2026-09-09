"""
BAS Autonomous HAR System - Agent 7: Reasoning & Guidance Agent
Interprets procedural state, produces next-step astronaut instructions,
and formulates immediate voice alerts and recovery prompts upon anomaly detection.
"""

import json
from typing import Optional, Tuple
from src.core.types import FSMStep, AnomalyType


class ReasoningAgent:
    """Provides conversational mission guidance, next-step recommendations, and recovery advice."""

    def __init__(self, config_path: str = "configs/experiment_fsm.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.step_instructions = {}
        for s_idx_str, s_info in self.config.get("states", {}).items():
            try:
                step_enum = FSMStep(int(s_idx_str))
                self.step_instructions[step_enum] = s_info["instruction"]
            except ValueError:
                pass

        self.last_guided_step: Optional[FSMStep] = None
        self.last_anomaly_alert: Optional[AnomalyType] = None

    def evaluate_guidance(
        self,
        current_step: FSMStep,
        anomaly: AnomalyType,
        transition_event: Optional[str]
    ) -> Tuple[str, Optional[str]]:
        """
        Determines current on-screen instruction and any pending spoken alert.
        Returns:
            - active_instruction: Text instruction displayed in the GUI HUD
            - voice_alert_to_speak: Voice utterance string if audio should be triggered, else None
        """
        voice_alert: Optional[str] = None
        instruction = self.step_instructions.get(current_step, "Awaiting instructions.")

        # 1. Handle Critical Anomalies (Highest Priority)
        if anomaly != AnomalyType.NONE and anomaly != self.last_anomaly_alert:
            self.last_anomaly_alert = anomaly
            anom_key = anomaly.value if hasattr(anomaly, "value") else str(anomaly)
            anom_dict = self.config.get("anomalies", {})
            anom_info = anom_dict.get(anom_key) or anom_dict.get(anomaly.name, {})

            if not anom_info:
                if "SKIP" in anom_key:
                    anom_info = anom_dict.get("ERROR_SKIP", anom_dict.get("ERROR_PREMATURE_CLOSE", {}))
                elif "SEQ" in anom_key:
                    anom_info = anom_dict.get("ERROR_SEQ", anom_dict.get("ERROR_UNRETURNED_CLOSE", {}))
                elif "STALL" in anom_key or "TIMEOUT" in anom_key:
                    anom_info = anom_dict.get("STALL_TIMEOUT", {})
                else:
                    anom_info = {}

            voice_alert = anom_info.get("alert_tts", f"Warning: Procedural deviation detected.")
            rec_prompt = anom_info.get("recovery_prompt", "Please resume nominal procedure.")
            instruction = f"ANOMALY: {rec_prompt}"
            return instruction, voice_alert

        # Reset anomaly memory once resolved
        if anomaly == AnomalyType.NONE:
            self.last_anomaly_alert = None

        # 2. Handle State Transitions / Next-Step Guidance
        if transition_event is not None or current_step != self.last_guided_step:
            self.last_guided_step = current_step
            voice_alert = instruction

        return instruction, voice_alert
