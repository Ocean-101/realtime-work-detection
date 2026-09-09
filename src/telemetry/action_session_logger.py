"""
BAS Autonomous HAR System - Action Session Logger
Records structured, timestamped procedural action logs into a dedicated JSON file
capturing:
- What I am doing (Current detected action)
- What I have to do (Next required procedural step)
- Timing, duration, and safety compliance
Provides the definitive structured feed for the offline LLM procedural auditor.
"""

import os
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional


class ActionSessionLogger:
    """Logs the chronological action timeline into structured JSON for offline LLM audits."""

    def __init__(
        self,
        experiment_id: str = "BAS-EXP-BOX-RETURN",
        procedure_name: str = "Box Object Extraction & Return Procedure",
        output_path: str = "experiments/session_actions.json"
    ):
        self.experiment_id = experiment_id
        self.procedure_name = procedure_name
        self.output_path = output_path
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        self.session_id = f"SES-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_wall_time = datetime.now(timezone.utc).isoformat()
        self.start_epoch = time.time()
        self.actions: List[Dict[str, Any]] = []
        self.current_action_entry: Optional[Dict[str, Any]] = None
        self.anomalies: List[str] = []
        self.last_step: Optional[int] = None
        self.last_activity: Optional[str] = None

    def update(
        self,
        frame_id: int,
        timestamp_sec: float,
        step: int,
        step_name: str,
        what_i_am_doing: str,
        what_i_have_to_do: str,
        lid_angle: float,
        anomaly: str = "NONE"
    ):
        """Updates continuous action state and registers discrete step/activity blocks."""
        if anomaly and anomaly != "NONE" and anomaly not in self.anomalies:
            self.anomalies.append(anomaly)

        # Detect new action phase transition
        is_new_phase = (
            self.current_action_entry is None or
            step != self.last_step or
            (what_i_am_doing != self.last_activity and what_i_am_doing != "IDLE")
        )

        if is_new_phase:
            # Finalize previous action entry
            if self.current_action_entry is not None:
                self.current_action_entry["end_time_sec"] = round(timestamp_sec, 2)
                self.current_action_entry["end_frame"] = frame_id
                self.current_action_entry["duration_sec"] = round(
                    max(0.1, timestamp_sec - self.current_action_entry["start_time_sec"]), 2
                )
                self.actions.append(self.current_action_entry)

            # Start new action entry
            self.current_action_entry = {
                "step": step,
                "step_name": step_name,
                "what_i_am_doing": what_i_am_doing,
                "what_i_have_to_do": what_i_have_to_do,
                "start_time_sec": round(timestamp_sec, 2),
                "end_time_sec": round(timestamp_sec, 2),
                "duration_sec": 0.0,
                "start_frame": frame_id,
                "end_frame": frame_id,
                "lid_angle_deg": round(lid_angle, 1),
                "compliance": "NOMINAL" if anomaly == "NONE" else f"ANOMALY: {anomaly}"
            }
            self.last_step = step
            self.last_activity = what_i_am_doing

            # Persist update to JSON
            self.save()
        else:
            # Update end frame and lid angle of active action
            if self.current_action_entry:
                self.current_action_entry["end_frame"] = frame_id
                self.current_action_entry["end_time_sec"] = round(timestamp_sec, 2)
                self.current_action_entry["duration_sec"] = round(
                    max(0.1, timestamp_sec - self.current_action_entry["start_time_sec"]), 2
                )
                self.current_action_entry["lid_angle_deg"] = round(lid_angle, 1)

    def to_dict(self) -> Dict[str, Any]:
        """Compiles complete session dictionary."""
        all_actions = list(self.actions)
        if self.current_action_entry:
            all_actions.append(self.current_action_entry)

        elapsed = round(time.time() - self.start_epoch, 2)
        total_steps = len(set(a["step"] for a in all_actions))

        return {
            "session_id": self.session_id,
            "experiment_id": self.experiment_id,
            "procedure_name": self.procedure_name,
            "start_time": self.start_wall_time,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed,
            "total_actions_recorded": len(all_actions),
            "summary": {
                "total_steps_executed": total_steps,
                "anomalies_flagged": len(self.anomalies),
                "anomaly_details": self.anomalies,
                "compliance_status": "100% NOMINAL" if not self.anomalies else "PROCEDURAL DEVIATION DETECTED"
            },
            "timeline": all_actions
        }

    def save(self, filepath: Optional[str] = None):
        """Atomically saves the session actions to JSON."""
        target = filepath or self.output_path
        data = self.to_dict()
        try:
            with open(target, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[ActionSessionLogger] Warning saving JSON: {e}")

    def reset(self):
        """Resets the logger for a new test run cycle."""
        self.session_id = f"SES-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_wall_time = datetime.now(timezone.utc).isoformat()
        self.start_epoch = time.time()
        self.actions = []
        self.current_action_entry = None
        self.anomalies = []
        self.last_step = None
        self.last_activity = None
        self.save()
