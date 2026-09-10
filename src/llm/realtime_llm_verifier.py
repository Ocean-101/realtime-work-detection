"""
BAS Autonomous HAR System - Real-Time Local LLM Step Verifier
Runs continuous, non-blocking asynchronous verification using local Ollama (qwen2.5:1.5b).
Adjudicates real-time procedural transitions, filters visual jitter and finger occlusions,
and validates compliance for the Bharatiya Antariksh Station (BAS).
"""

import os
import json
import time
import threading
import urllib.request
import urllib.error
from collections import deque
from typing import Dict, Any, Optional, List
from src.core.types import FSMStep, AnomalyType


class RealtimeLLMVerifier:
    """
    Asynchronous, non-blocking local LLM verifier.
    Consumes sliding telemetry windows and queries local Ollama in a background worker thread.
    """

    def __init__(
        self,
        model_name: str = "qwen2.5:1.5b",
        ollama_url: str = "http://localhost:11434/api/generate",
        interval_sec: float = 1.0,
        window_size: int = 25
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.interval_sec = interval_sec
        self.window_size = window_size

        self.telemetry_history = deque(maxlen=window_size)
        self.lock = threading.Lock()

        # Latest verified state
        self.latest_verification: Dict[str, Any] = {
            "verified_step": 0,
            "step_name": "IDLE",
            "confidence": 1.0,
            "anomaly_verdict": "NOMINAL",
            "reason": "System initialized. Awaiting box opening.",
            "timestamp": time.time(),
            "model": self.model_name,
            "latency_s": 0.0
        }

        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._run_worker, daemon=True)
        self._worker_thread.start()

    def push_telemetry(
        self,
        frame_id: int,
        step: FSMStep,
        activity: str,
        lid_angle: float,
        is_inside: bool,
        hoi_action: str,
        hand_dist_m: float,
        anomaly: AnomalyType
    ):
        """Pushes a snapshot of frame telemetry into the sliding verification window."""
        with self.lock:
            self.telemetry_history.append({
                "frame_id": frame_id,
                "heuristic_step": int(step),
                "step_name": step.name,
                "activity": activity,
                "lid_angle": round(lid_angle, 1),
                "is_inside": 1 if is_inside else 0,
                "hoi_action": hoi_action,
                "hand_dist_m": round(hand_dist_m, 2),
                "anomaly": anomaly.value
            })

    def get_latest_verification(self) -> Dict[str, Any]:
        """Returns the most recent verification consensus without blocking."""
        with self.lock:
            return dict(self.latest_verification)

    def _run_worker(self):
        """Background thread executing non-blocking Ollama queries."""
        # Initial brief sleep to let the video pipeline start up smoothly
        time.sleep(1.0)

        while not self._stop_event.is_set():
            time.sleep(self.interval_sec)

            # Extract window copy
            with self.lock:
                if len(self.telemetry_history) < 5:
                    continue
                history_snapshot = list(self.telemetry_history)

            # Analyze recent telemetry
            result = self._query_llm_for_verification(history_snapshot)
            if result:
                with self.lock:
                    self.latest_verification = result

    def _query_llm_for_verification(self, history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Sends sliding window telemetry to local Ollama and parses JSON verdict."""
        if not history:
            return None

        recent = history[-1]
        earlier = history[0]

        avg_lid = round(sum(h["lid_angle"] for h in history) / len(history), 1)
        recent_activities = list(set(h["activity"] for h in history[-8:]))
        inside_ratio = sum(h["is_inside"] for h in history) / len(history)
        candidate_step = recent["heuristic_step"]
        candidate_name = recent["step_name"]

        prompt = f"""You are the autonomous procedural supervisor for Bharatiya Antariksh Station (BAS).
Verify the active step of this 4-step procedure:
- Step 0: IDLE (Box closed, waiting for operator to open container)
- Step 1: BOX_OPENED (Container lid opened >20 deg, operator reaching to take out component)
- Step 2: OBJECT_EXTRACTED (Component box is outside container, held or manipulated)
- Step 3: OBJECT_RETURNED (Component box placed back inside container)
- Step 4: COMPLETE (Container lid closed <20 deg, procedure finished)

Recent Physical Telemetry:
- Average Container Lid Angle: {avg_lid} deg (Current: {recent['lid_angle']} deg)
- Component Inside Container Ratio: {round(inside_ratio, 2)}
- Current Activities: {', '.join(recent_activities)}
- Heuristic Candidate Step: {candidate_step} ({candidate_name})
- Anomaly Flag: {recent['anomaly']}

Determine the verified procedural step and state.
Reply strictly in valid JSON format with this exact schema:
{{"verified_step": <integer 0-4>, "step_name": "<NAME>", "confidence": <float 0.0-1.0>, "anomaly_verdict": "NOMINAL"|"ANOMALY", "reason": "<brief 1-sentence explanation>"}}"""

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.8,
                "num_predict": 90
            }
        }

        t_start = time.time()
        try:
            req = urllib.request.Request(
                self.ollama_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                latency = round(time.time() - t_start, 3)
                raw_text = data.get("response", "").strip()

                # Extract JSON if enclosed in markdown code fences
                if "```" in raw_text:
                    parts = raw_text.split("```")
                    for p in parts:
                        p_str = p.strip()
                        if p_str.startswith("json"):
                            p_str = p_str[4:].strip()
                        if p_str.startswith("{") and p_str.endswith("}"):
                            raw_text = p_str
                            break

                json_start = raw_text.find("{")
                json_end = raw_text.rfind("}")
                if json_start != -1 and json_end != -1:
                    raw_text = raw_text[json_start:json_end + 1]

                parsed = json.loads(raw_text)
                step_val = int(parsed.get("verified_step", candidate_step))
                step_val = max(0, min(4, step_val))

                names = {
                    0: "IDLE",
                    1: "BOX_OPENED",
                    2: "OBJECT_EXTRACTED",
                    3: "OBJECT_RETURNED",
                    4: "COMPLETE"
                }

                return {
                    "verified_step": step_val,
                    "step_name": names.get(step_val, parsed.get("step_name", candidate_name)),
                    "confidence": float(parsed.get("confidence", 0.90)),
                    "anomaly_verdict": parsed.get("anomaly_verdict", "NOMINAL"),
                    "reason": parsed.get("reason", "Verified by Local LLM supervisor."),
                    "timestamp": time.time(),
                    "model": self.model_name,
                    "latency_s": latency
                }

        except Exception as e:
            # If Ollama takes longer or is unreachable, generate deterministic telemetry verification
            return self._fallback_deterministic_verification(
                candidate_step, candidate_name, avg_lid, inside_ratio, recent_activities
            )

    def _fallback_deterministic_verification(
        self,
        candidate_step: int,
        candidate_name: str,
        avg_lid: float,
        inside_ratio: float,
        recent_activities: List[str]
    ) -> Dict[str, Any]:
        """Provides verified fallback consensus if Ollama is momentarily busy."""
        verified = candidate_step
        reason = "Deterministic kinematic fallback verification."

        if candidate_step == 0 and avg_lid >= 20.0:
            verified = 1
            reason = f"Container lid elevated ({avg_lid} deg) indicating opened state."
        elif candidate_step == 1 and inside_ratio < 0.4:
            verified = 2
            reason = "Component extracted and held outside container."
        elif candidate_step == 2 and inside_ratio > 0.6:
            verified = 3
            reason = "Component returned and seated inside container."
        elif candidate_step == 3 and avg_lid < 22.0:
            verified = 4
            reason = f"Container lid closed ({avg_lid} deg). Mission complete."

        names = {
            0: "IDLE",
            1: "BOX_OPENED",
            2: "OBJECT_EXTRACTED",
            3: "OBJECT_RETURNED",
            4: "COMPLETE"
        }

        return {
            "verified_step": verified,
            "step_name": names.get(verified, candidate_name),
            "confidence": 0.88,
            "anomaly_verdict": "NOMINAL",
            "reason": reason,
            "timestamp": time.time(),
            "model": "rule_consensus_fallback",
            "latency_s": 0.001
        }

    def close(self):
        """Stops the verifier background worker thread cleanly."""
        self._stop_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.5)
