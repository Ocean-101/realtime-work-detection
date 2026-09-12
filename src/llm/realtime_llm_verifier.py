"""
BAS Autonomous HAR System - Real-Time Multimodal VLM Step Verifier
Runs continuous, non-blocking asynchronous verification using local Qwen3-VL:2B-Instruct via Ollama.
Fuses YOLO bounding boxes and spatial metrics with visual reasoning to adjudicate
procedural transitions and diagnose "what wrong is going on".
"""

import os
import json
import time
import base64
import threading
import urllib.request
import urllib.error
from collections import deque
from typing import Dict, Any, Optional, List
import cv2
import numpy as np
from src.core.types import FSMStep, AnomalyType


class RealtimeLLMVerifier:
    """
    Asynchronous, non-blocking multimodal Vision-Language Model (VLM) verifier.
    Consumes sliding telemetry windows and downsampled camera frames, querying
    local Ollama (qwen3-vl:2b-instruct) in a background worker thread.
    """

    def __init__(
        self,
        model_name: str = "qwen3-vl:2b-instruct",
        ollama_url: str = "http://localhost:11434/api/generate",
        interval_sec: float = 1.0,
        window_size: int = 25,
        target_img_size: tuple = (384, 384)
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.interval_sec = interval_sec
        self.window_size = window_size
        self.target_img_size = target_img_size

        self.telemetry_history = deque(maxlen=window_size)
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_detected_objects: List[str] = []
        self.trigger_event = threading.Event()
        self.lock = threading.Lock()

        # Latest verified state & diagnostic verdict
        self.latest_verification: Dict[str, Any] = {
            "verified_step": 0,
            "step_name": "IDLE",
            "confidence": 1.0,
            "anomaly_verdict": "NOMINAL",
            "what_is_wrong": "None",
            "corrective_action": "System initialized. Awaiting box opening.",
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
        anomaly: AnomalyType,
        frame: Optional[np.ndarray] = None,
        detected_objects: Optional[List[str]] = None,
        force_priority: bool = False
    ):
        """
        Pushes a snapshot of frame telemetry and optional visual frame into the verifier.
        If force_priority is True (e.g. suspected anomaly or step candidate), wakes up
        the VLM worker immediately.
        """
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
                "anomaly": anomaly.value if hasattr(anomaly, "value") else str(anomaly)
            })

            if frame is not None:
                # Downsample image for high-speed multimodal VLM inference
                if frame.shape[0] != self.target_img_size[1] or frame.shape[1] != self.target_img_size[0]:
                    self.latest_frame = cv2.resize(frame, self.target_img_size, interpolation=cv2.INTER_AREA)
                else:
                    self.latest_frame = frame.copy()

            if detected_objects is not None:
                self.latest_detected_objects = list(detected_objects)

        if force_priority:
            self.trigger_event.set()

    def get_latest_verification(self) -> Dict[str, Any]:
        """Returns the most recent verification consensus without blocking."""
        with self.lock:
            return dict(self.latest_verification)

    def _run_worker(self):
        """Background thread executing non-blocking Ollama queries."""
        time.sleep(0.5)

        while not self._stop_event.is_set():
            # Wait for either periodic heartbeat interval or priority event trigger
            self.trigger_event.wait(timeout=self.interval_sec)
            self.trigger_event.clear()

            if self._stop_event.is_set():
                break

            # Extract snapshot copy under lock
            with self.lock:
                if len(self.telemetry_history) < 2:
                    continue
                history_snapshot = list(self.telemetry_history)
                frame_snapshot = self.latest_frame.copy() if self.latest_frame is not None else None
                objects_snapshot = list(self.latest_detected_objects)

            # Analyze with multimodal VLM
            result = self._query_llm_for_verification(history_snapshot, frame_snapshot, objects_snapshot)
            if result:
                with self.lock:
                    self.latest_verification = result

    def _query_llm_for_verification(
        self,
        history: List[Dict[str, Any]],
        frame: Optional[np.ndarray] = None,
        detected_objects: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Sends sliding window telemetry + image frame to local Ollama and parses JSON verdict."""
        if not history:
            return None

        recent = history[-1]
        avg_lid = round(sum(h["lid_angle"] for h in history) / len(history), 1)
        recent_activities = list(set(h["activity"] for h in history[-8:]))
        inside_ratio = sum(h["is_inside"] for h in history) / len(history)
        candidate_step = recent["heuristic_step"]
        candidate_name = recent["step_name"]
        detected_str = ", ".join(detected_objects) if detected_objects else "None"

        prompt = f"""You are the autonomous procedural supervisor for Bharatiya Antariksh Station (BAS).
Verify the active step and procedural compliance of this task:
- Step 0: IDLE (Box closed, waiting for operator to open container)
- Step 1: BOX_OPENED (Container lid opened >20 deg, operator preparing to extract object)
- Step 2: OBJECT_EXTRACTED (Object extracted outside container, held or manipulated)
- Step 3: OBJECT_RETURNED (Object placed back inside container)
- Step 4: COMPLETE (Container lid closed <20 deg, procedure finished)

Recent Physical & YOLO Observations:
- Average Container Lid Angle: {avg_lid} deg (Current: {recent['lid_angle']} deg)
- Component Inside Container Ratio: {round(inside_ratio, 2)}
- Current Activities: {', '.join(recent_activities)}
- Detected Physical Objects: {detected_str}
- Candidate Step: {candidate_step} ({candidate_name})
- YOLO Anomaly Flag: {recent['anomaly']}

Task:
1. Visually examine the scene to verify the actual physical step (0 to 4).
2. Check for errors (e.g. premature close, wrong sequence, skipped step, dropped object).
3. If an error is happening, state clearly WHAT IS WRONG and suggest the corrective action.

Reply strictly in valid JSON format with this exact schema:
{{
  "verified_step": <integer 0-4>,
  "step_name": "<IDLE|BOX_OPENED|OBJECT_EXTRACTED|OBJECT_RETURNED|COMPLETE>",
  "confidence": <float 0.0-1.0>,
  "anomaly_verdict": "NOMINAL" | "PROCEDURAL_ERROR" | "PHYSICAL_ANOMALY",
  "what_is_wrong": "<brief 1-sentence explanation of what error is occurring, or 'None'>",
  "corrective_action": "<brief 1-sentence corrective action, or 'Continue'>",
  "reason": "<1-sentence summary of visual evidence>"
}}"""

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.8,
                "num_predict": 120
            }
        }

        # Encode image frame to base64 if available
        if frame is not None:
            try:
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                b64_str = base64.b64encode(buf).decode("utf-8")
                payload["images"] = [b64_str]
            except Exception:
                pass

        t_start = time.time()
        try:
            req = urllib.request.Request(
                self.ollama_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            # 8s timeout allows responsive background completion
            with urllib.request.urlopen(req, timeout=8) as resp:
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

                step_name = names.get(step_val, parsed.get("step_name", candidate_name))
                anomaly_verdict = parsed.get("anomaly_verdict", "NOMINAL")
                what_is_wrong = parsed.get("what_is_wrong", "None")
                corrective_action = parsed.get("corrective_action", "Continue nominal procedure.")
                reason = parsed.get("reason", "Verified by Multimodal VLM supervisor.")

                return {
                    "verified_step": step_val,
                    "step_name": step_name,
                    "confidence": float(parsed.get("confidence", 0.92)),
                    "anomaly_verdict": anomaly_verdict,
                    "what_is_wrong": what_is_wrong,
                    "corrective_action": corrective_action,
                    "reason": reason,
                    "timestamp": time.time(),
                    "model": self.model_name,
                    "latency_s": latency
                }

        except Exception:
            # Fallback to deterministic verification if Ollama takes too long
            return self._fallback_deterministic_verification(
                candidate_step, candidate_name, avg_lid, inside_ratio, recent_activities, recent["anomaly"]
            )

    def _fallback_deterministic_verification(
        self,
        candidate_step: int,
        candidate_name: str,
        avg_lid: float,
        inside_ratio: float,
        recent_activities: List[str],
        anomaly_flag: str
    ) -> Dict[str, Any]:
        """Provides verified fallback consensus if Ollama is momentarily busy."""
        verified = candidate_step
        reason = "Deterministic kinematic fallback verification."
        anomaly_verdict = "NOMINAL"
        what_is_wrong = "None"
        corrective_action = "Continue nominal procedure."

        if anomaly_flag != "NONE":
            anomaly_verdict = "PROCEDURAL_ERROR"
            if "SKIP" in anomaly_flag or "PREMATURE" in anomaly_flag:
                what_is_wrong = "Container closed prematurely before completing extraction and return sequence."
                corrective_action = "Reopen the container lid and complete the step."
            elif "SEQ" in anomaly_flag:
                what_is_wrong = "Action performed out of procedural sequence."
                corrective_action = "Return component and follow standard sequence."
            else:
                what_is_wrong = f"Detected anomaly: {anomaly_flag}"
                corrective_action = "Check procedure checklist."

        elif candidate_step == 0 and avg_lid >= 20.0:
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
            "anomaly_verdict": anomaly_verdict,
            "what_is_wrong": what_is_wrong,
            "corrective_action": corrective_action,
            "reason": reason,
            "timestamp": time.time(),
            "model": "rule_consensus_fallback",
            "latency_s": 0.001
        }

    def close(self):
        """Stops the verifier background worker thread cleanly."""
        self._stop_event.set()
        self.trigger_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.5)
