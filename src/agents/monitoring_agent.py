"""
BAS Autonomous HAR System - Agent 8: Monitoring Agent
Coordinates mission outputs: Dual Video Pipeline (Local NVMe + RTSP IP Streaming),
Offline Speech Synthesis (TTS), Structured JSONL Telemetry, and Live Video Overlays.
"""

import time
import cv2
import numpy as np
from typing import Dict, Any, Optional
from src.core.types import (
    AstronautPose3D,
    ExperimentObject,
    FSMStep,
    AnomalyType,
    EntityState
)
from src.audio.offline_tts import OfflineTTS
from src.telemetry.jsonl_logger import JSONLTelemetryLogger
from src.telemetry.action_session_logger import ActionSessionLogger
from src.streaming.video_pipeline import DualVideoPipeline


class MonitoringAgent:
    """Master Egress and Mission Telemetry Coordinator."""

    def __init__(
        self,
        enable_tts: bool = True,
        enable_streaming: bool = True,
        stream_port: int = 8080,
        output_video_path: Optional[str] = None,
        output_telemetry_path: Optional[str] = None
    ):
        self.tts = OfflineTTS() if enable_tts else None
        self.telemetry = JSONLTelemetryLogger(output_telemetry_path)
        self.action_logger = ActionSessionLogger()
        self.video_pipeline = DualVideoPipeline(
            local_output_path=output_video_path,
            stream_port=stream_port
        ) if enable_streaming else None

        self.last_step_logged: Optional[FSMStep] = None

    def reset(self):
        """Resets action session logger and telemetry for new experiment run."""
        if self.action_logger:
            self.action_logger.reset()

    def check_reset_requested(self) -> bool:
        """Checks if web client requested an experiment reset."""
        if self.video_pipeline:
            return self.video_pipeline.check_and_clear_reset()
        return False

    def process_egress(
        self,
        raw_frame: np.ndarray,
        fused_pose: AstronautPose3D,
        objects: Dict[str, ExperimentObject],
        lid_angle: float,
        current_step: FSMStep,
        debounce_count: int,
        anomaly: AnomalyType,
        instruction: str,
        voice_alert: Optional[str],
        transition_event: Optional[str],
        frame_id: int,
        fps: float,
        latency_ms: float,
        twin_canvas: Optional[np.ndarray] = None,
        current_activity: str = "IDLE"
    ) -> np.ndarray:
        """
        Synthesizes audio alerts, logs session actions to JSON, and renders live HUD overlays.
        Returns the annotated display frame.
        """
        # 1. Trigger Voice Alerts / Spoken Guidance
        if voice_alert and self.tts:
            is_priority = anomaly != AnomalyType.NONE
            self.tts.speak(voice_alert, priority=is_priority)

        # 2. Update Structured Action Session Logger (What I'm doing vs What I have to do)
        timestamp_sec = frame_id / max(1.0, fps)
        self.action_logger.update(
            frame_id=frame_id,
            timestamp_sec=timestamp_sec,
            step=int(current_step),
            step_name=current_step.name,
            what_i_am_doing=current_activity,
            what_i_have_to_do=instruction,
            lid_angle=lid_angle,
            anomaly=anomaly.value
        )

        # 3. Emit Telemetry Records
        if transition_event is not None:
            self.telemetry.log_event(
                frame_id=frame_id,
                state_id=int(current_step),
                state_name=current_step.name,
                event_name=transition_event,
                status="SUCCESS",
                anomaly=anomaly.value,
                tts_prompt=instruction,
                details={"lid_angle": round(lid_angle, 1), "activity": current_activity}
            )
        elif anomaly != AnomalyType.NONE:
            self.telemetry.log_event(
                frame_id=frame_id,
                state_id=int(current_step),
                state_name=current_step.name,
                event_name="ANOMALY_DETECTED",
                status="ANOMALY",
                anomaly=anomaly.value,
                tts_prompt=instruction
            )

        # Periodic Heartbeat
        self.telemetry.log_heartbeat(frame_id, int(current_step), current_step.name, fps, latency_ms)

        # 4. Composite Live Stream HUD Overlay
        annotated_frame = self._draw_hud(
            raw_frame=raw_frame,
            pose=fused_pose,
            objects=objects,
            lid_angle=lid_angle,
            current_step=current_step,
            debounce_count=debounce_count,
            anomaly=anomaly,
            instruction=instruction,
            fps=fps,
            twin_canvas=twin_canvas,
            current_activity=current_activity
        )

        # 5. Dispatch to Dual Video Pipeline (Local MP4 + RTSP Stream + Web API)
        if self.video_pipeline:
            self.video_pipeline.write_frame(annotated_frame, telemetry={
                "step": int(current_step),
                "step_name": current_step.name,
                "debounce": debounce_count,
                "debounce_max": 12,
                "anomaly": anomaly.value,
                "instruction": instruction,
                "transition_event": transition_event,
                "fps": round(fps, 1),
                "latency_ms": round(latency_ms, 1),
                "lid_angle": round(lid_angle, 1),
                "frame_id": frame_id,
                "what_i_am_doing": current_activity,
                "what_i_have_to_do": instruction
            })

        return annotated_frame

    def _draw_hud(
        self,
        raw_frame: np.ndarray,
        pose: AstronautPose3D,
        objects: Dict[str, ExperimentObject],
        lid_angle: float,
        current_step: FSMStep,
        debounce_count: int,
        anomaly: AnomalyType,
        instruction: str,
        fps: float,
        twin_canvas: Optional[np.ndarray] = None,
        current_activity: str = "IDLE"
    ) -> np.ndarray:
        frame = raw_frame.copy()
        h, w, _ = frame.shape

        # 1. Draw 2D Object Bounding Boxes
        color_map = {
            "container_box": (160, 160, 160),
            "container_lid": (0, 240, 200),
            "component_box": (255, 140, 0),
            "red_box": (40, 40, 240),
            "yellow_box": (20, 220, 240)
        }

        for name, obj in objects.items():
            if obj.bbox:
                bx1, by1 = int(obj.bbox.xmin), int(obj.bbox.ymin)
                bx2, by2 = int(obj.bbox.xmax), int(obj.bbox.ymax)
                col = color_map.get(name, (200, 200, 200))
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), col, 2)
                
                # Label badge
                status_txt = f"{obj.name.upper()} [{obj.state.value}]"
                cv2.rectangle(frame, (bx1, max(0, by1 - 22)), (bx1 + len(status_txt)*9 + 10, by1), col, -1)
                cv2.putText(frame, status_txt, (bx1 + 5, max(15, by1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (10, 10, 10), 1)

        # 2. Draw Hand & 3D Pose Keypoints
        wrist = pose.joints.get("wrist")
        if wrist:
            # Reconstruct pixel from wrist camera coords
            wx = int(self.video_pipeline.resolution[0]/2 + wrist.pos_camera.x * 400) if self.video_pipeline else w//2
            wy = int(self.video_pipeline.resolution[1]/2 + wrist.pos_camera.y * 400) if self.video_pipeline else h//2
            wx = max(10, min(w - 10, wx))
            wy = max(10, min(h - 10, wy))
            cv2.circle(frame, (wx, wy), 9, (0, 240, 255), -1)
            cv2.putText(frame, "ASTRONAUT WRIST", (wx + 12, wy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 240, 255), 1)

        # 3. Top Mission Banner & Bottom Action Bar (Semi-transparent HUD overlay)
        banner_h = 75
        bot_h = 68
        bot_col = (12, 16, 24)
        if anomaly != AnomalyType.NONE:
            bot_col = (35, 10, 25)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (12, 16, 24), -1)
        cv2.rectangle(overlay, (0, h - bot_h), (w, h), bot_col, -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

        cv2.line(frame, (0, banner_h), (w, banner_h), (0, 180, 255), 2)
        border_col = (0, 0, 255) if anomaly != AnomalyType.NONE else (0, 180, 255)
        cv2.line(frame, (0, h - bot_h), (w, h - bot_h), border_col, 2)

        # Logo / Title
        cv2.putText(frame, "BHARATIYA ANTARIKSH STATION (BAS) | ON-BOARD HAR ASSISTANT", (20, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        
        # Step Progress Pills (Box Object Extraction & Return Procedure)
        steps = [
            ("S0: IDLE", FSMStep.IDLE),
            ("S1: OPEN BOX", FSMStep.BOX_OPENED),
            ("S2: EXTRACT", FSMStep.OBJECT_EXTRACTED),
            ("S3: RETURN", FSMStep.OBJECT_RETURNED),
            ("S4: COMPLETE", FSMStep.COMPLETE)
        ]
        
        px = 20
        for label, step_val in steps:
            if current_step == step_val:
                p_col = (0, 220, 255) # Active
                p_txt_col = (10, 10, 10)
            elif current_step > step_val:
                p_col = (40, 180, 50) # Completed
                p_txt_col = (255, 255, 255)
            else:
                p_col = (45, 50, 60) # Future
                p_txt_col = (160, 165, 175)

            p_w = len(label) * 9 + 20
            cv2.rectangle(frame, (px, 38), (px + p_w, 64), p_col, -1)
            cv2.rectangle(frame, (px, 38), (px + p_w, 64), (180, 190, 200), 1)
            cv2.putText(frame, label, (px + 8, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.42, p_txt_col, 1)
            px += p_w + 12

        # FPS & Telemetry status in top right
        cv2.putText(frame, f"FPS: {fps:.1f} | DEBOUNCE: {debounce_count}/12", (w - 240, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1)
        
        # Debounce progress bar
        bar_w = 140
        bar_h = 8
        bx = w - 240
        by = 34
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (40, 45, 55), -1)
        fill_w = int((min(12, debounce_count) / 12.0) * bar_w)
        if fill_w > 0:
            cv2.rectangle(frame, (bx, by), (bx + fill_w, by + bar_h), (0, 240, 255), -1)
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (100, 120, 140), 1)

        cv2.putText(frame, f"S-BAND JSONL: ACTIVE | LID: {lid_angle:.0f}deg", (w - 240, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 190, 200), 1)
        
        # Row 1: What I am Doing (Active Human Activity)
        cv2.rectangle(frame, (16, h - 58), (196, h - 36), (0, 210, 255), -1)
        cv2.putText(frame, "WHAT I AM DOING", (22, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (10, 10, 10), 1)
        act_display = current_activity if current_activity != "IDLE" else "OBSERVING WORKSPACE (IDLE)"
        cv2.putText(frame, act_display, (208, h - 41), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 240, 255), 2)

        # Row 2: What I Have to Do (Next Procedural Step Guidance)
        req_badge_col = (0, 180, 60) if anomaly == AnomalyType.NONE else (40, 40, 240)
        cv2.rectangle(frame, (16, h - 28), (196, h - 6), req_badge_col, -1)
        req_badge_text = "WHAT TO DO NEXT" if anomaly == AnomalyType.NONE else f"ALERT [{anomaly.value}]"
        cv2.putText(frame, req_badge_text, (22, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
        cv2.putText(frame, instruction, (208, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

        # 5. Compact Picture-in-Picture Digital Twin Canvas in bottom right
        if twin_canvas is not None:
            th, tw = twin_canvas.shape[:2]
            pip_w = min(260, max(160, int(w * 0.22)))
            pip_h = int(th * (pip_w / tw))
            pip_resized = cv2.resize(twin_canvas, (pip_w, pip_h))
            
            # Place neatly in bottom right above bottom action bar
            py1 = h - bot_h - pip_h - 10
            py2 = py1 + pip_h
            px1 = w - pip_w - 10
            px2 = px1 + pip_w

            frame[py1:py2, px1:px2] = pip_resized
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 220, 255), 2)
            cv2.putText(frame, "3D DIGITAL TWIN", (px1 + 6, py1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 220, 255), 1)

        return frame

    def close(self):
        if self.action_logger:
            self.action_logger.save()
        if self.video_pipeline:
            self.video_pipeline.close()
        if self.telemetry:
            self.telemetry.close()
        if self.tts:
            self.tts.shutdown()
