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
        self.video_pipeline = DualVideoPipeline(
            local_output_path=output_video_path,
            stream_port=stream_port
        ) if enable_streaming else None

        self.last_step_logged: Optional[FSMStep] = None

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
        twin_canvas: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Synthesizes audio alerts, emits JSONL telemetry, and renders live HUD overlays.
        Returns the annotated display frame.
        """
        # 1. Trigger Voice Alerts / Spoken Guidance
        if voice_alert and self.tts:
            is_priority = anomaly != AnomalyType.NONE
            self.tts.speak(voice_alert, priority=is_priority)

        # 2. Emit Telemetry Records
        if transition_event is not None:
            self.telemetry.log_event(
                frame_id=frame_id,
                state_id=int(current_step),
                state_name=current_step.name,
                event_name=transition_event,
                status="SUCCESS",
                anomaly=anomaly.value,
                tts_prompt=instruction,
                details={"lid_angle": round(lid_angle, 1)}
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

        # 3. Composite Live Stream HUD Overlay
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
            twin_canvas=twin_canvas
        )

        # 4. Dispatch to Dual Video Pipeline (Local MP4 + RTSP Stream)
        if self.video_pipeline:
            self.video_pipeline.write_frame(annotated_frame)

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
        twin_canvas: Optional[np.ndarray] = None
    ) -> np.ndarray:
        frame = raw_frame.copy()
        h, w, _ = frame.shape

        # 1. Draw 2D Object Bounding Boxes
        color_map = {
            "container_box": (160, 160, 160),
            "container_lid": (0, 240, 200),
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
            wx = int(w * 0.5 + wrist.pos_camera.x * 600)
            wy = int(h * 0.5 + wrist.pos_camera.y * 600)
            wx = max(10, min(w - 10, wx))
            wy = max(10, min(h - 10, wy))
            cv2.circle(frame, (wx, wy), 9, (255, 180, 0), -1)
            cv2.circle(frame, (wx, wy), 11, (255, 255, 255), 2)
            cv2.putText(frame, "ASTRONAUT WRIST", (wx + 15, wy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

        # 3. Top Mission Banner
        banner_h = 75
        cv2.rectangle(frame, (0, 0), (w, banner_h), (15, 18, 22), -1)
        cv2.line(frame, (0, banner_h), (w, banner_h), (0, 180, 255), 2)

        # Logo / Title
        cv2.putText(frame, "BHARATIYA ANTARIKSH STATION (BAS) | ON-BOARD HAR ASSISTANT", (20, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        
        # Step Progress Pills
        steps = [("S0: IDLE", FSMStep.IDLE), 
                 ("S1: OPEN", FSMStep.CONTAINER_OPEN), 
                 ("S2: EXTRACT RED", FSMStep.RED_EXTRACTED), 
                 ("S3: EXTRACT YEL", FSMStep.COMPLETE)]
        
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
        cv2.putText(frame, f"FPS: {fps:.1f} | DEBOUNCE: {debounce_count}/15", (w - 240, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 200), 1)
        cv2.putText(frame, f"S-BAND JSONL: ACTIVE", (w - 240, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 190, 200), 1)

        # 4. Bottom Instruction & Voice Prompt Bar
        bot_h = 55
        bot_col = (15, 18, 22)
        if anomaly != AnomalyType.NONE:
            bot_col = (20, 20, 160) # Red warning banner
        
        cv2.rectangle(frame, (0, h - bot_h), (w, h), bot_col, -1)
        cv2.line(frame, (0, h - bot_h), (w, h - bot_h), (0, 0, 255) if anomaly != AnomalyType.NONE else (0, 180, 255), 2)
        
        prefix = "ALERT: " if anomaly != AnomalyType.NONE else "NEXT STEP: "
        cv2.putText(frame, f"{prefix}{instruction}", (20, h - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

        # 5. Optional Picture-in-Picture Digital Twin Canvas in bottom right
        if twin_canvas is not None:
            th, tw = twin_canvas.shape[:2]
            pip_w = 340
            pip_h = int(th * (pip_w / tw))
            pip_resized = cv2.resize(twin_canvas, (pip_w, pip_h))
            
            # Place in bottom right above instruction bar
            py1 = h - bot_h - pip_h - 10
            py2 = py1 + pip_h
            px1 = w - pip_w - 10
            px2 = px1 + pip_w

            frame[py1:py2, px1:px2] = pip_resized
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 220, 255), 2)

        return frame

    def close(self):
        if self.video_pipeline:
            self.video_pipeline.close()
        if self.telemetry:
            self.telemetry.close()
        if self.tts:
            self.tts.shutdown()
