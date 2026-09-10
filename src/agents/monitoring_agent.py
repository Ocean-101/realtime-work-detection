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
from src.telemetry.csv_logger import RealtimeCSVTelemetryLogger
from src.streaming.video_pipeline import DualVideoPipeline


class MonitoringAgent:
    """Master Egress and Mission Telemetry Coordinator."""

    def __init__(
        self,
        enable_tts: bool = True,
        enable_streaming: bool = True,
        stream_port: int = 8080,
        output_video_path: Optional[str] = None,
        output_telemetry_path: Optional[str] = None,
        realtime_feed_dir: str = "realtime_feed"
    ):
        self.tts = OfflineTTS() if enable_tts else None
        self.telemetry = JSONLTelemetryLogger(output_telemetry_path)
        self.csv_logger = RealtimeCSVTelemetryLogger(realtime_feed_dir=realtime_feed_dir)
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
        current_activity: str = "IDLE",
        llm_verification: Optional[Dict[str, Any]] = None,
        source_type: str = "LIVE_WEBCAM",
        is_step_correct: bool = True,
        step_verdict: str = "CORRECT (NOMINAL)",
        experiment_id: str = "BAS-EXP-BOX-RETURN"
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

        # 3. Emit Frame-by-Frame CSV Telemetry for 3D & Digital Twin
        self.csv_logger.log_frame(
            frame_id=frame_id,
            fps=fps,
            step=current_step,
            activity=current_activity,
            anomaly=anomaly,
            instruction=instruction,
            lid_angle=lid_angle,
            pose=fused_pose,
            objects=objects,
            llm_verification=llm_verification
        )

        # 4. Emit Milestone Telemetry Records
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

        # 5. Composite Live Stream HUD Overlay
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
            current_activity=current_activity,
            llm_verification=llm_verification,
            source_type=source_type,
            is_step_correct=is_step_correct,
            step_verdict=step_verdict
        )

        # 6. Dispatch to Dual Video Pipeline (Local MP4 + RTSP Stream + Web API)
        if self.video_pipeline:
            self.video_pipeline.set_active_source_type(source_type)
            self.video_pipeline.set_active_experiment_id(experiment_id)
            self.video_pipeline.write_frame(annotated_frame, telemetry={
                "step": int(current_step),
                "step_name": current_step.name,
                "debounce": debounce_count,
                "debounce_max": 6,
                "anomaly": anomaly.value,
                "instruction": instruction,
                "transition_event": transition_event,
                "fps": round(fps, 1),
                "latency_ms": round(latency_ms, 1),
                "lid_angle": round(lid_angle, 1),
                "frame_id": frame_id,
                "what_i_am_doing": current_activity,
                "what_i_have_to_do": instruction,
                "source_type": source_type,
                "is_step_correct": is_step_correct,
                "step_verdict": step_verdict,
                "experiment_id": experiment_id
            })

        return annotated_frame

    def check_source_switch_requested(self) -> Optional[str]:
        """Returns requested source ('0' or 'clip1.mp4') if switched via UI, else None."""
        if self.video_pipeline:
            return self.video_pipeline.check_and_clear_source_switch()
        return None

    def check_experiment_switch_requested(self) -> Optional[str]:
        """Returns requested protocol path if switched via UI, else None."""
        if self.video_pipeline:
            return self.video_pipeline.check_and_clear_experiment_switch()
        return None

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
        current_activity: str = "IDLE",
        llm_verification: Optional[Dict[str, Any]] = None,
        source_type: str = "LIVE_WEBCAM",
        is_step_correct: bool = True,
        step_verdict: str = "CORRECT (NOMINAL)"
    ) -> np.ndarray:
        frame = raw_frame.copy()
        h, w, _ = frame.shape

        # Adaptive layout scaling based on resolution
        scale = max(0.40, min(0.70, w / 1280.0))
        thick = 1 if w < 1000 else 2
        banner_h = max(34, min(50, int(h * 0.065)))
        bot_h = max(32, min(48, int(h * 0.060)))

        # 1. Draw 2D Object Bounding Boxes & Staggered Badges (Zero Overlap)
        color_map = {
            "container_box": (160, 160, 160),
            "container_lid": (0, 240, 200),
            "component_box": (255, 140, 0),
            "red_box": (60, 70, 255),
            "yellow_box": (0, 230, 255),
            "person": (255, 215, 0),
            "cell phone": (0, 255, 128),
            "bottle": (255, 105, 180),
            "cup": (255, 140, 0),
            "book": (186, 85, 211),
            "laptop": (0, 191, 255),
            "scissors": (255, 20, 147),
            "mouse": (50, 205, 50),
            "chair": (169, 169, 169)
        }

        def get_obj_color(obj_name: str) -> Tuple[int, int, int]:
            base = obj_name.split("_")[0]
            if base in color_map:
                return color_map[base]
            h_val = int((abs(hash(obj_name)) * 37) % 180)
            hsv_pix = np.uint8([[[h_val, 220, 240]]])
            bgr = cv2.cvtColor(hsv_pix, cv2.COLOR_HSV2BGR)[0][0]
            return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

        occupied_badge_rects: List[Tuple[int, int, int, int]] = []

        for name, obj in objects.items():
            if obj.bbox:
                bx1, by1 = int(obj.bbox.xmin), int(obj.bbox.ymin)
                bx2, by2 = int(obj.bbox.xmax), int(obj.bbox.ymax)
                col = get_obj_color(name)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), col, max(1, int(1.5 * scale * 2)))

                # Text format: show confidence percentage if detected by neural model
                if obj.bbox.confidence and obj.bbox.confidence > 0.05 and name not in ("container_box", "container_lid"):
                    status_txt = f"{obj.name.upper()} {int(obj.bbox.confidence * 100)}%"
                else:
                    status_txt = f"{obj.name.upper()} [{obj.state.value}]"
                font_scale = max(0.34, scale * 0.85)
                (tw, th), _ = cv2.getTextSize(status_txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                pad_x, pad_y = 6, 4
                badge_w = tw + pad_x * 2
                badge_h = th + pad_y * 2

                # Target position: Above the box if there is room below top banner, otherwise inside
                rx1 = max(4, min(w - badge_w - 4, bx1))
                if by1 - badge_h - 2 >= banner_h + 2:
                    ry1 = by1 - badge_h - 2
                else:
                    ry1 = min(h - bot_h - badge_h - 2, by1 + 4)

                # Collision resolution with previously placed badges
                for (ox1, oy1, ox2, oy2) in occupied_badge_rects:
                    if not (rx1 + badge_w < ox1 or rx1 > ox2 or ry1 + badge_h < oy1 or ry1 > oy2):
                        ry1 = min(h - bot_h - badge_h - 2, oy2 + 3)

                rx2 = rx1 + badge_w
                ry2 = ry1 + badge_h
                occupied_badge_rects.append((rx1, ry1, rx2, ry2))

                # Draw badge background and high-contrast text
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), col, -1)
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (10, 10, 10), 1)
                cv2.putText(
                    frame, status_txt,
                    (rx1 + pad_x, ry1 + th + pad_y - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (10, 10, 10), 1, cv2.LINE_AA
                )

        # 2. Draw Hand & 3D Pose Keypoints with pill label
        wrist = pose.joints.get("wrist")
        if wrist:
            wx = int(self.video_pipeline.resolution[0]/2 + wrist.pos_camera.x * 400) if self.video_pipeline else w//2
            wy = int(self.video_pipeline.resolution[1]/2 + wrist.pos_camera.y * 400) if self.video_pipeline else h//2
            wx = max(15, min(w - 15, wx))
            wy = max(banner_h + 15, min(h - bot_h - 15, wy))

            cv2.circle(frame, (wx, wy), max(5, int(8 * scale)), (0, 240, 255), -1)
            cv2.circle(frame, (wx, wy), max(7, int(11 * scale)), (255, 255, 255), 1)

            # Clean pill badge for astronaut wrist
            w_txt = "ASTRONAUT WRIST"
            w_scale = max(0.32, scale * 0.78)
            (wtw, wth), _ = cv2.getTextSize(w_txt, cv2.FONT_HERSHEY_SIMPLEX, w_scale, 1)
            wrx1 = max(4, min(w - wtw - 10, wx + 10))
            wry1 = max(banner_h + 2, min(h - bot_h - wth - 8, wy - wth // 2 - 3))
            wrx2 = wrx1 + wtw + 8
            wry2 = wry1 + wth + 6

            for (ox1, oy1, ox2, oy2) in occupied_badge_rects:
                if not (wrx1 + (wrx2 - wrx1) < ox1 or wrx1 > ox2 or wry1 + (wry2 - wry1) < oy1 or wry1 > oy2):
                    wry1 = min(h - bot_h - (wry2 - wry1) - 2, oy2 + 3)
                    wry2 = wry1 + wth + 6

            sub = frame[wry1:wry2, wrx1:wrx2]
            if sub.size > 0:
                dark_rect = np.full_like(sub, 20)
                cv2.addWeighted(sub, 0.25, dark_rect, 0.75, 0, sub)
                cv2.rectangle(frame, (wrx1, wry1), (wrx2, wry2), (0, 240, 255), 1)
                cv2.putText(
                    frame, w_txt,
                    (wrx1 + 4, wry1 + wth + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, w_scale, (0, 240, 255), 1, cv2.LINE_AA
                )

        # 3. Streamlined Minimalist HUD Overlay (Header & Footer)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (8, 12, 18), -1)
        cv2.rectangle(overlay, (0, h - bot_h), (w, h), (8, 12, 18), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Separator lines
        sep_col = (0, 230, 120) if current_step == FSMStep.COMPLETE else ((0, 60, 220) if anomaly != AnomalyType.NONE else (0, 180, 230))
        cv2.line(frame, (0, banner_h), (w, banner_h), sep_col, 1)
        cv2.line(frame, (0, h - bot_h), (w, h - bot_h), sep_col, 1)

        # Header Zone 1 (Left): Mission Identifier & Real-Time Source Indicator
        src_tag = "LIVE #0" if source_type == "LIVE_WEBCAM" else "DEMO CLIP"
        title_txt = f"BAS HAR | {src_tag}"
        title_scale = max(0.38, scale * 0.90)
        (tw, th), _ = cv2.getTextSize(title_txt, cv2.FONT_HERSHEY_SIMPLEX, title_scale, thick)
        title_y = (banner_h + th) // 2
        title_col = (0, 240, 150) if source_type == "LIVE_WEBCAM" else (255, 220, 100)
        cv2.putText(frame, title_txt, (14, title_y), cv2.FONT_HERSHEY_SIMPLEX, title_scale, title_col, thick, cv2.LINE_AA)
        left_bound = 14 + tw + 14

        # Header Zone 3 (Right): Telemetry Status
        status_scale = max(0.34, scale * 0.82)
        llm_s_tag = ""
        if llm_verification:
            llm_v_step = llm_verification.get("verified_step", int(current_step))
            llm_s_tag = f" | LLM: S{llm_v_step}"
        status_txt = f"FPS: {fps:.0f} | LID: {lid_angle:.0f}d | DEB: {debounce_count}/6{llm_s_tag}"
        (sw, sh), _ = cv2.getTextSize(status_txt, cv2.FONT_HERSHEY_SIMPLEX, status_scale, 1)
        status_x = max(left_bound + 10, w - sw - 14)
        status_y = (banner_h + sh) // 2
        cv2.putText(frame, status_txt, (status_x, status_y), cv2.FONT_HERSHEY_SIMPLEX, status_scale, (180, 215, 235), 1, cv2.LINE_AA)
        right_bound = status_x - 14

        # Header Zone 2 (Center): Active Procedure Step (Guaranteed No Overlap)
        step_labels = {
            FSMStep.IDLE: "S0: STANDBY",
            FSMStep.BOX_OPENED: "S1: CONTAINER OPEN",
            FSMStep.OBJECT_EXTRACTED: "S2: OBJECT EXTRACTED",
            FSMStep.OBJECT_RETURNED: "S3: OBJECT RETURNED",
            FSMStep.COMPLETE: "S4: MISSION COMPLETE"
        }
        active_step_txt = step_labels.get(current_step, current_step.name)
        step_scale = max(0.36, scale * 0.86)
        (step_w, step_h), _ = cv2.getTextSize(active_step_txt, cv2.FONT_HERSHEY_SIMPLEX, step_scale, 1)

        # Calculate center position & clamp between left and right bounds
        ideal_center_x = (w - step_w) // 2
        center_x = max(left_bound, min(right_bound - step_w, ideal_center_x))
        if center_x + step_w <= right_bound:
            step_badge_col = (0, 230, 120) if current_step == FSMStep.COMPLETE else (0, 210, 255)
            pill_pad_x = 8
            pill_pad_y = 4
            px1 = center_x - pill_pad_x
            py1 = max(2, (banner_h - (step_h + pill_pad_y * 2)) // 2)
            px2 = center_x + step_w + pill_pad_x
            py2 = min(banner_h - 2, py1 + step_h + pill_pad_y * 2)
            cv2.rectangle(frame, (px1, py1), (px2, py2), (20, 30, 45), -1)
            cv2.rectangle(frame, (px1, py1), (px2, py2), step_badge_col, 1)
            cv2.putText(
                frame, active_step_txt,
                (center_x, (banner_h + step_h) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, step_scale, step_badge_col, 1, cv2.LINE_AA
            )

        # Footer: Action Bar & Step Verification Badge
        act_scale = max(0.36, scale * 0.85)
        act_display = current_activity if current_activity != "IDLE" else "OBSERVING"
        doing_txt = f"DOING: {act_display}"
        (dw, dh), _ = cv2.getTextSize(doing_txt, cv2.FONT_HERSHEY_SIMPLEX, act_scale, 1)
        bot_y = h - (bot_h - dh) // 2 - 2

        cv2.rectangle(frame, (10, h - bot_h + 4), (10 + dw + 12, h - 4), (20, 35, 50), -1)
        cv2.rectangle(frame, (10, h - bot_h + 4), (10 + dw + 12, h - 4), (0, 220, 255), 1)
        cv2.putText(frame, doing_txt, (16, bot_y), cv2.FONT_HERSHEY_SIMPLEX, act_scale, (0, 230, 255), 1, cv2.LINE_AA)

        doing_end = 10 + dw + 18

        # Real-time Step Correctness Pill
        verd_txt = "STEP: OK" if is_step_correct else "STEP: ERR"
        verd_col = (0, 230, 120) if is_step_correct else (40, 60, 255)
        (vw, vh), _ = cv2.getTextSize(verd_txt, cv2.FONT_HERSHEY_SIMPLEX, act_scale, 1)
        vx1 = doing_end
        cv2.rectangle(frame, (vx1, h - bot_h + 4), (vx1 + vw + 12, h - 4), (15, 30, 25) if is_step_correct else (45, 18, 22), -1)
        cv2.rectangle(frame, (vx1, h - bot_h + 4), (vx1 + vw + 12, h - 4), verd_col, 1)
        cv2.putText(frame, verd_txt, (vx1 + 6, bot_y), cv2.FONT_HERSHEY_SIMPLEX, act_scale, verd_col, 1, cv2.LINE_AA)

        guide_start = vx1 + vw + 18

        # Guidance / Alert Text on the right with safe truncation
        guide_col = (80, 100, 255) if anomaly != AnomalyType.NONE else (255, 255, 255)
        prefix = "ALERT: " if anomaly != AnomalyType.NONE else "NEXT: "
        full_guide = f"{prefix}{instruction}"

        guide_scale = max(0.34, scale * 0.82)
        avail_width = w - guide_start - 16

        (gw, gh), _ = cv2.getTextSize(full_guide, cv2.FONT_HERSHEY_SIMPLEX, guide_scale, 1)
        display_guide = full_guide
        if gw > avail_width and avail_width > 60:
            while len(display_guide) > 8:
                display_guide = display_guide[:-4] + "..."
                (gw, gh), _ = cv2.getTextSize(display_guide, cv2.FONT_HERSHEY_SIMPLEX, guide_scale, 1)
                if gw <= avail_width:
                    break

        if avail_width > 40:
            cv2.putText(frame, display_guide, (guide_start, bot_y), cv2.FONT_HERSHEY_SIMPLEX, guide_scale, guide_col, 1, cv2.LINE_AA)

        return frame

    def close(self):
        if self.action_logger:
            self.action_logger.save()
        if self.video_pipeline:
            self.video_pipeline.close()
        if self.telemetry:
            self.telemetry.close()
        if self.csv_logger:
            self.csv_logger.close()
        if self.tts:
            self.tts.shutdown()
