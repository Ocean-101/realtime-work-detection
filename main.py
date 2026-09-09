"""
BAS Autonomous HAR System - Master Multi-Agent Orchestrator
Bridges all 8 specialized agents over the Digital Twin Memory Blackboard,
providing real-time procedural tracking, deterministic validation, voice alerts,
and dual-stream video output for the Bharatiya Antariksh Station (BAS).
"""

import sys
import os

# Ensure .venv dependencies are accessible even if user runs system python
_venv_site = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Lib", "site-packages")
if os.path.exists(_venv_site) and _venv_site not in sys.path:
    sys.path.insert(0, _venv_site)

import time
import argparse
import cv2
import numpy as np

# Core & Blackboard
from src.core.types import FSMStep, AnomalyType
from src.core.shared_memory import DigitalTwinBlackboard

# 8 Specialized Agents
from src.agents.perception_agent import PerceptionAgent
from src.agents.imu_agent import IMUAgent
from src.agents.fusion_agent import FusionAgent
from src.agents.har_agent import HARAgent
from src.agents.digital_twin_agent import DigitalTwinAgent
from src.agents.validation_agent import ValidationAgent
from src.agents.reasoning_agent import ReasoningAgent
from src.agents.monitoring_agent import MonitoringAgent


def run_orchestrator(
    source="clip.mp4",
    config_path=None,
    use_desktop_gui=False,
    show_window=True,
    enable_tts=True,
    enable_streaming=True,
    stream_port=8080,
    max_frames=None
):
    print("=" * 70)
    print("   BHARATIYA ANTARIKSH STATION (BAS) - ON-BOARD HAR MULTI-AGENT SYSTEM")
    print("   Autonomous Sequence Validation & Digital Twin Architecture")
    print("   ISRO SIH Problem Statement #26174")
    print("=" * 70)

    # Automatically select FSM config if not explicitly provided
    if config_path is None:
        if "nominal" in str(source) or "anomaly" in str(source):
            config_path = "configs/experiment_fsm.json"
        else:
            config_path = "configs/box_return_fsm.json"

    print(f"[Orchestrator] Active Procedure Protocol: {config_path}")

    # 1. Initialize Shared Memory Blackboard
    blackboard = DigitalTwinBlackboard()

    # 2. Instantiate the 8 Specialized Agents
    print("[Orchestrator] Initializing 8 Specialized Agents...")
    agent_perception = PerceptionAgent()
    agent_imu = IMUAgent()
    agent_fusion = FusionAgent()
    agent_har = HARAgent()
    agent_twin = DigitalTwinAgent()
    agent_validation = ValidationAgent(config_path=config_path)
    agent_reasoning = ReasoningAgent(config_path=config_path)
    agent_monitoring = MonitoringAgent(
        enable_tts=enable_tts,
        enable_streaming=enable_streaming,
        stream_port=stream_port
    )

    # Optional Desktop GUI
    desktop_gui = None
    if use_desktop_gui:
        try:
            from src.gui.mission_gui import MissionControlGUI
            desktop_gui = MissionControlGUI()
            print("[Orchestrator] Desktop Mission Control GUI initialized.")
        except Exception as e:
            print(f"[Orchestrator] Desktop GUI warning: {e}. Falling back to Web/Headless mode.")

    # 3. Setup Video Source
    if str(source).isdigit():
        cam_idx = int(source)
        cap = None
        # Try MSMF, DirectShow, and Default backends on Windows
        for backend in [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]:
            try:
                c = cv2.VideoCapture(cam_idx, backend)
                if c.isOpened():
                    ret_test, _ = c.read()
                    if ret_test:
                        cap = c
                        break
                c.release()
            except Exception:
                pass

        # If requested index failed (e.g. index 1), auto-probe Camera #0
        if cap is None and cam_idx != 0:
            print(f"[Warning] Camera index #{cam_idx} not responding. Probing Camera #0...")
            for backend in [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]:
                try:
                    c = cv2.VideoCapture(0, backend)
                    if c.isOpened():
                        ret_test, _ = c.read()
                        if ret_test:
                            cap = c
                            source = 0
                            print(f"[Orchestrator] Successfully engaged active Camera #0!")
                            break
                    c.release()
                except Exception:
                    pass

        if cap is None or not cap.isOpened():
            print(f"[Warning] Web Camera device #{source} could not be opened (disconnected or in use).")
            fallback_source = "clip1.mp4" if os.path.exists("clip1.mp4") else "clip.mp4"
            if not os.path.exists(fallback_source):
                print(f"[Orchestrator] Generating default simulation video...")
                from tools.generate_synthetic_data import generate_experiment_video
                generate_experiment_video(fallback_source, anomaly=False)
            print(f"[Orchestrator] Automatically falling back to video: '{fallback_source}'...")
            cap = cv2.VideoCapture(fallback_source)
        else:
            # Preserve native capture resolution and set zero-delay buffer
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            cw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ch = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[Orchestrator] Ingesting from Web Camera device #{source} ({cw}x{ch}, buffer=1 real-time)...")
    else:
        if not os.path.exists(source):
            print(f"[Orchestrator] Video file '{source}' not found. Generating default simulation...")
            from tools.generate_synthetic_data import generate_experiment_video
            generate_experiment_video(source, anomaly=False)
        cap = cv2.VideoCapture(source)
        print(f"[Orchestrator] Ingesting from Video File: {source}...")

    if not cap.isOpened():
        print(f"[Error] Could not open video source: {source}")
        agent_monitoring.close()
        return

    frame_id = 0
    t_start = time.time()
    prev_frame_time = t_start

    print("[Orchestrator] Multi-Agent Pipeline Running. Press Ctrl+C or close window to exit.")
    if enable_streaming:
        print(f"[*] Open Browser Mission Dashboard at: http://localhost:{stream_port}/")

    try:
        while True:
            # Check manual reset request from web client or desktop GUI
            web_reset = enable_streaming and agent_monitoring.check_reset_requested()
            gui_reset = desktop_gui and desktop_gui.check_reset_requested()
            if web_reset or gui_reset:
                print("\n[Orchestrator] Reset requested from UI. Restarting real-time test from Step 0...")
                if not str(source).isdigit():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                agent_validation.reset()
                agent_har.reset()
                agent_monitoring.reset()
                agent_perception.last_lid_angle = 0.0
                frame_id = 0
                prev_frame_time = time.time()

            ret, raw_frame = cap.read()
            if not ret:
                # On live webcam, momentarily dropped frames should not exit or loop
                if str(source).isdigit():
                    time.sleep(0.01)
                    continue

                # If experiment reached COMPLETE, hold final completed state for 4s so user/web client sees completion
                if agent_validation.current_step == FSMStep.COMPLETE:
                    print("\n[Orchestrator] Procedure COMPLETED successfully! Holding final state for 4 seconds...")
                    t_hold_end = time.time() + 4.0
                    while time.time() < t_hold_end:
                        if agent_monitoring.check_reset_requested():
                            break
                        time.sleep(0.1)

                # Loop video file for continuous exhibition/testing
                print("\n[Orchestrator] Looping experiment from Step 0...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                agent_validation.reset()
                agent_har.reset()
                agent_monitoring.reset()
                agent_perception.last_lid_angle = 0.0
                frame_id = 0
                ret, raw_frame = cap.read()
                if not ret:
                    time.sleep(0.033)
                    continue

            frame_id += 1
            if max_frames and frame_id > max_frames:
                break

            now = time.time()
            dt = max(1e-4, now - prev_frame_time)
            fps = 1.0 / dt
            prev_frame_time = now

            # ==========================================
            # PIPELINE EXECUTION ACROSS THE 8 AGENTS
            # ==========================================
            t_infer_start = time.time()

            # AGENT 1: Perception Agent (YOLOv8n + 3D HMR)
            objects_cam, pose_cam, lid_angle = agent_perception.process_frame(raw_frame)

            # AGENT 2: IMU Agent (128 Hz ingestion / virtual kinematics + ZUPT)
            imu_telemetry = agent_imu.update_from_vision(pose_cam)

            # AGENT 3: Fusion Agent (Constrained UKF + Stationary Rack Frame R + ROM)
            fused_pose, objects_rack = agent_fusion.fuse_kinematics(
                pose_cam, imu_telemetry, objects_cam, dt=dt
            )

            # AGENT 4: HAR Agent (AdaSpot RoI + 3D Hand-Object Interaction Engine)
            active_hoi, objects_state, current_activity = agent_har.evaluate_interactions(
                fused_pose, objects_rack, lid_angle
            )

            # AGENT 5: Digital Twin Agent (3D Scene Synchronization & Render)
            scene_graph = agent_twin.sync_scene_state(
                fused_pose, objects_state, lid_angle, agent_validation.current_step
            )
            twin_canvas = agent_twin.render_digital_twin_canvas(scene_graph)

            # AGENT 6: Validation Agent (Deterministic FSM + 15-frame Debounce)
            step, deb_count, anomaly, anomaly_msg, trans_event = agent_validation.evaluate_step(
                objects_state, lid_angle, active_hoi, frame_id
            )

            # AGENT 7: Reasoning & Guidance Agent (Next-step suggestions + Anomaly alerts)
            instruction, voice_alert = agent_reasoning.evaluate_guidance(
                step, anomaly, trans_event
            )

            latency_ms = (time.time() - t_infer_start) * 1000.0

            # Update Central Shared Memory (Digital Twin Memory Blackboard)
            blackboard.update_frame_metadata(frame_id, fps, latency_ms)
            blackboard.update_objects(objects_state, lid_angle)
            blackboard.update_pose(fused_pose)
            blackboard.update_hoi(active_hoi, current_activity)
            blackboard.update_fsm_state(step, deb_count, anomaly, anomaly_msg, instruction)

            # AGENT 8: Monitoring Agent (Dual Video + Offline TTS + JSONL + HUD Overlay + Action Logger)
            annotated_frame = agent_monitoring.process_egress(
                raw_frame=raw_frame,
                fused_pose=fused_pose,
                objects=objects_state,
                lid_angle=lid_angle,
                current_step=step,
                debounce_count=deb_count,
                anomaly=anomaly,
                instruction=instruction,
                voice_alert=voice_alert,
                transition_event=trans_event,
                frame_id=frame_id,
                fps=fps,
                latency_ms=latency_ms,
                twin_canvas=twin_canvas,
                current_activity=current_activity
            )

            # On-Screen Video Feed Display (Native GUI Window)
            if show_window:
                try:
                    cv2.imshow("BAS Mission Control - Live Stream & Action Detection", annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n[Orchestrator] Exit requested by user (Q key pressed).")
                        break
                    elif key == ord('r'):
                        print("\n[Orchestrator] Manual reset requested (R key pressed).")
                        if not str(source).isdigit():
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        agent_validation.reset()
                        agent_har.reset()
                        agent_monitoring.reset()
                        agent_perception.last_lid_angle = 0.0
                        frame_id = 0
                    elif key == ord('a'):
                        print("\n[Orchestrator] Triggering Offline LLM Procedural Audit (A key pressed)...")
                        from src.llm.offline_llm_analyzer import analyze_session
                        analyze_session()
                except Exception:
                    pass

            # Update Desktop Tkinter GUI if active
            if desktop_gui:
                if not desktop_gui.is_alive():
                    print("\n[Orchestrator] Desktop GUI window closed by user.")
                    break
                desktop_gui.update_frame(annotated_frame)
                log_snippet = f"[{trans_event}] {instruction}" if trans_event else None
                desktop_gui.update_state(int(step), instruction, anomaly.value, log_snippet)

            # Procedural Completion: Trigger Automated Offline Local LLM Audit
            if trans_event == "BOX_CLOSED":
                print("\n" + "=" * 70)
                print("[Orchestrator] PROCEDURE COMPLETED! Triggering Offline Local LLM Audit...")
                print("=" * 70)
                try:
                    from src.llm.offline_llm_analyzer import analyze_session
                    audit_res = analyze_session()
                    if audit_res.get("report"):
                        print("\n[OFFLINE LLM AUDIT REPORT]:\n" + audit_res["report"])
                except Exception as e:
                    print(f"[Orchestrator] LLM Audit notice: {e}")

            # Console status line (updated every 30 frames)
            if frame_id % 30 == 0:
                print(f"[Frame {frame_id:04d}] Step: {step.name:15s} | Debounce: {deb_count:02d}/15 | "
                      f"Doing: {current_activity:22s} | Next: {instruction[:25]} | FPS: {fps:.1f}")

    except KeyboardInterrupt:
        print("\n[Orchestrator] Shutdown requested by user.")
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        agent_monitoring.close()
        
        # Calculate final telemetry compression audit
        total_time_sec = time.time() - t_start
        ratio = agent_monitoring.telemetry.calculate_compression_ratio(total_time_sec)
        print("=" * 70)
        print("MISSION TELEMETRY AUDIT")
        print(f"Total Session Duration : {total_time_sec:.1f} seconds")
        print(f"Total Frames Processed : {frame_id} frames")
        print(f"Structured JSONL Size  : {agent_monitoring.telemetry.total_bytes_written} bytes")
        print(f"Empirical Compression : {ratio:,.1f} : 1")
        print(f"Telemetry Log Saved At : {agent_monitoring.telemetry.output_path}")
        print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BAS Multi-Agent HAR Orchestrator")
    parser.add_argument("--source", default="clip.mp4",
                        help="Video source: camera index (0) or path to MP4 (default: clip.mp4)")
    parser.add_argument("--config", default=None,
                        help="Path to FSM configuration file (default: auto-detected)")
    parser.add_argument("--desktop-gui", action="store_true",
                        help="Launch native desktop Tkinter GUI")
    parser.add_argument("--no-window", action="store_true",
                        help="Disable native OpenCV desktop display window")
    parser.add_argument("--no-tts", action="store_true",
                        help="Disable voice synthesis")
    parser.add_argument("--no-stream", action="store_true",
                        help="Disable IP streaming server")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port for Web Mission Dashboard & RTSP/HTTP stream")
    parser.add_argument("--frames", type=int, default=None,
                        help="Maximum frames to process (useful for automated testing)")
    args = parser.parse_args()

    run_orchestrator(
        source=args.source,
        config_path=args.config,
        use_desktop_gui=args.desktop_gui,
        show_window=not args.no_window,
        enable_tts=not args.no_tts,
        enable_streaming=not args.no_stream,
        stream_port=args.port,
        max_frames=args.frames
    )
