"""
BAS Autonomous HAR System - Desktop Mission Control GUI (Tkinter)
Native desktop dashboard displaying live annotated camera feed,
active procedural step checklist, telemetry diagnostics, and manual controls.
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from typing import Optional


class MissionControlGUI:
    """Desktop Mission Control GUI built on Tkinter for instant Windows 11 compatibility."""

    def __init__(self, title: str = "BAS Mission Control - HAR & Digital Twin"):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("1366x800")
        self.root.configure(bg="#0a0d14")

        self.latest_frame: Optional[np.ndarray] = None
        self._setup_ui()

    def _setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#101622", height=60, bd=1, relief="solid")
        header_frame.pack(fill="x", side="top", padx=10, pady=5)

        title_lbl = tk.Label(
            header_frame,
            text="BHARATIYA ANTARIKSH STATION (BAS) | ON-BOARD HAR & DIGITAL TWIN",
            font=("Orbitron", 14, "bold"),
            fg="#00f0ff",
            bg="#101622"
        )
        title_lbl.pack(side="left", padx=15, pady=10)

        self.status_lbl = tk.Label(
            header_frame,
            text="S-BAND TELEMETRY: ACTIVE | 3,000,000:1 COMPRESSION",
            font=("Consolas", 10, "bold"),
            fg="#00e676",
            bg="#101622"
        )
        self.status_lbl.pack(side="right", padx=15)

        # Content Area (2 Columns)
        content_frame = tk.Frame(self.root, bg="#0a0d14")
        content_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Left Column: Video Viewport
        video_frame = tk.Frame(content_frame, bg="#000000", bd=2, relief="groove")
        video_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.video_canvas = tk.Label(video_frame, bg="#000000")
        self.video_canvas.pack(fill="both", expand=True)

        # Right Column: Procedure Checklist & Diagnostics
        sidebar_frame = tk.Frame(content_frame, bg="#101622", width=420, bd=1, relief="solid")
        sidebar_frame.pack(side="right", fill="y", padx=(0, 0))
        sidebar_frame.pack_propagate(False)

        # Steps Header
        steps_title = tk.Label(
            sidebar_frame,
            text="PROCEDURAL CHECKLIST (DETERMINISTIC FSM)",
            font=("Orbitron", 10, "bold"),
            fg="#ffffff",
            bg="#101622"
        )
        steps_title.pack(anchor="w", padx=15, pady=(15, 10))

        self.step_labels = []
        try:
            with open("configs/box_return_fsm.json", "r") as f:
                fsm_cfg = json.load(f)
            step_names = [f"Step {k}: {v['name'].replace('_', ' ').title()}" for k, v in fsm_cfg.get("states", {}).items()]
        except Exception:
            step_names = [
                "Step 0: Initialize Workspace",
                "Step 1: Open Box",
                "Step 2: Extract Object",
                "Step 3: Return Object",
                "Step 4: Close Box & Complete"
            ]
        for idx, s_name in enumerate(step_names):
            lbl = tk.Label(
                sidebar_frame,
                text=f"[ ] {s_name}",
                font=("Consolas", 10, "bold"),
                fg="#94a3b8",
                bg="#162032",
                anchor="w",
                padx=10,
                pady=8
            )
            lbl.pack(fill="x", padx=15, pady=4)
            self.step_labels.append(lbl)

        # Guidance Box
        g_title = tk.Label(
            sidebar_frame,
            text="ACTIVE INSTRUCTION & VOICE ALERT",
            font=("Orbitron", 10, "bold"),
            fg="#ffffff",
            bg="#101622"
        )
        g_title.pack(anchor="w", padx=15, pady=(20, 5))

        self.instruction_lbl = tk.Label(
            sidebar_frame,
            text="System initialized. Please open container box.",
            font=("Segoe UI", 10, "italic"),
            fg="#00f0ff",
            bg="#0d1b2a",
            wraplength=380,
            justify="left",
            padx=12,
            pady=12
        )
        self.instruction_lbl.pack(fill="x", padx=15, pady=5)

        # Diagnostics Log Box
        diag_title = tk.Label(
            sidebar_frame,
            text="TELEMETRY LOG STREAM",
            font=("Orbitron", 10, "bold"),
            fg="#ffffff",
            bg="#101622"
        )
        diag_title.pack(anchor="w", padx=15, pady=(20, 5))

        self.log_text = tk.Text(
            sidebar_frame,
            height=10,
            bg="#06080d",
            fg="#00e676",
            font=("Consolas", 9),
            bd=0,
            padx=8,
            pady=8
        )
        self.log_text.pack(fill="both", expand=True, padx=15, pady=(5, 15))

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        try:
            self.root.destroy()
        except Exception:
            pass
        self.root = None

    def is_alive(self) -> bool:
        try:
            return bool(self.root and self.root.winfo_exists())
        except Exception:
            return False

    def update_frame(self, frame_bgr: np.ndarray):
        """Updates the video canvas with the latest annotated frame."""
        if not self.is_alive():
            return
        self.latest_frame = frame_bgr
        h, w, _ = frame_bgr.shape
        target_w = 850
        target_h = int(h * (target_w / w))

        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (target_w, target_h)), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img_tk = ImageTk.PhotoImage(image=img, master=self.root)
        self.video_canvas.img_tk = img_tk
        self.video_canvas.configure(image=img_tk)

    def update_state(self, step_idx: int, instruction: str, anomaly: str, log_line: Optional[str] = None):
        """Updates step checklist highlighting, instruction text, and log messages."""
        if not self.is_alive():
            return

        for idx, lbl in enumerate(self.step_labels):
            if idx == step_idx:
                lbl.configure(text=lbl.cget("text").replace("[ ]", "[>>]").replace("[OK]", "[>>]"),
                              fg="#000000", bg="#00f0ff")
            elif idx < step_idx:
                lbl.configure(text=lbl.cget("text").replace("[ ]", "[OK]").replace("[>>]", "[OK]"),
                              fg="#000000", bg="#00e676")
            else:
                lbl.configure(fg="#94a3b8", bg="#162032")

        if anomaly != "NONE":
            self.instruction_lbl.configure(text=f"WARNING: {instruction}", fg="#ff1744", bg="#2a0d14")
        else:
            self.instruction_lbl.configure(text=instruction, fg="#00f0ff", bg="#0d1b2a")

        if log_line:
            self.log_text.insert("end", log_line + "\n")
            self.log_text.see("end")

        try:
            self.root.update_idletasks()
            self.root.update()
        except Exception:
            pass
