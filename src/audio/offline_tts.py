"""
BAS Autonomous HAR System - Offline Speech Synthesizer (TTS)
Provides non-blocking, sub-100ms offline voice guidance and priority alerts.
Runs natively on Windows 11 via Windows SAPI with zero external cloud dependencies.
"""

import threading
import queue
import subprocess
import time
from typing import Optional


class OfflineTTS:
    """Threaded offline Text-To-Speech engine using Windows SAPI."""

    def __init__(self, voice_gender: str = "Female", speech_rate: int = 1):
        self._queue: queue.Queue = queue.Queue()
        self._running: bool = True
        self._worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self._worker_thread.start()
        self._last_spoken_time = 0.0
        self._last_text = ""

    def speak(self, text: str, priority: bool = False):
        """Dispatches an utterance to the speech queue in a non-blocking manner."""
        if not text or not text.strip():
            return
        
        # Debounce duplicate utterances within 2 seconds
        now = time.time()
        if text == self._last_text and (now - self._last_spoken_time) < 2.5:
            return

        if priority:
            # Clear pending non-priority utterances for urgent safety alerts
            try:
                while not self._queue.empty():
                    self._queue.get_nowait()
            except queue.Empty:
                pass
        
        self._queue.put(text)
        self._last_text = text
        self._last_spoken_time = now

    def _speech_worker(self):
        """Worker thread processing speech items sequentially."""
        while self._running:
            try:
                text = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                # Sanitize text for PowerShell
                clean_text = text.replace("'", "").replace('"', "")
                ps_cmd = f"& {{ Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Rate = 1; $s.Speak('{clean_text}') }}"
                subprocess.run(
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_cmd],
                    capture_output=True,
                    timeout=5.0
                )
            except Exception as e:
                # Fallback to console print if speech fails
                print(f"[Offline TTS Audio Alert]: {text}")
            finally:
                self._queue.task_done()

    def shutdown(self):
        self._running = False
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
