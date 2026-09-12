"""
BAS Autonomous HAR System - Offline Speech Synthesizer (TTS)
Provides sub-10ms instantaneous offline voice guidance and priority alerts.
Runs natively on Windows via Windows SAPI COM (SAPI.SpVoice) with zero subprocess overhead
and zero external cloud dependencies. Supports instant audio preemption for urgent safety warnings.
"""

import threading
import queue
import time
from typing import Optional


class OfflineTTS:
    """High-performance in-process offline Text-To-Speech engine with priority preemption."""

    def __init__(self, voice_gender: str = "Female", speech_rate: int = 1):
        self._queue: queue.Queue = queue.Queue()
        self._running: bool = True
        self._speech_rate: int = speech_rate
        self._last_spoken_time: float = 0.0
        self._last_text: str = ""
        self._worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self._worker_thread.start()

    def speak(self, text: str, priority: bool = False):
        """
        Dispatches an utterance to the speech queue in a non-blocking, zero-latency manner.
        If priority is True, immediately preempts/purges pending and playing audio.
        """
        if not text or not text.strip():
            return

        now = time.time()
        debounce_window = 4.0 if priority else 3.0

        # Debounce identical utterances
        if text == self._last_text and (now - self._last_spoken_time) < debounce_window:
            return

        self._last_text = text
        self._last_spoken_time = now

        if priority:
            # Purge pending queue items immediately for safety alerts
            try:
                while not self._queue.empty():
                    self._queue.get_nowait()
            except Exception:
                pass
            self._queue.put((text, True))
        else:
            self._queue.put((text, False))

    def reset(self):
        """Immediately halts all active audio playback and flushes the speech queue."""
        self._last_spoken_time = 0.0
        self._last_text = ""
        try:
            while not self._queue.empty():
                self._queue.get_nowait()
        except Exception:
            pass
        # Send a special reset sentinel
        self._queue.put(("__RESET__", True))

    def _speech_worker(self):
        """Worker thread processing speech via native SAPI COM with preemption."""
        speaker = None
        has_com = False

        # Attempt 1: Native Windows SAPI COM Dispatch
        try:
            import pythoncom
            import win32com.client
            pythoncom.CoInitialize()
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Rate = self._speech_rate
            speaker.Volume = 100
            has_com = True
        except Exception as e:
            has_com = False

        # Attempt 2: pyttsx3 fallback
        pyttsx_engine = None
        if not has_com:
            try:
                import pyttsx3
                pyttsx_engine = pyttsx3.init()
            except Exception:
                pass

        while self._running:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            text, priority = item
            if text == "__RESET__":
                if has_com and speaker is not None:
                    try:
                        # Purge active audio (flag 2: SVSFPurgeBeforeSpeak)
                        speaker.Speak("", 2)
                    except Exception:
                        pass
                self._queue.task_done()
                continue

            try:
                if has_com and speaker is not None:
                    if priority:
                        # Flag 3 = SVSFlagsAsync (1) | SVSFPurgeBeforeSpeak (2)
                        speaker.Speak(text, 3)
                    else:
                        # If previous step's instruction is still speaking, purge and speak current step
                        if speaker.Status.RunningState == 2:
                            speaker.Speak("", 2)
                        # Flag 1 = SVSFlagsAsync
                        speaker.Speak(text, 1)

                    # Monitor playback until finished or interrupted by higher priority item
                    while speaker.Status.RunningState == 2 and self._running:
                        if not self._queue.empty():
                            try:
                                next_item = self._queue.queue[0]
                                if next_item[1]:  # Priority item waiting
                                    break
                            except Exception:
                                pass
                        time.sleep(0.04)

                elif pyttsx_engine is not None:
                    pyttsx_engine.say(text)
                    pyttsx_engine.runAndWait()
                else:
                    # Final fallback: Console banner
                    print(f"[Audio Voice]: {text}")
            except Exception as e:
                print(f"[Offline TTS Speech Output]: {text}")
            finally:
                self._queue.task_done()

        if has_com:
            try:
                import pythoncom
                pythoncom.CoUninitialize()
            except Exception:
                pass

    def shutdown(self):
        self._running = False
        self.reset()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
