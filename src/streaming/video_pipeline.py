"""
BAS Autonomous HAR System - Dual Video Pipeline
Simultaneously handles local H.264 MP4 recording and local-network IP streaming
for remote monitoring on tablet terminals and ground station consoles.
"""

import os
import cv2
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime
from typing import Optional, Tuple
import numpy as np


class StreamHandler(BaseHTTPRequestHandler):
    """Serves an MJPEG video stream to connected browser/tablet clients."""

    def do_GET(self):
        if self.path in ('/stream', '/video'):
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.end_headers()

            while getattr(self.server, 'running', True):
                frame_bytes = getattr(self.server, 'latest_jpeg', None)
                if frame_bytes is not None:
                    try:
                        self.wfile.write(b'--frame\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', str(len(frame_bytes)))
                        self.end_headers()
                        self.wfile.write(frame_bytes)
                        self.wfile.write(b'\r\n')
                    except (ConnectionResetError, BrokenPipeError):
                        break
                time.sleep(0.033) # ~30 FPS
        elif self.path in ('/', '/index.html'):
            # Serve Web Mission Control Dashboard
            html_path = os.path.join(os.path.dirname(__file__), "..", "gui", "web_twin", "index.html")
            try:
                with open(html_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                self.send_error(404, f"Dashboard file not found: {e}")
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        # Suppress noisy HTTP request logging
        return


class DualVideoPipeline:
    """Coordinates local video persistence and simultaneous network IP streaming."""

    def __init__(
        self,
        local_output_path: Optional[str] = None,
        stream_port: int = 8080,
        fps: float = 30.0,
        resolution: Tuple[int, int] = (1280, 720)
    ):
        self.fps = fps
        self.resolution = resolution
        self.stream_port = stream_port
        
        # 1. Local Storage Sink
        if local_output_path is None:
            os.makedirs("experiments", exist_ok=True)
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.local_output_path = f"experiments/video_{ts}.mp4"
        else:
            self.local_output_path = local_output_path
            os.makedirs(os.path.dirname(os.path.abspath(local_output_path)), exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._writer = cv2.VideoWriter(
            self.local_output_path, fourcc, self.fps, self.resolution
        )

        # 2. IP Streaming Sink (HTTP MJPEG Server)
        self._server = None
        self._server_thread = None
        self._running = True
        self._start_stream_server()

    def _start_stream_server(self):
        try:
            self._server = ThreadingHTTPServer(('0.0.0.0', self.stream_port), StreamHandler)
            self._server.running = True
            self._server.latest_jpeg = None
            self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._server_thread.start()
            print(f"[Dual Video Pipeline] IP Live Streaming active at: http://localhost:{self.stream_port}/stream")
        except Exception as e:
            print(f"[Dual Video Pipeline] IP Stream Server warning: {e}")

    def write_frame(self, frame: np.ndarray):
        """Dispatches frame to local MP4 writer and encodes JPEG for streaming clients."""
        if frame is None:
            return

        # Ensure frame matches target resolution
        h, w = frame.shape[:2]
        if (w, h) != self.resolution:
            frame_resized = cv2.resize(frame, self.resolution)
        else:
            frame_resized = frame

        # Write to local file
        if self._writer and self._writer.isOpened():
            self._writer.write(frame_resized)

        # Update streaming buffer
        if self._server:
            ret, jpeg = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                self._server.latest_jpeg = jpeg.tobytes()

    def close(self):
        self._running = False
        if self._writer:
            self._writer.release()
        if self._server:
            self._server.running = False
            self._server.shutdown()
        print(f"[Dual Video Pipeline] Recording saved to: {self.local_output_path}")
