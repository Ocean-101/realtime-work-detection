"""
Unit Tests for JSON Lines Telemetry Compression Ratio (3,000,000:1)
ISRO Smart India Hackathon (SIH) | Problem Statement ID: 26174
Audits deep-space telemetry bandwidth efficiency.
"""

import os
import sys
import unittest

# Ensure workspace root in path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.telemetry.jsonl_logger import JSONLTelemetryLogger


class TestTelemetryCompression(unittest.TestCase):

    def test_compression_ratio_calculation(self):
        log_path = os.path.join(WORKSPACE_ROOT, "experiments", "test_telemetry.jsonl")
        logger = JSONLTelemetryLogger(output_path=log_path)

        # Log typical 30-minute session events (~10 milestone lines)
        logger.log_event(0, 0, "IDLE", "SYSTEM_INIT", tts_prompt="Open container box")
        logger.log_event(375, 1, "BOX_OPENED", "BOX_OPENED", tts_prompt="Extract object")
        logger.log_event(1120, 2, "OBJECT_EXTRACTED", "OBJECT_EXTRACTED", tts_prompt="Return object")
        logger.log_event(1850, 4, "COMPLETE", "BOX_CLOSED", tts_prompt="Procedure complete")

        logger.close()

        # Check total bytes written
        file_size = os.path.getsize(log_path)
        self.assertLess(file_size, 15000, "Structured telemetry must remain under 15 KB")

        # 30 minute session at 30 fps, 1080p, 3 bytes per pixel
        duration_sec = 1800.0  # 30 minutes
        ratio = logger.calculate_compression_ratio(duration_sec, fps=30.0, width=1920, height=1080, bytes_per_pixel=3)

        print(f"\n[Telemetry Audit] 30-Minute Video Uncompressed: {duration_sec * 30 * 1920 * 1080 * 3 / (1024**3):.2f} GB")
        print(f"[Telemetry Audit] Emitted JSONL Telemetry: {file_size} bytes")
        print(f"[Telemetry Audit] Calculated Compression Ratio: {ratio:,.1f} : 1")

        self.assertGreater(ratio, 1_000_000.0, "Compression ratio must exceed 1,000,000:1")

        # Clean up
        if os.path.exists(log_path):
            os.remove(log_path)


if __name__ == "__main__":
    unittest.main()
