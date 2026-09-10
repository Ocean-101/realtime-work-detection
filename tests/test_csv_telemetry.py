"""
Unit Tests for Real-Time Frame Telemetry CSV Logger
Validates that every single frame records complete 3D kinematic, spatial, and Local LLM verification
parameters into both the archival folder and the dedicated realtime_feed/ folder for 3D animation
and Digital Twin model ingestion.
"""

import os
import csv
import shutil
import unittest
from src.core.types import (
    AstronautPose3D,
    Joint3D,
    Vector3D,
    ExperimentObject,
    BBox2D,
    FSMStep,
    AnomalyType,
    EntityState
)
from src.telemetry.csv_logger import RealtimeCSVTelemetryLogger


class TestCSVTelemetryLogger(unittest.TestCase):

    def setUp(self):
        self.test_csv_path = "experiments/test_telemetry_run.csv"
        self.test_realtime_dir = "experiments/test_realtime_feed"
        if os.path.exists(self.test_csv_path):
            os.remove(self.test_csv_path)
        if os.path.exists(self.test_realtime_dir):
            shutil.rmtree(self.test_realtime_dir, ignore_errors=True)

        self.logger = RealtimeCSVTelemetryLogger(
            output_csv_path=self.test_csv_path,
            realtime_feed_dir=self.test_realtime_dir
        )

    def tearDown(self):
        self.logger.close()
        if os.path.exists(self.test_csv_path):
            try:
                os.remove(self.test_csv_path)
            except Exception:
                pass
        if os.path.exists(self.test_realtime_dir):
            shutil.rmtree(self.test_realtime_dir, ignore_errors=True)

    def test_csv_header_format(self):
        """Validates that header contains all 42 kinematic, spatial, and LLM channels."""
        self.assertEqual(len(self.logger.CSV_HEADER), 42)
        self.assertIn("frame_id", self.logger.CSV_HEADER)
        self.assertIn("wrist_rack_x", self.logger.CSV_HEADER)
        self.assertIn("container_rack_x", self.logger.CSV_HEADER)
        self.assertIn("component_rack_x", self.logger.CSV_HEADER)
        self.assertIn("component_is_inside", self.logger.CSV_HEADER)
        self.assertIn("lid_angle_deg", self.logger.CSV_HEADER)
        self.assertIn("hand_to_object_dist_m", self.logger.CSV_HEADER)
        self.assertIn("llm_verified_step", self.logger.CSV_HEADER)
        self.assertIn("llm_verified_name", self.logger.CSV_HEADER)
        self.assertIn("llm_confidence", self.logger.CSV_HEADER)
        self.assertIn("llm_reason", self.logger.CSV_HEADER)

    def test_log_frames_and_verify_both_csv_outputs(self):
        """Simulates 20 frames of live telemetry logging and verifies both archival and realtime_feed outputs."""
        # Construct mock pose
        joints = {
            "wrist": Joint3D(name="wrist", pos_camera=Vector3D(0.1, -0.2, 1.2), pos_rack=Vector3D(0.5, 0.6, 0.7)),
            "elbow": Joint3D(name="elbow", pos_camera=Vector3D(0.15, -0.1, 1.3), pos_rack=Vector3D(0.55, 0.7, 0.8))
        }
        pose = AstronautPose3D(
            joints=joints,
            elbow_angle_deg=115.5,
            shoulder_angle_deg=45.2,
            body_orientation_deg=-90.0,
            rom_limits_violated=False
        )

        # Construct mock objects
        objects = {
            "container_box": ExperimentObject(
                name="container_box",
                class_name="container_box",
                pos_rack=Vector3D(0.0, 0.5, 0.0),
                bbox=BBox2D(100.0, 100.0, 400.0, 300.0, confidence=0.95, class_id=0, class_name="container_box")
            ),
            "component_box": ExperimentObject(
                name="component_box",
                class_name="component_box",
                pos_rack=Vector3D(0.05, 0.52, 0.05),
                bbox=BBox2D(150.0, 150.0, 250.0, 250.0, confidence=0.90, class_id=2, class_name="component_box"),
                state=EntityState.GRASPED,
                is_inside_container=False
            )
        }

        mock_llm_verif = {
            "verified_step": 2,
            "step_name": "OBJECT_EXTRACTED",
            "confidence": 0.96,
            "anomaly_verdict": "NOMINAL",
            "reason": "Operator grasped and extracted component."
        }

        # Log 20 frames
        for frame_id in range(1, 21):
            self.logger.log_frame(
                frame_id=frame_id,
                fps=30.0,
                step=FSMStep.OBJECT_EXTRACTED,
                activity="EXTRACT COMPONENT BOX",
                anomaly=AnomalyType.NONE,
                instruction="Extract component box from primary container.",
                lid_angle=72.5,
                pose=pose,
                objects=objects,
                llm_verification=mock_llm_verif
            )

        self.assertEqual(self.logger.rows_written, 20)
        self.logger.close()

        # 1. Read back archival CSV and verify contents
        self.assertTrue(os.path.exists(self.test_csv_path))
        with open(self.test_csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            header = reader[0]
            self.assertEqual(header, self.logger.CSV_HEADER)
            self.assertEqual(len(reader), 21)  # 1 header + 20 data rows

            first_row = reader[1]
            self.assertEqual(first_row[0], "1")  # frame_id
            self.assertEqual(first_row[5], "OBJECT_EXTRACTED")  # step_name
            self.assertEqual(first_row[6], "EXTRACT COMPONENT BOX")  # activity
            self.assertEqual(first_row[9], "72.5")  # lid_angle_deg
            self.assertEqual(first_row[10], "OPEN")  # lid_state
            self.assertEqual(first_row[14], "0.5")  # wrist_rack_x
            self.assertEqual(first_row[21], "0.0")  # container_rack_x
            self.assertEqual(first_row[28], "0.05")  # component_rack_x
            self.assertEqual(first_row[35], "GRASPED")  # component_state
            self.assertEqual(first_row[36], "0")  # component_is_inside
            self.assertEqual(first_row[38], "2")  # llm_verified_step
            self.assertEqual(first_row[39], "OBJECT_EXTRACTED")  # llm_verified_name
            self.assertEqual(first_row[40], "0.96")  # llm_confidence

        # 2. Read back dedicated realtime_feed/current_feed_telemetry.csv
        self.assertTrue(os.path.exists(self.logger.realtime_current_path))
        with open(self.logger.realtime_current_path, "r", encoding="utf-8") as rf:
            rt_reader = list(csv.reader(rf))
            self.assertEqual(len(rt_reader), 21)
            self.assertEqual(rt_reader[1][38], "2")  # verified step in real-time feed


if __name__ == "__main__":
    unittest.main()
