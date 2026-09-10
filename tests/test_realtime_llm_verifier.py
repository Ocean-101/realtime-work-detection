"""
Unit Tests for RealtimeLLMVerifier
Validates background worker lifecycle, sliding telemetry ingestion,
and real-time procedural consensus verification.
"""

import time
import unittest
from src.core.types import FSMStep, AnomalyType
from src.llm.realtime_llm_verifier import RealtimeLLMVerifier


class TestRealtimeLLMVerifier(unittest.TestCase):

    def setUp(self):
        self.verifier = RealtimeLLMVerifier(interval_sec=0.2, window_size=20)

    def tearDown(self):
        self.verifier.close()

    def test_initial_state(self):
        """Verifies initial verification state before telemetry ingestion."""
        verif = self.verifier.get_latest_verification()
        self.assertIn("verified_step", verif)
        self.assertIn("confidence", verif)
        self.assertIn("reason", verif)
        self.assertEqual(verif["verified_step"], 0)

    def test_push_telemetry_and_consensus(self):
        """Simulates pushing extraction telemetry and verifies structured consensus."""
        for frame_id in range(1, 15):
            self.verifier.push_telemetry(
                frame_id=frame_id,
                step=FSMStep.OBJECT_EXTRACTED,
                activity="EXTRACT COMPONENT BOX",
                lid_angle=70.0,
                is_inside=False,
                hoi_action="EXTRACT",
                hand_dist_m=0.15,
                anomaly=AnomalyType.NONE
            )

        # Allow background thread a moment to process window
        time.sleep(0.5)

        verif = self.verifier.get_latest_verification()
        self.assertIsInstance(verif["verified_step"], int)
        self.assertIn(verif["verified_step"], [0, 1, 2, 3, 4])
        self.assertGreaterEqual(verif["confidence"], 0.5)
        self.assertIsInstance(verif["reason"], str)

    def test_deterministic_fallback(self):
        """Validates fallback consensus generator logic."""
        fallback = self.verifier._fallback_deterministic_verification(
            candidate_step=1,
            candidate_name="BOX_OPENED",
            avg_lid=65.0,
            inside_ratio=0.1,  # Object is outside
            recent_activities=["EXTRACT COMPONENT BOX"]
        )
        self.assertEqual(fallback["verified_step"], 2)
        self.assertEqual(fallback["step_name"], "OBJECT_EXTRACTED")
        self.assertGreater(fallback["confidence"], 0.8)


if __name__ == "__main__":
    unittest.main()
