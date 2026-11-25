"""
NewsSentinel - Phase 5 Workstream 3: Uncertainty Detection Unit Tests
=====================================================================
Location: tests/test_uncertainty.py

Validates:
  1. High-confidence Real news classified as Real with is_uncertain=False.
  2. High-confidence Fake news classified as Fake with is_uncertain=False.
  3. Borderline ambiguous news (probabilities < 0.65) classified as Uncertain with is_uncertain=True.
  4. Preservation of underlying probabilities and raw binary predictions.
  5. Configurable uncertainty threshold behavior.
  6. Rejection and graceful error handling for missing/empty/invalid inputs.
  7. SQLite persistence of uncertainty status and distinct aggregate dashboard counts.
  8. Synchronous API endpoint response contract for uncertain evaluations.
"""

import sys
import unittest
import tempfile
import pathlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predictor import get_predictor, predict_news
from app.services.database import DatabaseManager
from app.services.analyzer_service import AnalyzerService
from app.routes.analyzer import analyze_text, AnalyzeRequest


class TestUncertaintyDetection(unittest.TestCase):
    """Test suite for Bayesian/calibrated uncertainty detection and thresholding."""

    @classmethod
    def setUpClass(cls):
        cls.predictor = get_predictor()

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = pathlib.Path(self.tmp_dir.name) / "test_uncertainty.db"
        self.db = DatabaseManager(self.db_path)
        self.service = AnalyzerService(db=self.db, uncertainty_threshold=0.65)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_high_confidence_real_news(self):
        """Verify that high-confidence wire reporting classifies as Real (is_uncertain=False)."""
        title = "Senate passes bipartisan budget bill to avert government shutdown"
        text = (
            "WASHINGTON (Reuters) - The U.S. Senate voted overwhelmingly on Thursday "
            "to approve a compromise spending bill, sending the measure to the president's "
            "desk and avoiding a looming federal agency shutdown ahead of the midnight deadline."
        )
        res = self.predictor.predict(title, text, uncertainty_threshold=0.65)
        self.assertEqual(res["prediction"], "Real")
        self.assertFalse(res["is_uncertain"])
        self.assertGreaterEqual(res["confidence"], 0.65)
        self.assertGreater(res["probabilities"]["Real"], 0.65)
        self.assertEqual(res["raw_prediction"], "Real")

    def test_high_confidence_fake_news(self):
        """Verify that high-confidence sensational clickbait classifies as Fake (is_uncertain=False)."""
        title = "SHOCKING! SATANIC PEDOPHILE RING EXPOSED IN SECRET UNDERGROUND BUNKER [VIDEO]"
        text = (
            "UNBELIEVABLE! Top government officials were caught red handed operating a secret "
            "satanic pedophile dungeon beneath a local pizza restaurant. Millions of paid Soros "
            "crisis actors and chemtrail operators have been mobilized to suppress this shocking truth!"
        )
        res = self.predictor.predict(title, text, uncertainty_threshold=0.65)
        self.assertEqual(res["prediction"], "Fake")
        self.assertFalse(res["is_uncertain"])
        self.assertGreaterEqual(res["confidence"], 0.65)
        self.assertGreater(res["probabilities"]["Fake"], 0.65)
        self.assertEqual(res["raw_prediction"], "Fake")

    def test_borderline_uncertain_prediction(self):
        """
        Verify that borderline ambiguous text whose probabilities fall below 0.65
        is classified as 'Uncertain' with is_uncertain=True while preserving raw probabilities.
        """
        borderline_text = "Officials spoke to reporters about general matters during a routine weekly gathering."
        res = self.predictor.predict("", borderline_text, uncertainty_threshold=0.65)

        self.assertTrue(res["is_uncertain"], "Borderline prediction must be flagged as uncertain")
        self.assertEqual(res["prediction"], "Uncertain")
        self.assertIn("UNCERTAIN", res["verdict"])
        self.assertLess(max(res["probabilities"].values()), 0.65)

        # Verify underlying information is preserved
        self.assertIn(res["raw_prediction"], ["Real", "Fake"])
        self.assertIn("Fake", res["probabilities"])
        self.assertIn("Real", res["probabilities"])
        self.assertAlmostEqual(res["probabilities"]["Fake"] + res["probabilities"]["Real"], 1.0, places=3)
        self.assertEqual(res["uncertainty_threshold"], 0.65)

    def test_configurable_uncertainty_threshold(self):
        """Verify that varying the threshold dynamically adjusts the uncertainty boundary."""
        borderline_text = "Officials spoke to reporters about general matters during a routine weekly gathering."

        # At default threshold 0.65, it is uncertain (~0.604 confidence)
        res_default = self.predictor.predict("", borderline_text, uncertainty_threshold=0.65)
        self.assertTrue(res_default["is_uncertain"])
        self.assertEqual(res_default["prediction"], "Uncertain")

        # Lower threshold to 0.50 -> max probability (~0.604) exceeds threshold, so not uncertain
        res_lenient = self.predictor.predict("", borderline_text, uncertainty_threshold=0.50)
        self.assertFalse(res_lenient["is_uncertain"])
        self.assertIn(res_lenient["prediction"], ["Real", "Fake"])

        # Strict threshold 0.99 -> even ~0.90 confident news becomes uncertain
        real_text = "The Department of State held bilateral trade discussions in Geneva on Wednesday."
        res_strict = self.predictor.predict("", real_text, uncertainty_threshold=0.99)
        self.assertTrue(res_strict["is_uncertain"])
        self.assertEqual(res_strict["prediction"], "Uncertain")

    def test_missing_and_invalid_inputs(self):
        """Verify input validation correctly rejects empty, whitespace, and out-of-bound inputs."""
        # Empty input
        valid, err = self.service.validate_input(text="", headline="")
        self.assertFalse(valid)
        self.assertIn("cannot be empty", err)

        # Whitespace only
        valid, err = self.service.validate_input(text="     \n\t   ", headline="")
        self.assertFalse(valid)
        self.assertIn("cannot be empty", err)

        # Below 10 characters
        valid, err = self.service.validate_input(text="Too short", headline="")
        self.assertFalse(valid)
        self.assertIn("too short", err)

        # Exceeds maximum 50,000 characters
        oversized = "a" * 50001
        valid, err = self.service.validate_input(text=oversized, headline="")
        self.assertFalse(valid)
        self.assertIn("maximum allowed length", err)

    def test_database_persistence_and_dashboard_counts(self):
        """
        Verify that analyses store is_uncertain in SQLite and that system stats
        distinctly report Fake, Real, and Uncertain counts.
        """
        # 1. High-confidence Real
        self.service.process_analysis_request(
            headline="White House Accord",
            text="WASHINGTON (Reuters) - The Senate approved the federal funding bill today."
        )

        # 2. High-confidence Fake
        self.service.process_analysis_request(
            headline="SECRET LIZARD PEOPLE",
            text="SHOCKING! Underground bunker exposes extraterrestrial lizard people controlling world banks!"
        )

        # 3. Borderline Uncertain
        self.service.process_analysis_request(
            headline="",
            text="Officials spoke to reporters about general matters during a routine weekly gathering."
        )

        # Verify SQLite counts
        counts = self.db.get_counts()
        self.assertEqual(counts["total"], 3)
        self.assertEqual(counts["real"], 1)
        self.assertEqual(counts["fake"], 1)
        self.assertEqual(counts["uncertain"], 1)

        # Verify service system stats
        stats = self.service.get_system_stats()
        self.assertEqual(stats["total_analyses"], 3)
        self.assertEqual(stats["real_detected"], 1)
        self.assertEqual(stats["fake_detected"], 1)
        self.assertEqual(stats["uncertain_detected"], 1)
        self.assertAlmostEqual(stats["uncertain_percent"], 33.3, places=1)

        # Verify history retrieval
        history = self.service.get_history()
        self.assertEqual(len(history), 3)
        # Newest record is the uncertain one
        self.assertEqual(history[0]["prediction"], "Uncertain")
        self.assertTrue(history[0]["is_uncertain"])

    def test_api_analyze_response_when_uncertain(self):
        """Verify that synchronous /api/analyze route returns structured uncertainty information."""
        # Borderline input
        payload = AnalyzeRequest(
            headline="",
            text="Officials spoke to reporters about general matters during a routine weekly gathering."
        )
        res = analyze_text(payload)
        self.assertTrue(res.success)
        self.assertTrue(res.data["is_uncertain"])
        self.assertEqual(res.data["prediction"], "Uncertain")
        self.assertIn("UNCERTAIN — NEEDS VERIFICATION", res.message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
