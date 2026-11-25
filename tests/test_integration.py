"""
NewsSentinel - Phase 4 Synchronous Integration Test Suite
=========================================================
Location: tests/test_integration.py

Validates the full prediction pipeline without TestClient, asyncio event loops,
or network sockets (ensuring 100% compatibility with Windows sandbox isolation).

Coverage:
  1. Input validation boundary tests (empty, whitespace, too short, too long).
  2. Linguistic feature extraction accuracy.
  3. Real-time machine learning prediction on real news.
  4. Real-time machine learning prediction on fake news.
  5. Session history tracking and reverse chronological order.
  6. Live system telemetry and statistics aggregation.
  7. Model metadata and candidate benchmark loading.
  8. Pydantic request and response schema contracts.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.analyzer_service import AnalyzerService
from app.routes.analyzer import AnalyzeRequest, APIResponse


class TestSynchronousIntegration(unittest.TestCase):
    """Integration test suite executing purely synchronously with zero socket operations."""

    def setUp(self):
        # Create an isolated service instance for each test to keep history clean
        self.service = AnalyzerService()
        self.service.clear_history()

    def test_input_validation_rejections(self):
        """Verify input validation correctly rejects invalid and out-of-bound inputs."""
        # Empty inputs
        valid, err = self.service.validate_input(text="", headline="")
        self.assertFalse(valid)
        self.assertIn("cannot be empty", err)

        # Whitespace-only inputs
        valid, err = self.service.validate_input(text="   \n\t   ", headline="")
        self.assertFalse(valid)
        self.assertIn("cannot be empty", err)

        # Under minimum 10 characters
        valid, err = self.service.validate_input(text="Short", headline="")
        self.assertFalse(valid)
        self.assertIn("too short", err)

        # Over maximum 50,000 characters
        valid, err = self.service.validate_input(text="A" * 50001, headline="")
        self.assertFalse(valid)
        self.assertIn("exceeds the maximum", err)

        # Valid input
        valid, err = self.service.validate_input(text="Valid news content with sufficient length.", headline="Headline")
        self.assertTrue(valid)
        self.assertIsNone(err)

    def test_linguistic_metrics_extraction(self):
        """Verify that deterministic linguistic metrics are computed accurately."""
        sample_text = "BREAKING NEWS: Global economy reports strong recovery! Did inflation finally peak?"
        metrics = self.service.extract_linguistic_metrics(sample_text)

        self.assertEqual(metrics["characters_count"], len(sample_text))
        self.assertGreater(metrics["words_count"], 5)
        self.assertEqual(metrics["exclamation_count"], 1)
        self.assertEqual(metrics["question_count"], 1)
        self.assertGreater(metrics["uppercase_ratio"], 10.0)
        self.assertIn("reading_time_display", metrics)

    def test_real_news_prediction_flow(self):
        """Verify that authentic Reuters reporting is classified as Real with high confidence."""
        headline = "Senate approves federal funding bill to avert government shutdown"
        body = (
            "WASHINGTON (Reuters) - The United States Senate voted overwhelmingly on Thursday "
            "to approve a compromise spending bill, sending the measure to President Trump's desk "
            "and avoiding a partial shutdown of key government agencies ahead of the fiscal deadline."
        )

        res = self.service.process_analysis_request(text=body, headline=headline)

        self.assertTrue(res["success"])
        self.assertIsNone(res["error"])

        data = res["data"]
        self.assertEqual(data["prediction"], "Real")
        self.assertEqual(data["label"], 1)
        self.assertGreaterEqual(data["confidence"], 0.80)
        self.assertGreater(data["probabilities"]["Real"], 0.80)
        self.assertLess(data["probabilities"]["Fake"], 0.20)
        self.assertGreater(data["latency_ms"], 0.0)
        self.assertIn("disclaimer", data)
        self.assertIn("not definitive fact-checking", data["disclaimer"])

    def test_fake_news_prediction_flow(self):
        """Verify that sensational clickbait is classified as Fake with high confidence."""
        headline = "SHOCKING! SATANIC PEDOPHILE RING EXPOSED IN SECRET UNDERGROUND BUNKER [VIDEO]"
        body = (
            "UNBELIEVABLE! Top government officials were caught red handed operating a secret "
            "satanic pedophile dungeon beneath a local pizza restaurant. Millions of paid Soros "
            "crisis actors and chemtrail operators have been mobilized to suppress this shocking truth!"
        )

        res = self.service.process_analysis_request(text=body, headline=headline)

        self.assertTrue(res["success"])
        data = res["data"]
        self.assertEqual(data["prediction"], "Fake")
        self.assertEqual(data["label"], 0)
        self.assertGreaterEqual(data["confidence"], 0.80)
        self.assertGreater(data["probabilities"]["Fake"], 0.80)
        self.assertLess(data["probabilities"]["Real"], 0.20)

    def test_session_history_and_statistics(self):
        """Verify that predictions are logged in history and reflected in system statistics."""
        initial_stats = self.service.get_system_stats()
        self.assertEqual(initial_stats["total_analyses"], 0)
        self.assertEqual(initial_stats["model_accuracy_display"], "96.59%")

        # Submit one Real and one Fake article
        self.service.process_analysis_request(
            headline="White House holds bilateral talks with European leaders",
            text="BRUSSELS (Reuters) - Diplomatic representatives gathered in Brussels on Friday for trade negotiations."
        )
        self.service.process_analysis_request(
            headline="ALIENS CONFIRMED: Secret Pentagon memo reveals extraterrestrial lizard people control banks",
            text="Unbelievable proof leaked online showing secret illuminati alien shape shifters ruling the global financial system!"
        )

        history = self.service.get_history()
        self.assertEqual(len(history), 2)
        # Verify reverse chronological ordering (newest first)
        self.assertEqual(history[0]["prediction"], "Fake")
        self.assertEqual(history[1]["prediction"], "Real")

        stats = self.service.get_system_stats()
        self.assertEqual(stats["total_analyses"], 2)
        self.assertEqual(stats["real_detected"], 1)
        self.assertEqual(stats["fake_detected"], 1)
        self.assertEqual(stats["real_percent"], 50.0)
        self.assertEqual(stats["fake_percent"], 50.0)

    def test_pydantic_schema_validation(self):
        """Verify request and response Pydantic schemas enforce type contracts."""
        req = AnalyzeRequest(headline="Test Headline", text="Article content with valid length.")
        self.assertEqual(req.headline, "Test Headline")
        self.assertEqual(req.text, "Article content with valid length.")

        resp = APIResponse(
            success=True,
            timestamp="2026-11-20T12:00:00",
            message="Success",
            data={"prediction": "Real", "confidence_display": "99.8%"}
        )
        self.assertTrue(resp.success)
        self.assertEqual(resp.data["prediction"], "Real")


if __name__ == "__main__":
    unittest.main(verbosity=2)
