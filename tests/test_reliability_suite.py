"""
NewsSentinel - Phase 5 Workstream 4: Robust Testing & Final Reliability Suite
=============================================================================
Location: tests/test_reliability_suite.py

Comprehensive system-level test suite validating:
  1. All REST and HTML API endpoints (analysis, history, stats, metrics, health, pages).
  2. Input validation edge cases: empty, whitespace, boundary lengths (9, 10, 50k, 50001),
     Unicode, emojis, HTML/XSS injection, URLs, punctuation.
  3. Database persistence reliability: cold restart persistence, pagination,
     concurrent multi-threaded access, and idempotent migrations.
  4. Cross-surface telemetry consistency across API, SQLite, history, and dashboard.
  5. Explainable AI feature attribution mathematical soundness and regression prevention.
  6. Uncertainty detection thresholding and probability preservation.
"""

import sys
import unittest
import tempfile
import pathlib
import threading
from pathlib import Path
from fastapi import Request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predictor import get_predictor, explain_news
from app.services.database import DatabaseManager
from app.services.analyzer_service import AnalyzerService
from app.routes.analyzer import (
    AnalyzeRequest,
    analyze_text,
    get_analysis_history,
    get_stats,
    get_model_metrics,
    health_check,
)
from app.routes.pages import (
    dashboard_page,
    analyzer_page,
    history_page,
    performance_page,
    about_page,
)


class TestSystemReliability(unittest.TestCase):
    """System-level test suite for Phase 5 verification and regression prevention."""

    @classmethod
    def setUpClass(cls):
        cls.predictor = get_predictor()

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = pathlib.Path(self.tmp_dir.name) / "sentinel_reliability.db"
        self.db = DatabaseManager(self.db_path)
        self.service = AnalyzerService(db=self.db, uncertainty_threshold=0.65)

    def tearDown(self):
        self.tmp_dir.cleanup()

    # =========================================================================
    # 1. API Endpoints & Page Rendering
    # =========================================================================

    def test_health_check_endpoint(self):
        """Verify health check returns operational status and application metadata."""
        res = health_check()
        self.assertEqual(res["status"], "healthy")
        self.assertEqual(res["app_name"], "NewsSentinel")
        self.assertTrue(res["model_loaded"])
        self.assertIn("timestamp", res)

    def test_stats_and_metrics_endpoints(self):
        """Verify platform statistics and Phase 3 model benchmark comparisons."""
        stats_res = get_stats()
        self.assertTrue(stats_res["success"])
        stats = stats_res["stats"]
        self.assertIn("total_analyses", stats)
        self.assertIn("fake_detected", stats)
        self.assertIn("real_detected", stats)
        self.assertIn("uncertain_detected", stats)
        self.assertEqual(stats["model_accuracy_display"], "96.59%")

        metrics_res = get_model_metrics()
        self.assertTrue(metrics_res["success"])
        self.assertIn("champion_test_metrics", metrics_res["metadata"])
        self.assertAlmostEqual(metrics_res["metadata"]["champion_test_metrics"]["accuracy"], 0.9659, places=3)
        self.assertIn("Linear SVM (Calibrated)", metrics_res["comparison"])

    def test_api_analyze_and_history_endpoints(self):
        """Verify analyze and history REST routes with pagination."""
        req = AnalyzeRequest(
            headline="Global Summit",
            text="GENEVA (Reuters) - Diplomats convened on Monday to discuss international maritime agreements."
        )
        res = analyze_text(req)
        self.assertTrue(res.success)
        self.assertEqual(res.data["prediction"], "Real")
        self.assertGreaterEqual(res.data["confidence"], 0.70)
        self.assertIn("explanation", res.data)

        # Query history endpoint
        hist_res = get_analysis_history(limit=5, offset=0)
        self.assertTrue(hist_res["success"])
        self.assertGreaterEqual(hist_res["total"], 1)
        self.assertGreaterEqual(len(hist_res["records"]), 1)

    def test_all_html_pages_render_synchronously(self):
        """Verify all Jinja2 HTML templates render status 200 with valid context."""
        def make_req(path: str):
            return Request({"type": "http", "method": "GET", "path": path, "headers": []})

        pages = [
            (dashboard_page, "/"),
            (analyzer_page, "/analyzer"),
            (history_page, "/history"),
            (performance_page, "/performance"),
            (about_page, "/about"),
        ]

        for page_fn, path in pages:
            response = page_fn(make_req(path))
            self.assertEqual(response.status_code, 200, f"Page {path} failed to render")

    # =========================================================================
    # 2. Input Validation & Edge Cases
    # =========================================================================

    def test_input_validation_boundaries(self):
        """Verify exact character boundary behavior (9 chars rejected, 10 accepted)."""
        # 9 chars -> rejected
        v9, err9 = self.service.validate_input("123456789")
        self.assertFalse(v9)
        self.assertIn("too short", err9)

        # 10 chars -> accepted
        v10, err10 = self.service.validate_input("1234567890")
        self.assertTrue(v10)
        self.assertIsNone(err10)

        # 50,000 chars -> accepted
        v50k, err50k = self.service.validate_input("a" * 50000)
        self.assertTrue(v50k)
        self.assertIsNone(err50k)

        # 50,001 chars -> rejected
        v50k1, err50k1 = self.service.validate_input("a" * 50001)
        self.assertFalse(v50k1)
        self.assertIn("maximum allowed length", err50k1)

    def test_whitespace_and_empty_inputs(self):
        """Verify complete rejection of whitespace-only and empty payloads."""
        for empty_val in ["", "   ", "\t\t\n\n", "   \r\n  "]:
            valid, err = self.service.validate_input(empty_val, "")
            self.assertFalse(valid)
            self.assertIn("cannot be empty", err)

    def test_multilingual_unicode_and_emojis(self):
        """Verify pipeline handles non-ASCII, Cyrillic, CJK, and emojis without crashing."""
        multilingual_text = (
            "Официальные лица сообщили о переговорах. 政治 ニュース. "
            "Breaking global announcement! 🚀🌐✨ Diplomatic meeting in progress."
        )
        res = self.service.process_analysis_request(text=multilingual_text, headline="Multilingual Report")
        self.assertTrue(res["success"])
        self.assertIn("prediction", res["data"])
        self.assertIn("confidence", res["data"])

    def test_html_and_url_sanitization(self):
        """Verify HTML script injection and URLs are stripped cleanly during preprocessing."""
        dirty_title = "<script>alert('XSS')</script> Treasury Announcement"
        dirty_text = (
            "Visit https://evil-phishing.com/scam?q=1 for details! "
            "<img src=x onerror=alert(1)> Federal officials concluded regular sessions."
        )
        cleaned = self.predictor.preprocess_text(dirty_title, dirty_text)
        self.assertNotIn("<script>", cleaned)
        self.assertNotIn("https://", cleaned)
        self.assertNotIn("<img", cleaned)
        self.assertNotIn("onerror", cleaned)

    def test_punctuation_heavy_inputs(self):
        """Verify excessive punctuation markers do not cause division by zero or errors."""
        punc_text = "!?!?!?!???? ...... ------ $$$$$$ $$$$ %%%%%%% &&&&&&& officials met"
        res = self.service.process_analysis_request(text=punc_text)
        self.assertTrue(res["success"])
        metrics = res["data"]["metrics"]
        self.assertGreater(metrics["exclamation_count"], 0)
        self.assertGreater(metrics["question_count"], 0)

    # =========================================================================
    # 3. Database Concurrency & Cold Restart Persistence
    # =========================================================================

    def test_cold_restart_persistence(self):
        """Verify that analysis records and statistics survive cold application restart."""
        # 1. Insert 2 records with initial service instance
        self.service.process_analysis_request(
            headline="Bipartisan Accord",
            text="WASHINGTON (Reuters) - Senate passed the federal appropriation bill."
        )
        self.service.process_analysis_request(
            headline="ALIEN CONSPIRACY",
            text="SHOCKING! Underground bunker proves lizard people rule the banking cartel!"
        )

        initial_stats = self.service.get_system_stats()
        self.assertEqual(initial_stats["total_analyses"], 2)

        # 2. Simulate complete restart: instantiate brand new AnalyzerService on same DB file
        restarted_service = AnalyzerService(db=DatabaseManager(self.db_path))
        restarted_stats = restarted_service.get_system_stats()

        self.assertEqual(restarted_stats["total_analyses"], 2)
        self.assertEqual(restarted_stats["real_detected"], 1)
        self.assertEqual(restarted_stats["fake_detected"], 1)

        history = restarted_service.get_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["prediction"], "Fake")
        self.assertEqual(history[1]["prediction"], "Real")

    def test_multi_threaded_database_concurrency(self):
        """Verify thread-safety and WAL mode with concurrent reading and writing."""
        num_threads = 8
        ops_per_thread = 5
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(ops_per_thread):
                    # Write
                    self.db.insert_analysis({
                        "headline": f"Thread {thread_id} Title {i}",
                        "snippet": f"Thread {thread_id} snippet {i}",
                        "prediction": "Real" if i % 2 == 0 else "Fake",
                        "label": 1 if i % 2 == 0 else 0,
                        "confidence": 0.85,
                        "confidence_percent": 85.0,
                        "confidence_display": "85.0%"
                    })
                    # Read
                    _ = self.db.get_history(limit=5)
                    _ = self.db.get_counts()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Concurrent database operations failed with: {errors}")
        self.assertEqual(self.db.count(), num_threads * ops_per_thread)

    def test_idempotent_migration_handling(self):
        """Verify init_db() can be invoked repeatedly without duplicating schema migrations."""
        for _ in range(3):
            self.db.init_db()

        with self.db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT version FROM schema_migrations ORDER BY version ASC")
            versions = [r[0] for r in cur.fetchall()]
            self.assertEqual(versions, [1, 2, 3], "Migrations must not be duplicated")

    # =========================================================================
    # 4. Prediction Reliability, Telemetry Consistency & Uncertainty
    # =========================================================================

    def test_cross_surface_telemetry_consistency(self):
        """
        Verify that prediction, uncertainty status, probabilities, and counts
        remain 100% consistent across API output, SQLite database, history, and dashboard stats.
        """
        # Test borderline uncertain text
        res = self.service.process_analysis_request(
            headline="",
            text="Officials spoke to reporters about general matters during a routine weekly gathering."
        )
        data = res["data"]
        self.assertTrue(data["is_uncertain"])
        self.assertEqual(data["prediction"], "Uncertain")
        rec_id = data["id"]

        # 1. Verify in SQLite
        db_rec = self.db.get_analysis_by_id(rec_id)
        self.assertIsNotNone(db_rec)
        self.assertEqual(db_rec["prediction"], "Uncertain")
        self.assertTrue(db_rec["is_uncertain"])
        self.assertAlmostEqual(db_rec["confidence"], data["confidence"], places=4)

        # 2. Verify in history
        history = self.service.get_history()
        self.assertEqual(history[0]["id"], rec_id)
        self.assertEqual(history[0]["prediction"], "Uncertain")
        self.assertTrue(history[0]["is_uncertain"])

        # 3. Verify in system stats
        stats = self.service.get_system_stats()
        self.assertEqual(stats["total_analyses"], 1)
        self.assertEqual(stats["uncertain_detected"], 1)
        self.assertEqual(stats["real_detected"], 0)
        self.assertEqual(stats["fake_detected"], 0)
        self.assertEqual(stats["uncertain_percent"], 100.0)

    def test_probability_distribution_normalization(self):
        """Verify that Fake and Real probabilities always sum to 1.0 within numerical precision."""
        test_articles = [
            ("Economy", "WASHINGTON (Reuters) - Federal officials released the budget update."),
            ("Conspiracy", "SHOCKING! Reptilian overlords control government departments!"),
            ("Borderline", "Officials spoke to reporters about general matters during a routine weekly gathering."),
            ("Short Snippet", "The quick brown fox jumps over the lazy dog in the woods today.")
        ]

        for title, text in test_articles:
            pred = self.predictor.predict(title, text)
            p_fake = pred["probabilities"]["Fake"]
            p_real = pred["probabilities"]["Real"]
            self.assertAlmostEqual(p_fake + p_real, 1.0, places=3)
            self.assertGreaterEqual(p_fake, 0.0)
            self.assertLessEqual(p_fake, 1.0)
            self.assertGreaterEqual(p_real, 0.0)
            self.assertLessEqual(p_real, 1.0)

    # =========================================================================
    # 5. Explainable AI Feature Attribution Verification
    # =========================================================================

    def test_xai_mathematical_identity_and_regression_prevention(self):
        """Verify token contribution = round(weight * tfidf, 4) and directional consistency."""
        exp = explain_news(
            "Economic Reform Accord",
            "The president signed the budget legislation following extensive negotiations with congressional leaders."
        )

        self.assertIn("top_real_signals", exp)
        self.assertIn("top_fake_signals", exp)
        self.assertEqual(exp["attribution_type"], "approximate_linear_svm_decision_attribution")

        for signal in exp["top_real_signals"]:
            self.assertGreater(signal["contribution"], 0.0)
            expected = round(signal["weight"] * signal["tfidf"], 4)
            self.assertAlmostEqual(signal["contribution"], expected, places=3)

        for signal in exp["top_fake_signals"]:
            self.assertLess(signal["contribution"], 0.0)
            expected = round(signal["weight"] * signal["tfidf"], 4)
            self.assertAlmostEqual(signal["contribution"], expected, places=3)

    def test_xai_zero_signal_graceful_handling(self):
        """Verify that zero matched features returns empty signal lists without error."""
        exp = explain_news("", "xyzqwer asdfzxcv poiulkhj")
        self.assertEqual(exp["total_features_matched"], 0)
        self.assertEqual(exp["top_real_signals"], [])
        self.assertEqual(exp["top_fake_signals"], [])
        self.assertEqual(exp["net_lexical_bias"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
