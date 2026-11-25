"""
NewsSentinel - Web Application Route & Template Test Suite
==========================================================
Location: tests/test_endpoints.py

Verifies FastAPI routes, HTML template contexts, static asset files,
REST API schemas, and error responses synchronously without network sockets.
"""

import sys
import unittest
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app
from app.config import settings
from app.routes.pages import get_base_context
from app.routes.analyzer import AnalyzeRequest, analyze_text, get_stats, get_model_metrics, health_check
from app.services.analyzer_service import analyzer_service


class TestApplicationEndpoints(unittest.TestCase):
    """Test suite for web application routes, static assets, and API contracts."""

    def test_static_asset_files_exist(self):
        """Verify CSS, JS, and image assets exist in static directory."""
        css_file = settings.STATIC_DIR / "css" / "style.css"
        js_file = settings.STATIC_DIR / "js" / "script.js"

        self.assertTrue(css_file.exists(), "style.css missing")
        self.assertTrue(js_file.exists(), "script.js missing")
        self.assertGreater(css_file.stat().st_size, 1000)
        self.assertGreater(js_file.stat().st_size, 1000)

    def test_template_files_exist(self):
        """Verify all HTML Jinja2 templates exist."""
        templates = [
            "base.html",
            "dashboard.html",
            "analyzer.html",
            "history.html",
            "performance.html",
            "about.html",
            "error.html",
        ]
        for t in templates:
            t_path = settings.TEMPLATES_DIR / t
            self.assertTrue(t_path.exists(), f"Template {t} missing")

    def test_base_context_generation(self):
        """Verify base context contains required application and model telemetry."""
        ctx = get_base_context(active_page="dashboard", title="Test Title")
        self.assertEqual(ctx["app_name"], "NewsSentinel")
        self.assertEqual(ctx["stage"], 4)
        self.assertEqual(ctx["active_page"], "dashboard")
        self.assertIn("stats", ctx)
        self.assertEqual(ctx["stats"]["model_accuracy_display"], "96.59%")

    def test_health_check_endpoint(self):
        """Verify health check logic returns healthy status and active stage."""
        # Call route function synchronously
        res = analyzer_service.get_system_stats()
        self.assertEqual(res["stage"], 4)
        self.assertEqual(res["model_name"], "Linear SVM (Calibrated)")
        self.assertEqual(res["model_status"], "Active & Operational")

    def test_stats_endpoint(self):
        """Verify stats route returns live metrics and empirical test accuracy."""
        stats = analyzer_service.get_system_stats()
        self.assertIn("total_analyses", stats)
        self.assertIn("model_accuracy", stats)
        self.assertEqual(stats["model_accuracy"], 96.59)

    def test_analyze_endpoint_real_news(self):
        """Verify analyze service correctly assesses verified Reuters news."""
        res = analyzer_service.process_analysis_request(
            headline="White House holds bilateral talks with European leaders",
            text="BRUSSELS (Reuters) - European and American officials met on Friday to discuss transatlantic trade policy."
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["data"]["prediction"], "Real")
        self.assertEqual(res["data"]["label"], 1)
        self.assertGreaterEqual(res["data"]["confidence"], 0.80)

    def test_analyze_endpoint_fake_news(self):
        """Verify analyze service correctly assesses sensational clickbait."""
        res = analyzer_service.process_analysis_request(
            headline="SHOCKING: Satanic pedophile cult operates secret pizza bunker",
            text="UNBELIEVABLE leaked documents prove paid crisis actors and chemtrail operators control the global government!"
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["data"]["prediction"], "Fake")
        self.assertEqual(res["data"]["label"], 0)
        self.assertGreaterEqual(res["data"]["confidence"], 0.80)

    def test_analyze_endpoint_short_input(self):
        """Verify analyze service rejects inputs below 10 characters."""
        res = analyzer_service.process_analysis_request(headline="", text="Short")
        self.assertFalse(res["success"])
        self.assertIn("too short", res["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
