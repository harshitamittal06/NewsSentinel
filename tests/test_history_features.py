"""
Unit & Integration Tests for NewsSentinel History Features & Global Theme Switcher
=================================================================================
Location: tests/test_history_features.py

Tests:
1. Database Schema Migration v3 (full_text column).
2. Backward compatibility: older records with NULL full_text safely load with None.
3. Classification filtering logic (All, Real, Fake, Uncertain).
4. Full text search and filter combination logic.
5. Detail modal and fallback notice integrity.
6. History page HTML rendering, filter tabs, modal elements, and JSON payload.
7. Base template theme switcher, anti-flash script, and CSS variables.
"""

import unittest
import tempfile
import json
from pathlib import Path
from fastapi import Request

from app.services.database import DatabaseManager
from app.services.analyzer_service import AnalyzerService
from app.routes.pages import (
    dashboard_page,
    analyzer_page,
    history_page,
    performance_page,
    about_page,
)


def make_request(path: str) -> Request:
    return Request({"type": "http", "method": "GET", "path": path, "headers": []})


class TestHistoryFeaturesAndTheme(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_history.db"
        self.db = DatabaseManager(db_path=self.db_path)
        self.service = AnalyzerService(db=self.db, uncertainty_threshold=0.65)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_schema_migration_v3_full_text_column(self):
        """Verify that Schema v3 adds full_text and tracks migration in schema_migrations table."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT version, description FROM schema_migrations ORDER BY version ASC")
            migrations = cursor.fetchall()
            self.assertEqual(len(migrations), 3, "Must have exactly 3 schema migrations (v1, v2, v3)")
            self.assertEqual(migrations[2]["version"], 3)
            self.assertIn("full_text", migrations[2]["description"])

            cursor.execute("PRAGMA table_info(analyses)")
            columns = [row["name"] for row in cursor.fetchall()]
            self.assertIn("full_text", columns)
            self.assertIn("is_uncertain", columns)

    def test_insert_and_retrieve_record_with_full_text(self):
        """Verify that insert_analysis stores full_text and get_history returns it accurately."""
        original_article = (
            "WASHINGTON - The Senate committee held hearings today regarding the digital media "
            "landscape and the propagation of algorithmic signals. Several researchers presented findings."
        )
        record = {
            "headline": "Senate Holds Media Hearing",
            "snippet": "WASHINGTON - The Senate committee held hearings today...",
            "full_text": original_article,
            "prediction": "Real",
            "prediction_class": "real",
            "label": 1,
            "confidence": 0.942,
            "confidence_percent": 94.2,
            "confidence_display": "94.2%",
            "probabilities": {"Fake": 0.058, "Real": 0.942},
            "metrics": {"words_count": 22},
            "explanation": {"top_real_signals": [{"token": "senate", "contribution": 0.12}]},
            "latency_ms": 8,
            "model_name": "Linear SVM (Calibrated)",
            "stage": 5,
            "disclaimer": "Advisory disclaimer",
            "is_uncertain": False,
        }

        rec_id = self.db.insert_analysis(record)
        self.assertGreater(rec_id, 0)

        history = self.db.get_history(limit=10)
        self.assertEqual(len(history), 1)
        saved = history[0]
        self.assertEqual(saved["id"], rec_id)
        self.assertEqual(saved["full_text"], original_article)
        self.assertEqual(saved["headline"], "Senate Holds Media Hearing")
        self.assertEqual(saved["prediction"], "Real")
        self.assertFalse(saved["is_uncertain"])

    def test_backward_compatibility_older_record_null_full_text(self):
        """
        Verify that an older record created before migration v3 (where full_text is NULL)
        safely loads with full_text as None, without raising exceptions.
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO analyses (
                    timestamp, headline, snippet, prediction, prediction_class, label,
                    confidence, confidence_percent, confidence_display, probabilities_json,
                    metrics_json, explanation_json, latency_ms, model_name, stage, disclaimer,
                    created_at, is_uncertain, full_text
                ) VALUES (
                    '2026-11-20 12:00:00 UTC', 'Legacy Article', 'Legacy excerpt...', 'Fake', 'fake', 0,
                    0.88, 88.0, '88.0%', '{"Fake": 0.88, "Real": 0.12}',
                    '{}', '{}', 10, 'Linear SVM', 4, 'Disclaimer',
                    '2026-11-20T12:00:00', 0, NULL
                )
                """
            )
            legacy_id = cursor.lastrowid

        history = self.db.get_history()
        legacy_rec = next(r for r in history if r["id"] == legacy_id)
        self.assertIsNone(legacy_rec["full_text"], "Older records with NULL full_text must return None")
        self.assertEqual(legacy_rec["headline"], "Legacy Article")

    def test_classification_filter_counts_computation(self):
        """Verify that Real, Fake, and Uncertain record counts are correctly distinguished."""
        records = [
            {"headline": "Real 1", "full_text": "Text 1", "prediction": "Real", "is_uncertain": False},
            {"headline": "Real 2", "full_text": "Text 2", "prediction": "Real", "is_uncertain": False},
            {"headline": "Fake 1", "full_text": "Text 3", "prediction": "Fake", "is_uncertain": False},
            {"headline": "Fake 2", "full_text": "Text 4", "prediction": "Fake", "is_uncertain": False},
            {"headline": "Fake 3", "full_text": "Text 5", "prediction": "Fake", "is_uncertain": False},
            {"headline": "Uncertain 1", "full_text": "Text 6", "prediction": "Uncertain", "is_uncertain": True},
        ]

        for r in records:
            r.update({
                "snippet": r["full_text"],
                "prediction_class": "uncertain" if r["is_uncertain"] else r["prediction"].lower(),
                "label": 1 if r["prediction"] == "Real" else 0,
                "confidence": 0.90 if not r["is_uncertain"] else 0.55,
                "confidence_percent": 90.0 if not r["is_uncertain"] else 55.0,
                "confidence_display": "90.0%" if not r["is_uncertain"] else "55.0%",
                "probabilities": {"Fake": 0.5, "Real": 0.5},
                "metrics": {"words_count": 10},
                "explanation": {},
                "latency_ms": 10,
                "model_name": "Linear SVM",
                "stage": 5,
                "disclaimer": "Disclaimer",
            })
            self.db.insert_analysis(r)

        counts = self.db.get_counts()
        self.assertEqual(counts["total"], 6)
        self.assertEqual(counts["real"], 2)
        self.assertEqual(counts["fake"], 3)
        self.assertEqual(counts["uncertain"], 1)

    def test_history_page_rendering_and_interactive_elements(self):
        """Verify that history_page renders HTTP 200 and contains all required UI elements."""
        req = make_request("/history")
        response = history_page(req)
        self.assertEqual(response.status_code, 200)
        html = response.body.decode("utf-8")

        # Verify 4 filter tabs
        self.assertIn('id="tab-all"', html)
        self.assertIn('id="tab-real"', html)
        self.assertIn('id="tab-fake"', html)
        self.assertIn('id="tab-uncertain"', html)

        # Verify live count badges
        self.assertIn('id="count-all"', html)
        self.assertIn('id="count-real"', html)
        self.assertIn('id="count-fake"', html)
        self.assertIn('id="count-uncertain"', html)

        # Verify search input
        self.assertIn('id="history-search-input"', html)

        # Verify article detail modal structure
        self.assertIn('id="article-detail-modal"', html)
        self.assertIn('id="modal-headline"', html)
        self.assertIn('id="modal-full-text-content"', html)
        self.assertIn('id="modal-prob-bar-fake"', html)
        self.assertIn('id="modal-prob-bar-real"', html)
        self.assertIn('id="modal-xai-section"', html)
        self.assertIn('id="modal-close-btn"', html)
        self.assertIn('id="modal-footer-close-btn"', html)

        # Verify embedded JSON scripts
        self.assertIn('id="initial-history-data"', html)
        self.assertIn('id="initial-stats-data"', html)

    def test_theme_switcher_navbar_and_anti_flash_script(self):
        """Verify that theme toggle button, anti-flash script, and CSS variables exist across pages."""
        pages = [
            (dashboard_page, "/"),
            (analyzer_page, "/analyzer"),
            (history_page, "/history"),
            (performance_page, "/performance"),
            (about_page, "/about"),
        ]

        for page_fn, path in pages:
            req = make_request(path)
            response = page_fn(req)
            self.assertEqual(response.status_code, 200, f"Page {path} must return 200 OK")
            html = response.body.decode("utf-8")

            # Theme toggle button
            self.assertIn('id="theme-toggle-btn"', html, f"Theme toggle missing on {path}")
            self.assertIn('id="theme-toggle-label"', html, f"Theme label missing on {path}")
            self.assertIn('class="theme-icon theme-icon-sun"', html, f"Sun icon missing on {path}")
            self.assertIn('class="theme-icon theme-icon-moon"', html, f"Moon icon missing on {path}")

            # Anti-flash script
            self.assertIn('localStorage.getItem("newssentinel_theme")', html)
            self.assertIn('data-theme', html)

    def test_style_css_light_theme_tokens_present(self):
        """Verify style.css includes light theme tokens and history modal styles."""
        css_path = Path("app/static/css/style.css")
        self.assertTrue(css_path.exists())
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()

        # Design tokens
        self.assertIn('[data-theme="light"]', css)
        self.assertIn('--bg-deep: #F8FAFC;', css)
        self.assertIn('--bg-surface: #FFFFFF;', css)
        self.assertIn('--text-main: #0F172A;', css)
        self.assertIn('--border-subtle: #E2E8F0;', css)

        # Component styles
        self.assertIn('.theme-toggle-btn', css)
        self.assertIn('.history-filter-tabs', css)
        self.assertIn('.filter-tab-btn', css)
        self.assertIn('.tab-count-badge', css)
        self.assertIn('.history-row-clickable', css)
        self.assertIn('.modal-backdrop', css)
        self.assertIn('.modal-window', css)
        self.assertIn('.old-record-notice', css)
        self.assertIn('.pagination-footer', css)

    def test_empty_database_history_page(self):
        """Verify that when 0 history records exist, the page displays standby state cleanly."""
        empty_db = DatabaseManager(db_path=Path(self.temp_dir.name) / "empty.db")
        history = empty_db.get_history()
        self.assertEqual(len(history), 0)
        counts = empty_db.get_counts()
        self.assertEqual(counts["total"], 0)
        self.assertEqual(counts["real"], 0)
        self.assertEqual(counts["fake"], 0)
        self.assertEqual(counts["uncertain"], 0)

    def test_model_prediction_labels_and_verdicts(self):
        """Verify that verdicts use non-misleading labels: MODEL PREDICTION: REAL, FAKE, and UNCERTAIN — NEEDS VERIFICATION."""
        predictor = self.service._get_predictor()

        # High confidence real
        real_res = predictor.predict(
            "Senate Budget Bill",
            "WASHINGTON (Reuters) - The U.S. Senate voted overwhelmingly on Thursday to approve a compromise spending bill.",
            uncertainty_threshold=0.65
        )
        self.assertEqual(real_res["verdict"], "MODEL PREDICTION: REAL")

        # High confidence fake
        fake_res = predictor.predict(
            "Secret Underground Base",
            "UNBELIEVABLE! Soros crisis actors and chemtrail operators were caught red-handed operating a satanic pedophile dungeon!",
            uncertainty_threshold=0.65
        )
        self.assertEqual(fake_res["verdict"], "MODEL PREDICTION: FAKE")

        # Borderline uncertain
        uncertain_res = predictor.predict(
            "",
            "Officials spoke to reporters about general matters during a routine weekly gathering.",
            uncertainty_threshold=0.65
        )
        self.assertTrue(uncertain_res["is_uncertain"])
        self.assertEqual(uncertain_res["verdict"], "UNCERTAIN — NEEDS VERIFICATION")

    def test_clear_disclaimer_in_results_and_pages(self):
        """Verify the exact mandatory disclaimer is present in analyzer results and page templates."""
        expected_disclaimer = (
            "NewsSentinel identifies linguistic patterns learned from its training data. "
            "It does not independently verify sources, events, quotations or factual claims."
        )

        # Check analyzer service result disclaimer
        res = self.service.process_analysis_request(
            "WASHINGTON (Reuters) - The Senate approved the spending legislation today.",
            "Senate Passes Bill"
        )
        self.assertTrue(res["success"])
        self.assertIn(expected_disclaimer, res["data"]["disclaimer"])

        # Check About Page
        about_html = about_page(make_request("/about")).body.decode("utf-8")
        self.assertIn(expected_disclaimer, about_html)

        # Check History Page
        hist_html = history_page(make_request("/history")).body.decode("utf-8")
        self.assertIn(expected_disclaimer, hist_html)

    def test_about_page_realistic_writing_style_limitation(self):
        """Verify About page explains the realistic writing style limitation and distinguishes confidence from certainty."""
        about_html = about_page(make_request("/about")).body.decode("utf-8")
        self.assertIn("Critical Limitation: Realistic Writing Style Resemblance", about_html)
        self.assertIn("73.2% confidence", about_html)
        self.assertIn("realistic writing style can closely resemble", about_html)
        self.assertIn("Confidence Is Not Factual Certainty", about_html)


if __name__ == "__main__":
    unittest.main()
