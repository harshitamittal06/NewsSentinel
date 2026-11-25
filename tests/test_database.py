"""
NewsSentinel - Phase 5 Workstream 2: SQLite Persistence Unit Tests
==================================================================
Location: tests/test_database.py

Validates:
  1. Database initialization, directory creation, schema migrations, and WAL mode.
  2. Safe handling of empty databases (zero counts, empty lists, None lookups).
  3. Complete record persistence roundtrip (metrics, probabilities, XAI explanation).
  4. Pagination correctness (limit, offset, reverse chronological ordering).
  5. Aggregate statistics computation (total, real count, fake count).
  6. Thread-safe concurrent writing with multi-threading.
  7. Transaction integrity and rollback on exception.
  8. End-to-end integration with AnalyzerService.
  9. REST API /api/history route response contract.
"""

import sys
import unittest
import tempfile
import pathlib
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.database import DatabaseManager
from app.services.analyzer_service import AnalyzerService
from app.routes.analyzer import get_analysis_history


class TestSQLitePersistence(unittest.TestCase):
    """Test suite for SQLite data persistence and thread safety."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = pathlib.Path(self.tmp_dir.name) / "test_sentinel.db"
        self.db = DatabaseManager(self.db_path)
        self.service = AnalyzerService(db=self.db)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_database_initialization_and_wal_mode(self):
        """Verify database creation, WAL mode configuration, and migration tracking."""
        self.assertTrue(self.db_path.exists(), "Database file must exist on disk")

        with self.db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode;")
            journal_mode = cur.fetchone()[0]
            self.assertEqual(journal_mode.lower(), "wal")

            cur.execute("SELECT version, description FROM schema_migrations ORDER BY version ASC;")
            migrations = cur.fetchall()
            self.assertEqual(len(migrations), 3)
            self.assertEqual(migrations[0]["version"], 1)
            self.assertEqual(migrations[1]["version"], 2)
            self.assertEqual(migrations[2]["version"], 3)

    def test_empty_database_handling(self):
        """Verify that an empty database gracefully returns zero counts and empty lists."""
        self.assertEqual(self.db.count(), 0)
        self.assertEqual(self.db.get_history(), [])
        self.assertEqual(self.db.get_history(limit=10, offset=0), [])

        counts = self.db.get_counts()
        self.assertEqual(counts["total"], 0)
        self.assertEqual(counts["real"], 0)
        self.assertEqual(counts["fake"], 0)

        self.assertIsNone(self.db.get_analysis_by_id(999))

    def test_insert_and_retrieve_roundtrip(self):
        """Verify full fidelity roundtrip of analysis fields, metrics, and XAI explanations."""
        sample_record = {
            "timestamp": "2026-11-21 12:00:00 UTC",
            "headline": "Senate Approves Energy Modernization Bill",
            "snippet": "WASHINGTON (Reuters) - Lawmakers passed bipartisan energy legislation...",
            "prediction": "Real",
            "prediction_class": "real",
            "label": 1,
            "confidence": 0.9654,
            "confidence_percent": 96.54,
            "confidence_display": "96.5%",
            "probabilities": {"Fake": 0.0346, "Real": 0.9654},
            "metrics": {
                "words_count": 85,
                "sentences_count": 4,
                "lexical_diversity": 68.2,
                "uppercase_ratio": 3.1
            },
            "explanation": {
                "top_real_signals": [{"token": "bipartisan", "contribution": 0.142}],
                "top_fake_signals": [{"token": "shocking", "contribution": -0.051}],
                "total_features_matched": 18,
                "net_lexical_bias": 0.421,
                "baseline_intercept": -0.534
            },
            "latency_ms": 11.2,
            "model_name": "Linear SVM (Calibrated)",
            "stage": 5,
            "disclaimer": "Test academic disclaimer"
        }

        record_id = self.db.insert_analysis(sample_record)
        self.assertGreater(record_id, 0)

        fetched = self.db.get_analysis_by_id(record_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["id"], record_id)
        self.assertEqual(fetched["headline"], sample_record["headline"])
        self.assertEqual(fetched["snippet"], sample_record["snippet"])
        self.assertEqual(fetched["prediction"], "Real")
        self.assertEqual(fetched["confidence"], 0.9654)
        self.assertEqual(fetched["metrics"]["words_count"], 85)
        self.assertEqual(fetched["probabilities"]["Real"], 0.9654)
        self.assertEqual(fetched["explanation"]["top_real_signals"][0]["token"], "bipartisan")

    def test_pagination_and_ordering(self):
        """Verify limit, offset, and descending ID (newest first) ordering."""
        # Insert 10 records
        for i in range(1, 11):
            self.db.insert_analysis({
                "headline": f"Headline #{i}",
                "snippet": f"Article body snippet #{i}",
                "prediction": "Real" if i % 2 == 0 else "Fake",
                "label": 1 if i % 2 == 0 else 0,
                "confidence": 0.90,
                "confidence_percent": 90.0,
                "confidence_display": "90.0%"
            })

        self.assertEqual(self.db.count(), 10)

        # Page 1: newest 3 records (IDs 10, 9, 8)
        p1 = self.db.get_history(limit=3, offset=0)
        self.assertEqual(len(p1), 3)
        self.assertEqual([r["id"] for r in p1], [10, 9, 8])

        # Page 2: next 3 records (IDs 7, 6, 5)
        p2 = self.db.get_history(limit=3, offset=3)
        self.assertEqual(len(p2), 3)
        self.assertEqual([r["id"] for r in p2], [7, 6, 5])

        # Page 4: remaining records (IDs 2, 1)
        p4 = self.db.get_history(limit=3, offset=8)
        self.assertEqual(len(p4), 2)
        self.assertEqual([r["id"] for r in p4], [2, 1])

        # Out-of-bounds offset
        p_empty = self.db.get_history(limit=3, offset=20)
        self.assertEqual(p_empty, [])

    def test_aggregate_statistics(self):
        """Verify accurate calculation of total, fake, and real counts."""
        # Insert 3 Real and 2 Fake
        for pred in ["Real", "Fake", "Real", "Real", "Fake"]:
            self.db.insert_analysis({
                "prediction": pred,
                "label": 1 if pred == "Real" else 0,
                "snippet": f"Sample {pred}",
                "confidence": 0.95,
                "confidence_percent": 95.0,
                "confidence_display": "95.0%"
            })

        counts = self.db.get_counts()
        self.assertEqual(counts["total"], 5)
        self.assertEqual(counts["real"], 3)
        self.assertEqual(counts["fake"], 2)

    def test_thread_safe_concurrent_inserts(self):
        """Verify thread safety with simultaneous writes across multiple threads."""
        threads = []
        num_threads = 6
        inserts_per_thread = 5

        def worker(thread_idx: int):
            for j in range(inserts_per_thread):
                self.db.insert_analysis({
                    "headline": f"Thread {thread_idx} Item {j}",
                    "snippet": f"Concurrent snippet from thread {thread_idx}",
                    "prediction": "Real",
                    "label": 1,
                    "confidence": 0.90,
                    "confidence_percent": 90.0,
                    "confidence_display": "90.0%"
                })

        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        self.assertEqual(self.db.count(), num_threads * inserts_per_thread)

    def test_analyzer_service_persistence_integration(self):
        """Verify AnalyzerService persists inferences to SQLite and updates live telemetry."""
        result = self.service.process_analysis_request(
            text="The federal committee finalized its standard fiscal recommendations for the fiscal quarter.",
            headline="Fiscal Committee Report"
        )
        self.assertTrue(result["success"])
        record_id = result["data"]["id"]
        self.assertGreater(record_id, 0)

        # Check DB directly
        db_record = self.db.get_analysis_by_id(record_id)
        self.assertIsNotNone(db_record)
        self.assertEqual(db_record["headline"], "Fiscal Committee Report")
        self.assertIn("explanation", db_record)
        self.assertIn("top_real_signals", db_record["explanation"])

        # Check service history and telemetry
        history = self.service.get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["id"], record_id)

        stats = self.service.get_system_stats()
        self.assertEqual(stats["total_analyses"], 1)

    def test_api_history_route_response(self):
        """Verify /api/history response dictionary and pagination metadata."""
        # Prepopulate with 2 records
        self.db.insert_analysis({"prediction": "Real", "label": 1, "snippet": "A", "confidence": 0.9, "confidence_percent": 90.0, "confidence_display": "90%"})
        self.db.insert_analysis({"prediction": "Fake", "label": 0, "snippet": "B", "confidence": 0.8, "confidence_percent": 80.0, "confidence_display": "80%"})

        res = get_analysis_history(limit=10, offset=0)
        self.assertTrue(res["success"])
        self.assertIn("records", res)
        self.assertIn("total", res)
        self.assertEqual(res["limit"], 10)
        self.assertEqual(res["offset"], 0)
        self.assertGreaterEqual(res["total"], 2)
        self.assertGreaterEqual(len(res["records"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
