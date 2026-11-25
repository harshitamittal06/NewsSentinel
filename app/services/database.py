"""
NewsSentinel - SQLite Database Service
======================================
Location: app/services/database.py

Provides thread-safe persistence for news analysis records using Python's
built-in sqlite3 with Write-Ahead Logging (WAL) mode, connection pooling via
context management, schema migrations, and paginated query capabilities.
"""

import os
import sys
import json
import sqlite3
import threading
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "news_sentinel.db"

CURRENT_SCHEMA_VERSION = 3


class DatabaseManager:
    """Thread-safe SQLite manager for NewsSentinel analysis telemetry and history."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._lock = threading.Lock()
        self._ensure_db_dir()
        self.init_db()

    def _ensure_db_dir(self) -> None:
        """Ensure parent directory for the database file exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def get_connection(self):
        """
        Context manager for acquiring a thread-isolated SQLite connection.
        Enforces WAL mode, foreign keys, and automatic transaction management.
        """
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self) -> None:
        """Initializes database schema and executes necessary migrations safely."""
        with self._lock:
            with self.get_connection() as conn:
                cur = conn.cursor()

                # Migration tracking table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version INTEGER PRIMARY KEY,
                        description TEXT NOT NULL,
                        applied_at TEXT NOT NULL
                    )
                """)

                # Analysis history table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS analyses (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        headline TEXT,
                        snippet TEXT NOT NULL,
                        prediction TEXT NOT NULL,
                        prediction_class TEXT NOT NULL,
                        label INTEGER NOT NULL,
                        confidence REAL NOT NULL,
                        confidence_percent REAL NOT NULL,
                        confidence_display TEXT NOT NULL,
                        probabilities_json TEXT NOT NULL,
                        metrics_json TEXT NOT NULL,
                        explanation_json TEXT,
                        latency_ms REAL NOT NULL,
                        model_name TEXT NOT NULL,
                        stage INTEGER DEFAULT 5,
                        disclaimer TEXT NOT NULL,
                        is_uncertain INTEGER DEFAULT 0,
                        full_text TEXT,
                        created_at TEXT DEFAULT (datetime('now', 'utc'))
                    )
                """)

                # Performance indices for sorting and filtering
                cur.execute("CREATE INDEX IF NOT EXISTS idx_analyses_id_desc ON analyses(id DESC)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_analyses_prediction ON analyses(prediction)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_analyses_timestamp ON analyses(timestamp)")

                # Migration v1
                cur.execute("SELECT version FROM schema_migrations WHERE version = 1")
                if not cur.fetchone():
                    cur.execute("""
                        INSERT INTO schema_migrations (version, description, applied_at)
                        VALUES (1, 'Initial schema for analyses with WAL mode and XAI explanation JSON', ?)
                    """, (datetime.now(timezone.utc).isoformat(),))

                # Migration v2 (Workstream 3: Uncertainty Detection)
                cur.execute("SELECT version FROM schema_migrations WHERE version = 2")
                if not cur.fetchone():
                    cur.execute("PRAGMA table_info(analyses)")
                    columns = [col[1] for col in cur.fetchall()]
                    if "is_uncertain" not in columns:
                        cur.execute("ALTER TABLE analyses ADD COLUMN is_uncertain INTEGER DEFAULT 0")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_analyses_uncertain ON analyses(is_uncertain)")
                    cur.execute("""
                        INSERT INTO schema_migrations (version, description, applied_at)
                        VALUES (2, 'Add is_uncertain flag and index for Workstream 3', ?)
                    """, (datetime.now(timezone.utc).isoformat(),))
                else:
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_analyses_uncertain ON analyses(is_uncertain)")

                # Migration v3: Full original article text preservation
                cur.execute("SELECT version FROM schema_migrations WHERE version = 3")
                if not cur.fetchone():
                    cur.execute("PRAGMA table_info(analyses)")
                    columns = [col[1] for col in cur.fetchall()]
                    if "full_text" not in columns:
                        cur.execute("ALTER TABLE analyses ADD COLUMN full_text TEXT")
                    cur.execute("""
                        INSERT INTO schema_migrations (version, description, applied_at)
                        VALUES (3, 'Add full_text column for complete original article text storage', ?)
                    """, (datetime.now(timezone.utc).isoformat(),))

    def insert_analysis(self, record: Dict[str, Any]) -> int:
        """
        Persists an analysis record into SQLite.
        Returns the generated database primary key ID.
        """
        probabilities_json = json.dumps(record.get("probabilities", {}))
        metrics_json = json.dumps(record.get("metrics", {}))
        explanation_json = json.dumps(record.get("explanation", {}))
        full_text_val = record.get("full_text") or record.get("text") or ""

        is_uncertain = 1 if record.get("is_uncertain") or record.get("prediction") == "Uncertain" else 0

        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO analyses (
                    timestamp,
                    headline,
                    snippet,
                    prediction,
                    prediction_class,
                    label,
                    confidence,
                    confidence_percent,
                    confidence_display,
                    probabilities_json,
                    metrics_json,
                    explanation_json,
                    latency_ms,
                    model_name,
                    stage,
                    disclaimer,
                    is_uncertain,
                    full_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.get("timestamp", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")),
                record.get("headline", ""),
                record.get("snippet", ""),
                record.get("prediction", "Unknown"),
                record.get("prediction_class", "unknown"),
                int(record.get("label", 0)),
                float(record.get("confidence", 0.0)),
                float(record.get("confidence_percent", 0.0)),
                record.get("confidence_display", "0.0%"),
                probabilities_json,
                metrics_json,
                explanation_json,
                float(record.get("latency_ms", 0.0)),
                record.get("model_name", "Linear SVM (Calibrated)"),
                int(record.get("stage", 5)),
                record.get("disclaimer", ""),
                is_uncertain,
                full_text_val
            ))
            return cur.lastrowid

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Converts an SQLite row back into the full analysis record dictionary."""
        metrics = json.loads(row["metrics_json"]) if row["metrics_json"] else {}
        probabilities = json.loads(row["probabilities_json"]) if row["probabilities_json"] else {}
        explanation = json.loads(row["explanation_json"]) if row["explanation_json"] else {}

        # Safe extraction of is_uncertain and full_text columns
        row_keys = row.keys()
        is_uncertain = bool(row["is_uncertain"]) if "is_uncertain" in row_keys else (row["prediction"] == "Uncertain")
        full_text = row["full_text"] if "full_text" in row_keys and row["full_text"] else None

        return {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "headline": row["headline"] or "",
            "snippet": row["snippet"],
            "full_text": full_text,
            "metrics": metrics,
            "prediction": row["prediction"],
            "prediction_class": row["prediction_class"],
            "is_uncertain": is_uncertain,
            "raw_prediction": "Real" if row["label"] == 1 else "Fake",
            "label": row["label"],
            "confidence": row["confidence"],
            "confidence_percent": row["confidence_percent"],
            "confidence_display": row["confidence_display"],
            "probabilities": probabilities,
            "latency_ms": row["latency_ms"],
            "explanation": explanation,
            "model_name": row["model_name"],
            "stage": row["stage"],
            "disclaimer": row["disclaimer"]
        }

    def get_history(self, limit: Optional[int] = None, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Retrieves analysis history records ordered newest first (id DESC).
        Supports pagination via limit and offset.
        """
        query = "SELECT * FROM analyses ORDER BY id DESC"
        params: List[Any] = []

        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params.extend([max(1, limit), max(0, offset)])
        elif offset > 0:
            query += " LIMIT -1 OFFSET ?"
            params.append(max(0, offset))

        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(query, params)
            rows = cur.fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_analysis_by_id(self, record_id: int) -> Optional[Dict[str, Any]]:
        """Fetches a specific record by its primary key ID."""
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM analyses WHERE id = ?", (record_id,))
            row = cur.fetchone()
            if row:
                return self._row_to_dict(row)
        return None

    def get_counts(self) -> Dict[str, int]:
        """
        Returns live aggregated statistics: total count, fake count, real count, and uncertain count.
        Safe and performant on empty tables.
        """
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN (is_uncertain = 0 OR is_uncertain IS NULL) AND prediction = 'Real' THEN 1 ELSE 0 END) as real_count,
                    SUM(CASE WHEN (is_uncertain = 0 OR is_uncertain IS NULL) AND prediction = 'Fake' THEN 1 ELSE 0 END) as fake_count,
                    SUM(CASE WHEN is_uncertain = 1 OR prediction = 'Uncertain' THEN 1 ELSE 0 END) as uncertain_count
                FROM analyses
            """)
            row = cur.fetchone()
            total = row["total"] or 0
            real_count = row["real_count"] or 0
            fake_count = row["fake_count"] or 0
            uncertain_count = row["uncertain_count"] or 0

            return {
                "total": total,
                "real": real_count,
                "fake": fake_count,
                "uncertain": uncertain_count
            }

    def count(self) -> int:
        """Returns the total number of records in the database."""
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM analyses")
            return cur.fetchone()[0] or 0

    def clear_history(self) -> int:
        """Deletes all records from the analyses table. Returns number of rows deleted."""
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM analyses")
            deleted = cur.rowcount
            # Reset auto-increment sequence
            cur.execute("DELETE FROM sqlite_sequence WHERE name = 'analyses'")
            return deleted


# Default shared database manager singleton
db_manager = DatabaseManager()
