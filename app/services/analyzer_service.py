"""
NewsSentinel - Text Analysis & Prediction Service
Stage 4: Full ML Model Integration & Real Telemetry
===================================================
Connects the trained Linear SVM model and TF-IDF vectorizer (from Phase 3)
to the web application backend. Computes real-time inference, probability
breakdowns, linguistic metrics, and maintains session analysis history.
"""

import re
import math
import json
import time
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timezone
from pathlib import Path

from app.services.database import db_manager, DatabaseManager
from src.predictor import get_predictor, NewsPredictor
from app.config import settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
COMPARISON_PATH = PROJECT_ROOT / "reports" / "model_comparison.json"


class AnalyzerService:
    def __init__(self, db: Optional[DatabaseManager] = None, uncertainty_threshold: Optional[float] = None):
        self._db: DatabaseManager = db or db_manager
        self._analysis_log: List[Dict[str, Any]] = list(reversed(self._db.get_history()))
        self._metadata = self._load_metadata()
        self._comparison = self._load_comparison()
        self._threshold = uncertainty_threshold if uncertainty_threshold is not None else getattr(settings, "UNCERTAINTY_THRESHOLD", 0.65)
        self._predictor: Optional[NewsPredictor] = None

    def _load_metadata(self) -> Dict[str, Any]:
        """Loads Phase 3 trained model metadata from disk."""
        if METADATA_PATH.exists():
            try:
                with open(METADATA_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _load_comparison(self) -> Dict[str, Any]:
        """Loads Phase 3 candidate model comparison metrics from disk."""
        if COMPARISON_PATH.exists():
            try:
                with open(COMPARISON_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _get_predictor(self) -> NewsPredictor:
        """Lazy-loads the model predictor singleton."""
        if self._predictor is None:
            self._predictor = get_predictor()
        return self._predictor

    def validate_input(self, text: str, headline: str = "") -> Tuple[bool, Optional[str]]:
        """Validate submitted text and headline length and structure."""
        combined = f"{headline or ''} {text or ''}".strip()
        if not combined:
            return False, "Input cannot be empty. Please provide a headline, article body, or claim."

        if len(combined) < 10:
            return False, "Input is too short for meaningful analysis (minimum 10 characters required)."

        if len(combined) > 50000:
            return False, "Input exceeds the maximum allowed length of 50,000 characters."

        return True, None

    def extract_linguistic_metrics(self, text: str) -> Dict[str, Any]:
        """
        Extract deterministic linguistic and stylometric features.
        Quantifies structural traits that correlate with credibility patterns.
        """
        cleaned = text.strip()
        chars_count = len(cleaned)

        words = re.findall(r"\b[a-zA-Z0-9_\'-]+\b", cleaned)
        word_count = len(words)

        sentences = [s.strip() for s in re.split(r"[.!?]+", cleaned) if s.strip()]
        sentence_count = max(1, len(sentences))

        alpha_chars = [c for c in cleaned if c.isalpha()]
        uppercase_chars = [c for c in alpha_chars if c.isupper()]
        uppercase_ratio = round((len(uppercase_chars) / len(alpha_chars)) * 100, 1) if alpha_chars else 0.0

        exclamation_count = cleaned.count("!")
        question_count = cleaned.count("?")

        avg_word_len = round(sum(len(w) for w in words) / word_count, 1) if word_count > 0 else 0.0
        unique_words = len(set(w.lower() for w in words))
        lexical_diversity = round((unique_words / word_count) * 100, 1) if word_count > 0 else 0.0
        reading_time_sec = max(1, math.ceil((word_count / 200) * 60)) if word_count > 0 else 0

        return {
            "characters_count": chars_count,
            "words_count": word_count,
            "sentences_count": sentence_count,
            "avg_word_length": avg_word_len,
            "lexical_diversity": lexical_diversity,
            "uppercase_ratio": uppercase_ratio,
            "exclamation_count": exclamation_count,
            "question_count": question_count,
            "reading_time_sec": reading_time_sec,
            "reading_time_display": f"{reading_time_sec}s" if reading_time_sec < 60 else f"{round(reading_time_sec / 60, 1)}m",
        }

    def process_analysis_request(self, text: str, headline: str = "") -> Dict[str, Any]:
        """
        Executes real-time machine learning inference using the Phase 3 trained champion model.
        Returns prediction, confidence percentage, class probabilities, and linguistic metrics.
        """
        is_valid, error = self.validate_input(text, headline)
        if not is_valid:
            return {
                "success": False,
                "error": error,
                "data": None
            }

        # Combined text for linguistic feature extraction
        full_text = f"{headline or ''} {text or ''}".strip()
        metrics = self.extract_linguistic_metrics(full_text)

        # Display snippet
        snippet_src = headline if headline else text
        snippet = snippet_src.strip()[:140] + "..." if len(snippet_src.strip()) > 140 else snippet_src.strip()

        # Run real-time machine learning inference
        try:
            predictor = self._get_predictor()
            pred = predictor.predict(
                title=headline or "",
                text=text or "",
                uncertainty_threshold=self._threshold
            )
        except Exception as e:
            return {
                "success": False,
                "error": f"Model inference error: {str(e)}",
                "data": None
            }

        pred_class = "uncertain" if pred.get("is_uncertain") else pred["prediction"].lower()

        record = {
            "id": 0,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "headline": headline or "",
            "snippet": snippet,
            "full_text": text or "",
            "metrics": metrics,
            "prediction": pred["prediction"],                 # "Real", "Fake", or "Uncertain"
            "raw_prediction": pred.get("raw_prediction", pred["prediction"]),
            "prediction_class": pred_class,                   # "real", "fake", or "uncertain"
            "is_uncertain": pred.get("is_uncertain", False),
            "verdict": pred.get("verdict", ""),
            "label": pred["label"],                           # 1 or 0
            "confidence": pred["confidence"],                 # e.g. 0.985
            "confidence_percent": pred["confidence_percent"], # e.g. 98.5
            "confidence_display": f"{pred['confidence_percent']:.1f}%",
            "probabilities": pred["probabilities"],           # {"Fake": 0.015, "Real": 0.985}
            "latency_ms": pred["latency_ms"],
            "explanation": pred.get("explanation", {}),
            "model_name": self._metadata.get("model_name", "Linear SVM (Calibrated)"),
            "stage": 5,
            "disclaimer": (
                "NewsSentinel identifies linguistic patterns learned from its training data. "
                "It does not independently verify sources, events, quotations or factual claims. "
                "Model confidence indicates pattern matching probability within the WELFake training corpus, "
                "not factual certainty. Predictions are ML-based statistical signals, not definitive fact-checking."
            )
        }

        # Persist in SQLite database
        try:
            db_id = self._db.insert_analysis(record)
            record["id"] = db_id
        except Exception:
            record["id"] = len(self._analysis_log) + 1

        # Store in historical in-memory cache
        self._analysis_log.append(record)

        return {
            "success": True,
            "error": None,
            "data": record
        }

    def get_system_stats(self) -> Dict[str, Any]:
        """
        Returns live platform statistics computed from SQLite persistence
        and empirical Phase 3 model metadata.
        """
        try:
            counts = self._db.get_counts()
            total = counts["total"]
            fake_count = counts["fake"]
            real_count = counts["real"]
            uncertain_count = counts.get("uncertain", 0)
        except Exception:
            total = len(self._analysis_log)
            fake_count = sum(1 for r in self._analysis_log if r.get("prediction") == "Fake" and not r.get("is_uncertain"))
            real_count = sum(1 for r in self._analysis_log if r.get("prediction") == "Real" and not r.get("is_uncertain"))
            uncertain_count = sum(1 for r in self._analysis_log if r.get("is_uncertain") or r.get("prediction") == "Uncertain")

        # Load real test accuracy from Phase 3 metadata
        accuracy_val = None
        accuracy_display = "N/A"
        if self._metadata and "champion_test_metrics" in self._metadata:
            acc = self._metadata["champion_test_metrics"].get("accuracy", 0.0)
            accuracy_val = round(acc * 100, 2)
            accuracy_display = f"{accuracy_val:.2f}%"

        model_name = self._metadata.get("model_name", "Linear SVM (Calibrated)")

        return {
            "total_analyses": total,
            "fake_detected": fake_count,
            "real_detected": real_count,
            "uncertain_detected": uncertain_count,
            "fake_percent": round((fake_count / total * 100), 1) if total > 0 else 0.0,
            "real_percent": round((real_count / total * 100), 1) if total > 0 else 0.0,
            "uncertain_percent": round((uncertain_count / total * 100), 1) if total > 0 else 0.0,
            "uncertainty_threshold": self._threshold,
            "model_accuracy": accuracy_val,
            "model_accuracy_display": accuracy_display,
            "model_name": model_name,
            "model_status": "Active & Operational",
            "stage": 4
        }

    def get_history(self, limit: Optional[int] = None, offset: int = 0) -> List[Dict[str, Any]]:
        """Return historical analysis records (most recent first) from SQLite with pagination support."""
        try:
            return self._db.get_history(limit=limit, offset=offset)
        except Exception:
            history = list(reversed(self._analysis_log))
            if limit is not None:
                return history[offset:offset + limit]
            return history[offset:] if offset > 0 else history

    def get_history_count(self) -> int:
        """Return total count of historical analyses recorded in SQLite."""
        try:
            return self._db.count()
        except Exception:
            return len(self._analysis_log)

    def clear_history(self) -> int:
        """Clears all historical records from database and memory."""
        self._analysis_log.clear()
        try:
            return self._db.clear_history()
        except Exception:
            return 0

    def get_model_metadata(self) -> Dict[str, Any]:
        """Returns the full Phase 3 model evaluation metadata."""
        return self._metadata

    def get_model_comparison(self) -> Dict[str, Any]:
        """Returns the Phase 3 candidate model benchmark results."""
        return self._comparison


# Singleton instance
analyzer_service = AnalyzerService()
