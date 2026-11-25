"""
NewsSentinel - ML Prediction Service
====================================
Location: src/predictor.py

Provides thread-safe model inference, confidence scoring,
and input text preprocessing for the trained NewsSentinel classifier.
"""

import sys
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib"


class NewsPredictor:
    """Predictor service for evaluating news veracity using the trained champion model."""

    _instance = None
    _model = None
    _vectorizer = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NewsPredictor, cls).__new__(cls)
            cls._instance._load_artifacts()
        return cls._instance

    def _load_artifacts(self) -> None:
        """Loads serialized model and vectorizer from disk."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        if not VECTORIZER_PATH.exists():
            raise FileNotFoundError(f"Vectorizer file not found at {VECTORIZER_PATH}")

        self._model = joblib.load(MODEL_PATH)
        self._vectorizer = joblib.load(VECTORIZER_PATH)

        # Precompute mean linear SVM weights and feature names across calibrated folds
        # for instant O(NNZ) explainable AI token attribution calculations
        self._weights = np.mean([clf.estimator.coef_[0] for clf in self._model.calibrated_classifiers_], axis=0)
        self._intercept = float(np.mean([clf.estimator.intercept_[0] for clf in self._model.calibrated_classifiers_]))
        self._feature_names = self._vectorizer.get_feature_names_out()

    @staticmethod
    def preprocess_text(title: str, text: str) -> str:
        """Applies exact preprocessing used during model training."""
        combined = f"{title or ''} {text or ''}"
        cleaned = combined.lower()
        cleaned = re.sub(r"https?://\S+|www\.\S+", "", cleaned)
        cleaned = re.sub(r"<.*?>", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _compute_attribution(self, X_vec, top_k: int = 5) -> Dict[str, Any]:
        """
        Computes approximate lexical token attributions from the sparse TF-IDF vector:
        contribution_j = weight_j * tfidf_j

        Where weight_j is the mean LinearSVC weight across the 3 calibrated cross-validation folds.
        - Positive contribution (> 0): increases the linear decision score, pushing toward Real News (class 1).
        - Negative contribution (< 0): decreases the linear decision score, pushing toward Fake News (class 0).

        Mathematical Note:
        This is an exact linear decomposition of the mean uncalibrated decision score
        f_bar(x) = w_bar^T x + b_bar. For the ensemble's non-linear Platt-calibrated output
        P(y=1|x) = (1/3) * sum_{k=1}^3 sigma(-a_k f_k(x) - b_k), it serves as an approximate
        first-order directional explanation of lexical influence and does NOT directly quantify
        percentage changes in calibrated probability.
        """
        indices = X_vec.indices
        data = X_vec.data

        if len(indices) == 0:
            return {
                "top_real_signals": [],
                "top_fake_signals": [],
                "total_features_matched": 0,
                "net_lexical_bias": 0.0,
                "baseline_intercept": round(self._intercept, 4),
                "attribution_type": "approximate_linear_svm_decision_attribution",
                "disclaimer": (
                    "NewsSentinel identifies linguistic patterns learned from its training data. "
                    "It does not independently verify sources, events, quotations or factual claims. "
                    "Token attributions reflect approximate lexical contributions (w_j * x_j) based on "
                    "underlying LinearSVC weights across calibration folds. Positive values push the decision "
                    "boundary toward Real; negative values push toward Fake. They reflect statistical "
                    "correlations in the WELFake training corpus and do not directly quantify changes in "
                    "calibrated probability or verify factual truth."
                ),
            }

        weights = self._weights[indices]
        contributions = weights * data

        real_candidates = []
        fake_candidates = []

        for idx, w, val, contrib in zip(indices, weights, data, contributions):
            feat_name = self._feature_names[idx]
            item = {
                "token": feat_name,
                "contribution": round(float(contrib), 4),
                "weight": round(float(w), 4),
                "tfidf": round(float(val), 4),
            }
            if contrib > 0:
                real_candidates.append(item)
            elif contrib < 0:
                fake_candidates.append(item)

        # Sort real candidates descending (highest positive impact first)
        real_signals = sorted(real_candidates, key=lambda x: x["contribution"], reverse=True)[:top_k]
        # Sort fake candidates ascending (most negative impact first)
        fake_signals = sorted(fake_candidates, key=lambda x: x["contribution"])[:top_k]

        total_bias = float(np.sum(contributions))

        return {
            "top_real_signals": real_signals,
            "top_fake_signals": fake_signals,
            "total_features_matched": int(X_vec.nnz),
            "net_lexical_bias": round(total_bias, 4),
            "baseline_intercept": round(self._intercept, 4),
            "attribution_type": "approximate_linear_svm_decision_attribution",
            "disclaimer": (
                "NewsSentinel identifies linguistic patterns learned from its training data. "
                "It does not independently verify sources, events, quotations or factual claims. "
                "Token attributions reflect approximate lexical contributions (w_j * x_j) based on "
                "underlying LinearSVC weights across calibration folds. Positive values push the decision "
                "boundary toward Real; negative values push toward Fake. They reflect statistical "
                "correlations in the WELFake training corpus and do not directly quantify changes in "
                "calibrated probability or verify factual truth."
            ),
        }

    def explain(self, title: str, text: str, top_k: int = 5) -> Dict[str, Any]:
        """Provides explainable AI feature attribution for input text."""
        cleaned_input = self.preprocess_text(title, text)
        X_vec = self._vectorizer.transform([cleaned_input])
        return self._compute_attribution(X_vec, top_k=top_k)

    def predict(self, title: str, text: str, include_explanation: bool = True, uncertainty_threshold: Optional[float] = 0.65) -> Dict[str, Any]:
        """
        Predicts whether an article is Real or Fake news, or Uncertain if probabilities
        fall below the confidence threshold.

        Label convention:
          0 = Fake News
          1 = Real News
        """
        start_time = time.time()
        cleaned_input = self.preprocess_text(title, text)
        words = cleaned_input.split()
        word_count = len(words)

        # Vectorize using loaded TF-IDF vectorizer
        X_vec = self._vectorizer.transform([cleaned_input])

        # Predict label
        pred_label = int(self._model.predict(X_vec)[0])
        label_name = "Real" if pred_label == 1 else "Fake"

        # Calculate calibrated probabilities
        if hasattr(self._model, "predict_proba"):
            probs = self._model.predict_proba(X_vec)[0]
            fake_prob = float(probs[0])
            real_prob = float(probs[1])
        else:
            # Fallback for uncalibrated models
            decision = float(self._model.decision_function(X_vec)[0])
            real_prob = 1.0 / (1.0 + np.exp(-decision))
            fake_prob = 1.0 - real_prob

        confidence = real_prob if pred_label == 1 else fake_prob

        # Determine uncertainty status
        threshold = uncertainty_threshold if uncertainty_threshold is not None else 0.65
        max_prob = max(fake_prob, real_prob)
        is_uncertain = bool(max_prob < threshold)

        final_prediction = "Uncertain" if is_uncertain else label_name
        verdict = (
            "UNCERTAIN — NEEDS VERIFICATION"
            if is_uncertain
            else ("MODEL PREDICTION: REAL" if pred_label == 1 else "MODEL PREDICTION: FAKE")
        )

        # Compute explainability feature attribution
        explanation = self._compute_attribution(X_vec) if include_explanation else None

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        res = {
            "prediction": final_prediction,
            "raw_prediction": label_name,
            "verdict": verdict,
            "is_uncertain": is_uncertain,
            "uncertainty_threshold": threshold,
            "label": pred_label,
            "confidence": round(confidence, 4),
            "confidence_percent": round(confidence * 100, 2),
            "probabilities": {
                "Fake": round(fake_prob, 4),
                "Real": round(real_prob, 4),
            },
            "input_word_count": word_count,
            "latency_ms": elapsed_ms,
        }
        if explanation is not None:
            res["explanation"] = explanation
        return res


# Convenience module-level predictor function
_default_predictor = None


def get_predictor() -> NewsPredictor:
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = NewsPredictor()
    return _default_predictor


def predict_news(title: str, text: str, uncertainty_threshold: Optional[float] = 0.65) -> Dict[str, Any]:
    predictor = get_predictor()
    return predictor.predict(title, text, uncertainty_threshold=uncertainty_threshold)


def explain_news(title: str, text: str, top_k: int = 5) -> Dict[str, Any]:
    predictor = get_predictor()
    return predictor.explain(title, text, top_k=top_k)
