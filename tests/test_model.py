"""
NewsSentinel - Phase 3 Automated Model Validation & Unit Tests
==============================================================
Location: tests/test_model.py

Validates:
  1. Existence and integrity of serialized model artifacts in models/.
  2. TF-IDF vectorizer configuration and transformation dimension.
  3. Preprocessing robustness (HTML tags, URLs, whitespace, casing).
  4. Predictor inference contract (data types, probability bounds, sum to 1.0).
  5. Sanity classification tests on canonical real news and sensational fake news.
  6. Existence and completeness of evaluation reports in reports/.
"""

import sys
import unittest
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predictor import NewsPredictor, get_predictor, predict_news


class TestModelArtifactsAndInference(unittest.TestCase):
    """Test suite for machine learning pipeline artifacts and inference engine."""

    @classmethod
    def setUpClass(cls):
        cls.models_dir = PROJECT_ROOT / "models"
        cls.reports_dir = PROJECT_ROOT / "reports"
        cls.model_path = cls.models_dir / "best_model.joblib"
        cls.vec_path = cls.models_dir / "tfidf_vectorizer.joblib"
        cls.meta_path = cls.models_dir / "model_metadata.json"
        cls.rep_txt = cls.reports_dir / "model_evaluation_report.txt"
        cls.rep_json = cls.reports_dir / "model_comparison.json"
        cls.predictor = get_predictor()

    def test_artifacts_exist_on_disk(self):
        """Verify that all required Phase 3 artifacts exist on disk."""
        self.assertTrue(self.model_path.exists(), "best_model.joblib missing")
        self.assertTrue(self.vec_path.exists(), "tfidf_vectorizer.joblib missing")
        self.assertTrue(self.meta_path.exists(), "model_metadata.json missing")
        self.assertTrue(self.rep_txt.exists(), "model_evaluation_report.txt missing")
        self.assertTrue(self.rep_json.exists(), "model_comparison.json missing")

    def test_model_metadata_contents(self):
        """Verify that metadata json contains all required schema keys."""
        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.assertIn("model_name", meta)
        self.assertIn("splits", meta)
        self.assertEqual(meta["splits"]["ratio"], "70/15/15")
        self.assertEqual(meta["labels"], {"0": "Fake", "1": "Real"})
        self.assertIn("champion_validation_metrics", meta)
        self.assertIn("champion_test_metrics", meta)
        self.assertGreaterEqual(meta["champion_test_metrics"]["accuracy"], 0.90)

    def test_vectorizer_properties(self):
        """Verify vectorizer feature vocabulary size and n-gram range."""
        vec = self.predictor._vectorizer
        self.assertEqual(len(vec.vocabulary_), 50000)
        self.assertEqual(vec.ngram_range, (1, 2))
        self.assertTrue(vec.sublinear_tf)

    def test_text_preprocessing(self):
        """Verify that text preprocessing cleans markup, URLs, and excess whitespace."""
        raw_title = "BREAKING: Aliens Land In Washington! <script>alert(1)</script>"
        raw_text = "Visit http://fake-news.xyz for more details!   Huge story   here.   "
        cleaned = NewsPredictor.preprocess_text(raw_title, raw_text)

        self.assertNotIn("<script>", cleaned)
        self.assertNotIn("http://", cleaned)
        self.assertNotIn("  ", cleaned)
        self.assertTrue(cleaned.islower())

    def test_prediction_output_contract(self):
        """Verify the dictionary contract returned by the prediction service."""
        res = self.predictor.predict(
            title="Federal Reserve Keeps Interest Rates Steady",
            text="WASHINGTON (Reuters) - The Federal Reserve kept benchmark interest rates unchanged on Wednesday..."
        )

        self.assertIn("prediction", res)
        self.assertIn(res["prediction"], ["Real", "Fake"])
        self.assertIn("label", res)
        self.assertIn(res["label"], [0, 1])
        self.assertIn("confidence", res)
        self.assertGreaterEqual(res["confidence"], 0.5)
        self.assertLessEqual(res["confidence"], 1.0)

        self.assertIn("probabilities", res)
        probs = res["probabilities"]
        self.assertIn("Real", probs)
        self.assertIn("Fake", probs)
        self.assertAlmostEqual(probs["Real"] + probs["Fake"], 1.0, places=3)
        self.assertGreater(res["latency_ms"], 0.0)

    def test_sanity_real_news_classification(self):
        """Verify that standard Reuters journalistic wire report classifies as Real (1)."""
        title = "Senate passes bipartisan budget bill to avert government shutdown"
        text = (
            "WASHINGTON (Reuters) - The U.S. Senate voted overwhelmingly on Thursday "
            "to approve a compromise spending bill, sending the measure to the president's "
            "desk and avoiding a looming federal agency shutdown ahead of the midnight deadline."
        )
        res = self.predictor.predict(title, text)
        self.assertEqual(res["prediction"], "Real")
        self.assertEqual(res["label"], 1)
        self.assertGreater(res["probabilities"]["Real"], 0.80)

    def test_sanity_fake_news_conspiracy_classification(self):
        """Verify that sensational conspiracy clickbait classifies as Fake (0)."""
        title = "SHOCKING! SATANIC PEDOPHILE RING EXPOSED IN SECRET UNDERGROUND BUNKER [VIDEO]"
        text = (
            "UNBELIEVABLE! Top government officials were caught red handed operating a secret "
            "satanic pedophile dungeon beneath a local pizza restaurant. Millions of paid Soros "
            "crisis actors and chemtrail operators have been mobilized to suppress this shocking truth!"
        )
        res = self.predictor.predict(title, text)
        self.assertEqual(res["prediction"], "Fake")
        self.assertEqual(res["label"], 0)
        self.assertGreater(res["probabilities"]["Fake"], 0.80)


if __name__ == "__main__":
    unittest.main(verbosity=2)
