"""
NewsSentinel - Phase 5 Workstream 1: Explainable AI Unit Tests
==============================================================
Location: tests/test_explainability.py

Validates:
  1. Prediction and explain API contracts for feature attribution.
  2. Mathematical correctness: contribution_j = weight_j * tfidf_j.
  3. Directional correctness: Real signals > 0, Fake signals < 0.
  4. Sorting order: Real descending (highest positive first), Fake ascending (most negative first).
  5. Handling of out-of-vocabulary inputs and edge cases.
  6. Integration with AnalyzerService output schema.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predictor import get_predictor, explain_news, predict_news
from app.services.analyzer_service import analyzer_service


class TestExplainableAI(unittest.TestCase):
    """Test suite for token-level linear SVM feature attribution."""

    @classmethod
    def setUpClass(cls):
        cls.predictor = get_predictor()

    def test_explain_contract_schema(self):
        """Verify that explain() returns the expected schema and field types."""
        res = self.predictor.explain(
            title="Bipartisan Senate Accord",
            text="Lawmakers reached a budget agreement to fund federal government agencies."
        )

        self.assertIn("top_real_signals", res)
        self.assertIn("top_fake_signals", res)
        self.assertIn("total_features_matched", res)
        self.assertIn("net_lexical_bias", res)
        self.assertIn("baseline_intercept", res)
        self.assertIn("attribution_type", res)
        self.assertIn("disclaimer", res)

        self.assertIsInstance(res["top_real_signals"], list)
        self.assertIsInstance(res["top_fake_signals"], list)
        self.assertIsInstance(res["total_features_matched"], int)
        self.assertIsInstance(res["net_lexical_bias"], float)
        self.assertIsInstance(res["baseline_intercept"], float)
        self.assertEqual(res["attribution_type"], "approximate_linear_svm_decision_attribution")
        self.assertIn("WELFake", res["disclaimer"])
        self.assertIn("calibrated probability", res["disclaimer"])

    def test_attribution_mathematical_soundness(self):
        """
        Verify mathematical identity: contribution == round(weight * tfidf, 4).
        Also verify positive contribution for Real signals and negative for Fake signals.
        """
        res = self.predictor.explain(
            title="White House Fiscal Announcement",
            text="The president signed the economic reform legislation following weeks of debate in Congress."
        )

        # Real signals
        for item in res["top_real_signals"]:
            self.assertIn("token", item)
            self.assertIn("contribution", item)
            self.assertIn("weight", item)
            self.assertIn("tfidf", item)
            self.assertGreater(item["contribution"], 0.0, f"Token {item['token']} should have positive contribution")
            expected_contrib = round(item["weight"] * item["tfidf"], 4)
            self.assertAlmostEqual(item["contribution"], expected_contrib, places=4)

        # Fake signals
        for item in res["top_fake_signals"]:
            self.assertIn("token", item)
            self.assertIn("contribution", item)
            self.assertIn("weight", item)
            self.assertIn("tfidf", item)
            self.assertLess(item["contribution"], 0.0, f"Token {item['token']} should have negative contribution")
            expected_contrib = round(item["weight"] * item["tfidf"], 4)
            self.assertAlmostEqual(item["contribution"], expected_contrib, places=4)

    def test_sorting_order(self):
        """Verify Real signals sorted descending and Fake signals sorted ascending (most negative first)."""
        res = self.predictor.explain(
            title="President Addresses Capitol Press Corps",
            text="Officials held a briefing discussing ongoing diplomatic discussions and geopolitical developments."
        )

        real_contribs = [item["contribution"] for item in res["top_real_signals"]]
        self.assertEqual(real_contribs, sorted(real_contribs, reverse=True))

        fake_contribs = [item["contribution"] for item in res["top_fake_signals"]]
        self.assertEqual(fake_contribs, sorted(fake_contribs))

    def test_out_of_vocabulary_edge_case(self):
        """Verify graceful handling when no input tokens match the TF-IDF vocabulary."""
        res = self.predictor.explain(
            title="!@#$%^ &*()",
            text="zzqqxxjjkww mmnnppyytt"
        )

        self.assertEqual(res["total_features_matched"], 0)
        self.assertEqual(res["top_real_signals"], [])
        self.assertEqual(res["top_fake_signals"], [])
        self.assertEqual(res["net_lexical_bias"], 0.0)

    def test_predict_includes_explanation(self):
        """Verify predict() returns explanation by default and respects toggle flag."""
        res_with = self.predictor.predict("Senate vote", "Lawmakers voted on the bill.", include_explanation=True)
        self.assertIn("explanation", res_with)
        self.assertIn("top_real_signals", res_with["explanation"])

        res_without = self.predictor.predict("Senate vote", "Lawmakers voted on the bill.", include_explanation=False)
        self.assertNotIn("explanation", res_without)

    def test_analyzer_service_integration(self):
        """Verify AnalyzerService delivers explanation dictionary in API payload."""
        result = analyzer_service.process_analysis_request(
            text="The Department of State issued formal guidance on regional treaty commitments.",
            headline="Treaty Update"
        )
        self.assertTrue(result["success"])
        data = result["data"]
        self.assertIn("explanation", data)
        self.assertIn("top_real_signals", data["explanation"])
        self.assertIn("top_fake_signals", data["explanation"])
        self.assertIn("total_features_matched", data["explanation"])

    def test_convenience_explain_news(self):
        """Verify module-level explain_news wrapper."""
        res = explain_news("Treasury Report", "Economic indicators show robust employment and stable inflation metrics.")
        self.assertIsInstance(res, dict)
        self.assertIn("top_real_signals", res)
        self.assertGreater(len(res["top_real_signals"]), 0)

    def test_directional_monotonicity_with_predict_proba(self):
        """
        Verify that positive-attribution tokens monotonically increase Real calibrated probability,
        and negative-attribution tokens monotonically increase Fake calibrated probability.
        """
        base_title = "Economic Report"
        base_text = "The committee released its findings on employment and industry conditions."
        base_pred = self.predictor.predict(base_title, base_text)
        explanation = base_pred["explanation"]

        self.assertGreater(len(explanation["top_real_signals"]), 0)
        self.assertGreater(len(explanation["top_fake_signals"]), 0)

        # Pick top positive token
        top_real_token = explanation["top_real_signals"][0]["token"]
        boosted_real_text = f"{base_text} {(' ' + top_real_token) * 5}"
        pred_boosted_real = self.predictor.predict(base_title, boosted_real_text)
        self.assertGreater(
            pred_boosted_real["probabilities"]["Real"],
            base_pred["probabilities"]["Real"],
            f"Injecting positive token '{top_real_token}' must increase Real probability"
        )

        # Pick top negative token
        top_fake_token = explanation["top_fake_signals"][0]["token"]
        boosted_fake_text = f"{base_text} {(' ' + top_fake_token) * 5}"
        pred_boosted_fake = self.predictor.predict(base_title, boosted_fake_text)
        self.assertGreater(
            pred_boosted_fake["probabilities"]["Fake"],
            base_pred["probabilities"]["Fake"],
            f"Injecting negative token '{top_fake_token}' must increase Fake probability"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
