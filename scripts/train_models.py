"""
NewsSentinel - Phase 3: Machine Learning Model Training & Evaluation Pipeline
=============================================================================
Location: scripts/train_models.py

Pipeline Stages:
  1. Ingests cleaned dataset from data/processed/WELFake_cleaned.csv.
  2. Combines 'title' and 'text' into single unified feature 'full_text'.
  3. Preprocesses text (lowercasing, URL/HTML stripping, whitespace normalization).
  4. Performs stratified 70/15/15 train/validation/test split.
  5. Fits TF-IDF vectorizer (unigrams + bigrams, max_features=50,000) on train set only.
  6. Trains and tunes three candidate models:
     - Logistic Regression
     - Multinomial Naive Bayes
     - Linear SVM (Calibrated for well-behaved probabilities)
  7. Evaluates all models on the Validation Set (Accuracy, Precision, Recall, F1, Confusion Matrix).
  8. Selects the champion model based STRICTLY on validation performance.
  9. Evaluates the champion model (and all candidates) on the held-out Test Set.
  10. Serializes the best model and TF-IDF vectorizer to models/.
  11. Exports detailed reports to reports/model_evaluation_report.txt and reports/model_comparison.json.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Configuration Paths
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "WELFake_cleaned.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

REPORT_TXT_PATH = REPORTS_DIR / "model_evaluation_report.txt"
COMPARISON_JSON_PATH = REPORTS_DIR / "model_comparison.json"


def preprocess_text_series(title_series: pd.Series, text_series: pd.Series) -> pd.Series:
    """Combines title and text and applies standardized cleaning."""
    print("    * Combining 'title' and 'text' into unified input feature...")
    combined = title_series.fillna("").astype(str) + " " + text_series.fillna("").astype(str)

    print("    * Cleaning text (lowercasing, stripping URLs, HTML tags, excess whitespace)...")
    combined = combined.str.lower()
    combined = combined.str.replace(r"https?://\S+|www\.\S+", "", regex=True)
    combined = combined.str.replace(r"<.*?>", "", regex=True)
    combined = combined.str.replace(r"\s+", " ", regex=True).str.strip()
    return combined


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Computes comprehensive binary classification metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred).tolist()  # [[TN, FP], [FN, TP]]

    # Class 0: Fake
    p_fake = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    r_fake = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    f1_fake = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))

    # Class 1: Real
    p_real = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    r_real = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1_real = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    # Macro and weighted
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    return {
        "accuracy": acc,
        "confusion_matrix": {
            "TN_fake": cm[0][0],
            "FP_predicted_real": cm[0][1],
            "FN_predicted_fake": cm[1][0],
            "TP_real": cm[1][1],
            "raw_matrix": cm,
        },
        "fake_class_0": {
            "precision": p_fake,
            "recall": r_fake,
            "f1_score": f1_fake,
        },
        "real_class_1": {
            "precision": p_real,
            "recall": r_real,
            "f1_score": f1_real,
        },
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def run_pipeline() -> None:
    start_time = time.time()
    print("=" * 80)
    print(" NEWSSENTINEL - PHASE 3: ML MODEL TRAINING & BENCHMARKING PIPELINE")
    print("=" * 80)

    # 1. Load Data
    print("\n[Stage 1/6] Ingesting Primary Cleaned Dataset...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, low_memory=False, encoding="utf-8")
    total_records = len(df)
    print(f"    * Ingested: {total_records:,} records from {DATA_PATH.relative_to(PROJECT_ROOT)}")
    print(f"    * Label Distribution: Real (1) = {(df['label'] == 1).sum():,} | Fake (0) = {(df['label'] == 0).sum():,}")

    # 2. Preprocess & Feature Combination
    print("\n[Stage 2/6] Feature Engineering & Text Preprocessing...")
    X_text = preprocess_text_series(df["title"], df["text"])
    y = df["label"].values

    # 3. Stratified Train / Validation / Test Split (70 / 15 / 15)
    print("\n[Stage 3/6] Stratified Train/Val/Test Split (70% / 15% / 15%)...")
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X_text, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    print(f"    * Training Set:   {len(X_train_raw):,} records ({len(X_train_raw)/total_records*100:.1f}%) [Real: {(y_train==1).sum():,}, Fake: {(y_train==0).sum():,}]")
    print(f"    * Validation Set: {len(X_val_raw):,} records ({len(X_val_raw)/total_records*100:.1f}%) [Real: {(y_val==1).sum():,}, Fake: {(y_val==0).sum():,}]")
    print(f"    * Test Set:       {len(X_test_raw):,} records ({len(X_test_raw)/total_records*100:.1f}%) [Real: {(y_test==1).sum():,}, Fake: {(y_test==0).sum():,}]")

    # 4. TF-IDF Feature Extraction (Fit on Train ONLY)
    print("\n[Stage 4/6] Fitting TF-IDF Vectorizer (Unigrams + Bigrams)...")
    t_tfidf = time.time()
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        min_df=3,
        sublinear_tf=True,
        stop_words="english",
    )
    X_train_vec = vectorizer.fit_transform(X_train_raw)
    X_val_vec = vectorizer.transform(X_val_raw)
    X_test_vec = vectorizer.transform(X_test_raw)
    print(f"    * Vocabulary Size:   {len(vectorizer.vocabulary_):,} features")
    print(f"    * N-gram Range:      (1, 2) unigrams & bigrams")
    print(f"    * Sublinear TF:      Enabled (logarithmic scaling)")
    print(f"    * Vectorized Matrix: {X_train_vec.shape} (Time: {time.time() - t_tfidf:.2f}s)")

    # 5. Model Definitions
    print("\n[Stage 5/6] Training & Evaluating Candidate Models...")
    candidates = {
        "Multinomial Naive Bayes": {
            "model": MultinomialNB(alpha=0.1),
            "description": "Multinomial Naive Bayes with Laplace smoothing (alpha=0.1)",
        },
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, C=1.0, random_state=42, solver="lbfgs"),
            "description": "Logistic Regression with L2 regularization (C=1.0, solver='lbfgs')",
        },
        "Linear SVM (Calibrated)": {
            "model": CalibratedClassifierCV(
                estimator=LinearSVC(C=1.0, random_state=42, max_iter=2000),
                cv=3,
            ),
            "description": "Linear Support Vector Classifier with 3-fold Platt probability calibration",
        },
    }

    results = {}
    best_model_name = None
    best_val_f1 = -1.0

    for name, item in candidates.items():
        print(f"\n  ---> Training {name}...")
        clf = item["model"]
        t_train_start = time.time()
        clf.fit(X_train_vec, y_train)
        train_duration = time.time() - t_train_start

        # Validation Set Evaluation (STRICT Model Selection Gate)
        y_val_pred = clf.predict(X_val_vec)
        val_metrics = compute_metrics(y_val, y_val_pred)
        val_metrics["train_time_seconds"] = round(train_duration, 3)

        print(f"       Trained in: {train_duration:.2f}s")
        print(f"       Validation Accuracy: {val_metrics['accuracy']*100:.2f}%")
        print(f"       Validation Macro F1: {val_metrics['macro_f1']:.4f}")
        print(f"       Fake F1: {val_metrics['fake_class_0']['f1_score']:.4f} | Real F1: {val_metrics['real_class_1']['f1_score']:.4f}")

        results[name] = {
            "description": item["description"],
            "validation_metrics": val_metrics,
        }

        # Track best model strictly on validation F1 score
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_model_name = name

    print("\n" + "-" * 80)
    print(f"[*] Champion Model Selected: '{best_model_name}' (Validation Macro F1: {best_val_f1:.4f})")
    print("-" * 80)

    # 6. Final Evaluation on Held-Out Test Set (Unseen)
    print("\n[Stage 6/6] Final Evaluation on Held-Out Test Set (Leakage-Free)...")
    for name, item in candidates.items():
        clf = item["model"]
        y_test_pred = clf.predict(X_test_vec)
        test_metrics = compute_metrics(y_test, y_test_pred)
        results[name]["test_metrics"] = test_metrics

    # Print Champion Test Metrics
    champ_test = results[best_model_name]["test_metrics"]
    print(f"    * Champion: '{best_model_name}' on Test Set:")
    print(f"      - Test Accuracy: {champ_test['accuracy']*100:.2f}%")
    print(f"      - Test Macro F1: {champ_test['macro_f1']:.4f}")
    print(f"      - Fake Class (0) -> Precision: {champ_test['fake_class_0']['precision']*100:.2f}%, Recall: {champ_test['fake_class_0']['recall']*100:.2f}%, F1: {champ_test['fake_class_0']['f1_score']:.4f}")
    print(f"      - Real Class (1) -> Precision: {champ_test['real_class_1']['precision']*100:.2f}%, Recall: {champ_test['real_class_1']['recall']*100:.2f}%, F1: {champ_test['real_class_1']['f1_score']:.4f}")
    print(f"      - Confusion Matrix: TN={champ_test['confusion_matrix']['TN_fake']}, FP={champ_test['confusion_matrix']['FP_predicted_real']}, FN={champ_test['confusion_matrix']['FN_predicted_fake']}, TP={champ_test['confusion_matrix']['TP_real']}")

    # 7. Model Serialization
    print("\n[+] Saving Champion Model & TF-IDF Vectorizer...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    champion_clf = candidates[best_model_name]["model"]
    joblib.dump(champion_clf, BEST_MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    model_size_mb = BEST_MODEL_PATH.stat().st_size / (1024 * 1024)
    vec_size_mb = VECTORIZER_PATH.stat().st_size / (1024 * 1024)
    print(f"    * Saved best model to:      {BEST_MODEL_PATH.relative_to(PROJECT_ROOT)} ({model_size_mb:.2f} MB)")
    print(f"    * Saved TF-IDF vectorizer:  {VECTORIZER_PATH.relative_to(PROJECT_ROOT)} ({vec_size_mb:.2f} MB)")

    # Save metadata JSON
    metadata = {
        "model_name": best_model_name,
        "model_file": str(BEST_MODEL_PATH.name),
        "vectorizer_file": str(VECTORIZER_PATH.name),
        "training_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "WELFake_cleaned.csv",
        "dataset_rows": total_records,
        "splits": {
            "train": len(X_train_raw),
            "val": len(X_val_raw),
            "test": len(X_test_raw),
            "ratio": "70/15/15",
            "stratified": True,
            "random_state": 42,
        },
        "features": {
            "type": "TF-IDF",
            "ngram_range": [1, 2],
            "max_features": 50000,
            "sublinear_tf": True,
            "stop_words": "english",
        },
        "labels": {
            "0": "Fake",
            "1": "Real",
        },
        "champion_validation_metrics": results[best_model_name]["validation_metrics"],
        "champion_test_metrics": results[best_model_name]["test_metrics"],
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"    * Saved model metadata to:  {METADATA_PATH.relative_to(PROJECT_ROOT)}")

    # 8. Save Comparison JSON
    with open(COMPARISON_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"    * Saved model metrics JSON: {COMPARISON_JSON_PATH.relative_to(PROJECT_ROOT)}")

    # 9. Generate Detailed Text Report
    generate_txt_report(results, best_model_name, total_records, len(X_train_raw), len(X_val_raw), len(X_test_raw), time.time() - start_time)


def generate_txt_report(
    results: Dict[str, Any],
    best_name: str,
    total_rows: int,
    train_rows: int,
    val_rows: int,
    test_rows: int,
    elapsed_time: float,
) -> None:
    """Generates an exhaustive, research-grade evaluation report."""
    rep = []
    rep.append("=" * 80)
    rep.append("NEWSSENTINEL - PHASE 3: ML MODEL BENCHMARK & EVALUATION REPORT")
    rep.append("=" * 80)
    rep.append(f"Execution Date:      {time.strftime('%Y-%m-%d %H:%M:%S')}")
    rep.append(f"Total Corpus:        {total_rows:,} records (data/processed/WELFake_cleaned.csv)")
    rep.append(f"Train / Val / Test:  {train_rows:,} (70%) / {val_rows:,} (15%) / {test_rows:,} (15%) [Stratified]")
    rep.append(f"Feature Extractor:   TF-IDF Unigrams + Bigrams (50,000 max features, sublinear TF)")
    rep.append(f"Label Convention:    0 = Fake News, 1 = Real News")
    rep.append("=" * 80)
    rep.append("")

    rep.append("--------------------------------------------------------------------------------")
    rep.append("1. CANDIDATE MODEL COMPARISON (VALIDATION SET - MODEL SELECTION GATE)")
    rep.append("--------------------------------------------------------------------------------")
    rep.append(f"{'Model Name':<28} | {'Accuracy':<9} | {'Macro F1':<9} | {'Fake F1':<8} | {'Real F1':<8} | {'Train Time':<10}")
    rep.append("-" * 80)
    for name, data in results.items():
        vm = data["validation_metrics"]
        rep.append(
            f"{name:<28} | {vm['accuracy']*100:>7.2f}% | {vm['macro_f1']:>9.4f} | "
            f"{vm['fake_class_0']['f1_score']:>8.4f} | {vm['real_class_1']['f1_score']:>8.4f} | "
            f"{vm['train_time_seconds']:>8.2f}s"
        )
    rep.append("-" * 80)
    rep.append(f"CHAMPION MODEL SELECTED BASED ON VALIDATION SET: '{best_name}'")
    rep.append("")

    rep.append("--------------------------------------------------------------------------------")
    rep.append("2. FINAL HELD-OUT TEST SET EVALUATION (GENERALIZATION PERFORMANCE)")
    rep.append("--------------------------------------------------------------------------------")
    rep.append(f"{'Model Name':<28} | {'Accuracy':<9} | {'Macro F1':<9} | {'Fake F1':<8} | {'Real F1':<8} | {'Status':<10}")
    rep.append("-" * 80)
    for name, data in results.items():
        tm = data["test_metrics"]
        status = "CHAMPION" if name == best_name else "Candidate"
        rep.append(
            f"{name:<28} | {tm['accuracy']*100:>7.2f}% | {tm['macro_f1']:>9.4f} | "
            f"{tm['fake_class_0']['f1_score']:>8.4f} | {tm['real_class_1']['f1_score']:>8.4f} | "
            f"{status:<10}"
        )
    rep.append("-" * 80)
    rep.append("")

    rep.append("--------------------------------------------------------------------------------")
    rep.append(f"3. IN-DEPTH AUDIT OF CHAMPION MODEL: {best_name}")
    rep.append("--------------------------------------------------------------------------------")
    c_val = results[best_name]["validation_metrics"]
    c_test = results[best_name]["test_metrics"]
    rep.append("Validation Set Performance:")
    rep.append(f"  * Accuracy:     {c_val['accuracy']*100:.2f}%")
    rep.append(f"  * Macro F1:     {c_val['macro_f1']:.4f}")
    rep.append(f"  * Fake (0):     Precision = {c_val['fake_class_0']['precision']*100:.2f}% | Recall = {c_val['fake_class_0']['recall']*100:.2f}% | F1 = {c_val['fake_class_0']['f1_score']:.4f}")
    rep.append(f"  * Real (1):     Precision = {c_val['real_class_1']['precision']*100:.2f}% | Recall = {c_val['real_class_1']['recall']*100:.2f}% | F1 = {c_val['real_class_1']['f1_score']:.4f}")
    rep.append(f"  * Confusion Matrix: TN (Fake)={c_val['confusion_matrix']['TN_fake']:,}, FP (Fake as Real)={c_val['confusion_matrix']['FP_predicted_real']:,}, FN (Real as Fake)={c_val['confusion_matrix']['FN_predicted_fake']:,}, TP (Real)={c_val['confusion_matrix']['TP_real']:,}")
    rep.append("")
    rep.append("Test Set Performance (Held-Out Final Evaluation):")
    rep.append(f"  * Accuracy:     {c_test['accuracy']*100:.2f}%")
    rep.append(f"  * Macro F1:     {c_test['macro_f1']:.4f}")
    rep.append(f"  * Fake (0):     Precision = {c_test['fake_class_0']['precision']*100:.2f}% | Recall = {c_test['fake_class_0']['recall']*100:.2f}% | F1 = {c_test['fake_class_0']['f1_score']:.4f}")
    rep.append(f"  * Real (1):     Precision = {c_test['real_class_1']['precision']*100:.2f}% | Recall = {c_test['real_class_1']['recall']*100:.2f}% | F1 = {c_test['real_class_1']['f1_score']:.4f}")
    rep.append(f"  * Confusion Matrix: TN (Fake)={c_test['confusion_matrix']['TN_fake']:,}, FP (Fake as Real)={c_test['confusion_matrix']['FP_predicted_real']:,}, FN (Real as Fake)={c_test['confusion_matrix']['FN_predicted_fake']:,}, TP (Real)={c_test['confusion_matrix']['TP_real']:,}")
    rep.append("")

    rep.append("--------------------------------------------------------------------------------")
    rep.append("4. ARTIFACT PERSISTENCE & DEPLOYMENT READINESS")
    rep.append("--------------------------------------------------------------------------------")
    rep.append("Serialized Artifacts:")
    rep.append(f"  * Model Binary:      models/{BEST_MODEL_PATH.name}")
    rep.append(f"  * TF-IDF Vectorizer: models/{VECTORIZER_PATH.name}")
    rep.append(f"  * Model Metadata:    models/{METADATA_PATH.name}")
    rep.append("")
    rep.append("Phase 3 Execution Summary:")
    rep.append(f"  * Total Execution Time: {elapsed_time:.2f} seconds")
    rep.append("  * Training Integrity:   Zero data leakage; test set isolated until after model selection.")
    rep.append("  * Production Status:    READY FOR FASTAPI INFERENCE INTEGRATION (PHASE 4)")
    rep.append("=" * 80)

    with open(REPORT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(rep) + "\n")
    print(f"    * Saved human-readable report: {REPORT_TXT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    run_pipeline()
