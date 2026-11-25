# NewsSentinel 🛡️
### AI-Powered Fake News and Misinformation Pattern Detection System
*Phase 5: Explainable AI, SQLite Persistence, Uncertainty Detection & Reliability Suite*

---

## 📌 Project Overview

**NewsSentinel** is a modern, research-grade, AI-powered intelligence platform engineered to detect, quantify, and analyze misinformation patterns in news headlines, journalistic articles, and digital media reports.

The project is structured across five systematic engineering stages:
- **Phase 1 (Complete)**: Project foundation, modular FastAPI architecture, premium dark futuristic dashboard interface, deterministic linguistic feature extraction, and RESTful telemetry API.
- **Phase 2 (Complete)**: Primary dataset engineering, data sanitization, deduplication (62,586 clean master records), forensic ground-truth label verification, and ISOT duplicate containment audit.
- **Phase 3 (Complete)**: Benchmark model training, Platt calibration, and Linear Support Vector Machine champion model deployment (96.59% accuracy on 9,388 unseen test articles).
- **Phase 4 (Complete)**: Live machine learning inferencing pipeline integration and synchronous real-time web telemetry.
- **Phase 5 (Complete)**: Explainable AI (XAI token attribution), thread-safe SQLite WAL persistence, Bayesian uncertainty detection (<0.65 threshold), global dark/light theme engine, and comprehensive 68-test automated reliability suite.

---

## 📊 Primary Dataset: WELFake (Master Corpus)

Following comprehensive exploratory data inspection and duplicate contamination analysis, **WELFake** (*Word Embedding over Linguistic Features for Fake News Detection*) was selected as the **sole primary dataset** for NewsSentinel.

### Why WELFake?
- **Unified Representation**: Incorporates and synthesizes news from multiple historical repositories (Kaggle Fake, McIntire, Reuters, and BuzzFeed) into a massive 72,134-article master corpus.
- **Complete Ingestion**: Empirical analysis proved that secondary datasets (such as ISOT) are already 100% contained within WELFake, rendering multi-dataset merging redundant and prone to cross-split data leakage.

### Dataset Pipeline & Cleaning Summary:
| Metric | Raw Master (`WELFake_Dataset.csv`) | Cleaned Corpus (`WELFake_cleaned.csv`) |
| :--- | :--- | :--- |
| **Storage Location** | `data/raw/WELFake_Dataset.csv` | `data/processed/WELFake_cleaned.csv` |
| **Disk Size** | `233.73 MB` | `206.72 MB` |
| **Total Rows** | `72,134` | **`62,586`** (100% Unique & Complete) |
| **Removed Missing / Empty** | — | `-1,341` (-1.86% null/whitespace titles & bodies) |
| **Removed Duplicate Pairs** | — | `-8,207` (-11.38% redundant syndicated stories) |
| **Label Mapping** | Raw inverted export | **Standardized: 0 = Fake, 1 = Real** |
| **Class Distribution** | 35,028 (0) / 37,106 (1) | **Real (1): 34,788 (55.6%) \| Fake (0): 27,798 (44.4%)** |

> [!NOTE]
> **Forensic Label Mapping Verification**:
> The official published Kaggle/Zenodo benchmark specification (Verma et al.) defines `0 = Fake, 1 = Real`. In the raw exported CSV, Reuters and NYT articles were exported with raw value `0` and fake blogs with `1`. In `data/processed/WELFake_cleaned.csv`, the dataset is formally standardized to the official benchmark standard (`0 = Fake`, `1 = Real`) and augmented with an unambiguous `label_name` column (`'Fake'` / `'Real'`).

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend Framework** | Python 3.10+, [FastAPI](https://fastapi.tiangolo.com/) |
| **ASGI Server** | [Uvicorn](https://www.uvicorn.org/) |
| **Templating Engine** | [Jinja2](https://palletsprojects.com/p/jinja/) |
| **Frontend UI/UX** | HTML5, Modern CSS3 (Grid & Flexbox), Vanilla JavaScript (ES6+) |
| **Data Processing** | [Pandas](https://pandas.pydata.org/), NumPy |
| **Typography** | Plus Jakarta Sans, JetBrains Mono |
| **Data Validation** | [Pydantic v2](https://docs.pydantic.dev/), Pydantic Settings |
| **API Specification** | OpenAPI (Swagger UI) & ReDoc |
| **Upcoming ML Stack** | Scikit-learn, NLTK, Plotly, SQLite |

---

## 🎨 Visual Identity & Design System

NewsSentinel adopts a premium dark futuristic aesthetic inspired by high-stakes intelligence and cybersecurity operations centers:
- **Primary Canvas**: Deep Space Dark (`#0B0D14`)
- **Card Surface**: Obsidian Navy (`#151925`)
- **Elevated Surface**: Slate Void (`#1C2234`)
- **Accent Signals**: Electric Violet (`#8B5CF6`) and Cyan/Neon Blue (`#38BDF8`)
- **Typography**: Crisp white primary headings (`#F8FAFC`) with subtle violet-blue glowing borders

---

## 📁 Project Directory Structure

```
NewsSentinel/
│
├── app/
│   ├── __init__.py               # Package metadata and version definition
│   ├── config.py                 # Pydantic configuration & environment settings
│   ├── main.py                   # FastAPI application factory and error handlers
│   │
│   ├── routes/
│   │   ├── __init__.py           # Route exports
│   │   ├── pages.py              # Web page controllers (Jinja2 template rendering)
│   │   └── analyzer.py           # RESTful API endpoints (/api/analyze, /api/stats, /api/health)
│   │
│   ├── services/
│   │   ├── __init__.py           # Service layer exports
│   │   └── analyzer_service.py   # Deterministic linguistic telemetry & input validation
│   │
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css         # Custom futuristic dark UI stylesheet
│   │   ├── js/
│   │   │   └── script.js         # Client-side reactivity, live telemetry, and API integration
│   │   └── images/
│   │       └── .gitkeep          # Static assets & brand graphics
│   │
│   └── templates/
│       ├── base.html             # Shell template with glowing sidebar & responsive top header
│       ├── dashboard.html        # Command dashboard with statistics & quick console
│       ├── analyzer.html         # Dedicated news analyzer workspace with metrics
│       ├── history.html          # Historical telemetry & audit log archive
│       ├── performance.html      # Model performance blueprints & confusion matrix standby
│       ├── about.html            # Academic methodology & multi-stage roadmap
│       └── error.html            # Branded dark error presentation (404/500)
│
├── data/
│   ├── raw/
│   │   └── WELFake_Dataset.csv   # Primary raw master dataset (72,134 records, preserved)
│   └── processed/
│       └── WELFake_cleaned.csv   # Deduplicated, standardized clean corpus (62,586 records)
│
├── reports/
│   ├── duplicate_analysis.txt    # Cross-dataset duplicate & contamination audit
│   └── welfake_cleaning_report.txt # Detailed cleaning, deduplication & label audit report
│
├── scripts/
│   ├── __init__.py               # Scripts module init
│   ├── inspect_datasets.py       # Non-destructive inspection of raw and cleaned datasets
│   ├── clean_welfake.py          # Data cleaning, deduplication & standardization pipeline
│   └── analyze_duplicates.py     # Deep duplicate & data leakage analysis
│
├── tests/
│   └── test_endpoints.py         # Automated test suite for web routes and REST endpoints
├── .env.example                  # Environment configuration template
├── .gitignore                    # Version control exclusion rules
├── requirements.txt              # Core Python dependencies
└── README.md                     # Comprehensive project documentation
```

---

## 🚀 Setup & Execution Guide

### Prerequisites
- Python **3.10** or higher installed on your system.

### Step 1: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Data Pipeline Operations (Phase 2)
To inspect the raw and cleaned datasets:
```bash
python scripts/inspect_datasets.py
```

To run the data cleaning and deduplication pipeline:
```bash
python scripts/clean_welfake.py
```

To run the duplicate and data leakage analysis:
```bash
python scripts/analyze_duplicates.py
```

### Step 3: Run the Web Dashboard
Launch the server using Uvicorn:
```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
Or simply:
```bash
python -m app.main
```

### Step 4: Access the Application
Open your web browser and navigate to:
- **Command Dashboard**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **News Analyzer**: [http://127.0.0.1:8000/analyzer](http://127.0.0.1:8000/analyzer)
- **Analysis History**: [http://127.0.0.1:8000/history](http://127.0.0.1:8000/history)
- **Model Performance**: [http://127.0.0.1:8000/performance](http://127.0.0.1:8000/performance)
- **About Project**: [http://127.0.0.1:8000/about](http://127.0.0.1:8000/about)
- **Interactive OpenAPI Documentation**: [http://127.0.0.1:8000/api/docs](http://127.0.0.1:8000/api/docs)

---

---

## ⚠️ Model Scope, Core Limitations & Operational Disclaimer

> [!IMPORTANT]
> **Mandatory Operational Disclaimer**:
> **NewsSentinel identifies linguistic patterns learned from its training data. It does not independently verify sources, events, quotations or factual claims.**

### 1. The Realistic Writing Style Limitation
Manual evaluation demonstrated that when a **deliberately fabricated scientific article** is composed in clean, realistic journalistic language (e.g. standard reporting structure, neutral vocabulary, formal attribution, lack of sensational punctuation), the model may classify it as **Real** with high confidence (e.g., 73.2%).

**Why this occurs**:
- Realistic writing style closely resembles linguistic and stylometric patterns learned from genuine-news training examples (such as Reuters wire reports).
- The model evaluates statistical n-gram token distributions, not real-world event occurrences or empirical reality.
- Fabricated claims that mimic professional journalism do not trigger the lexical patterns associated with overt sensational clickbait.

### 2. Model Confidence Is Not Factual Certainty
- "Model Confidence" represents the calibrated posterior probability $P(y = 1 \mid x)$ under the model's TF-IDF feature space.
- It measures how strongly input tokens correlate with patterns in the WELFake training corpus, **never the factual certainty or ground truth of the claims**.
- NewsSentinel is designed as a pattern-detection triage aid, not an autonomous fact-checker.

### 3. Clear Result Labels
To avoid misleading users, NewsSentinel strictly adheres to descriptive, non-dogmatic result labels:
- **`MODEL PREDICTION: REAL`**: The input exhibits linguistic patterns matching genuine news in the training corpus.
- **`MODEL PREDICTION: FAKE`**: The input exhibits linguistic patterns matching misinformation in the training corpus.
- **`UNCERTAIN — NEEDS VERIFICATION`**: Neither class reached the 0.65 probability threshold, indicating weak, mixed, or borderline lexical signals.

---

## 🔬 Core Features & Academic Integrity Principles

1. **No Fabricated Machine Learning Predictions**:
   - The platform strictly displays empirical telemetry computed from actual model inference and evaluated test sets.
2. **Explainable AI (XAI)**:
   - Approximate lexical token attributions ($\bar{w}_j \cdot x_j$) reveal the directional influence of specific words pushing toward Real or Fake.
3. **Calibrated Uncertainty Detection**:
   - Borderline predictions below the 0.65 threshold trigger an explicit "UNCERTAIN — NEEDS VERIFICATION" warning while preserving underlying probabilities.
4. **Persistent Audit Trail**:
   - Thread-safe SQLite database running in WAL mode maintains a complete, paginated historical log with full original text.
5. **Leakage-Free Corpus Architecture**:
   - Primary corpus is completely deduplicated (8,207 identical title-text pairs eliminated), and ISOT is documented as being contained within WELFake.
6. **Standardized Ground Truth**:
   - Target encoding is explicitly verified and aligned with benchmark literature (`0 = Fake`, `1 = Real`).
