<div align="center">

<img src="https://img.shields.io/badge/AI-Healthcare-blue?style=for-the-badge&logo=heart&logoColor=white" />
<img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
<img src="https://img.shields.io/badge/Gradio-UI-F97316?style=for-the-badge&logo=gradio&logoColor=white" />
<img src="https://img.shields.io/badge/Gemini-LLM-4285F4?style=for-the-badge&logo=google&logoColor=white" />
<img src="https://img.shields.io/badge/RAG-ChromaDB-8B5CF6?style=for-the-badge" />

# 🏥 AI-Powered Multi-Modal Healthcare Diagnostic System

**A multi-modal AI diagnostic system that combines classical Machine Learning, Deep Learning, Retrieval-Augmented Generation, and a Gemini-powered LLM to deliver comprehensive, evidence-backed clinical reports.**

</div>

---

> ⚠️ **Medical Disclaimer:** This system is built strictly for academic and educational purposes. All outputs are AI-generated predictions and **must not** be used as a substitute for professional medical advice, diagnosis, or treatment. A qualified clinician must review any healthcare decision.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [ML Pipeline — Phase 01](#-ml-pipeline--phase-01)
- [Deep Learning Pipeline — Phase 02](#-deep-learning-pipeline--phase-02)
- [RAG & LLM Report Engine — Phase 03](#-rag--llm-report-engine--phase-03)
- [Dataset Sources](#-dataset-sources)
- [Installation & Setup](#-installation--setup)
- [Configuration Reference](#-configuration-reference)
- [Saving Models from the Notebook](#-saving-models-from-the-notebook)
- [Running the Application](#-running-the-application)
- [Technology Stack](#-technology-stack)
- [Important Notes](#-important-notes)

---

## 🔍 Overview

The **Smart Healthcare Diagnostic Hub** is an end-to-end AI pipeline designed around two clinical prediction tasks:

| Task | Model Type | Dataset | Output |
|------|-----------|---------|--------|
| **Diabetes Risk Prediction** | Classical ML (Random Forest / XGBoost) | CDC BRFSS 2015 Health Indicators | Binary risk label + probability |
| **Pneumonia Detection** | Deep Learning (DenseNet121 CNN) | Chest X-Ray Images (Kaggle) | Normal vs Pneumonia + confidence |

After the models generate their predictions, a **Retrieval-Augmented Generation (RAG)** system queries a curated medical knowledge base and a **Gemini LLM** synthesizes all evidence into a structured, Arabic-language clinical report with citations — all served through an interactive **Gradio** web interface.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Gradio Web Interface                         │
│          (app.py — tabular inputs + chest X-ray upload)             │
└──────────────┬────────────────────────────────┬────────────────────┘
               │                                │
               ▼                                ▼
 ┌─────────────────────────┐      ┌─────────────────────────────────┐
 │    ML Model Adapter     │      │        DL Model Adapter         │
 │  (sklearn / joblib)     │      │  (DenseNet121 — Keras / PyTorch)│
 │  Diabetes Risk Predict  │      │    Chest X-Ray Classification   │
 └──────────┬──────────────┘      └──────────────┬──────────────────┘
            │                                     │
            └──────────────┬──────────────────────┘
                           │ ML Result + DL Result
                           ▼
             ┌─────────────────────────┐
             │      RAG Engine         │
             │  ┌───────────────────┐  │
             │  │ Hybrid Search     │  │
             │  │ ChromaDB (dense)  │  │
             │  │ BM25 (sparse)     │  │
             │  │ Cross-Encoder     │  │
             │  │   Reranker        │  │
             │  └───────────────────┘  │
             │  Medical PDF Knowledge  │
             │  Base (6 documents)     │
             └──────────┬──────────────┘
                        │ Retrieved Evidence
                        ▼
             ┌─────────────────────────┐
             │   Gemini LLM Service    │
             │  (gemini-2.5-flash-lite)│
             │  Structured Arabic      │
             │  Clinical Report        │
             └─────────────────────────┘
```

---

## ✨ Key Features

- **🤖 Multi-Modal Inference** — Combines structured tabular health data with medical image analysis in a single pipeline.
- **📚 Evidence-Grounded Reports** — Every LLM recommendation is tied to real retrieved evidence via `[Evidence N]` citations, preventing hallucination.
- **🔍 Hybrid Search RAG** — Uses both dense semantic vector search (ChromaDB + Sentence Transformers) and sparse BM25 keyword search, fused and re-ranked by a Cross-Encoder for maximum retrieval precision.
- **🛡️ Safety-First LLM Prompting** — The system prompt enforces cautious clinical language, forbids inventing diagnoses, and triggers urgent-care warnings for red-flag symptoms.
- **⚙️ Fully Configurable** — Every threshold, model path, label, and API key is managed via `.env` — no code changes needed to swap models or tune sensitivity.
- **🌐 Interactive UI** — Clean Gradio interface with side-by-side input panels and real-time report generation.
- **📊 Advanced Preprocessing** — SMOTE oversampling, NearMiss undersampling, VIF analysis, Chi-Square feature selection, and dimensionality visualization (PCA, t-SNE, UMAP).

---

## 📁 Project Structure

```
Smart-Healthcare-Diagnostic-Hub/
│
├── app.py                          # Gradio UI entry point
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
├── .env                            # Your local secrets (gitignored)
│
├── core/
│   ├── config.py                   # Centralised Settings (dataclass + dotenv)
│   ├── model_adapters.py           # ML & DL model loading + inference wrappers
│   ├── llm_service.py              # Gemini client, prompt builder, report generator
│   ├── rag_engine.py               # RAG orchestration (retrieve → rerank → return)
│   ├── hybrid_search.py            # BM25 index management + fusion search
│   ├── retriever.py                # ChromaDB semantic search
│   ├── reranker.py                 # Cross-encoder re-ranking
│   ├── ingest.py                   # PDF ingestion pipeline
│   ├── loader.py                   # Document loader utilities
│   ├── chunker.py                  # Text chunking strategy
│   ├── embedding.py                # Sentence Transformer embedding wrapper
│   ├── vector_db.py                # ChromaDB client setup
│   └── __init__.py
│
├── data/                           # Medical PDF knowledge base
│   ├── Classification and Diagnosis.pdf
│   ├── Introduction and Methodology.pdf
│   ├── Medical Evaluation.pdf
│   ├── Standards of Care in Diabetes.pdf
│   ├── Summary of Revisions Standards.pdf
│   └── X-Ray paper.pdf
│
├── models/
│   ├── ml_model.pkl                # Serialised sklearn pipeline (Random Forest)
│   ├── model.weights.h5            # DenseNet121 Keras weights
│   ├── config.json                 # DenseNet121 model architecture config
│   └── metadata.json               # Model metadata
│
├── vectordb/
│   ├── bm25_index.pkl              # Persisted BM25 sparse index
│   └── chroma/                     # ChromaDB persistent vector store
│
└── Smart_Healthcare_Diagnostic_Hub.ipynb   # Full ML/DL training notebook
```

---

## 🧪 ML Pipeline — Phase 01

The Jupyter notebook (`Smart_Healthcare_Diagnostic_Hub.ipynb`) covers a complete machine learning pipeline for diabetes risk classification.

### Dataset
**CDC Diabetes Health Indicators Dataset** (Kaggle — BRFSS 2015)
- **253,680** survey responses
- **21 features** including BMI, Blood Pressure, Cholesterol, Physical Activity, General Health, Age, etc.
- Binary target: `Diabetes_binary` (0 = No Diabetes, 1 = Diabetes Risk)

### Pipeline Steps

**1. Data Cleaning & EDA**
- Missing value analysis and removal
- Duplicate detection and removal
- Target distribution analysis
- Correlation heatmap

**2. Feature Analysis**
- **VIF (Variance Inflation Factor)** — multicollinearity detection
- **ANOVA F-test** — continuous feature selection
- **Chi-Square** — categorical feature scoring

**3. Feature Engineering**
Three interaction features were engineered to capture non-linear clinical relationships:

| Feature | Formula | Clinical Rationale |
|---------|---------|-------------------|
| `GenHlth_BMI` | `GenHlth × BMI` | Captures compound risk when poor health co-occurs with high BMI |
| `Age_HighBP` | `Age × HighBP` | Amplifies hypertension risk in older patients |
| `PhysHlth_NoActivity` | `PhysHlth × (1 - PhysActivity)` | Flags physically ill patients who are also inactive |

**4. Class Imbalance Handling**
- **SMOTE** (Synthetic Minority Over-sampling Technique) on the full dataset split
- **NearMiss v1 Undersampling** for the model comparison pipeline

**5. Dimensionality Reduction & Visualization**
Applied on a 5,000-sample subset for visual separability analysis:
- **PCA** (Principal Component Analysis)
- **t-SNE** (t-Distributed Stochastic Neighbor Embedding)
- **UMAP** (Uniform Manifold Approximation and Projection)

**6. Model Comparison**
All models evaluated with Confusion Matrix, Classification Report, and RMSE:

| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline linear model |
| Decision Tree | Depth-limited to 12 |
| Random Forest | 10 estimators, depth 12 |
| XGBoost | Learning rate 0.1 |

**7. Deep Learning Extension (Tabular)**
- **Conv1D Neural Network** — local feature dependency extraction on reshaped tabular vectors
- **Variational Autoencoder (VAE)** — unsupervised representation learning

---

## 🫁 Deep Learning Pipeline — Phase 02

### Dataset
**Chest X-Ray Images (Pneumonia)** — Paul Mooney, Kaggle
- **5,863** JPEG chest X-ray images
- Binary classes: `NORMAL` / `PNEUMONIA`
- Pre-split into train / validation / test directories

### Model Architecture: DenseNet121

DenseNet121 was selected for its dense connection pattern — every layer receives feature maps from all preceding layers — which encourages feature reuse, reduces parameter count, and provides strong gradient flow. These properties are especially valuable for medical imaging on relatively small datasets.

```
DenseNet121 Backbone (ImageNet pretrained, frozen in Phase 1)
        ↓
GlobalAveragePooling2D
        ↓
BatchNormalization
        ↓
Dense(256, relu) → Dropout(0.5)
        ↓
Dense(128, relu) → Dropout(0.25)
        ↓
Dense(1, sigmoid)  ← Binary: NORMAL vs PNEUMONIA
```

### Training Strategy

| Phase | Backbone | Epochs | Optimizer | LR |
|-------|----------|--------|-----------|-----|
| Phase 1 (Head training) | Frozen | 15 | Adam | 1e-3 |

**Callbacks used:**
- `ModelCheckpoint` — saves best model by `val_recall` and `val_auc` separately
- `EarlyStopping` — monitors `val_recall` with patience=5
- `ReduceLROnPlateau` — halves LR on `val_loss` plateau with patience=3
- `CSVLogger` — full epoch-by-epoch training log

**Class Imbalance:** `compute_class_weight('balanced')` applied during training to compensate for the pneumonia-heavy distribution.

### Clinical Threshold Optimization

Rather than using a default 0.5 threshold, the system performs a threshold sweep from 0.20 to 0.75 and selects the **lowest threshold where Recall ≥ 0.95**. In medical screening, false negatives (missed pneumonia cases) carry far greater risk than false positives.

```
Default threshold: 0.5
Clinical threshold: 0.3  (configured in .env as DL_THRESHOLD)
```

---

## 🔬 RAG & LLM Report Engine — Phase 03

### Knowledge Base

Six curated medical PDFs are ingested, chunked, embedded, and indexed:

| Document | Content |
|----------|---------|
| Standards of Care in Diabetes | ADA clinical guidelines |
| Classification and Diagnosis | Diabetes diagnostic criteria |
| Introduction and Methodology | Study methodology reference |
| Medical Evaluation | Clinical evaluation standards |
| Summary of Revisions Standards | Standards revision summary |
| X-Ray Paper | Chest X-ray diagnostic reference |

### Hybrid Search Architecture

```
User query
    │
    ├──► ChromaDB (Sentence Transformers dense embeddings)
    │         └──► Top-K semantic matches
    │
    └──► BM25 (Okapi BM25 sparse index)
              └──► Top-K keyword matches
                         │
              ┌──────────▼──────────┐
              │   Score Fusion      │
              │ (RRF / weighted)    │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  Cross-Encoder      │
              │    Re-Ranker        │
              └──────────┬──────────┘
                         │
              Final Top-K Evidence Chunks
```

### Gemini Report Structure

The LLM produces a structured 8-section Arabic clinical report:

| Section | Content |
|---------|---------|
| **1. Result Summary** | ML + DL prediction labels and probabilities |
| **2. Why ML predicted Diabetes Risk** | Feature-linked explanation with input values |
| **3. Why DL predicted Pneumonia/Normal** | Image confidence interpretation |
| **4. RAG Medical Evidence** | Summarised retrieved evidence with `[Evidence N]` citations |
| **5. What this means for the user** | Safe, general next steps (no prescriptions) |
| **6. Red-Flag Warning Signs** | Urgent symptoms requiring immediate clinical review |
| **7. Interpretation Limitations** | Model quality, data quality, and RAG caveats |
| **8. Medical Disclaimer** | Mandatory non-diagnostic disclaimer |

---

## 📊 Dataset Sources

| Dataset | Platform | Link |
|---------|----------|------|
| Diabetes Health Indicators (BRFSS 2015) | Kaggle | [alexteboul/diabetes-health-indicators-dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) |
| Chest X-Ray Images (Pneumonia) | Kaggle | [paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.10 or higher
- pip
- A valid **Google Gemini API Key** (free tier available at [aistudio.google.com](https://aistudio.google.com))

### Step 1 — Clone the Repository

```bash
git clone https://github.com/your-username/smart-healthcare-diagnostic-hub.git
cd smart-healthcare-diagnostic-hub
```

### Step 2 — Create a Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Do **not** install `google-generativeai` alongside TensorFlow 2.21 — it causes `protobuf` version conflicts. This project uses the newer `google-genai` package exclusively.

### Step 4 — Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` and fill in your credentials:

```env
GEMINI_API_KEY=your_real_gemini_api_key_here
GEMINI_MODEL_NAME=gemini-2.5-flash-lite
```

---

## ⚙️ Configuration Reference

All settings are loaded from `.env` via `core/config.py`. Below is the full reference:

### LLM Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | *(required)* | Your Google Gemini API key |
| `GEMINI_MODEL_NAME` | `gemini-2.5-flash-lite` | Gemini model to use |

### ML Model Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ML_MODEL_PATH` | `models/ml_model.pkl` | Path to the serialised sklearn pipeline |
| `ML_THRESHOLD` | `0.5` | Decision threshold for Diabetes Risk |
| `ML_POSITIVE_CLASS_INDEX` | `1` | Index of the positive class |
| `ML_LABEL_NEGATIVE` | `No Diabetes Risk` | Label for class 0 |
| `ML_LABEL_POSITIVE` | `Diabetes Risk` | Label for class 1 |
| `ML_FEATURE_NAMES` | *(19 features, comma-separated)* | Feature input order |

### DL Model Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `DL_MODEL_PATH` | `models/dl_model.keras` | Path to full Keras model or weights file |
| `DL_MODEL_CONFIG_PATH` | `models/config.json` | Architecture config (required for `.h5` weights) |
| `DL_FRAMEWORK` | `auto` | `keras`, `keras_weights`, `torch`, or `auto` |
| `DL_IMAGE_SIZE` | `224,224` | Input image dimensions |
| `DL_THRESHOLD` | `0.3` | Clinical threshold (optimised for recall ≥ 0.95) |
| `DL_LABEL_NEGATIVE` | `Normal X-ray` | Label for class 0 |
| `DL_LABEL_POSITIVE` | `Pneumonia` | Label for class 1 |

### RAG Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_TOP_K` | `4` | Number of evidence chunks to retrieve and cite |

---

## 💾 Saving Models from the Notebook

### Saving the ML Model

After running the final feature engineering and model training cells in the notebook, save your best model using:

```python
import os
import joblib
from sklearn.ensemble import RandomForestClassifier

os.makedirs("models", exist_ok=True)

# Use the feature names from the engineered dataframe
feature_names = list(X_train_p_raw.columns)

best_ml_model = RandomForestClassifier(
    max_depth=12,
    n_estimators=10,
    random_state=42,
)
best_ml_model.fit(X_train_p, y_train_p)

joblib.dump(
    {
        "model": best_ml_model,
        "preprocessor": scaler_p,
        "feature_names": feature_names,
        "threshold": 0.5,
        "labels": ["No Diabetes Risk", "Diabetes Risk"],
    },
    "models/ml_model.pkl",
)

print("Saved ML model to models/ml_model.pkl")
print(feature_names)
```

**Expected feature order (19 features):**

```
HighBP, HighChol, BMI, Smoker, Stroke, HeartDiseaseorAttack,
PhysActivity, HvyAlcoholConsump, NoDocbcCost, GenHlth, MentHlth,
PhysHlth, DiffWalk, Age, Education, Income,
GenHlth_BMI, Age_HighBP, PhysHlth_NoActivity
```

### Saving the DL Model

The notebook automatically saves the best model by `val_recall` to:

```
checkpoints/best_model_recall.keras
```

**Option A** — Copy the full model:
```bash
cp checkpoints/best_model_recall.keras models/dl_model.keras
```

**Option B** — Use weights + config (already present in `models/`), then configure `.env`:
```env
DL_MODEL_PATH=models/model.weights.h5
DL_MODEL_CONFIG_PATH=models/config.json
DL_FRAMEWORK=keras_weights
DL_IMAGE_SIZE=224,224
DL_THRESHOLD=0.3
DL_LABEL_NEGATIVE=Normal X-ray
DL_LABEL_POSITIVE=Pneumonia
```

> The adapter applies the same preprocessing as the notebook's validation/test generator: resize to (224, 224) and normalize by dividing pixel values by 255.

---

## ▶️ Running the Application

```bash
python app.py
```

The Gradio interface will launch at `http://127.0.0.1:7860` by default.

### Input Fields

**ML Inputs (Diabetes Risk)**

| Input | Type | Range |
|-------|------|-------|
| HighBP | Radio | 0 = No, 1 = Yes |
| HighChol | Radio | 0 = No, 1 = Yes |
| BMI | Number | Any positive float |
| Smoker | Radio | 0 = No, 1 = Yes |
| Stroke | Radio | 0 = No, 1 = Yes |
| HeartDiseaseorAttack | Radio | 0 = No, 1 = Yes |
| PhysActivity | Radio | 0 = No, 1 = Yes |
| HvyAlcoholConsump | Radio | 0 = No, 1 = Yes |
| NoDocbcCost | Radio | 0 = No, 1 = Yes |
| GenHlth | Slider | 1 (Excellent) → 5 (Poor) |
| MentHlth | Slider | 0–30 days |
| PhysHlth | Slider | 0–30 days |
| DiffWalk | Radio | 0 = No, 1 = Yes |
| Age category | Slider | 1–13 (CDC age bands) |
| Education | Slider | 1–6 |
| Income | Slider | 1–8 |
| Symptoms / Notes | Text | Free text |

**DL Input (Chest X-Ray)**

| Input | Type | Notes |
|-------|------|-------|
| Chest X-ray Image | Image upload | JPEG / PNG, auto-resized to 224×224 |

### Output

| Output | Description |
|--------|-------------|
| **Gemini Report** | Structured 8-section Arabic clinical report with citations |
| **Raw Model / RAG Outputs** | JSON debug view of ML result, DL result, RAG query, and evidence list |

---

## 🛠️ Technology Stack

| Category | Technology |
|----------|------------|
| **Web UI** | Gradio |
| **ML Models** | scikit-learn, XGBoost, imbalanced-learn |
| **Deep Learning** | TensorFlow / Keras (DenseNet121) |
| **LLM** | Google Gemini (`google-genai`) |
| **Vector Store** | ChromaDB |
| **Embeddings** | Sentence Transformers |
| **Sparse Search** | rank-bm25 (BM25Okapi) |
| **PDF Parsing** | pypdf |
| **Data Processing** | pandas, numpy |
| **Visualisation** | matplotlib, seaborn |
| **Dimensionality Reduction** | scikit-learn PCA, t-SNE; UMAP-learn |
| **Config Management** | python-dotenv |
| **Model Serialisation** | joblib |
| **Image Processing** | Pillow |

---

## 📌 Important Notes

- **`google-generativeai` conflict:** Do **not** install this package in the same environment as TensorFlow 2.21. It conflicts via `protobuf`. Always use `google-genai` as listed in `requirements.txt`.
- **Gradio form tags:** The UI avoids native HTML `<form>` elements; all interactions are handled through Gradio's event system.
- **BM25 caching:** The BM25 model is cached in memory after the first build and only rebuilt when new documents are added, preventing redundant IDF recalculation on every query.
- **ChromaDB persistence:** The vector store is persisted to `vectordb/chroma/` and loaded automatically on startup — you do not need to re-ingest documents unless the knowledge base changes.
- **Threshold tuning:** The DL clinical threshold (default `0.3`) is intentionally lower than the standard `0.5` to prioritise recall in a medical screening context. Adjust `DL_THRESHOLD` in `.env` if needed.
- **Educational use only:** The models, RAG system, and generated reports are for university/project demonstration purposes. This system is **not validated for clinical deployment**.

---

<div align="center">

Built with ❤️ for AI in Healthcare Research

</div>