"""Application configuration loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"

if load_dotenv:
    load_dotenv(ENV_PATH)


NOTEBOOK_ML_FEATURES = [
    "HighBP",
    "HighChol",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "HvyAlcoholConsump",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Age",
    "Education",
    "Income",
    "GenHlth_BMI",
    "Age_HighBP",
    "PhysHlth_NoActivity",
]


def _csv(name: str, default: str) -> List[str]:
    raw = os.getenv(name, default)
    return [x.strip() for x in raw.split(",") if x.strip()]


def _float_csv(name: str, default: str) -> List[float]:
    return [float(x) for x in _csv(name, default)]


@dataclass(frozen=True)
class Settings:
    # Gemini / LLM
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "").strip()
    gemini_model_name: str = os.getenv(
        "GEMINI_MODEL_NAME",
        "gemini-2.5-flash-lite",
    ).strip()

    # ML config follows the notebook target: Diabetes_binary.
    ml_model_path: str = os.getenv("ML_MODEL_PATH", "models/ml_model.pkl").strip()
    ml_preprocessor_path: str = os.getenv("ML_PREPROCESSOR_PATH", "").strip()
    ml_feature_names: List[str] = None  # set below
    ml_positive_class_index: int = int(os.getenv("ML_POSITIVE_CLASS_INDEX", "1"))
    ml_threshold: float = float(os.getenv("ML_THRESHOLD", "0.5"))
    ml_label_negative: str = os.getenv(
        "ML_LABEL_NEGATIVE",
        "No Diabetes Risk",
    ).strip()
    ml_label_positive: str = os.getenv(
        "ML_LABEL_POSITIVE",
        "Diabetes Risk",
    ).strip()

    # DL config follows the notebook:
    # DenseNet121, binary NORMAL/PNEUMONIA, 224x224, image / 255.
    #
    # Supports either:
    # 1. Full model:
    #    DL_MODEL_PATH=models/dl_model.keras
    #
    # 2. Weights + config:
    #    DL_MODEL_PATH=models/model.weights.h5
    #    DL_MODEL_CONFIG_PATH=models/config.json
    #    DL_FRAMEWORK=keras_weights
    dl_model_path: str = os.getenv("DL_MODEL_PATH", "models/dl_model.keras").strip()
    dl_model_config_path: Optional[str] = os.getenv(
        "DL_MODEL_CONFIG_PATH",
        "",
    ).strip() or None

    dl_framework: str = os.getenv("DL_FRAMEWORK", "auto").strip().lower()
    dl_image_size: List[str] = None  # set below
    dl_mean: List[float] = None  # set below
    dl_std: List[float] = None  # set below
    dl_threshold: float = float(os.getenv("DL_THRESHOLD", "0.3"))
    dl_label_negative: str = os.getenv(
        "DL_LABEL_NEGATIVE",
        "Normal X-ray",
    ).strip()
    dl_label_positive: str = os.getenv(
        "DL_LABEL_POSITIVE",
        "Pneumonia",
    ).strip()

    # RAG
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "4"))


settings = Settings(
    ml_feature_names=_csv("ML_FEATURE_NAMES", ",".join(NOTEBOOK_ML_FEATURES)),
    dl_image_size=_csv("DL_IMAGE_SIZE", "224,224"),
    dl_mean=_float_csv("DL_MEAN", "0.485,0.456,0.406"),
    dl_std=_float_csv("DL_STD", "0.229,0.224,0.225"),
)