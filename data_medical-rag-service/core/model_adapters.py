"""Real ML/DL adapters for the Smart Healthcare Diagnostic Hub."""
from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.config import ROOT_DIR, settings


@dataclass
class MLResult:
    status: str
    label: str
    probability: Optional[float]
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DLResult:
    status: str
    label: str
    probability: Optional[float]
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _resolve_path(path_value: str | None) -> Optional[Path]:
    if not path_value:
        return None
    p = Path(path_value)
    return p if p.is_absolute() else ROOT_DIR / p


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


class MLModelAdapter:
    """Generic sklearn/joblib tabular model adapter."""

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = _resolve_path(model_path or settings.ml_model_path)
        self.preprocessor_path = _resolve_path(settings.ml_preprocessor_path)
        self.feature_names: List[str] = list(settings.ml_feature_names)
        self.threshold = settings.ml_threshold
        self.negative_label = settings.ml_label_negative
        self.positive_label = settings.ml_label_positive
        self.model: Any = None
        self.preprocessor: Any = None
        self.load_error: Optional[str] = None
        self._load()

    def _load_pickle_or_joblib(self, path: Path) -> Any:
        try:
            import joblib
            return joblib.load(path)
        except Exception:
            with open(path, "rb") as f:
                return pickle.load(f)

    def _load(self) -> None:
        if self.model_path is None or not self.model_path.exists():
            self.load_error = f"ML model not found at {self.model_path}"
            return

        try:
            artifact = self._load_pickle_or_joblib(self.model_path)

            if isinstance(artifact, dict):
                self.model = (
                    artifact.get("model")
                    or artifact.get("estimator")
                    or artifact.get("pipeline")
                )
                self.preprocessor = artifact.get("preprocessor") or artifact.get("scaler")
                self.feature_names = list(artifact.get("feature_names", self.feature_names))
                self.threshold = float(artifact.get("threshold", self.threshold))

                labels = artifact.get("labels") or artifact.get("class_names")
                if isinstance(labels, Sequence) and len(labels) >= 2:
                    self.negative_label = str(labels[0])
                    self.positive_label = str(labels[-1])
            else:
                self.model = artifact

            if self.model is None:
                raise ValueError("Could not find a model object in the ML artifact.")

            if self.preprocessor is None and self.preprocessor_path and self.preprocessor_path.exists():
                self.preprocessor = self._load_pickle_or_joblib(self.preprocessor_path)

        except Exception as exc:
            self.load_error = str(exc)
            self.model = None

    def _to_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except Exception:
            return default

    def _build_input(self, features: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, float]]:
        clean = {
            name: self._to_float(features.get(name), 0.0)
            for name in self.feature_names
        }
        arr = np.array([[clean[name] for name in self.feature_names]], dtype=float)
        return arr, clean

    def _maybe_dataframe(self, arr: np.ndarray, clean: Dict[str, float]) -> Any:
        try:
            import pandas as pd
            return pd.DataFrame([clean], columns=self.feature_names)
        except Exception:
            return arr

    def _probability_from_output(self, output: Any) -> float:
        values = np.asarray(output).astype(float)

        if values.ndim == 2:
            row = values[0]
            if row.size == 1:
                return float(row[0])

            idx = min(max(settings.ml_positive_class_index, 0), row.size - 1)
            return float(row[idx])

        return float(values.ravel()[0])

    def predict(self, features: Dict[str, Any]) -> MLResult:
        arr, clean = self._build_input(features)

        if self.model is None:
            return MLResult(
                status="model_missing",
                label="ML model not connected",
                probability=None,
                details={
                    "message": "Place your trained ML model file and set ML_MODEL_PATH in .env.",
                    "expected_features": self.feature_names,
                    "received_features": clean,
                    "load_error": self.load_error,
                },
            )

        try:
            x: Any = self._maybe_dataframe(arr, clean)

            if self.preprocessor is not None:
                x = self.preprocessor.transform(x)

            probability: Optional[float] = None

            if hasattr(self.model, "predict_proba"):
                probability = self._probability_from_output(self.model.predict_proba(x))
                label = self.positive_label if probability >= self.threshold else self.negative_label

            elif hasattr(self.model, "decision_function"):
                probability = _sigmoid(self._probability_from_output(self.model.decision_function(x)))
                label = self.positive_label if probability >= self.threshold else self.negative_label

            elif hasattr(self.model, "predict"):
                pred = np.asarray(self.model.predict(x)).ravel()[0]

                try:
                    probability = float(pred)
                    label = self.positive_label if probability >= self.threshold else self.negative_label
                except Exception:
                    label = str(pred)

            else:
                raise TypeError("ML model must expose predict_proba, decision_function, or predict.")

            return MLResult(
                status="ok",
                label=label,
                probability=probability,
                details={
                    "features_used": clean,
                    "feature_order": self.feature_names,
                    "threshold": self.threshold,
                    "model_path": str(self.model_path),
                },
            )

        except Exception as exc:
            return MLResult(
                status="error",
                label="ML inference failed",
                probability=None,
                details={
                    "error": str(exc),
                    "features_used": clean,
                    "feature_order": self.feature_names,
                },
            )


class DLModelAdapter:
    """Keras/PyTorch image classification adapter."""

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = _resolve_path(model_path or settings.dl_model_path)
        self.config_path = _resolve_path(getattr(settings, "dl_model_config_path", None))

        self.framework = settings.dl_framework
        self.image_size = (
            int(settings.dl_image_size[0]),
            int(settings.dl_image_size[1]),
        )
        self.mean = np.array(settings.dl_mean, dtype=np.float32)
        self.std = np.array(settings.dl_std, dtype=np.float32)
        self.threshold = settings.dl_threshold
        self.negative_label = settings.dl_label_negative
        self.positive_label = settings.dl_label_positive

        self.model: Any = None
        self.load_error: Optional[str] = None
        self.loaded_framework: Optional[str] = None

        self._load()

    def _infer_framework(self) -> str:
        if self.framework in {"keras", "tensorflow", "tf", "keras_weights"}:
            return "keras"

        if self.framework in {"pytorch", "torch"}:
            return "pytorch"

        name = self.model_path.name.lower() if self.model_path else ""
        suffix = self.model_path.suffix.lower() if self.model_path else ""

        if name.endswith(".weights.h5"):
            return "keras"

        if suffix in {".keras", ".h5", ".hdf5"}:
            return "keras"

        if suffix in {".pt", ".pth", ".jit", ".ts"}:
            return "pytorch"

        return "keras"

    def _load_keras_from_weights_and_config(self) -> Any:
        from tensorflow import keras

        if self.config_path is None or not self.config_path.exists():
            raise FileNotFoundError(
                "DL_MODEL_CONFIG_PATH is required when using model.weights.h5. "
                "Example: DL_MODEL_CONFIG_PATH=models/config.json"
            )

        if self.model_path is None or not self.model_path.exists():
            raise FileNotFoundError("DL_MODEL_PATH must point to model.weights.h5.")

        with open(self.config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        model = keras.models.model_from_json(json.dumps(config_data))
        model.load_weights(str(self.model_path))
        return model

    def _load(self) -> None:
        if self.model_path is None or not self.model_path.exists():
            self.load_error = f"DL model not found at {self.model_path}"
            return

        fw = self._infer_framework()

        try:
            if fw == "keras":
                from tensorflow import keras

                if self.model_path.name.lower().endswith(".weights.h5"):
                    self.model = self._load_keras_from_weights_and_config()
                else:
                    self.model = keras.models.load_model(str(self.model_path))

                self.loaded_framework = "keras"

            else:
                import torch

                try:
                    self.model = torch.jit.load(str(self.model_path), map_location="cpu")
                except Exception:
                    self.model = torch.load(str(self.model_path), map_location="cpu")

                if isinstance(self.model, dict):
                    raise ValueError(
                        "This looks like a PyTorch state_dict/checkpoint. "
                        "Save the full model or TorchScript model, or add your architecture class."
                    )

                self.model.eval()
                self.loaded_framework = "pytorch"

        except Exception as exc:
            self.load_error = str(exc)
            self.model = None

    def _prepare_pil(self, image: Any):
        from PIL import Image

        if isinstance(image, Image.Image):
            img = image
        else:
            img = Image.fromarray(image)

        return img.convert("RGB").resize(self.image_size)

    def _keras_predict(self, image: Any) -> np.ndarray:
        img = self._prepare_pil(image)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        batch = np.expand_dims(arr, axis=0)
        pred = self.model.predict(batch, verbose=0)
        return np.asarray(pred).ravel()

    def _torch_predict(self, image: Any) -> np.ndarray:
        import torch

        img = self._prepare_pil(image)

        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std

        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()

        with torch.no_grad():
            out = self.model(tensor)

            if isinstance(out, (tuple, list)):
                out = out[0]

            out = out.detach().cpu()

            if out.ndim == 2 and out.shape[1] > 1:
                out = torch.softmax(out, dim=1)
            else:
                out = torch.sigmoid(out)

        return out.numpy().ravel()

    def _score_and_label(self, raw: np.ndarray) -> Tuple[float, str]:
        if raw.size == 0:
            raise ValueError("DL model returned an empty prediction.")

        if raw.size == 1:
            score = float(raw[0])
        else:
            score = float(raw[-1])

        label = self.positive_label if score >= self.threshold else self.negative_label
        return score, label

    def predict(self, image: Any) -> DLResult:
        if image is None:
            return DLResult(
                status="skipped",
                label="No X-ray image uploaded",
                probability=None,
                details={},
            )

        if self.model is None:
            return DLResult(
                status="model_missing",
                label="DL model not connected",
                probability=None,
                details={
                    "message": "Place your trained DL model files and set DL_MODEL_PATH/DL_MODEL_CONFIG_PATH in .env.",
                    "supported": [
                        "Keras full model: .keras/.h5/.hdf5",
                        "Keras weights + config: model.weights.h5 + config.json",
                        "PyTorch full model/TorchScript: .pt/.pth",
                    ],
                    "model_path": str(self.model_path),
                    "config_path": str(self.config_path),
                    "load_error": self.load_error,
                },
            )

        try:
            if self.loaded_framework == "keras":
                raw = self._keras_predict(image)
            else:
                raw = self._torch_predict(image)

            score, label = self._score_and_label(raw)

            return DLResult(
                status="ok",
                label=label,
                probability=score,
                details={
                    "framework": self.loaded_framework,
                    "model_path": str(self.model_path),
                    "config_path": str(self.config_path),
                    "image_size": self.image_size,
                    "threshold": self.threshold,
                    "raw_prediction": raw.astype(float).tolist(),
                },
            )

        except Exception as exc:
            return DLResult(
                status="error",
                label="DL inference failed",
                probability=None,
                details={"error": str(exc)},
            )