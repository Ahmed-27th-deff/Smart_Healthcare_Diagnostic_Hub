# Smart Healthcare Diagnostic Hub

This version is aligned with the uploaded notebook.

## What the models do

- **ML model:** predicts `Diabetes_binary` / diabetes risk from BRFSS-style tabular features.
- **DL model:** predicts chest X-ray `Normal X-ray` vs `Pneumonia` using the DenseNet121 Keras model from the notebook.
- **RAG + Gemini:** explains why the predictions may have happened, using retrieved evidence and citations.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Then set your Gemini key in `.env`:

```env
GEMINI_API_KEY=your_real_key_here
GEMINI_MODEL_NAME=gemini-2.5-flash-lite
```

## Save the ML model from the notebook

The notebook ML target is `Diabetes_binary`. After the final feature engineering cell and after selecting/training your best sklearn model, save it like this:

```python
import os
import joblib
from sklearn.ensemble import RandomForestClassifier

os.makedirs("models", exist_ok=True)

# The notebook uses X_train_p_raw after adding engineered features.
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

Expected default feature order:

```text
HighBP, HighChol, BMI, Smoker, Stroke, HeartDiseaseorAttack,
PhysActivity, HvyAlcoholConsump, NoDocbcCost, GenHlth, MentHlth,
PhysHlth, DiffWalk, Age, Education, Income,
GenHlth_BMI, Age_HighBP, PhysHlth_NoActivity
```

## Save the DL model from the notebook

The notebook saves the best DenseNet121 recall model here:

```text
checkpoints/best_model_recall.keras
```

Copy it into:

```text
models/dl_model.keras
```

or update `.env`:

```env
DL_MODEL_PATH=checkpoints/best_model_recall.keras
DL_FRAMEWORK=keras
DL_IMAGE_SIZE=224,224
DL_THRESHOLD=0.3
DL_LABEL_NEGATIVE=Normal X-ray
DL_LABEL_POSITIVE=Pneumonia
```

The adapter uses the same inference preprocessing as the notebook validation/test generator:

```python
image = image.resize((224, 224))
array = array / 255.0
```

## Run

```bash
python app.py
```

## Notes

- Do not install `google-generativeai` in this project with TensorFlow 2.21 because it can conflict through `protobuf`.
- This project uses the newer `google-genai` package only.
- The report is educational/project support only and is not a medical diagnosis.
