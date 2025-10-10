# src/api/app.py

from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import json
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = Path("/Users/admin/Desktop/faers-ae-prediction/models")
# -----------------------------
# Load models and features
# -----------------------------
logreg = joblib.load(MODEL_DIR / "logreg_baseline.joblib")
rf = joblib.load(MODEL_DIR / "rf_model.joblib")
gb = joblib.load(MODEL_DIR / "gb_model.joblib")

with open(MODEL_DIR / "logreg_features.json") as f:
    feature_names = json.load(f)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="FAERS Adverse Event Prediction API")

# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
    
    # Age bins - MUST match exactly what was used during training
    # The error shows the model expects: age_bin_35-49, age_bin_50-64, age_bin_65+
    # But you're creating: age_bin_0-34, age_bin_35-49, age_bin_50-64, age_bin_65+
    
    # Create ALL age bins that the model expects
    df["age_bin_0-34"] = (df["age"] <= 34).astype(int)
    df["age_bin_35-49"] = ((df["age"] >= 35) & (df["age"] <= 49)).astype(int)
    df["age_bin_50-64"] = ((df["age"] >= 50) & (df["age"] <= 64)).astype(int)
    df["age_bin_65+"] = (df["age"] >= 65).astype(int)

    # Sex encoding
    df["sex_male"] = (df["sex"].str.lower() == "male").astype(int)
    df["sex_female"] = (df["sex"].str.lower() == "female").astype(int)

    # Drug and reaction encoding
    for col in feature_names:
        if col.startswith("drug_") and "drug_name" in df:
            drug_value = col.replace("drug_", "").lower()
            df[col] = (df["drug_name"].str.lower() == drug_value).astype(int)
        elif col.startswith("reaction_") and "reaction" in df:
            reaction_value = col.replace("reaction_", "").lower()
            df[col] = (df["reaction"].str.lower() == reaction_value).astype(int)
        elif col not in df.columns:
            # Fill missing columns with 0
            df[col] = 0

    # Ensure all feature columns exist and are in correct order
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    # Align columns to training features
    df = df.reindex(columns=feature_names, fill_value=0)
    
    return df

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def home():
    return {"message": "FAERS Adverse Event Prediction API is running"}

@app.post("/predict")
def predict(data: dict, model: str = "logreg"):
    try:
        X_input = preprocess_input(data)

        if model == "logreg":
            probs = logreg.predict_proba(X_input)[:, 1][0]
        elif model == "rf":
            probs = rf.predict_proba(X_input)[:, 1][0]
        elif model == "gb":
            probs = gb.predict_proba(X_input)[:, 1][0]
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

        return {"probability": float(probs)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))