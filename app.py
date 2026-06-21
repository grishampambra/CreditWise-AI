"""CreditWise — AI loan approval prediction web application."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"

OHE_COLS = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Gender",
    "Employer_Category",
]

app = Flask(__name__)

_model = None
_scaler = None
_ohe = None
_education_encoder = None
_target_encoder = None
_feature_columns = None
_metrics = None


def load_artifacts():
    global _model, _scaler, _ohe, _education_encoder, _target_encoder, _feature_columns, _metrics

    if _model is not None:
        return

    _model = joblib.load(MODEL_DIR / "model.pkl")
    _scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    _ohe = joblib.load(MODEL_DIR / "ohe.pkl")
    _education_encoder = joblib.load(MODEL_DIR / "education_encoder.pkl")
    _target_encoder = joblib.load(MODEL_DIR / "target_encoder.pkl")
    _feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")

    metrics_path = MODEL_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            _metrics = json.load(f)


def preprocess_input(data: dict) -> np.ndarray:
    row = {
        "Applicant_Income": float(data["applicant_income"]),
        "Coapplicant_Income": float(data["coapplicant_income"]),
        "Employment_Status": data["employment_status"],
        "Age": float(data["age"]),
        "Marital_Status": data["marital_status"],
        "Dependents": float(data["dependents"]),
        "Credit_Score": float(data["credit_score"]),
        "Existing_Loans": float(data["existing_loans"]),
        "DTI_Ratio": float(data["dti_ratio"]),
        "Savings": float(data["savings"]),
        "Collateral_Value": float(data["collateral_value"]),
        "Loan_Amount": float(data["loan_amount"]),
        "Loan_Term": float(data["loan_term"]),
        "Loan_Purpose": data["loan_purpose"],
        "Property_Area": data["property_area"],
        "Education_Level": data["education_level"],
        "Gender": data["gender"],
        "Employer_Category": data["employer_category"],
    }

    df = pd.DataFrame([row])

    df["Education_Level"] = _education_encoder.transform(df["Education_Level"])

    encoded = _ohe.transform(df[OHE_COLS])
    encoded_df = pd.DataFrame(
        encoded, columns=_ohe.get_feature_names_out(OHE_COLS), index=df.index
    )
    df = pd.concat([df.drop(columns=OHE_COLS), encoded_df], axis=1)

    df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
    df["Credit_Score_sq"] = df["Credit_Score"] ** 2

    df = df.drop(columns=["Credit_Score", "DTI_Ratio"])
    df = df[_feature_columns]

    scaled = _scaler.transform(df)
    return scaled


@app.route("/")
def index():
    load_artifacts()
    return render_template("index.html", metrics=_metrics or {})


@app.route("/api/predict", methods=["POST"])
def predict():
    load_artifacts()

    try:
        data = request.get_json()
        features = preprocess_input(data)
        proba = float(_model.predict_proba(features)[0][1])
        prediction = int(_model.predict(features)[0])
        label = _target_encoder.inverse_transform([prediction])[0]

        return jsonify(
            {
                "approved": label == "Yes",
                "label": label,
                "confidence": round(proba if label == "Yes" else 1 - proba, 4),
                "probability_approved": round(proba, 4),
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/metrics")
def metrics():
    load_artifacts()
    return jsonify(_metrics or {})


if __name__ == "__main__":
    if not (MODEL_DIR / "model.pkl").exists():
        print("Model not found. Run: python train_model.py")
    app.run(debug=True, port=5000)
