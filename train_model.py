"""Train and persist the CreditWise loan approval model (matches credit_wise.ipynb)."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).parent / "loan_approval_data.csv"
MODEL_DIR = Path(__file__).parent / "models"

OHE_COLS = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Gender",
    "Employer_Category",
]


def load_and_preprocess():
    df = pd.read_csv(DATA_PATH)

    categorical_cols = df.select_dtypes(include=["object"]).columns
    numerical_cols = df.select_dtypes(include=["number"]).columns

    num_imp = SimpleImputer(strategy="mean")
    df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])

    cat_imp = SimpleImputer(strategy="most_frequent")
    df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])

    df = df.drop("Applicant_ID", axis=1)

    education_encoder = LabelEncoder()
    df["Education_Level"] = education_encoder.fit_transform(df["Education_Level"])

    target_encoder = LabelEncoder()
    df["Loan_Approved"] = target_encoder.fit_transform(df["Loan_Approved"])

    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoded = ohe.fit_transform(df[OHE_COLS])
    encoded_df = pd.DataFrame(
        encoded, columns=ohe.get_feature_names_out(OHE_COLS), index=df.index
    )
    df = pd.concat([df.drop(columns=OHE_COLS), encoded_df], axis=1)

    df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
    df["Credit_Score_sq"] = df["Credit_Score"] ** 2

    X = df.drop(columns=["Loan_Approved", "Credit_Score", "DTI_Ratio"])
    y = df["Loan_Approved"]

    return (
        X,
        y,
        num_imp,
        cat_imp,
        education_encoder,
        target_encoder,
        ohe,
    )


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)

    X, y, num_imp, cat_imp, education_encoder, target_encoder, ohe = load_and_preprocess()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "model": "Logistic Regression",
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "samples_trained": int(len(X_train)),
        "features": int(X.shape[1]),
    }

    joblib.dump(model, MODEL_DIR / "model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(ohe, MODEL_DIR / "ohe.pkl")
    joblib.dump(education_encoder, MODEL_DIR / "education_encoder.pkl")
    joblib.dump(target_encoder, MODEL_DIR / "target_encoder.pkl")
    joblib.dump(num_imp, MODEL_DIR / "num_imputer.pkl")
    joblib.dump(cat_imp, MODEL_DIR / "cat_imputer.pkl")
    joblib.dump(list(X.columns), MODEL_DIR / "feature_columns.pkl")

    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("CreditWise model trained and saved.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
