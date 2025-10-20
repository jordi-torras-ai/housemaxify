import os
import json
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ===================== CONFIG =====================
CLEAN_CSV_PATH = "../data/kc_house_data_clean.csv"
MODELS_DIR = "../models"
MODEL_PATH = os.path.join(MODELS_DIR, "linreg_kc_house.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "linreg_kc_house_metrics.json")

TARGET = "price"
TIME_SPLIT = "2015-01-01"
# ==================================================


def load_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "sale_date" not in df.columns:
        raise ValueError("Expected 'sale_date' column in the clean dataset.")
    df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")

    # Ensure zipcode is categorical (string) to avoid encoder dtype issues
    if "zipcode" in df.columns:
        df["zipcode"] = df["zipcode"].astype(str)

    return df


def time_based_split(df: pd.DataFrame):
    split_dt = pd.to_datetime(TIME_SPLIT)
    train_df = df[df["sale_date"] < split_dt].copy()
    test_df  = df[df["sale_date"] >= split_dt].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split yielded empty set(s). Adjust TIME_SPLIT.")

    X_train = train_df.drop(columns=[TARGET, "sale_date"])
    y_train = train_df[TARGET].values
    X_test  = test_df.drop(columns=[TARGET, "sale_date"])
    y_test  = test_df[TARGET].values

    # Safety: enforce dtype again after slicing
    if "zipcode" in X_train.columns:
        X_train["zipcode"] = X_train["zipcode"].astype(str)
        X_test["zipcode"] = X_test["zipcode"].astype(str)

    return X_train, y_train, X_test, y_test


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    # Explicitly treat zipcode as categorical
    categorical_cols = [c for c in ["zipcode"] if c in X.columns]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    model = LinearRegression()

    return Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"r2": r2, "mae": mae, "rmse": rmse}


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1) Load data
    df = load_clean_data(CLEAN_CSV_PATH)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2) Time-based split
    X_train, y_train, X_test, y_test = time_based_split(df)
    print(f"✅ Time-based split: {X_train.shape[0]} train, {X_test.shape[0]} test")

    # 3) Build pipeline
    pipe = build_pipeline(X_train)

    # 4) Fit (measure training time)
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    t1 = time.perf_counter()
    train_seconds = t1 - t0
    print(f"✅ Linear Regression model trained in {train_seconds:.3f} seconds")

    # 5) Predict & evaluate
    y_pred = pipe.predict(X_test)
    metrics = evaluate(y_test, y_pred)

    print("\n📊 Model Evaluation Metrics")
    print(f"   R²   : {metrics['r2']:.4f}")
    print(f"   MAE  : ${metrics['mae']:,.2f}")
    print(f"   RMSE : ${metrics['rmse']:,.2f}")

    # 6) Save model and metrics
    joblib.dump(pipe, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(
            {**metrics, "train_time_seconds": round(train_seconds, 3)},
            f,
            indent=2
        )

    print(f"\n💾 Model saved to:   {MODEL_PATH}")
    print(f"💾 Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()