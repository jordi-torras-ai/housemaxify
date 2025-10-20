import os
import json
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor

# ===================== CONFIG =====================
CLEAN_CSV_PATH = "../data/kc_house_data_clean.csv"
MODELS_DIR = "../models"
MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_kc_house.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "xgboost_kc_house_metrics.json")

TARGET = "price"
TIME_SPLIT = "2015-01-01"
RANDOM_STATE = 42
# ==================================================


def load_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "sale_date" not in df.columns:
        raise ValueError("Expected 'sale_date' in clean dataset.")
    df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")

    # Ensure 'zipcode' is string/categorical to avoid dtype issues in OHE
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


def build_pipeline(X_train: pd.DataFrame) -> Pipeline:
    categorical_cols = [c for c in ["zipcode"] if c in X_train.columns]
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    xgb = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist"
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", xgb),
    ])
    return pipe


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
    pipeline = build_pipeline(X_train)

    # 4) Train with timing
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    t1 = time.perf_counter()
    train_seconds = t1 - t0
    print(f"✅ XGBoost model trained in {train_seconds:.3f} seconds")

    # 5) Predict & evaluate
    y_pred = pipeline.predict(X_test)
    metrics = evaluate(y_test, y_pred)

    print("\n📊 XGBoost Model Evaluation")
    print(f"   R²   : {metrics['r2']:.4f}")
    print(f"   MAE  : ${metrics['mae']:,.2f}")
    print(f"   RMSE : ${metrics['rmse']:,.2f}")

    # 6) Save model & metrics (+ training time)
    joblib.dump(pipeline, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump({**metrics, "train_time_seconds": round(train_seconds, 3)}, f, indent=2)

    print(f"\n💾 Model saved to:   {MODEL_PATH}")
    print(f"💾 Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()