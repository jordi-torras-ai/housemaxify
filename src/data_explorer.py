import pandas as pd

# Load the dataset
file_path = "../data/kc_house_data.csv"
df = pd.read_csv(file_path)

# Show a few rows
print(df.head())

import pandas as pd
import numpy as np

# ========= CONFIG =========
CSV_PATH = "../data/kc_house_data.csv"   # <-- Updated path here
TARGET = "price"
ID_LIKE = {"id"}
DATE_LIKE = {"date"}
NONNEGATIVE_COLS = {"bedrooms", "bathrooms", "sqft_living", "sqft_lot",
                    "floors", "sqft_above", "sqft_basement", "yr_built",
                    "yr_renovated", "sqft_living15", "sqft_lot15"}
RANGE_CHECKS = {
    "bedrooms": (0, 15),
    "bathrooms": (0, 10),
    "sqft_living": (150, 20000),
    "sqft_lot": (300, 2000000),
    "floors": (0, 4),
    "yr_built": (1800, 2025),
    "yr_renovated": (0, 2025),
    "lat": (45.0, 49.0),
    "long": (-124.0, -120.0)
}
Z_THRESHOLD = 5.0
# ==========================

def validate_for_regression(df: pd.DataFrame) -> None:
    problems = []
    warnings = []

    # 1) Target check
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: '{TARGET}'")

    if df[TARGET].isna().any():
        problems.append(f"Target '{TARGET}' contains {df[TARGET].isna().sum()} missing values.")

    if (df[TARGET] <= 0).any():
        warnings.append(f"Target '{TARGET}' has non-positive values: {(df[TARGET] <= 0).sum()} rows.")

    # 2) Leakage columns
    leakage_cols = list((ID_LIKE | DATE_LIKE) & set(df.columns))
    if leakage_cols:
        warnings.append(f"Possible leakage columns (drop before training): {leakage_cols}")

    # 3) Missing values
    na_counts = df.isna().sum()
    if na_counts.any():
        problems.append(f"Missing values detected:\n{na_counts[na_counts > 0]}")

    # 4) Duplicates
    if df.duplicated().sum() > 0:
        warnings.append(f"{df.duplicated().sum()} duplicate rows found.")

    # 5) Validation summary
    print("\n============== DATA VALIDATION REPORT ==============")
    if problems:
        print("\n❌ Problems found:")
        for p in problems:
            print(" -", p)
    else:
        print("\n✅ No critical problems found.")

    if warnings:
        print("\n⚠️ Warnings:")
        for w in warnings:
            print(" -", w)
    print("====================================================\n")

    if problems:
        print("⚠️ Fix the problems before training a model.")
    else:
        print("✅ Dataset is OK to continue preprocessing for regression.\n")

# ==== CALL THE FUNCTION HERE ====
if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    validate_for_regression(df)