import pandas as pd
from pathlib import Path
import zipfile

# ========= CONFIG =========
CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "kc_house_data.csv"
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
    csv_path = CSV_PATH
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Run download_data.py first.")

    if zipfile.is_zipfile(csv_path):
        # Handle case where Kaggle delivered a zip archive named like the CSV.
        with zipfile.ZipFile(csv_path, "r") as archive:
            members = [name for name in archive.namelist() if name.endswith(".csv")]
            if not members:
                raise ValueError(f"No CSV files found inside archive {csv_path}")
            extracted_name = Path(members[0]).name
            archive.extract(members[0], csv_path.parent)
        csv_path.unlink()
        csv_path = csv_path.with_name(extracted_name)

    df = pd.read_csv(csv_path)
    print(df.head())
    validate_for_regression(df)
