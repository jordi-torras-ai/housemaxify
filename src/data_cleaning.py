import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# ===================== CONFIG =====================
CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "kc_house_data.csv"
CLEAN_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "kc_house_data_clean.csv"
TARGET = "price"
# ==================================================


def ensure_csv(csv_path: Path) -> Path:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Run download_data.py first.")

    if zipfile.is_zipfile(csv_path):
        with zipfile.ZipFile(csv_path, "r") as archive:
            members = [name for name in archive.namelist() if name.endswith(".csv")]
            if not members:
                raise ValueError(f"No CSV files found inside archive {csv_path}")
            extracted_name = Path(members[0]).name
            archive.extract(members[0], csv_path.parent)
        csv_path.unlink()
        return csv_path.with_name(extracted_name)

    return csv_path


def load_data():
    actual_csv = ensure_csv(CSV_PATH)
    df = pd.read_csv(actual_csv)
    print("✅ Loaded dataset:", df.shape)
    return df


def engineer_time_features(df):
    # Parse date
    df["sale_date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%S", errors="coerce")

    # Extract meaningful features
    df["sale_year"] = df["sale_date"].dt.year
    df["sale_month"] = df["sale_date"].dt.month
    df["sale_dow"] = df["sale_date"].dt.dayofweek  # day of week

    # House age feature
    df["house_age_at_sale"] = df["sale_year"] - df["yr_built"]

    # Renovation flag + time since renovation
    df["was_renovated"] = (df["yr_renovated"] > 0).astype(int)
    df["years_since_renov"] = np.where(df["yr_renovated"] > 0,
                                       df["sale_year"] - df["yr_renovated"], 0)
    return df


def clean_data(df):
    # Remove leakage columns
    df = df.drop(columns=["id", "date"])

    # Optional: drop rows without a sale_date
    df = df.dropna(subset=["sale_date"])

    return df


def save_clean_data(df):
    df.to_csv(CLEAN_CSV_PATH, index=False)
    print(f"✅ Clean dataset saved to: {CLEAN_CSV_PATH}")


def main():
    df = load_data()
    df = engineer_time_features(df)
    df = clean_data(df)

    print("\n✅ Final clean dataset shape:", df.shape)
    print("✅ Columns now are:", list(df.columns), "\n")

    save_clean_data(df)


if __name__ == "__main__":
    main()
