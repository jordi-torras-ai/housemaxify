import pandas as pd
import numpy as np

# ===================== CONFIG =====================
CSV_PATH = "../data/kc_house_data.csv"
CLEAN_CSV_PATH = "../data/kc_house_data_clean.csv"
TARGET = "price"
# ==================================================


def load_data():
    df = pd.read_csv(CSV_PATH)
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