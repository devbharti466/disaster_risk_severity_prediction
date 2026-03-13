"""
Data Preprocessing Module for Disaster Risk Severity Prediction.

Handles data loading, cleaning, missing value imputation, and feature engineering.
"""

import os
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

RAW_DATA_PATH = os.path.join(
    "datasets",
    "improved_ordinal_regression_dataset_with_location.xlsx - Sheet1.csv",
)
PROCESSED_DATA_PATH = os.path.join("data", "processed", "cleaned_data.csv")

# India bounding box (approximate)
INDIA_LAT_MIN, INDIA_LAT_MAX = 6.0, 37.0
INDIA_LON_MIN, INDIA_LON_MAX = 68.0, 98.0

# Season mapping
SEASON_MAP = {
    1: "Winter", 2: "Winter",
    3: "Summer", 4: "Summer", 5: "Summer",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Winter", 11: "Winter", 12: "Winter",
}


# ──────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────
def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV dataset."""
    df = pd.read_csv(path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ──────────────────────────────────────────────
# Cleaning
# ──────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset:
    1. Remove columns with extreme missingness (>40 %).
    2. Remove duplicate disaster records.
    3. Remove impossible values (negative deaths, etc.).
    4. Validate lat/lon ranges for India.
    5. Normalize location names.
    """
    df = df.copy()

    # 1. Drop columns with >40 % missing
    threshold = 0.40 * len(df)
    cols_before = set(df.columns)
    df = df.dropna(axis=1, thresh=len(df) - int(threshold))
    dropped = cols_before - set(df.columns)
    if dropped:
        print(f"Dropped columns with >40% missing: {dropped}")

    # 2. Remove duplicates
    n_before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {n_before - len(df)} duplicate rows")

    # 3. Remove impossible values
    impact_cols = [c for c in ["Total Deaths", "No. Injured", "No. Affected",
                               "No. Homeless", "Total Affected"] if c in df.columns]
    for col in impact_cols:
        neg_mask = df[col] < 0
        if neg_mask.any():
            print(f"Removing {neg_mask.sum()} rows with negative {col}")
            df = df[~neg_mask]

    # 4. Validate lat/lon for India
    if "Latitude" in df.columns and "Longitude" in df.columns:
        valid = (
            (df["Latitude"].between(INDIA_LAT_MIN, INDIA_LAT_MAX))
            & (df["Longitude"].between(INDIA_LON_MIN, INDIA_LON_MAX))
        )
        n_invalid = (~valid).sum()
        if n_invalid:
            print(f"Removing {n_invalid} rows with coordinates outside India")
            df = df[valid]

    # 5. Normalize location names
    if "Location" in df.columns:
        df["Location"] = df["Location"].str.strip().str.title()

    df = df.reset_index(drop=True)
    print(f"After cleaning: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ──────────────────────────────────────────────
# Imputation
# ──────────────────────────────────────────────
def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values – median/KNN for numeric, mode for categorical."""
    df = df.copy()

    # Numeric columns – median imputation (fast, robust)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  Imputed {col} with median = {median_val:.2f}")

    # Categorical columns – mode imputation
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"  Imputed {col} with mode = {mode_val}")

    return df


# ──────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create India-specific features:
    - Season (Monsoon / Winter / Summer)
    - Duration (End Date − Start Date in days)
    - Year_Trend (Start_Year − 1900)
    - Log_Total_Effect (log(1 + Deaths + Total_Affected))
    """
    df = df.copy()

    # Season
    if "Start Month" in df.columns:
        df["Season"] = df["Start Month"].map(SEASON_MAP)
        print("Created Season feature")

    # Duration
    date_cols = {"Start Year", "Start Month", "Start Day",
                 "End Year", "End Month", "End Day"}
    if date_cols.issubset(df.columns):
        start = pd.to_datetime(
            df[["Start Year", "Start Month", "Start Day"]].rename(
                columns={"Start Year": "year", "Start Month": "month", "Start Day": "day"}
            ),
            errors="coerce",
        )
        end = pd.to_datetime(
            df[["End Year", "End Month", "End Day"]].rename(
                columns={"End Year": "year", "End Month": "month", "End Day": "day"}
            ),
            errors="coerce",
        )
        df["Duration"] = (end - start).dt.days.clip(lower=0)
        df["Duration"] = df["Duration"].fillna(0).astype(int)
        print("Created Duration feature")

    # Year trend
    if "Start Year" in df.columns:
        df["Year_Trend"] = df["Start Year"] - 1900
        print("Created Year_Trend feature")

    # Log Total Effect
    if "Total Deaths" in df.columns and "Total Affected" in df.columns:
        df["Total_Effect"] = df["Total Deaths"] + df["Total Affected"]
        df["Log_Total_Effect"] = np.log1p(df["Total_Effect"])
        print("Created Log_Total_Effect feature")

    return df


# ──────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────
def preprocess(raw_path: str = RAW_DATA_PATH,
               save_path: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Run the full preprocessing pipeline and save the result."""
    print("=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)

    df = load_raw_data(raw_path)
    df = clean_data(df)
    df = impute_missing(df)
    df = engineer_features(df)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"\nSaved processed data to {save_path}")
    return df


if __name__ == "__main__":
    preprocess()
