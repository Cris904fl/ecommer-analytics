"""
E-Commerce Data Cleaning & Transformation Pipeline
===================================================
Handles: deduplication, null imputation, type casting,
derived metrics (revenue, discount_amount, margin proxy).
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

RAW_PATH = Path("data/raw/orders_raw.csv")
PROCESSED_PATH = Path("data/processed/orders_clean.csv")


# ─────────────────────────────────────────────
# 1. Ingestion
# ─────────────────────────────────────────────
def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    logger.info(f"Loading raw data from {path}")
    df = pd.read_csv(path, parse_dates=["order_date"])
    logger.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
# 2. Deduplication
# ─────────────────────────────────────────────
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["order_id"], keep="first")
    removed = before - len(df)
    logger.info(f"Duplicates removed: {removed} ({removed/before:.1%} of total)")
    return df


# ─────────────────────────────────────────────
# 3. Null Imputation
# ─────────────────────────────────────────────
def impute_nulls(df: pd.DataFrame) -> pd.DataFrame:
    # discount_pct: fill with median per category
    df["discount_pct"] = df.groupby("category")["discount_pct"].transform(
        lambda x: x.fillna(x.median())
    )
    # region: fill with mode overall
    mode_region = df["region"].mode()[0]
    df["region"] = df["region"].fillna(mode_region)
    logger.info(f"Null imputation complete. Remaining nulls: {df.isnull().sum().sum()}")
    return df


# ─────────────────────────────────────────────
# 4. Type Casting & Standardisation
# ─────────────────────────────────────────────
def standardise_types(df: pd.DataFrame) -> pd.DataFrame:
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["is_new_customer"] = df["is_new_customer"].astype(bool)
    df["quantity"] = df["quantity"].astype(int)
    df["unit_price"] = df["unit_price"].astype(float)
    df["discount_pct"] = df["discount_pct"].astype(float)
    df["category"] = df["category"].str.strip().str.title()
    df["region"] = df["region"].str.strip().str.upper()
    df["channel"] = df["channel"].str.strip().str.lower()
    df["status"] = df["status"].str.strip().str.lower()
    return df


# ─────────────────────────────────────────────
# 5. Derived Metrics
# ─────────────────────────────────────────────
def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df["gross_revenue"] = df["quantity"] * df["unit_price"]
    df["discount_amount"] = df["gross_revenue"] * df["discount_pct"] / 100
    df["net_revenue"] = df["gross_revenue"] - df["discount_amount"]

    # Time dimensions
    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.month
    df["quarter"] = df["order_date"].dt.quarter
    df["week"] = df["order_date"].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df["order_date"].dt.day_name()

    # Flag: revenue-positive (exclude returns and cancellations)
    df["is_revenue_positive"] = df["status"] == "completed"

    logger.info("Derived metrics added: gross_revenue, net_revenue, discount_amount, time dims")
    return df


# ─────────────────────────────────────────────
# 6. Validation assertions
# ─────────────────────────────────────────────
def validate(df: pd.DataFrame) -> None:
    assert df["order_id"].is_unique, "order_id must be unique after dedup"
    assert df["unit_price"].gt(0).all(), "unit_price must be > 0"
    assert df["quantity"].gt(0).all(), "quantity must be > 0"
    assert df["discount_pct"].between(0, 100).all(), "discount_pct out of range [0, 100]"
    assert df.isnull().sum().sum() == 0, "Unexpected nulls after imputation"
    logger.info("All validation assertions passed ✓")


# ─────────────────────────────────────────────
# 7. Save
# ─────────────────────────────────────────────
def save_clean(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Clean data saved → {path}  ({len(df):,} rows)")


# ─────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────
def run_cleaning_pipeline(
    raw_path: Path = RAW_PATH,
    output_path: Path = PROCESSED_PATH,
) -> pd.DataFrame:
    logger.info("=" * 55)
    logger.info("Starting E-Commerce Data Cleaning Pipeline")
    logger.info("=" * 55)

    df = load_raw(raw_path)
    df = remove_duplicates(df)
    df = impute_nulls(df)
    df = standardise_types(df)
    df = add_derived_metrics(df)
    validate(df)
    save_clean(df, output_path)

    logger.info(f"Pipeline complete. Final dataset: {len(df):,} rows")
    return df


if __name__ == "__main__":
    run_cleaning_pipeline()
