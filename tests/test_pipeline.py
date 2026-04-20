"""
Unit Tests — E-Commerce Analytics Pipeline
==========================================
Tests covering: cleaning, KPI computation, ML models, API endpoints.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.pipeline.cleaner import (
    remove_duplicates,
    impute_nulls,
    standardise_types,
    add_derived_metrics,
    validate,
)
from src.pipeline.kpis import (
    revenue_summary,
    monthly_revenue,
    revenue_by_category,
    repeat_purchase_rate,
)
from src.ml.models import AnomalyDetector, RevenueForecaster


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────
@pytest.fixture
def raw_df():
    """Minimal raw DataFrame simulating ingested orders."""
    np.random.seed(0)
    n = 200
    from datetime import datetime, timedelta
    import random

    dates = [datetime(2023, 1, 1) + timedelta(days=i * 2) for i in range(n)]
    df = pd.DataFrame({
        "order_id": [f"ORD-{str(i).zfill(5)}" for i in range(n)],
        "customer_id": [f"CUST-{str(random.randint(1, 50)).zfill(4)}" for _ in range(n)],
        "product_id": [f"PROD-{str(random.randint(1, 30)).zfill(4)}" for _ in range(n)],
        "category": np.random.choice(["electronica", "ropa", "hogar"], n),
        "channel": np.random.choice(["web", "mobile"], n),
        "quantity": np.random.randint(1, 5, n),
        "unit_price": np.random.uniform(100, 2000, n).round(2),
        "discount_pct": np.random.choice([0.0, 10.0, 20.0, None], n, p=[0.5, 0.2, 0.2, 0.1]),
        "status": np.random.choice(["completed", "returned", "cancelled"], n, p=[0.7, 0.15, 0.15]),
        "order_date": [d.strftime("%Y-%m-%d") for d in dates],
        "region": np.random.choice(["CDMX", "GDL", None], n, p=[0.45, 0.45, 0.10]),
        "customer_age": np.random.randint(18, 60, n),
        "is_new_customer": np.random.choice([True, False], n),
    })
    # Add 10 exact duplicates
    dupes = df.sample(10)
    return pd.concat([df, dupes], ignore_index=True)


@pytest.fixture
def clean_df(raw_df):
    """Fully cleaned DataFrame."""
    df = remove_duplicates(raw_df)
    df = impute_nulls(df)
    df = standardise_types(df)
    df = add_derived_metrics(df)
    return df


# ─────────────────────────────────────────────
# Cleaning Tests
# ─────────────────────────────────────────────
class TestCleaning:

    def test_remove_duplicates_reduces_rows(self, raw_df):
        before = len(raw_df)
        df = remove_duplicates(raw_df)
        assert len(df) < before, "Deduplication should reduce row count"
        assert df["order_id"].is_unique, "order_id must be unique after deduplication"

    def test_impute_nulls_removes_all_nulls(self, raw_df):
        df = remove_duplicates(raw_df)
        df = impute_nulls(df)
        assert df["discount_pct"].isnull().sum() == 0
        assert df["region"].isnull().sum() == 0

    def test_standardise_types(self, raw_df):
        df = remove_duplicates(raw_df)
        df = impute_nulls(df)
        df = standardise_types(df)
        assert pd.api.types.is_datetime64_any_dtype(df["order_date"])
        assert df["category"].str.istitle().all()
        assert df["channel"].str.islower().all()

    def test_derived_metrics_non_negative(self, clean_df):
        assert (clean_df["gross_revenue"] >= 0).all()
        assert (clean_df["net_revenue"] >= 0).all()
        assert (clean_df["discount_amount"] >= 0).all()

    def test_derived_metrics_math(self, clean_df):
        expected = clean_df["quantity"] * clean_df["unit_price"]
        pd.testing.assert_series_equal(
            clean_df["gross_revenue"].round(2),
            expected.round(2),
            check_names=False,
        )

    def test_validate_passes_on_clean_data(self, clean_df):
        """validate() should not raise on clean data."""
        validate(clean_df)  # Should not throw


# ─────────────────────────────────────────────
# KPI Tests
# ─────────────────────────────────────────────
class TestKPIs:

    def test_revenue_summary_keys(self, clean_df):
        completed = clean_df[clean_df["status"] == "completed"]
        summary = revenue_summary(completed)
        assert "total_net_revenue" in summary
        assert "avg_order_value" in summary
        assert "unique_customers" in summary

    def test_revenue_summary_values_positive(self, clean_df):
        completed = clean_df[clean_df["status"] == "completed"]
        summary = revenue_summary(completed)
        assert summary["total_net_revenue"] > 0
        assert summary["avg_order_value"] > 0

    def test_monthly_revenue_has_mom_growth(self, clean_df):
        df = monthly_revenue(clean_df)
        assert "mom_growth_pct" in df.columns
        assert len(df) > 0

    def test_revenue_by_category_sorted(self, clean_df):
        completed = clean_df[clean_df["status"] == "completed"]
        cat_df = revenue_by_category(completed)
        revenues = cat_df["net_revenue"].tolist()
        assert revenues == sorted(revenues, reverse=True), "Categories should be sorted by revenue desc"

    def test_repeat_purchase_rate_bounds(self, clean_df):
        completed = clean_df[clean_df["status"] == "completed"]
        rpr = repeat_purchase_rate(completed)
        assert 0 <= rpr["repeat_purchase_rate_pct"] <= 100


# ─────────────────────────────────────────────
# ML Model Tests
# ─────────────────────────────────────────────
class TestAnomalyDetector:

    def test_fit_predict_returns_flag_column(self, clean_df):
        detector = AnomalyDetector()
        detector.fit(clean_df)
        result = detector.predict(clean_df)
        assert "is_anomaly" in result.columns
        assert "anomaly_score" in result.columns

    def test_anomaly_rate_within_expected_range(self, clean_df):
        detector = AnomalyDetector()
        detector.fit(clean_df)
        result = detector.predict(clean_df)
        rate = result["is_anomaly"].mean()
        # Should be roughly near contamination param (0.03), allow wide range in test
        assert 0 < rate < 0.20, f"Unexpected anomaly rate: {rate:.2%}"

    def test_predict_requires_fit(self, clean_df):
        detector = AnomalyDetector()
        with pytest.raises(AssertionError):
            detector.predict(clean_df)


class TestRevenueForecaster:

    def test_forecast_returns_correct_n_weeks(self, clean_df):
        forecaster = RevenueForecaster()
        forecaster.fit(clean_df)
        forecast = forecaster.forecast(n_weeks=8)
        assert len(forecast) == 8

    def test_forecast_non_negative(self, clean_df):
        forecaster = RevenueForecaster()
        forecaster.fit(clean_df)
        forecast = forecaster.forecast(n_weeks=4)
        assert (forecast["forecasted_net_revenue"] >= 0).all()

    def test_metrics_available_after_fit(self, clean_df):
        forecaster = RevenueForecaster()
        forecaster.fit(clean_df)
        assert "mae" in forecaster._metrics
        assert "mape" in forecaster._metrics
        assert forecaster._metrics["mae"] >= 0
