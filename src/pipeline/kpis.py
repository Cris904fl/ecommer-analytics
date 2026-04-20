"""
Revenue & Sales KPI Calculator
================================
Computes business KPIs from clean order data:
  - Total & net revenue
  - Monthly/quarterly growth (MoM, QoQ)
  - AOV (Average Order Value)
  - Conversion rate by channel
  - Top categories & regions
  - Repeat purchase rate
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CLEAN_PATH = Path("data/processed/orders_clean.csv")


def load_clean(path: Path = CLEAN_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["order_date"])
    return df[df["is_revenue_positive"]].copy()


# ─────────────────────────────────────────────
# Revenue Summary
# ─────────────────────────────────────────────
def revenue_summary(df: pd.DataFrame) -> dict:
    return {
        "total_orders": int(len(df)),
        "total_gross_revenue": round(df["gross_revenue"].sum(), 2),
        "total_net_revenue": round(df["net_revenue"].sum(), 2),
        "total_discount_given": round(df["discount_amount"].sum(), 2),
        "avg_order_value": round(df["net_revenue"].mean(), 2),
        "avg_discount_pct": round(df["discount_pct"].mean(), 2),
        "unique_customers": int(df["customer_id"].nunique()),
        "unique_products": int(df["product_id"].nunique()),
    }


# ─────────────────────────────────────────────
# Monthly Revenue + MoM Growth
# ─────────────────────────────────────────────
def monthly_revenue(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.groupby(["year", "month"])
        .agg(
            orders=("order_id", "count"),
            gross_revenue=("gross_revenue", "sum"),
            net_revenue=("net_revenue", "sum"),
            avg_order_value=("net_revenue", "mean"),
        )
        .reset_index()
    )
    monthly["period"] = pd.to_datetime(
        monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
    )
    monthly = monthly.sort_values("period")
    monthly["mom_growth_pct"] = monthly["net_revenue"].pct_change() * 100
    return monthly.round(2)


# ─────────────────────────────────────────────
# Quarterly Revenue + QoQ Growth
# ─────────────────────────────────────────────
def quarterly_revenue(df: pd.DataFrame) -> pd.DataFrame:
    quarterly = (
        df.groupby(["year", "quarter"])
        .agg(
            orders=("order_id", "count"),
            net_revenue=("net_revenue", "sum"),
            avg_order_value=("net_revenue", "mean"),
        )
        .reset_index()
    )
    quarterly = quarterly.sort_values(["year", "quarter"])
    quarterly["qoq_growth_pct"] = quarterly["net_revenue"].pct_change() * 100
    quarterly["label"] = (
        "Q" + quarterly["quarter"].astype(str) + " " + quarterly["year"].astype(str)
    )
    return quarterly.round(2)


# ─────────────────────────────────────────────
# Revenue by Category
# ─────────────────────────────────────────────
def revenue_by_category(df: pd.DataFrame) -> pd.DataFrame:
    cat = (
        df.groupby("category")
        .agg(
            orders=("order_id", "count"),
            net_revenue=("net_revenue", "sum"),
            avg_order_value=("net_revenue", "mean"),
            avg_discount=("discount_pct", "mean"),
        )
        .reset_index()
        .sort_values("net_revenue", ascending=False)
    )
    cat["revenue_share_pct"] = (cat["net_revenue"] / cat["net_revenue"].sum() * 100).round(2)
    return cat.round(2)


# ─────────────────────────────────────────────
# Revenue by Channel
# ─────────────────────────────────────────────
def revenue_by_channel(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("channel")
        .agg(
            orders=("order_id", "count"),
            net_revenue=("net_revenue", "sum"),
            avg_order_value=("net_revenue", "mean"),
            new_customer_rate=("is_new_customer", "mean"),
        )
        .reset_index()
        .sort_values("net_revenue", ascending=False)
        .round(2)
    )


# ─────────────────────────────────────────────
# Revenue by Region
# ─────────────────────────────────────────────
def revenue_by_region(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("region")
        .agg(
            orders=("order_id", "count"),
            net_revenue=("net_revenue", "sum"),
            unique_customers=("customer_id", "nunique"),
            avg_order_value=("net_revenue", "mean"),
        )
        .reset_index()
        .sort_values("net_revenue", ascending=False)
        .round(2)
    )


# ─────────────────────────────────────────────
# Repeat Purchase Rate
# ─────────────────────────────────────────────
def repeat_purchase_rate(df: pd.DataFrame) -> dict:
    orders_per_customer = df.groupby("customer_id")["order_id"].count()
    repeat_customers = (orders_per_customer > 1).sum()
    total_customers = len(orders_per_customer)
    return {
        "total_customers": int(total_customers),
        "repeat_customers": int(repeat_customers),
        "repeat_purchase_rate_pct": round(repeat_customers / total_customers * 100, 2),
        "avg_orders_per_customer": round(orders_per_customer.mean(), 2),
    }


# ─────────────────────────────────────────────
# Run all KPIs
# ─────────────────────────────────────────────
def compute_all_kpis(path: Path = CLEAN_PATH) -> dict:
    df = load_clean(path)
    logger.info(f"Computing KPIs on {len(df):,} completed orders")

    kpis = {
        "summary": revenue_summary(df),
        "monthly": monthly_revenue(df).to_dict(orient="records"),
        "quarterly": quarterly_revenue(df).to_dict(orient="records"),
        "by_category": revenue_by_category(df).to_dict(orient="records"),
        "by_channel": revenue_by_channel(df).to_dict(orient="records"),
        "by_region": revenue_by_region(df).to_dict(orient="records"),
        "repeat_purchase": repeat_purchase_rate(df),
    }
    return kpis


if __name__ == "__main__":
    import json

    kpis = compute_all_kpis()
    print(json.dumps(kpis["summary"], indent=2))
    print("\nTop 3 categories by revenue:")
    for row in kpis["by_category"][:3]:
        print(f"  {row['category']}: ${row['net_revenue']:,.0f}  ({row['revenue_share_pct']}%)")
