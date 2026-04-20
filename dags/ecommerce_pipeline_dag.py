"""
Apache Airflow DAG: E-Commerce Analytics Pipeline
===================================================
Schedule: Daily at 06:00 UTC

Tasks:
  1. validate_raw_data       → check file exists and schema is valid
  2. run_cleaning_pipeline   → deduplicate, impute, derive metrics
  3. train_anomaly_detector  → fit IsolationForest
  4. train_revenue_forecaster→ fit Linear Regression forecaster
  5. generate_kpi_report     → compute KPIs and write HTML report
  6. notify_success          → log completion summary
"""

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import logging

logger = logging.getLogger(__name__)

# ─── Default Args ────────────────────────────
default_args = {
    "owner": "data-team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "start_date": days_ago(1),
}

# ─── DAG Definition ──────────────────────────
dag = DAG(
    dag_id="ecommerce_analytics_pipeline",
    description="Daily ETL + ML pipeline for e-commerce revenue analytics",
    default_args=default_args,
    schedule_interval="0 6 * * *",   # 06:00 UTC every day
    catchup=False,
    max_active_runs=1,
    tags=["ecommerce", "etl", "ml", "revenue"],
)


# ─── Task Functions ───────────────────────────

def _validate_raw_data(**context):
    """Validates that raw data file exists and has the expected schema."""
    import pandas as pd

    RAW_PATH = Path("data/raw/orders_raw.csv")

    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw data not found: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH, nrows=5)

    REQUIRED_COLS = {
        "order_id", "customer_id", "product_id", "category",
        "channel", "quantity", "unit_price", "discount_pct",
        "status", "order_date", "region", "customer_age", "is_new_customer"
    }
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in raw data: {missing}")

    full_df = pd.read_csv(RAW_PATH)
    logger.info(f"Raw data validated: {len(full_df):,} rows, {full_df.shape[1]} columns")
    context["ti"].xcom_push(key="raw_row_count", value=len(full_df))


def _run_cleaning_pipeline(**context):
    """Runs the data cleaning and transformation pipeline."""
    from src.pipeline.cleaner import run_cleaning_pipeline

    df = run_cleaning_pipeline()
    context["ti"].xcom_push(key="clean_row_count", value=len(df))
    logger.info(f"Cleaning complete. Clean rows: {len(df):,}")


def _train_anomaly_detector(**context):
    """Trains IsolationForest anomaly detector on clean data."""
    import pandas as pd
    from src.ml.models import AnomalyDetector

    df = pd.read_csv("data/processed/orders_clean.csv")
    detector = AnomalyDetector()
    detector.fit(df)
    df_flagged = detector.predict(df)
    detector.save()

    anomaly_count = int(df_flagged["is_anomaly"].sum())
    context["ti"].xcom_push(key="anomaly_count", value=anomaly_count)
    logger.info(f"Anomalies flagged: {anomaly_count}")


def _train_revenue_forecaster(**context):
    """Trains the weekly revenue forecaster."""
    import pandas as pd
    from src.ml.models import RevenueForecaster

    df = pd.read_csv("data/processed/orders_clean.csv", parse_dates=["order_date"])
    forecaster = RevenueForecaster()
    forecaster.fit(df)
    forecaster.save()

    forecast = forecaster.forecast(n_weeks=4)
    next_week_rev = forecast["forecasted_net_revenue"].iloc[0]
    context["ti"].xcom_push(key="next_week_forecast", value=round(float(next_week_rev), 2))
    logger.info(f"Next week forecast: ${next_week_rev:,.2f}")


def _generate_kpi_report(**context):
    """Computes all KPIs and writes an HTML summary report."""
    import json
    from src.pipeline.kpis import compute_all_kpis

    kpis = compute_all_kpis()
    summary = kpis["summary"]

    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)

    run_date = datetime.utcnow().strftime("%Y-%m-%d")
    report_path = report_dir / f"kpi_report_{run_date}.json"

    with open(report_path, "w") as f:
        json.dump(kpis, f, indent=2, default=str)

    logger.info(f"KPI report saved → {report_path}")
    logger.info(f"Summary: {json.dumps(summary, indent=2)}")

    context["ti"].xcom_push(key="total_net_revenue", value=summary["total_net_revenue"])
    context["ti"].xcom_push(key="total_orders", value=summary["total_orders"])


def _notify_success(**context):
    """Logs pipeline completion with key metrics from XCom."""
    ti = context["ti"]
    raw_rows = ti.xcom_pull(task_ids="validate_raw_data", key="raw_row_count")
    clean_rows = ti.xcom_pull(task_ids="run_cleaning_pipeline", key="clean_row_count")
    anomalies = ti.xcom_pull(task_ids="train_anomaly_detector", key="anomaly_count")
    next_week = ti.xcom_pull(task_ids="train_revenue_forecaster", key="next_week_forecast")
    net_revenue = ti.xcom_pull(task_ids="generate_kpi_report", key="total_net_revenue")
    orders = ti.xcom_pull(task_ids="generate_kpi_report", key="total_orders")

    logger.info("=" * 60)
    logger.info("✅  E-Commerce Pipeline — Run Complete")
    logger.info(f"   Raw rows ingested    : {raw_rows:,}")
    logger.info(f"   Clean rows produced  : {clean_rows:,}")
    logger.info(f"   Anomalies flagged    : {anomalies}")
    logger.info(f"   Next week forecast   : ${next_week:,.2f}")
    logger.info(f"   Total net revenue    : ${net_revenue:,.2f}")
    logger.info(f"   Total orders (YTD)   : {orders:,}")
    logger.info("=" * 60)


# ─── Task Definitions ────────────────────────

t1 = PythonOperator(
    task_id="validate_raw_data",
    python_callable=_validate_raw_data,
    dag=dag,
)

t2 = PythonOperator(
    task_id="run_cleaning_pipeline",
    python_callable=_run_cleaning_pipeline,
    dag=dag,
)

t3 = PythonOperator(
    task_id="train_anomaly_detector",
    python_callable=_train_anomaly_detector,
    dag=dag,
)

t4 = PythonOperator(
    task_id="train_revenue_forecaster",
    python_callable=_train_revenue_forecaster,
    dag=dag,
)

t5 = PythonOperator(
    task_id="generate_kpi_report",
    python_callable=_generate_kpi_report,
    dag=dag,
)

t6 = PythonOperator(
    task_id="notify_success",
    python_callable=_notify_success,
    dag=dag,
)

# ─── Task Dependencies ───────────────────────
#
#  t1 → t2 → t3 ─┐
#              t4 ─┤→ t5 → t6
#
t1 >> t2 >> [t3, t4] >> t5 >> t6
