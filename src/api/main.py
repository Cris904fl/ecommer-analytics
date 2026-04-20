"""
E-Commerce Analytics REST API
==============================
FastAPI application exposing KPIs, forecasts, and anomaly reports.

Endpoints:
  GET /                    → health check
  GET /kpis/summary        → revenue summary
  GET /kpis/monthly        → monthly revenue trend
  GET /kpis/quarterly      → quarterly revenue + QoQ growth
  GET /kpis/by-category    → breakdown by product category
  GET /kpis/by-channel     → breakdown by sales channel
  GET /kpis/by-region      → breakdown by region
  GET /ml/forecast         → 12-week revenue forecast
  GET /ml/anomalies        → anomalous orders report
  POST /pipeline/run       → trigger full pipeline (admin)
"""

import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parents[2]))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import logging

from src.pipeline.kpis import compute_all_kpis, load_clean, CLEAN_PATH
from src.ml.models import AnomalyDetector, RevenueForecaster, MODEL_DIR

logger = logging.getLogger(__name__)

app = FastAPI(
    title="E-Commerce Analytics API",
    description="Revenue KPIs, ML Forecasting & Anomaly Detection for e-commerce data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Startup: load models into memory
# ─────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("Loading ML models...")
    try:
        app.state.detector = AnomalyDetector.load(MODEL_DIR / "anomaly_detector.pkl")
        app.state.forecaster = RevenueForecaster.load(MODEL_DIR / "revenue_forecaster.pkl")
        logger.info("Models loaded successfully ✓")
    except FileNotFoundError:
        logger.warning("Models not found. Run `python -m src.ml.models` to train them first.")
        app.state.detector = None
        app.state.forecaster = None


# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "service": "E-Commerce Analytics API",
        "version": "1.0.0",
        "models_loaded": app.state.detector is not None,
    }


# ─────────────────────────────────────────────
# KPI Endpoints
# ─────────────────────────────────────────────
@app.get("/kpis/summary", tags=["KPIs"])
def kpi_summary():
    """Overall revenue summary: total orders, GMV, AOV, unique customers."""
    try:
        kpis = compute_all_kpis(CLEAN_PATH)
        return kpis["summary"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/kpis/monthly", tags=["KPIs"])
def kpi_monthly():
    """Monthly revenue with Month-over-Month growth %."""
    kpis = compute_all_kpis(CLEAN_PATH)
    return kpis["monthly"]


@app.get("/kpis/quarterly", tags=["KPIs"])
def kpi_quarterly():
    """Quarterly revenue with Quarter-over-Quarter growth %."""
    kpis = compute_all_kpis(CLEAN_PATH)
    return kpis["quarterly"]


@app.get("/kpis/by-category", tags=["KPIs"])
def kpi_by_category():
    """Revenue breakdown by product category, sorted by net revenue."""
    kpis = compute_all_kpis(CLEAN_PATH)
    return kpis["by_category"]


@app.get("/kpis/by-channel", tags=["KPIs"])
def kpi_by_channel():
    """Revenue and new customer rate by sales channel."""
    kpis = compute_all_kpis(CLEAN_PATH)
    return kpis["by_channel"]


@app.get("/kpis/by-region", tags=["KPIs"])
def kpi_by_region():
    """Revenue breakdown by region."""
    kpis = compute_all_kpis(CLEAN_PATH)
    return kpis["by_region"]


@app.get("/kpis/repeat-purchase", tags=["KPIs"])
def kpi_repeat_purchase():
    """Repeat purchase rate and average orders per customer."""
    kpis = compute_all_kpis(CLEAN_PATH)
    return kpis["repeat_purchase"]


# ─────────────────────────────────────────────
# ML Endpoints
# ─────────────────────────────────────────────
@app.get("/ml/forecast", tags=["ML"])
def revenue_forecast(
    weeks: int = Query(default=12, ge=1, le=52, description="Number of weeks to forecast")
):
    """
    Forecasts net revenue for the next N weeks using
    a Linear Regression model with Fourier seasonality features.
    """
    if app.state.forecaster is None:
        raise HTTPException(
            status_code=503,
            detail="Forecaster model not loaded. Run training first.",
        )
    forecast_df = app.state.forecaster.forecast(n_weeks=weeks)
    return {
        "model": "LinearRegression + Fourier Seasonality",
        "metrics": app.state.forecaster._metrics,
        "forecast": forecast_df.to_dict(orient="records"),
    }


@app.get("/ml/anomalies", tags=["ML"])
def anomaly_report(
    top_n: int = Query(default=20, ge=1, le=200, description="Number of top anomalies to return")
):
    """
    Returns the most anomalous orders detected by IsolationForest,
    sorted by anomaly score (lowest = most anomalous).
    """
    if app.state.detector is None:
        raise HTTPException(
            status_code=503,
            detail="Anomaly detector not loaded. Run training first.",
        )
    df = pd.read_csv(CLEAN_PATH)
    df_flagged = app.state.detector.predict(df)
    anomalies = (
        df_flagged[df_flagged["is_anomaly"]]
        .sort_values("anomaly_score")
        .head(top_n)[
            ["order_id", "category", "channel", "unit_price", "quantity",
             "gross_revenue", "discount_pct", "anomaly_score"]
        ]
    )
    return {
        "total_anomalies": int(df_flagged["is_anomaly"].sum()),
        "contamination_rate": round(df_flagged["is_anomaly"].mean() * 100, 2),
        "top_anomalies": anomalies.to_dict(orient="records"),
    }


# ─────────────────────────────────────────────
# Pipeline trigger (admin)
# ─────────────────────────────────────────────
@app.post("/pipeline/run", tags=["Admin"])
def run_pipeline():
    """Triggers the full ETL + ML training pipeline (use with caution in production)."""
    try:
        from src.pipeline.cleaner import run_cleaning_pipeline
        from src.ml.models import train_all

        run_cleaning_pipeline()
        results = train_all()

        # Reload models
        app.state.detector = AnomalyDetector.load(MODEL_DIR / "anomaly_detector.pkl")
        app.state.forecaster = RevenueForecaster.load(MODEL_DIR / "revenue_forecaster.pkl")

        return {"status": "success", "ml_results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
