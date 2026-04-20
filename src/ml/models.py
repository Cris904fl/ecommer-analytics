"""
ML Module: Revenue Forecasting & Anomaly Detection
====================================================
Models:
  1. IsolationForest  → detects anomalous orders (price spikes, fraud signals)
  2. Linear Regression (+ seasonality features) → weekly revenue forecast
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import joblib

logger = logging.getLogger(__name__)

CLEAN_PATH = Path("data/processed/orders_clean.csv")
MODEL_DIR = Path("models")


# ─────────────────────────────────────────────
# 1. Anomaly Detection
# ─────────────────────────────────────────────
class AnomalyDetector:
    """
    Uses IsolationForest to flag orders with unusual
    combinations of price, quantity, and discount.
    """

    FEATURES = ["unit_price", "quantity", "discount_pct", "gross_revenue"]
    CONTAMINATION = 0.03  # ~3% expected anomaly rate

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=200,
            contamination=self.CONTAMINATION,
            random_state=42,
            n_jobs=-1,
        )
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "AnomalyDetector":
        X = df[self.FEATURES].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True
        logger.info("IsolationForest trained on %d samples", len(df))
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.is_fitted, "Call .fit() first"
        X = df[self.FEATURES].fillna(0)
        X_scaled = self.scaler.transform(X)
        scores = self.model.decision_function(X_scaled)  # lower = more anomalous
        labels = self.model.predict(X_scaled)  # -1 = anomaly, 1 = normal

        result = df.copy()
        result["anomaly_score"] = scores.round(4)
        result["is_anomaly"] = labels == -1
        n_anomalies = result["is_anomaly"].sum()
        logger.info(f"Anomalies detected: {n_anomalies} ({n_anomalies/len(df):.1%})")
        return result

    def save(self, path: Path = MODEL_DIR / "anomaly_detector.pkl") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self.scaler, "model": self.model}, path)
        logger.info(f"Model saved → {path}")

    @classmethod
    def load(cls, path: Path = MODEL_DIR / "anomaly_detector.pkl") -> "AnomalyDetector":
        data = joblib.load(path)
        instance = cls()
        instance.scaler = data["scaler"]
        instance.model = data["model"]
        instance.is_fitted = True
        return instance


# ─────────────────────────────────────────────
# 2. Revenue Forecaster
# ─────────────────────────────────────────────
class RevenueForecaster:
    """
    Forecasts weekly net revenue using:
    - Time trend (week index)
    - Seasonality encoded as Fourier terms (sin/cos)
    - Linear Regression baseline
    """

    def __init__(self, n_fourier: int = 3):
        self.n_fourier = n_fourier
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_fitted = False
        self._period = 52  # weeks per year

    def _build_features(self, week_index: np.ndarray) -> np.ndarray:
        features = [week_index.reshape(-1, 1)]
        for k in range(1, self.n_fourier + 1):
            features.append(
                np.sin(2 * np.pi * k * week_index / self._period).reshape(-1, 1)
            )
            features.append(
                np.cos(2 * np.pi * k * week_index / self._period).reshape(-1, 1)
            )
        return np.hstack(features)

    def prepare_weekly_series(self, df: pd.DataFrame) -> pd.DataFrame:
        completed = df[df["status"] == "completed"].copy()
        weekly = (
            completed.groupby(["year", "week"])
            .agg(net_revenue=("net_revenue", "sum"), orders=("order_id", "count"))
            .reset_index()
            .sort_values(["year", "week"])
            .reset_index(drop=True)
        )
        weekly["week_index"] = np.arange(len(weekly))
        return weekly

    def fit(self, df: pd.DataFrame) -> "RevenueForecaster":
        weekly = self.prepare_weekly_series(df)
        X = self._build_features(weekly["week_index"].values)
        y = weekly["net_revenue"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        X_train_sc = self.scaler.fit_transform(X_train)
        X_test_sc = self.scaler.transform(X_test)

        self.model.fit(X_train_sc, y_train)
        y_pred = self.model.predict(X_test_sc)

        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        logger.info(f"Forecaster trained | MAE: ${mae:,.0f} | MAPE: {mape:.1%}")

        self._last_week_index = int(weekly["week_index"].max())
        self._metrics = {"mae": round(mae, 2), "mape": round(mape, 4)}
        self.is_fitted = True
        return self

    def forecast(self, n_weeks: int = 12) -> pd.DataFrame:
        assert self.is_fitted, "Call .fit() first"
        future_indices = np.arange(
            self._last_week_index + 1,
            self._last_week_index + 1 + n_weeks,
        )
        X_future = self._build_features(future_indices)
        X_future_sc = self.scaler.transform(X_future)
        predictions = self.model.predict(X_future_sc)

        return pd.DataFrame(
            {
                "week_ahead": range(1, n_weeks + 1),
                "forecasted_net_revenue": np.maximum(predictions, 0).round(2),
            }
        )

    def save(self, path: Path = MODEL_DIR / "revenue_forecaster.pkl") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.__dict__, path)
        logger.info(f"Forecaster saved → {path}")

    @classmethod
    def load(
        cls, path: Path = MODEL_DIR / "revenue_forecaster.pkl"
    ) -> "RevenueForecaster":
        instance = cls()
        instance.__dict__.update(joblib.load(path))
        return instance


# ─────────────────────────────────────────────
# Train & evaluate both models
# ─────────────────────────────────────────────
def train_all(path: Path = CLEAN_PATH) -> dict:
    df = pd.read_csv(path, parse_dates=["order_date"])

    # Anomaly Detection
    detector = AnomalyDetector()
    detector.fit(df)
    df_flagged = detector.predict(df)
    detector.save()

    anomaly_summary = (
        df_flagged[df_flagged["is_anomaly"]]
        .groupby("category")
        .agg(
            count=("order_id", "count"),
            avg_revenue=("gross_revenue", "mean"),
        )
        .reset_index()
        .to_dict(orient="records")
    )

    # Revenue Forecasting
    forecaster = RevenueForecaster()
    forecaster.fit(df)
    forecast_df = forecaster.forecast(n_weeks=12)
    forecaster.save()

    logger.info("\n12-Week Revenue Forecast:")
    logger.info(forecast_df.to_string(index=False))

    return {
        "anomaly_summary": anomaly_summary,
        "forecast": forecast_df.to_dict(orient="records"),
        "forecaster_metrics": forecaster._metrics,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    results = train_all()
    print("\nForecaster metrics:", results["forecaster_metrics"])
    print("Anomalies by category:")
    for row in results["anomaly_summary"]:
        print(f"  {row['category']}: {row['count']} orders flagged")
