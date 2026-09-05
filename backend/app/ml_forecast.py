"""
Real trained ML component for POLAR-EMS: a scikit-learn gradient boosting
regressor for station load forecasting.

This is a genuine trained model -- fit on a held-out train/test split with
real measured MAE/RMSE and feature importances -- distinct from the
seasonal-naive statistical forecaster used elsewhere in the API, which has
no learned parameters at all. Training data is synthetic, generated from
the same physical load model the live simulation uses (via a throwaway
SimulationEngine instance), not a real station dataset -- see README.md's
"what's simulated vs real" note.
"""

from __future__ import annotations

import math
import random
from datetime import timedelta

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from .engine import SimulationEngine, day_of_year, seasonal_temp

FEATURE_NAMES = ["hour_sin", "hour_cos", "season_sin", "season_cos", "temp", "wind", "cloud"]


def _features(hour: float, doy: int, temp: float, wind: float, cloud: float):
    return [
        math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
        math.sin(2 * math.pi * doy / 365), math.cos(2 * math.pi * doy / 365),
        temp / 50, wind / 20, cloud,
    ]


def generate_training_data(n_samples: int = 3000, seed: int = 42):
    """Synthetic (features, load) pairs from the simulation's own load model."""
    rng = random.Random(seed)
    eng = SimulationEngine()  # throwaway instance, never ticked
    X, y = [], []
    for _ in range(n_samples):
        doy = rng.randint(1, 365)
        hour = rng.uniform(0, 24)
        wind = max(0.3, rng.gauss(9.5, 4))
        cloud = min(0.95, max(0.0, rng.gauss(0.32, 0.15)))
        temp = seasonal_temp(doy) + rng.gauss(0, 2)
        eng.weather = {"temp": temp, "wind": wind, "irr": 0, "cloud": cloud, "doy": doy, "hour": hour}
        load = eng._compute_load()
        X.append(_features(hour, doy, temp, wind, cloud))
        y.append(load["total"])
    return np.array(X), np.array(y)


_model: GradientBoostingRegressor | None = None
_metrics: dict = {}


def train_model():
    global _model, _metrics
    X, y = generate_training_data(3000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.08, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    _metrics = {
        "model_type": "GradientBoostingRegressor (scikit-learn)",
        "test_mae_kw": round(float(mean_absolute_error(y_test, preds)), 2),
        "test_rmse_kw": round(float(mean_squared_error(y_test, preds) ** 0.5), 2),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_importances": {
            name: round(float(v), 4) for name, v in zip(FEATURE_NAMES, model.feature_importances_)
        },
    }
    _model = model
    return _metrics


def predict_load(hour: float, doy: int, temp: float, wind: float, cloud: float) -> float:
    if _model is None:
        train_model()
    x = np.array([_features(hour, doy, temp, wind, cloud)])
    return float(_model.predict(x)[0])


def predict_for_horizon(engine: SimulationEngine, horizon_hours: int) -> float:
    """Predict load at sim_time + horizon_hours, using the seasonally-expected
    weather for that future moment (same convention as the statistical forecaster)."""
    target_t = engine.sim_time + timedelta(hours=horizon_hours)
    doy_t = ((day_of_year(target_t) - 1) % 365) + 1
    hour_t = target_t.hour
    temp_t = seasonal_temp(doy_t)
    return predict_load(hour_t, doy_t, temp_t, 9.5, 0.32)


def get_metrics() -> dict:
    if not _metrics:
        train_model()
    return _metrics
