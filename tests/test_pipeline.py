"""Integration tests for scikit-learn Pipeline, serialization, and evaluation metrics."""

import joblib
import numpy as np
import xgboost as xgb

from src.config import TrainingConfig
from src.evaluation import compute_regression_metrics, evaluate_conformal_intervals
from src.modeling import build_pipeline
from tests.test_features import _sample_df


def test_pipeline_fit_predict_and_joblib_roundtrip(tmp_path):
    df, y = _sample_df(80)
    X_train, X_test = df.iloc[:60].reset_index(drop=True), df.iloc[60:].reset_index(drop=True)
    y_train = y[:60]

    config = TrainingConfig(spatial_knn_neighbors=5, geo_clusters=4, cv_folds=3)
    pipe = build_pipeline(
        xgb.XGBRegressor(n_estimators=20, max_depth=3, random_state=42),
        config=config,
        use_feature_engineering=True,
    )
    pipe.fit(X_train, y_train)
    preds_before = pipe.predict(X_test)

    artifact_path = tmp_path / "test_pipe.joblib"
    joblib.dump(pipe, artifact_path)
    loaded_pipe = joblib.load(artifact_path)
    preds_after = loaded_pipe.predict(X_test)

    np.testing.assert_allclose(preds_before, preds_after, rtol=1e-6)
    assert np.isfinite(preds_after).all()


def test_metrics_and_conformal_coverage():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics["RMSE"] > 0.0
    assert metrics["MAE"] > 0.0
    assert 0.9 < metrics["R2"] <= 1.0

    low = np.array([0.8, 1.5, 2.8, 3.5])
    high = np.array([1.3, 2.3, 3.4, 3.9])  # 4.0 is outside [3.5, 3.9]
    cqr = evaluate_conformal_intervals(y_true, low, high)
    assert np.isclose(cqr["empirical_coverage"], 0.75)
    assert cqr["mean_interval_width_100k"] > 0.0
