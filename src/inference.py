"""Production inference & local TreeSHAP explanation service."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import StackingRegressor

from src.config import (
    METADATA_PATH,
    MODEL_PIPELINE_PATH,
    QUANTILE_HIGH_PATH,
    QUANTILE_LOW_PATH,
    RAW_FEATURES,
)


class HousePricePredictor:
    """Thread-safe inference wrapper bundling prediction, CQR intervals, and TreeSHAP."""

    def __init__(
        self,
        model_path: Path = MODEL_PIPELINE_PATH,
        low_path: Path = QUANTILE_LOW_PATH,
        high_path: Path = QUANTILE_HIGH_PATH,
        metadata_path: Path = METADATA_PATH,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained pipeline not found at {model_path}. Run `python train.py` first."
            )

        self.pipeline = joblib.load(model_path)
        self.low_pipeline = joblib.load(low_path) if low_path.exists() else None
        self.high_pipeline = joblib.load(high_path) if high_path.exists() else None

        self.metadata: Dict[str, Any] = {}
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        self.cqr_q_hat: float = float(self.metadata.get("cqr_q_hat", 0.0))
        self.champion_name: str = str(self.metadata.get("champion_model", "Champion Pipeline"))
        self._shap_explainer: Optional[shap.TreeExplainer] = self._init_tree_explainer()

    def _init_tree_explainer(self) -> Optional[shap.TreeExplainer]:
        """Extract exact TreeExplainer from champion model (or XGB base learner if stacked)."""
        model_step = self.pipeline.named_steps.get("model")
        if model_step is None:
            return None
        if isinstance(model_step, StackingRegressor):
            tree_estimator = model_step.named_estimators_.get("xgb")
        else:
            tree_estimator = model_step

        if tree_estimator is None:
            return None
        try:
            return shap.TreeExplainer(tree_estimator)
        except Exception:
            return None

    def _transform_to_model_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature_engineer and scaler steps cleanly while preserving column names."""
        Xt = df[RAW_FEATURES].copy()
        if "feature_engineer" in self.pipeline.named_steps:
            Xt = self.pipeline.named_steps["feature_engineer"].transform(Xt)
        if "scaler" in self.pipeline.named_steps:
            Xt = self.pipeline.named_steps["scaler"].transform(Xt)
        return Xt

    def _raw_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute unscaled engineered feature values for human-readable SHAP explanations."""
        Xt = df[RAW_FEATURES].copy()
        if "feature_engineer" in self.pipeline.named_steps:
            Xt = self.pipeline.named_steps["feature_engineer"].transform(Xt)
        return Xt

    def predict(
        self,
        records: pd.DataFrame | List[Dict[str, Any]] | Dict[str, Any],
        include_shap: bool = True,
        top_k_drivers: int = 5,
    ) -> List[Dict[str, Any]]:
        """Generate point predictions, 90% CQR intervals, and local SHAP explanations."""
        if isinstance(records, dict):
            df = pd.DataFrame([records])
        elif isinstance(records, list):
            df = pd.DataFrame(records)
        else:
            df = records.copy()

        missing = [c for c in RAW_FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"Input is missing required columns: {missing}")

        df_input = df[RAW_FEATURES].astype(np.float64)
        point_preds = np.maximum(0.0, self.pipeline.predict(df_input))

        if self.low_pipeline is not None and self.high_pipeline is not None:
            raw_low = self.low_pipeline.predict(df_input) - self.cqr_q_hat
            raw_high = self.high_pipeline.predict(df_input) + self.cqr_q_hat
            low_preds = np.maximum(0.0, raw_low)
            high_preds = np.maximum(low_preds, raw_high)
        else:
            low_preds = point_preds
            high_preds = point_preds

        shap_vals = None
        base_val = None
        eng_df = None
        if include_shap and self._shap_explainer is not None:
            model_input_df = self._transform_to_model_input(df_input)
            eng_df = self._raw_engineered_features(df_input)
            shap_vals = self._shap_explainer.shap_values(model_input_df)
            expected = self._shap_explainer.expected_value
            base_val = float(np.asarray(expected).ravel()[0])

        outputs: List[Dict[str, Any]] = []
        for idx in range(len(df_input)):
            pred_100k = float(point_preds[idx])
            low_100k = float(low_preds[idx])
            high_100k = float(high_preds[idx])

            item: Dict[str, Any] = {
                "predicted_value_100k": round(pred_100k, 4),
                "predicted_price_usd": round(pred_100k * 100_000.0, 2),
                "interval_90_low_100k": round(low_100k, 4),
                "interval_90_high_100k": round(high_100k, 4),
                "interval_90_low_usd": round(low_100k * 100_000.0, 2),
                "interval_90_high_usd": round(high_100k * 100_000.0, 2),
                "model_name": self.champion_name,
            }

            if shap_vals is not None and eng_df is not None:
                row_shap = shap_vals[idx]
                row_eng = eng_df.iloc[idx]
                order = np.argsort(np.abs(row_shap))[::-1][:top_k_drivers]
                drivers = [
                    {
                        "feature": str(eng_df.columns[col_idx]),
                        "feature_value": round(float(row_eng.iloc[col_idx]), 4),
                        "shap_impact_100k": round(float(row_shap[col_idx]), 4),
                        "shap_impact_usd": round(float(row_shap[col_idx] * 100_000.0), 2),
                    }
                    for col_idx in order
                ]
                item["shap_base_value_100k"] = round(base_val, 4) if base_val is not None else None
                item["top_drivers"] = drivers

            outputs.append(item)

        return outputs
