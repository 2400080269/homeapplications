"""Model definition, Optuna hyperparameter tuning, cross-validation, and CQR uncertainty."""

import time
from typing import Any, Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import ALPHA_CONFIDENCE, TrainingConfig
from src.features import CaliforniaHousingFeatureEngineer

optuna.logging.set_verbosity(optuna.logging.WARNING)


def build_pipeline(
    estimator: Any,
    config: TrainingConfig,
    use_feature_engineering: bool = True,
) -> Pipeline:
    """Create a leak-free end-to-end scikit-learn Pipeline."""
    steps: List[Tuple[str, Any]] = []
    if use_feature_engineering:
        steps.append(
            (
                "feature_engineer",
                CaliforniaHousingFeatureEngineer(
                    include_spatial_comps=config.include_spatial_comps,
                    spatial_knn_neighbors=config.spatial_knn_neighbors,
                    geo_clusters=config.geo_clusters,
                    oof_folds=config.cv_folds,
                    random_state=config.random_state,
                ),
            )
        )
    scaler = StandardScaler()
    scaler.set_output(transform="pandas")
    steps.append(("scaler", scaler))
    steps.append(("model", estimator))
    return Pipeline(steps)


def _prepare_pretransformed_folds(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: TrainingConfig,
) -> List[Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]]:
    """Precompute leak-free per-fold feature transformations for fast Optuna tuning."""
    kf = KFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )
    folds = []
    for tr_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx].to_numpy(), y_train.iloc[val_idx].to_numpy()

        fe = CaliforniaHousingFeatureEngineer(
            include_spatial_comps=config.include_spatial_comps,
            spatial_knn_neighbors=config.spatial_knn_neighbors,
            geo_clusters=config.geo_clusters,
            oof_folds=config.cv_folds,
            random_state=config.random_state,
        )
        scaler = StandardScaler()
        scaler.set_output(transform="pandas")

        X_tr_fe = fe.fit_transform(X_tr, y_tr)
        X_val_fe = fe.transform(X_val)

        X_tr_sc = scaler.fit_transform(X_tr_fe)
        X_val_sc = scaler.transform(X_val_fe)
        folds.append((X_tr_sc, y_tr, X_val_sc, y_val))
    return folds


def tune_xgboost_optuna(
    folds: List[Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]],
    config: TrainingConfig,
) -> Tuple[Dict[str, Any], float]:
    """Tune XGBRegressor hyperparameters via Optuna TPE minimizing 5-fold CV RMSE."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 380),
            "max_depth": trial.suggest_int("max_depth", 4, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        fold_rmses = []
        for X_tr, y_tr, X_val, y_val in folds:
            model = xgb.XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",
                random_state=config.random_state,
                n_jobs=4,
                **params,
            )
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            fold_rmses.append(float(np.sqrt(mean_squared_error(y_val, preds))))
        mean_rmse = float(np.mean(fold_rmses))
        if (trial.number + 1) % 5 == 0 or trial.number == 0:
            print(f"    XGBoost Optuna trial {trial.number + 1}/{config.optuna_trials} -> CV RMSE: {mean_rmse:.4f}")
        return mean_rmse

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.random_state),
    )
    study.optimize(objective, n_trials=config.optuna_trials, show_progress_bar=False)
    return dict(study.best_params), float(study.best_value)


def tune_lightgbm_optuna(
    folds: List[Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]],
    config: TrainingConfig,
) -> Tuple[Dict[str, Any], float]:
    """Tune LGBMRegressor hyperparameters via Optuna TPE minimizing 5-fold CV RMSE."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 420),
            "num_leaves": trial.suggest_int("num_leaves", 20, 75),
            "max_depth": trial.suggest_int("max_depth", 4, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        fold_rmses = []
        for X_tr, y_tr, X_val, y_val in folds:
            model = lgb.LGBMRegressor(
                objective="regression",
                random_state=config.random_state,
                n_jobs=4,
                verbose=-1,
                **params,
            )
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            fold_rmses.append(float(np.sqrt(mean_squared_error(y_val, preds))))
        mean_rmse = float(np.mean(fold_rmses))
        if (trial.number + 1) % 5 == 0 or trial.number == 0:
            print(f"    LightGBM Optuna trial {trial.number + 1}/{config.optuna_trials} -> CV RMSE: {mean_rmse:.4f}")
        return mean_rmse

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.random_state + 1),
    )
    study.optimize(objective, n_trials=config.optuna_trials, show_progress_bar=False)
    return dict(study.best_params), float(study.best_value)


def build_candidate_pipelines(
    xgb_params: Dict[str, Any],
    lgb_params: Dict[str, Any],
    config: TrainingConfig,
) -> Dict[str, Pipeline]:
    """Assemble the complete model benchmark suite."""
    xgb_tuned = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=config.random_state,
        n_jobs=4,
        **xgb_params,
    )
    lgb_tuned = lgb.LGBMRegressor(
        objective="regression",
        random_state=config.random_state,
        n_jobs=4,
        verbose=-1,
        **lgb_params,
    )
    rf_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=20,
        min_samples_leaf=2,
        random_state=config.random_state,
        n_jobs=4,
    )

    stacking_model = StackingRegressor(
        estimators=[
            (
                "xgb",
                xgb.XGBRegressor(
                    objective="reg:squarederror",
                    tree_method="hist",
                    random_state=config.random_state,
                    n_jobs=4,
                    **xgb_params,
                ),
            ),
            (
                "lgbm",
                lgb.LGBMRegressor(
                    objective="regression",
                    random_state=config.random_state,
                    n_jobs=4,
                    verbose=-1,
                    **lgb_params,
                ),
            ),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=16,
                    min_samples_leaf=2,
                    random_state=config.random_state,
                    n_jobs=4,
                ),
            ),
        ],
        final_estimator=RidgeCV(alphas=np.logspace(-3, 2, 20)),
        cv=3,
        n_jobs=1,
        passthrough=False,
    )

    return {
        "Dummy Baseline (Mean)": build_pipeline(
            DummyRegressor(strategy="mean"),
            config=config,
            use_feature_engineering=False,
        ),
        "Linear Regression (Raw)": build_pipeline(
            LinearRegression(),
            config=config,
            use_feature_engineering=False,
        ),
        "Ridge Regression (Engineered)": build_pipeline(
            RidgeCV(alphas=np.logspace(-2, 4, 25)),
            config=config,
            use_feature_engineering=True,
        ),
        "Random Forest (Engineered)": build_pipeline(
            rf_model,
            config=config,
            use_feature_engineering=True,
        ),
        "LightGBM (Optuna-Tuned)": build_pipeline(
            lgb_tuned,
            config=config,
            use_feature_engineering=True,
        ),
        "XGBoost (Optuna-Tuned)": build_pipeline(
            xgb_tuned,
            config=config,
            use_feature_engineering=True,
        ),
        "Stacked Ensemble (XGB+LGBM+RF)": build_pipeline(
            stacking_model,
            config=config,
            use_feature_engineering=True,
        ),
    }


def cross_validate_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: TrainingConfig,
) -> Dict[str, float]:
    """Run leak-free K-fold cross-validation end-to-end across the full pipeline."""
    from sklearn.base import clone

    kf = KFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )
    rmses, maes, r2s = [], [], []
    start = time.perf_counter()

    for tr_idx, val_idx in kf.split(X_train):
        fold_pipe = clone(pipeline)
        fold_pipe.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        preds = fold_pipe.predict(X_train.iloc[val_idx])
        y_val = y_train.iloc[val_idx]

        rmses.append(float(np.sqrt(mean_squared_error(y_val, preds))))
        maes.append(float(mean_absolute_error(y_val, preds)))
        r2s.append(float(r2_score(y_val, preds)))

    elapsed = time.perf_counter() - start
    return {
        "cv_rmse_mean": float(np.mean(rmses)),
        "cv_rmse_std": float(np.std(rmses)),
        "cv_mae_mean": float(np.mean(maes)),
        "cv_r2_mean": float(np.mean(r2s)),
        "cv_r2_std": float(np.std(r2s)),
        "cv_time_sec": float(elapsed),
    }


def train_conformal_quantile_intervals(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    xgb_params: Dict[str, Any],
    config: TrainingConfig,
    confidence: float = ALPHA_CONFIDENCE,
) -> Tuple[Pipeline, Pipeline, float]:
    """Train lower/upper Quantile XGBoost pipelines with Conformalized Quantile Regression (CQR).

    Computes 5-fold out-of-fold conformity scores inside X_train to determine the exact
    conformal correction margin q_hat for guaranteed finite-sample coverage, then refits
    the lower (5th percentile) and upper (95th percentile) pipelines on 100% of X_train.
    """
    alpha = 1.0 - confidence
    q_low_alpha = alpha / 2.0
    q_high_alpha = 1.0 - (alpha / 2.0)

    quantile_params = {
        k: v
        for k, v in xgb_params.items()
        if k in {"n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight", "reg_alpha", "reg_lambda"}
    }

    def _make_quantile_pipe(q_alpha: float) -> Pipeline:
        model = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=q_alpha,
            tree_method="hist",
            random_state=config.random_state,
            n_jobs=4,
            **quantile_params,
        )
        return build_pipeline(model, config=config, use_feature_engineering=True)

    kf = KFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )
    oof_low = np.zeros(len(X_train), dtype=np.float64)
    oof_high = np.zeros(len(X_train), dtype=np.float64)

    for tr_idx, val_idx in kf.split(X_train):
        p_low = _make_quantile_pipe(q_low_alpha)
        p_high = _make_quantile_pipe(q_high_alpha)
        p_low.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        p_high.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        oof_low[val_idx] = p_low.predict(X_train.iloc[val_idx])
        oof_high[val_idx] = p_high.predict(X_train.iloc[val_idx])

    y_arr = y_train.to_numpy()
    conformity_scores = np.maximum(oof_low - y_arr, y_arr - oof_high)
    n_cal = len(conformity_scores)
    q_level = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / n_cal))
    q_hat = float(np.quantile(conformity_scores, q_level))

    final_low = _make_quantile_pipe(q_low_alpha)
    final_high = _make_quantile_pipe(q_high_alpha)
    final_low.fit(X_train, y_train)
    final_high.fit(X_train, y_train)

    return final_low, final_high, q_hat
