"""CLI entrypoint for training, Optuna tuning, CQR calibration, evaluation, and artifact export."""

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor

from src.config import (
    ARTIFACTS_DIR,
    FIGURES_DIR,
    METADATA_PATH,
    MODEL_PIPELINE_PATH,
    QUANTILE_HIGH_PATH,
    QUANTILE_LOW_PATH,
    TrainingConfig,
)
from src.data_loader import (
    create_train_test_split,
    inspect_data_quality,
    load_california_housing_data,
)
from src.evaluation import (
    bootstrap_metric_cis,
    compute_and_plot_shap,
    compute_regression_metrics,
    evaluate_conformal_intervals,
    plot_conformal_intervals,
    plot_diagnostics_suite,
    plot_model_comparison,
    plot_slice_errors,
    slice_error_analysis,
)
from src.modeling import (
    _prepare_pretransformed_folds,
    build_candidate_pipelines,
    cross_validate_pipeline,
    train_conformal_quantile_intervals,
    tune_lightgbm_optuna,
    tune_xgboost_optuna,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and benchmark California House Price Prediction pipelines."
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=15,
        help="Number of Optuna hyperparameter tuning trials per boosted model (default: 15).",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for 95%% confidence intervals (default: 1000).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(optuna_trials=args.trials, n_bootstrap=args.bootstrap)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("California House Price Predictor — Advanced End-to-End Training Pipeline")
    print("=" * 72)

    # 1. Data Loading & Audit
    print("\n[1/7] Loading dataset & auditing data quality...")
    df, raw_features, target_col = load_california_housing_data()
    quality = inspect_data_quality(df)
    print(
        f"  Samples: {quality['n_rows']:,} | Columns: {quality['n_columns']} | "
        f"Missing: {quality['total_missing']} | Duplicates: {quality['duplicate_rows']}"
    )
    print(
        f"  Censored Target (>= $500k): {quality['target_capped_count']} "
        f"({quality['target_capped_pct']:.2f}%)"
    )
    print(
        f"  Extreme Outliers -> AveRooms>20: {quality['extreme_ave_rooms_gt_20']} | "
        f"AveOccup>15: {quality['extreme_ave_occup_gt_15']}"
    )

    # 2. Strict Train-Test Split BEFORE any featurization
    print("\n[2/7] Performing target-decile stratified 80/20 train-test split...")
    X_train, X_test, y_train, y_test = create_train_test_split(
        df,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify_by_target_decile=True,
    )
    print(f"  Train samples: {len(X_train):,} | Holdout Test samples: {len(X_test):,}")

    # 3. Hyperparameter Optimization via Optuna on Leak-Free CV Folds
    print(f"\n[3/7] Tuning XGBoost & LightGBM via Optuna ({config.optuna_trials} trials, 5-fold CV)...")
    folds = _prepare_pretransformed_folds(X_train, y_train, config)
    xgb_params, xgb_cv_best = tune_xgboost_optuna(folds, config)
    lgb_params, lgb_cv_best = tune_lightgbm_optuna(folds, config)
    print(f"  Best XGBoost 5-Fold CV RMSE:  {xgb_cv_best:.4f}")
    print(f"  Best LightGBM 5-Fold CV RMSE: {lgb_cv_best:.4f}")

    # 4. Full End-to-End 5-Fold Cross-Validation & Holdout Benchmarking
    print("\n[4/7] Benchmarking 7 end-to-end pipelines (5-Fold CV + Holdout Test + 95% Bootstrap CI)...")
    candidates = build_candidate_pipelines(xgb_params, lgb_params, config)
    benchmark_rows = []
    fitted_pipelines: Dict[str, Any] = {}
    test_predictions: Dict[str, np.ndarray] = {}

    for name, pipe in candidates.items():
        print(f"  --> Evaluating {name}...")
        cv_stats = cross_validate_pipeline(pipe, X_train, y_train, config)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        test_stats = compute_regression_metrics(y_test.to_numpy(), y_pred)
        cis = bootstrap_metric_cis(
            y_test.to_numpy(),
            y_pred,
            n_bootstrap=config.n_bootstrap,
            random_state=config.random_state,
        )

        fitted_pipelines[name] = pipe
        test_predictions[name] = y_pred

        benchmark_rows.append(
            {
                "Model": name,
                "CV_RMSE_Mean": round(cv_stats["cv_rmse_mean"], 4),
                "CV_RMSE_Std": round(cv_stats["cv_rmse_std"], 4),
                "CV_R2_Mean": round(cv_stats["cv_r2_mean"], 4),
                "Test_RMSE": round(test_stats["RMSE"], 4),
                "RMSE_CI95_Low": round(cis["RMSE_ci95"][0], 4),
                "RMSE_CI95_High": round(cis["RMSE_ci95"][1], 4),
                "Test_MAE": round(test_stats["MAE"], 4),
                "Test_MedAE": round(test_stats["MedAE"], 4),
                "Test_MAPE_pct": round(test_stats["MAPE_pct"], 2),
                "Test_R2": round(test_stats["R2"], 4),
                "R2_CI95_Low": round(cis["R2_ci95"][0], 4),
                "R2_CI95_High": round(cis["R2_ci95"][1], 4),
                "CV_Time_Sec": round(cv_stats["cv_time_sec"], 2),
            }
        )

    benchmark_df = pd.DataFrame(benchmark_rows).sort_values("Test_RMSE", ascending=True).reset_index(drop=True)
    print("\n" + "=" * 72)
    print("Model Benchmark Summary (Sorted by Holdout Test RMSE):")
    print("=" * 72)
    display_cols = ["Model", "CV_RMSE_Mean", "Test_RMSE", "Test_MAE", "Test_MAPE_pct", "Test_R2"]
    print(benchmark_df[display_cols].to_string(index=False))
    print("=" * 72)

    champion_name = str(benchmark_df.iloc[0]["Model"])
    champion_pipe = fitted_pipelines[champion_name]
    champion_pred = test_predictions[champion_name]
    baseline_pred = test_predictions["Linear Regression (Raw)"]
    print(
        f"\n[Champion Selected] {champion_name} | "
        f"Test RMSE: {benchmark_df.iloc[0]['Test_RMSE']:.4f} | "
        f"Test R²: {benchmark_df.iloc[0]['Test_R2']:.4f}"
    )

    # 5. Calibrated Uncertainty Quantification via Conformalized Quantile Regression (CQR)
    print("\n[5/7] Training & calibrating 90% CQR Prediction Intervals...")
    low_pipe, high_pipe, cqr_q_hat = train_conformal_quantile_intervals(
        X_train, y_train, xgb_params, config
    )
    test_low = np.maximum(0.0, low_pipe.predict(X_test) - cqr_q_hat)
    test_high = np.maximum(test_low, high_pipe.predict(X_test) + cqr_q_hat)
    cqr_eval = evaluate_conformal_intervals(y_test.to_numpy(), test_low, test_high)
    print(
        f"  CQR q_hat adjustment: {cqr_q_hat:.4f} | "
        f"Empirical Test Coverage: {cqr_eval['empirical_coverage'] * 100:.2f}% (Target: 90.0%) | "
        f"Mean Interval Width: ${cqr_eval['mean_interval_width_100k'] * 100_000:,.0f}"
    )

    # 6. Slice-Based Error Analysis & TreeSHAP Explainability
    print("\n[6/7] Running slice-based error analysis & computing TreeSHAP explanations...")
    slice_df = slice_error_analysis(X_test, y_test, champion_pred)
    print("\nSlice Error Analysis:")
    print(slice_df.to_string(index=False))

    # Extract engineered scaled features on test set for exact TreeSHAP
    fe_step = champion_pipe.named_steps["feature_engineer"]
    sc_step = champion_pipe.named_steps["scaler"]
    model_step = champion_pipe.named_steps["model"]
    tree_for_shap = (
        model_step.named_estimators_["xgb"]
        if isinstance(model_step, StackingRegressor)
        else model_step
    )
    X_test_transformed = sc_step.transform(fe_step.transform(X_test))

    plot_model_comparison(benchmark_df, FIGURES_DIR)
    plot_diagnostics_suite(y_test.to_numpy(), champion_name, champion_pred, baseline_pred, FIGURES_DIR)
    plot_slice_errors(slice_df, FIGURES_DIR)
    plot_conformal_intervals(
        y_test.to_numpy(),
        champion_pred,
        test_low,
        test_high,
        cqr_eval["empirical_coverage"],
        FIGURES_DIR,
    )
    _, _, shap_df = compute_and_plot_shap(tree_for_shap, X_test_transformed, FIGURES_DIR)
    print("  Saved all evaluation & SHAP figures to ./figures/")

    # 7. Persist Production Artifacts & Metadata
    print("\n[7/7] Saving serialized pipelines and training metadata to ./artifacts/...")
    joblib.dump(champion_pipe, MODEL_PIPELINE_PATH)
    joblib.dump(low_pipe, QUANTILE_LOW_PATH)
    joblib.dump(high_pipe, QUANTILE_HIGH_PATH)

    metadata: Dict[str, Any] = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "California Housing (20,640 block groups, 8 raw features)",
        "engineered_feature_count": len(fe_step.get_feature_names_out()),
        "engineered_features": [str(f) for f in fe_step.get_feature_names_out()],
        "data_quality": quality,
        "champion_model": champion_name,
        "xgb_best_params": xgb_params,
        "lgb_best_params": lgb_params,
        "cqr_q_hat": cqr_q_hat,
        "cqr_evaluation": cqr_eval,
        "benchmarks": benchmark_df.to_dict(orient="records"),
        "slice_errors": slice_df.where(pd.notnull(slice_df), None).to_dict(orient="records"),
        "top_shap_features": shap_df.head(15).to_dict(orient="records"),
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved champion pipeline: {MODEL_PIPELINE_PATH}")
    print(f"  Saved quantile pipelines: {QUANTILE_LOW_PATH.name}, {QUANTILE_HIGH_PATH.name}")
    print(f"  Saved metadata JSON:     {METADATA_PATH}")
    print("\nTraining pipeline completed successfully!")


if __name__ == "__main__":
    main()
