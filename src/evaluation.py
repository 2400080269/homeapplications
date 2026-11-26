"""Statistical evaluation, bootstrap CIs, slice error analysis, CQR calibration, and SHAP plots."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

from src.config import TARGET_CAP_VALUE
from src.features import CALIFORNIA_COASTLINE_POINTS, haversine_distance_km

sns.set_theme(style="whitegrid")


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute core regression evaluation metrics."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    mae = float(mean_absolute_error(y_t, y_p))
    medae = float(median_absolute_error(y_t, y_p))
    mape = float(mean_absolute_percentage_error(y_t, y_p) * 100.0)
    r2 = float(r2_score(y_t, y_p))
    return {
        "RMSE": rmse,
        "MAE": mae,
        "MedAE": medae,
        "MAPE_pct": mape,
        "R2": r2,
    }


def bootstrap_metric_cis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Compute 95% nonparametric bootstrap confidence intervals for RMSE, MAE, and R2."""
    rng = np.random.default_rng(random_state)
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    n = len(y_t)

    rmses = np.empty(n_bootstrap, dtype=np.float64)
    maes = np.empty(n_bootstrap, dtype=np.float64)
    r2s = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt_b = y_t[idx]
        yp_b = y_p[idx]
        rmses[i] = np.sqrt(mean_squared_error(yt_b, yp_b))
        maes[i] = mean_absolute_error(yt_b, yp_b)
        r2s[i] = r2_score(yt_b, yp_b)

    return {
        "RMSE_ci95": (float(np.percentile(rmses, 2.5)), float(np.percentile(rmses, 97.5))),
        "MAE_ci95": (float(np.percentile(maes, 2.5)), float(np.percentile(maes, 97.5))),
        "R2_ci95": (float(np.percentile(r2s, 2.5)), float(np.percentile(r2s, 97.5))),
    }


def evaluate_conformal_intervals(
    y_true: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> Dict[str, float]:
    """Audit empirical test-set coverage and sharpness of calibrated prediction intervals."""
    y_t = np.asarray(y_true, dtype=np.float64)
    low = np.asarray(lower_bounds, dtype=np.float64)
    high = np.asarray(upper_bounds, dtype=np.float64)

    covered = (y_t >= low) & (y_t <= high)
    widths = high - low
    return {
        "empirical_coverage": float(np.mean(covered)),
        "mean_interval_width_100k": float(np.mean(widths)),
        "median_interval_width_100k": float(np.median(widths)),
    }


def slice_error_analysis(
    X_test: pd.DataFrame,
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Evaluate champion model accuracy across key domain subpopulations."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    lat = X_test["Latitude"].to_numpy()
    lon = X_test["Longitude"].to_numpy()
    coast_dists = np.min(
        np.vstack(
            [
                haversine_distance_km(lat, lon, c_lat, c_lon)
                for c_lat, c_lon in CALIFORNIA_COASTLINE_POINTS
            ]
        ),
        axis=0,
    )

    slices: List[Tuple[str, np.ndarray]] = [
        ("Target: Uncensored (< $500k)", y_t < TARGET_CAP_VALUE),
        ("Target: Censored Ceiling (>= $500k)", y_t >= TARGET_CAP_VALUE),
        ("Income: Low (MedInc < 2.5)", X_test["MedInc"].to_numpy() < 2.5),
        (
            "Income: Middle (2.5 <= MedInc <= 5.0)",
            (X_test["MedInc"].to_numpy() >= 2.5) & (X_test["MedInc"].to_numpy() <= 5.0),
        ),
        ("Income: High (MedInc > 5.0)", X_test["MedInc"].to_numpy() > 5.0),
        ("Geography: Coastal Belt (<= 25 km)", coast_dists <= 25.0),
        ("Geography: Inland (> 25 km)", coast_dists > 25.0),
        ("Housing Age: Newer (< 35 yrs)", X_test["HouseAge"].to_numpy() < 35.0),
        ("Housing Age: Older (>= 35 yrs)", X_test["HouseAge"].to_numpy() >= 35.0),
    ]

    rows = []
    for label, mask in slices:
        count = int(np.sum(mask))
        if count < 5:
            continue
        yt_s = y_t[mask]
        yp_s = y_p[mask]
        rmse = float(np.sqrt(mean_squared_error(yt_s, yp_s)))
        mae = float(mean_absolute_error(yt_s, yp_s))
        # R2 is degenerate when target variance is ~0 (e.g. censored $500k ceiling)
        r2 = float(r2_score(yt_s, yp_s)) if np.std(yt_s) > 1e-2 else float("nan")
        rows.append(
            {
                "Slice": label,
                "Count": count,
                "Share_pct": round(100.0 * count / len(y_t), 1),
                "RMSE": round(rmse, 4),
                "MAE": round(mae, 4),
                "R2": round(r2, 4) if np.isfinite(r2) else None,
            }
        )
    return pd.DataFrame(rows)


def _save_fig(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(benchmark_df: pd.DataFrame, figures_dir: Path) -> Path:
    """Visualize RMSE (with 95% bootstrap error bars) and R2 across all benchmarked models."""
    plt.ioff()
    df_plot = benchmark_df.sort_values("Test_RMSE", ascending=True).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # RMSE with 95% CI whiskers
    y_pos = np.arange(len(df_plot))
    err_low = df_plot["Test_RMSE"] - df_plot["RMSE_CI95_Low"]
    err_high = df_plot["RMSE_CI95_High"] - df_plot["Test_RMSE"]
    colors = ["#0f766e" if i == 0 else "#3b82f6" if "Optuna" in m or "Ensemble" in m else "#64748b" for i, m in enumerate(df_plot["Model"])]

    axes[0].barh(
        y_pos,
        df_plot["Test_RMSE"],
        xerr=[err_low, err_high],
        color=colors,
        capsize=4,
        alpha=0.9,
    )
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(df_plot["Model"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Holdout Test RMSE ($100k, lower is better)")
    axes[0].set_title("Model Comparison — Test RMSE with 95% Bootstrap CI")
    for idx, val in enumerate(df_plot["Test_RMSE"]):
        axes[0].text(val + 0.02, idx, f"{val:.4f}", va="center", fontsize=9, fontweight="bold")

    # R2 comparison
    axes[1].barh(y_pos, df_plot["Test_R2"], color=colors, alpha=0.9)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(df_plot["Model"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Holdout Test R² Score (higher is better)")
    axes[1].set_title("Model Comparison — Test R² Variance Explained")
    axes[1].set_xlim(min(0.0, df_plot["Test_R2"].min() - 0.05), 1.0)
    for idx, val in enumerate(df_plot["Test_R2"]):
        axes[1].text(max(val, 0) + 0.015, idx, f"{val:.4f}", va="center", fontsize=9, fontweight="bold")

    out_path = figures_dir / "model_benchmark_comparison.png"
    _save_fig(fig, out_path)
    return out_path


def plot_diagnostics_suite(
    y_test: np.ndarray,
    champion_name: str,
    champion_pred: np.ndarray,
    baseline_pred: np.ndarray,
    figures_dir: Path,
) -> Path:
    """Create 3-panel regression diagnostic plot: Actual vs Pred, Residuals vs Pred, Residual Histogram."""
    plt.ioff()
    y_t = np.asarray(y_test, dtype=np.float64)
    residuals = y_t - champion_pred

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: Baseline vs Champion Actual vs Predicted
    axes[0].scatter(y_t, baseline_pred, alpha=0.18, s=12, color="#94a3b8", label="Linear Baseline (Raw)")
    axes[0].scatter(y_t, champion_pred, alpha=0.30, s=14, color="#0f766e", label=champion_name)
    axes[0].plot([y_t.min(), y_t.max()], [y_t.min(), y_t.max()], "r--", lw=2, label="Ideal (y = ŷ)")
    axes[0].set_title("Actual vs. Predicted Values")
    axes[0].set_xlabel("Actual Median Value ($100k)")
    axes[0].set_ylabel("Predicted Median Value ($100k)")
    axes[0].legend(loc="upper left", fontsize=8.5)

    # Panel 2: Residuals vs Predicted (Homoscedasticity check)
    axes[1].scatter(champion_pred, residuals, alpha=0.25, s=12, color="#2563eb")
    axes[1].axhline(0.0, color="red", linestyle="--", lw=1.8)
    axes[1].set_title(f"{champion_name} — Residuals vs. Predicted")
    axes[1].set_xlabel("Predicted Value ($100k)")
    axes[1].set_ylabel("Residual (Actual - Predicted)")

    # Panel 3: Residual Distribution
    sns.histplot(residuals, bins=50, kde=True, color="#0f766e", ax=axes[2])
    axes[2].axvline(0.0, color="red", linestyle="--", lw=1.5)
    axes[2].set_title(f"Residual Distribution (Mean: {np.mean(residuals):.3f}, Std: {np.std(residuals):.3f})")
    axes[2].set_xlabel("Prediction Error ($100k)")
    axes[2].set_ylabel("Count")

    out_path = figures_dir / "actual_vs_predicted.png"
    _save_fig(fig, out_path)
    return out_path


def plot_slice_errors(slice_df: pd.DataFrame, figures_dir: Path) -> Path:
    """Visualize RMSE and MAE across domain subpopulations."""
    plt.ioff()
    fig, ax = plt.subplots(figsize=(11, 6))
    y_pos = np.arange(len(slice_df))
    height = 0.36

    ax.barh(y_pos - height / 2, slice_df["RMSE"], height=height, label="RMSE ($100k)", color="#0284c7")
    ax.barh(y_pos + height / 2, slice_df["MAE"], height=height, label="MAE ($100k)", color="#14b8a6")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{row['Slice']} (n={row['Count']})" for _, row in slice_df.iterrows()])
    ax.invert_yaxis()
    ax.set_xlabel("Error ($100k)")
    ax.set_title("Slice-Based Error Analysis Across Market Segments")
    ax.legend()

    out_path = figures_dir / "slice_error_analysis.png"
    _save_fig(fig, out_path)
    return out_path


def plot_conformal_intervals(
    y_test: np.ndarray,
    point_pred: np.ndarray,
    low_pred: np.ndarray,
    high_pred: np.ndarray,
    coverage: float,
    figures_dir: Path,
    sample_size: int = 120,
) -> Path:
    """Visualize calibrated 90% CQR prediction intervals on sorted test samples."""
    plt.ioff()
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_test), size=min(sample_size, len(y_test)), replace=False)
    order = np.argsort(point_pred[idx])
    sel = idx[order]

    x_axis = np.arange(len(sel))
    yt_s = np.asarray(y_test)[sel]
    yp_s = point_pred[sel]
    low_s = low_pred[sel]
    high_s = high_pred[sel]
    in_band = (yt_s >= low_s) & (yt_s <= high_s)

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.fill_between(
        x_axis,
        low_s,
        high_s,
        color="#93c5fd",
        alpha=0.45,
        label="Calibrated 90% Prediction Interval (CQR)",
    )
    ax.plot(x_axis, yp_s, color="#1d4ed8", lw=1.6, label="Point Prediction")
    ax.scatter(
        x_axis[in_band],
        yt_s[in_band],
        color="#15803d",
        s=20,
        zorder=4,
        label="Actual (Inside Interval)",
    )
    ax.scatter(
        x_axis[~in_band],
        yt_s[~in_band],
        color="#dc2626",
        marker="x",
        s=32,
        zorder=5,
        label="Actual (Outside Interval)",
    )
    ax.set_title(
        f"Conformalized Quantile Regression (CQR) — Test Coverage: {coverage * 100:.1f}% (Target: 90.0%)"
    )
    ax.set_xlabel("Test Block Groups (Sorted by Predicted Value)")
    ax.set_ylabel("Median House Value ($100k)")
    ax.legend(loc="upper left")

    out_path = figures_dir / "conformal_prediction_intervals.png"
    _save_fig(fig, out_path)
    return out_path


def compute_and_plot_shap(
    tree_model: Any,
    X_transformed: pd.DataFrame,
    figures_dir: Path,
    max_samples: int = 1500,
) -> Tuple[Path, Path, pd.DataFrame]:
    """Compute exact TreeSHAP values and save global bar + beeswarm visualizations."""
    plt.ioff()
    if len(X_transformed) > max_samples:
        X_sample = X_transformed.sample(n=max_samples, random_state=42)
    else:
        X_sample = X_transformed

    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(X_sample)

    # 1. SHAP Beeswarm Summary
    plt.figure(figsize=(11, 8))
    shap.summary_plot(shap_values, X_sample, max_display=16, show=False)
    plt.title("SHAP Feature Impact Summary (Top 16 Engineered & Geospatial Features)", pad=14)
    beeswarm_path = figures_dir / "shap_summary.png"
    _save_fig(plt.gcf(), beeswarm_path)

    # 2. Mean |SHAP| Global Importance Bar Chart
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = (
        pd.DataFrame({"Feature": X_sample.columns, "MeanAbsSHAP": mean_abs_shap})
        .sort_values("MeanAbsSHAP", ascending=False)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    top_df = shap_df.head(16).iloc[::-1]
    ax.barh(top_df["Feature"], top_df["MeanAbsSHAP"], color="#0f766e")
    ax.set_xlabel("Mean |SHAP Value| (Average Impact on Price in $100k)")
    ax.set_title("Global Feature Importance via Exact TreeSHAP")
    bar_path = figures_dir / "feature_importance.png"
    _save_fig(fig, bar_path)

    return beeswarm_path, bar_path, shap_df
