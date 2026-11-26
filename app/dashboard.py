"""Interactive Streamlit Valuation & Explainability Dashboard."""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    px = None
    go = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, FIGURES_DIR, METADATA_PATH
from src.data_loader import load_california_housing_data
from src.inference import HousePricePredictor

st.set_page_config(
    page_title="California Housing Valuation & ML Observatory",
    page_icon="🏠",
    layout="wide",
)

PRESET_LOCATIONS = {
    "Custom Input": None,
    "San Francisco (High Income / Coastal)": {
        "MedInc": 9.5,
        "HouseAge": 45.0,
        "AveRooms": 6.4,
        "AveBedrms": 1.05,
        "Population": 920.0,
        "AveOccup": 2.3,
        "Latitude": 37.78,
        "Longitude": -122.42,
    },
    "Silicon Valley / Palo Alto": {
        "MedInc": 11.2,
        "HouseAge": 36.0,
        "AveRooms": 7.1,
        "AveBedrms": 1.08,
        "Population": 850.0,
        "AveOccup": 2.6,
        "Latitude": 37.44,
        "Longitude": -122.14,
    },
    "Los Angeles (Westside)": {
        "MedInc": 7.2,
        "HouseAge": 38.0,
        "AveRooms": 5.8,
        "AveBedrms": 1.06,
        "Population": 1240.0,
        "AveOccup": 2.5,
        "Latitude": 34.05,
        "Longitude": -118.44,
    },
    "San Diego (Coastal)": {
        "MedInc": 6.4,
        "HouseAge": 28.0,
        "AveRooms": 5.9,
        "AveBedrms": 1.07,
        "Population": 1100.0,
        "AveOccup": 2.6,
        "Latitude": 32.75,
        "Longitude": -117.22,
    },
    "Fresno (Central Valley Inland)": {
        "MedInc": 2.6,
        "HouseAge": 24.0,
        "AveRooms": 4.9,
        "AveBedrms": 1.04,
        "Population": 1450.0,
        "AveOccup": 3.3,
        "Latitude": 36.74,
        "Longitude": -119.79,
    },
}


@st.cache_resource
def load_predictor() -> HousePricePredictor:
    return HousePricePredictor()


@st.cache_data
def load_reference_data() -> pd.DataFrame:
    df, _, _ = load_california_housing_data(DATA_DIR)
    return df.sample(n=min(4000, len(df)), random_state=42)


@st.cache_data
def load_metadata() -> dict:
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def main() -> None:
    st.title("🏠 California Housing Valuation & ML Observatory")
    st.caption(
        "Geospatial Feature Engineering • Optuna-Tuned Gradient Boosting & Stacking • "
        "Conformalized Quantile Regression (90% CQR Intervals) • Real-Time TreeSHAP"
    )

    try:
        predictor = load_predictor()
    except FileNotFoundError:
        st.error("Model artifacts not found. Run `python train.py` first to generate trained pipelines.")
        return

    metadata = load_metadata()
    ref_df = load_reference_data()

    # Sidebar controls
    st.sidebar.header("Neighborhood & Property Profile")
    preset_choice = st.sidebar.selectbox("Location Preset", list(PRESET_LOCATIONS.keys()))
    defaults = PRESET_LOCATIONS[preset_choice] or PRESET_LOCATIONS["San Francisco (High Income / Coastal)"]

    med_inc = st.sidebar.slider("Median Income ($10k units)", 0.5, 15.0, float(defaults["MedInc"]), 0.1)
    house_age = st.sidebar.slider("Median House Age (years)", 1.0, 52.0, float(defaults["HouseAge"]), 1.0)
    ave_rooms = st.sidebar.slider("Avg Rooms / Household", 1.5, 12.0, float(defaults["AveRooms"]), 0.1)
    ave_bedrms = st.sidebar.slider("Avg Bedrooms / Household", 0.5, 4.0, float(defaults["AveBedrms"]), 0.05)
    population = st.sidebar.slider("Block Group Population", 50.0, 6000.0, float(defaults["Population"]), 50.0)
    ave_occup = st.sidebar.slider("Avg Occupancy / Household", 1.0, 8.0, float(defaults["AveOccup"]), 0.1)
    latitude = st.sidebar.slider("Latitude", 32.5, 42.0, float(defaults["Latitude"]), 0.02)
    longitude = st.sidebar.slider("Longitude", -124.3, -114.3, float(defaults["Longitude"]), 0.02)

    payload = {
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude,
    }

    tab_predict, tab_benchmark, tab_explain = st.tabs(
        ["Live Valuation & SHAP", "Model Benchmarks & Slices", "Global Diagnostics"]
    )

    with tab_predict:
        result = predictor.predict(payload, include_shap=True, top_k_drivers=8)[0]

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Estimated Median House Value",
            f"${result['predicted_price_usd']:,.0f}",
            f"{result['predicted_value_100k']:.3f} ($100k)",
        )
        c2.metric(
            "90% CQR Prediction Interval",
            f"${result['interval_90_low_usd']:,.0f} – ${result['interval_90_high_usd']:,.0f}",
            f"Width: ${(result['interval_90_high_usd'] - result['interval_90_low_usd']):,.0f}",
        )
        c3.metric(
            "Champion Architecture",
            result["model_name"],
            f"CQR Test Coverage: {metadata.get('cqr_evaluation', {}).get('empirical_coverage', 0.9) * 100:.1f}%",
        )

        col_map, col_shap = st.columns([1.1, 1.0])

        with col_map:
            st.subheader("Geospatial Context (Selected Point vs. California Blocks)")
            fig_map = px.scatter_map(
                ref_df,
                lat="Latitude",
                lon="Longitude",
                color="MedHouseVal",
                color_continuous_scale="Viridis",
                opacity=0.5,
                zoom=4.8,
                center={"lat": latitude, "lon": longitude},
                map_style="carto-positron",
                labels={"MedHouseVal": "Value ($100k)"},
                height=440,
            )
            fig_map.add_trace(
                go.Scattermap(
                    lat=[latitude],
                    lon=[longitude],
                    mode="markers",
                    marker=go.scattermap.Marker(size=16, color="red"),
                    name="Selected Target Block",
                    text=[f"Predicted: ${result['predicted_price_usd']:,.0f}"],
                )
            )
            fig_map.update_layout(margin={"r": 0, "t": 10, "l": 0, "b": 0})
            st.plotly_chart(fig_map, width="stretch")

        with col_shap:
            st.subheader("Local TreeSHAP Valuation Drivers")
            drivers = pd.DataFrame(result.get("top_drivers", []))
            if not drivers.empty:
                drivers["Direction"] = drivers["shap_impact_usd"].apply(
                    lambda x: "Positive (+)" if x >= 0 else "Negative (-)"
                )
                drivers_sorted = drivers.sort_values("shap_impact_usd", key=abs, ascending=True)
                fig_shap = px.bar(
                    drivers_sorted,
                    x="shap_impact_usd",
                    y="feature",
                    orientation="h",
                    color="Direction",
                    color_discrete_map={"Positive (+)": "#0f766e", "Negative (-)": "#dc2626"},
                    hover_data=["feature_value", "shap_impact_100k"],
                    labels={"shap_impact_usd": "SHAP Impact on Price ($USD)", "feature": "Feature"},
                    height=440,
                )
                st.plotly_chart(fig_shap, width="stretch")

    with tab_benchmark:
        st.subheader("5-Fold Cross-Validation & Holdout Test Benchmark")
        if "benchmarks" in metadata:
            bench_df = pd.DataFrame(metadata["benchmarks"])
            st.dataframe(bench_df, width="stretch", hide_index=True)

        st.subheader("Slice-Based Error Analysis (Holdout Test Set)")
        if "slice_errors" in metadata:
            slice_df = pd.DataFrame(metadata["slice_errors"])
            st.dataframe(slice_df, width="stretch", hide_index=True)

    with tab_explain:
        st.subheader("Saved Diagnostic & Explainability Plots")
        fig_files = [
            ("Model Benchmark Comparison", "model_benchmark_comparison.png"),
            ("Conformal 90% Prediction Intervals", "conformal_prediction_intervals.png"),
            ("Global SHAP Beeswarm Summary", "shap_summary.png"),
            ("Global Feature Importance (TreeSHAP)", "feature_importance.png"),
            ("Actual vs. Predicted & Residuals", "actual_vs_predicted.png"),
            ("Slice Error Analysis", "slice_error_analysis.png"),
        ]
        for idx in range(0, len(fig_files), 2):
            cols = st.columns(2)
            for col_idx in range(2):
                if idx + col_idx < len(fig_files):
                    title, fname = fig_files[idx + col_idx]
                    fpath = FIGURES_DIR / fname
                    if fpath.exists():
                        cols[col_idx].image(str(fpath), caption=title, width="stretch")


if __name__ == "__main__":
    main()
