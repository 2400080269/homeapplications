# California House Price Valuation & Geospatial ML System

A production-grade, end-to-end Machine Learning system for predicting California residential block-group values with **leak-free geospatial feature engineering**, **Optuna hyperparameter optimization**, **multi-model stacking**, **calibrated 90% prediction intervals via Conformalized Quantile Regression (CQR)**, **real-time TreeSHAP explainability**, a **FastAPI REST microservice**, and an **interactive Streamlit dashboard**.

---

## System Architecture

```
Raw Input (8 Census Features)
        │
        ▼
[CaliforniaHousingFeatureEngineer]  ──► 29 Features (Haversine Hub/Coast Distances,
        │                                 30°/45° Rotated Coordinates, Structural Ratios,
        │                                 K-Means Cluster Proximity, & 5-Fold OOF Spatial Comps)
        ▼
[StandardScaler (pandas output)]
        │
        ├──► [Champion: Stacked Ensemble (XGBoost + LightGBM + RF -> RidgeCV)] ──► Point Valuation ($USD)
        ├──► [Conformalized Quantile XGBoost (q=0.05, q=0.95) ± q̂=0.0549]    ──► 90% Prediction Interval
        └──► [Exact TreeSHAP Explainer]                                        ──► Local $ Attribution
        │
        ▼
Serving Layers:  CLI (`predict.py`)  |  REST API (`app/api.py`)  |  Dashboard (`app/dashboard.py`)
```

---

## Verified Model Benchmarks

Evaluated using identical **5-Fold Cross-Validation** ($N = 16,512$) and a **target-decile stratified 20% Holdout Test Set** ($N = 4,128$) with **1,000 nonparametric bootstrap resamples** for 95% confidence intervals:

| Model | CV RMSE (mean ± std) | Test RMSE (95% CI) | Test MAE | Test MAPE | Test $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Stacked Ensemble (XGB+LGBM+RF)** | **0.4032 ± 0.0112** | **0.4038** `[0.3847, 0.4264]` | **0.2535** | **14.51%** | **0.8765** |
| **LightGBM (Optuna-Tuned)** | 0.4066 ± 0.0113 | 0.4063 `[0.3874, 0.4283]` | 0.2572 | 14.61% | 0.8749 |
| **XGBoost (Optuna-Tuned)** | 0.4061 ± 0.0103 | 0.4065 `[0.3872, 0.4286]` | 0.2550 | 14.66% | 0.8748 |
| **Random Forest (Engineered)** | 0.4211 ± 0.0110 | 0.4218 `[0.4015, 0.4424]` | 0.2645 | 15.11% | 0.8652 |
| **Ridge Regression (Engineered)** | 0.4627 ± 0.0164 | 0.4599 `[0.4414, 0.4800]` | 0.3044 | 17.25% | 0.8398 |
| **Linear Regression (Raw)** | 0.7215 ± 0.0074 | 0.7428 `[0.7109, 0.7823]` | 0.5364 | 32.59% | 0.5820 |
| **Dummy Baseline (Mean)** | 1.1552 ± 0.0076 | 1.1488 `[1.1225, 1.1747]` | 0.9091 | 62.98% | 0.0000 |

### Key Technical Highlights
- **45.6% RMSE Reduction over Baseline**: Test RMSE dropped from `0.7428` ($74,280) on raw Linear Regression down to **`0.4038` ($40,380)**, raising explained variance ($R^2$) from `0.5820` to **`0.8765`**.
- **Feature Engineering Proof**: Adding our 21 engineered geospatial & domain features lifted even a linear `RidgeCV` model from $R^2 = 0.5820$ to **$R^2 = 0.8398$** (`RMSE = 0.4599`).
- **Calibrated Uncertainty (CQR)**: Finite-sample Conformalized Quantile Regression achieved **`90.19%` empirical test coverage** (against a `90.0%` target) with a mean interval width of **`$113,690`**.
- **Slice-Based Error Rigor**: Explicitly audits uncensored (`< $500k`, `RMSE = 0.3814`, `R² = 0.8489`) vs. census top-coded (`>= $500k`, `RMSE = 0.7358`), coastal (`<= 25 km`, `RMSE = 0.4642`), and inland (`> 25 km`, `RMSE = 0.3543`) sub-markets.

---

## Project Structure

```
house-price-predictor/
├── src/
│   ├── __init__.py
│   ├── config.py                # Paths, domain bounds, CA economic hubs & coastline anchor coordinates
│   ├── data_loader.py           # Dataset loading, quality/censoring audit, stratified splitting
│   ├── features.py              # Custom sklearn Transformer (29 features, 5-fold OOF spatial comps)
│   ├── modeling.py              # Optuna TPE tuning, 7-model benchmark suite, CQR calibration
│   ├── evaluation.py            # Bootstrap CIs, slice error analysis, diagnostics, TreeSHAP plots
│   └── inference.py             # Thread-safe HousePricePredictor + local TreeSHAP attributions
├── app/
│   ├── api.py                   # FastAPI REST service (/health, /model-info, /predict, /predict/batch)
│   └── dashboard.py             # Interactive Streamlit map, live valuation & SHAP explorer
├── notebooks/
│   ├── 01_EDA.ipynb             # Executed Exploratory Data Analysis & censoring audit
│   └── 02_Modeling.ipynb        # Executed modeling benchmark, CQR & SHAP walkthrough
├── tests/
│   ├── test_features.py         # Unit tests for Haversine math & zero target-leakage invariants
│   ├── test_pipeline.py         # End-to-end Pipeline, joblib roundtrip & CQR coverage tests
│   └── test_api.py              # FastAPI TestClient integration & Pydantic 422 validation tests
├── artifacts/                   # Trained joblib pipelines + verified model_metadata.json
├── figures/                     # High-DPI benchmark, CQR interval, slice error & SHAP charts
├── train.py                     # CLI training & evaluation orchestrator
├── predict.py                   # CLI valuation & local SHAP driver script
├── Dockerfile                   # Containerized FastAPI deployment specification
└── requirements.txt             # Pinned environment dependencies
```

---

## Step-by-Step Execution & Output Guide

### 1. Activate the Virtual Environment
```powershell
cd house-price-predictor
.\venv\Scripts\activate
```

### 2. Run the Full Training, Tuning & Evaluation Pipeline
```powershell
python train.py
```
**Outputs generated:**
- Console summary of data quality, 7-model 5-fold CV + test benchmark, CQR coverage, and slice errors.
- Serialized pipelines and JSON metadata in `artifacts/`.
- Diagnostic and SHAP visualizations in `figures/`.

### 3. Run CLI Inference (Point Estimate + 90% Interval + Top 5 SHAP Drivers)
```powershell
python predict.py
```
Or pass custom property data via JSON:
```powershell
python predict.py --json '{"MedInc": 8.5, "HouseAge": 30, "AveRooms": 6.5, "AveBedrms": 1.05, "Population": 900, "AveOccup": 2.5, "Latitude": 37.77, "Longitude": -122.42}'
```

### 4. Launch the Interactive Streamlit Dashboard
```powershell
streamlit run app/dashboard.py
```
Opens a browser UI with interactive sliders, California map visualization, live 90% CQR valuation range, local SHAP breakdown, and benchmark tables.

### 5. Launch the FastAPI REST Microservice
```powershell
uvicorn app.api:app --reload --port 8000
```
Open interactive Swagger API docs at `http://127.0.0.1:8000/docs`.

### 6. Run the Automated Test Suite
```powershell
pytest tests -v
```

### 7. Explore Executed Jupyter Notebooks
```powershell
jupyter notebook
```
Open `notebooks/01_EDA.ipynb` and `notebooks/02_Modeling.ipynb` (or view them directly in your IDE).
