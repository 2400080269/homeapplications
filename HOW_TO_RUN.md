# How to Execute the California House Price Predictor Project

This guide provides **every exact command** to run, test, and inspect all outputs of the project on Windows PowerShell.

---

## Quick Overview: Where Each Output Lives

| What You Want to See | Command / File | Output Type |
| :--- | :--- | :--- |
| **Interactive Web App (Map + UI)** | `streamlit run app/dashboard.py` | Browser UI (`http://localhost:8501`) |
| **REST API & Swagger Docs** | `uvicorn app.api:app --reload --port 8000` | Web API (`http://127.0.0.1:8000/docs`) |
| **Instant Price Predictions + SHAP** | `python predict.py` | Terminal Output |
| **EDA & Modeling Notebooks** | `notebooks/01_EDA.ipynb`, `notebooks/02_Modeling.ipynb` | Pre-executed Notebooks |
| **Charts & Diagnostic Plots** | `figures/*.png` | High-DPI Images |
| **Automated Tests (9 Tests)** | `pytest tests -v` | Terminal Pass/Fail Report |
| **Re-train All 7 Models** | `python -u train.py` | Trains models + updates `artifacts/` & `figures/` |

---

## Step 1: Open PowerShell & Activate the Virtual Environment

Open **Windows PowerShell** inside the project root (`house-price-predictor/`) and run:

```powershell
cd house-price-predictor
.\venv\Scripts\activate
```

You will see `(venv)` appear at the start of your prompt.

> **Troubleshooting (PowerShell Script Policy):**  
> If Windows blocks `.\venv\Scripts\activate` with an execution policy error, run this once:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Then run `.\venv\Scripts\activate` again. *(Alternatively, you can always call `.\venv\Scripts\python.exe` directly without activating.)*

### (Only If Setting Up on a New Computer)
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python -u train.py
```

---

## Step 2: Run CLI Predictions (Point Estimate + 90% Interval + SHAP)

The trained models are already saved in `artifacts/`, so you can predict immediately:

```powershell
python predict.py
```

**What you will see in the terminal:**
- Estimated median house value in `$USD` and `$100k` units for sample San Francisco, Los Angeles, and Fresno properties.
- Calibrated **90% CQR Prediction Interval** (low to high dollar range).
- **Top 5 TreeSHAP Drivers** showing exact `+$` / `-$` dollar impact of each feature.

### Predict for Your Own Custom Property
Pass a JSON object using `--json`:
```powershell
python predict.py --json '{"MedInc": 9.0, "HouseAge": 25, "AveRooms": 6.5, "AveBedrms": 1.1, "Population": 800, "AveOccup": 2.4, "Latitude": 37.77, "Longitude": -122.42}'
```

---

## Step 3: Launch the Interactive Streamlit Dashboard

Run:
```powershell
streamlit run app/dashboard.py
```

**What happens:**
1. Your browser automatically opens **`http://localhost:8501`**.
2. Use the **left sidebar** to select a California city preset (San Francisco, Silicon Valley, Los Angeles, San Diego, Fresno) or adjust sliders manually.
3. Explore the **3 tabs**:
   - **Live Valuation & SHAP**: Shows predicted value, 90% prediction interval, interactive California map, and local TreeSHAP bar chart.
   - **Model Benchmarks & Slices**: Shows the 7-model benchmark table and subgroup slice error table.
   - **Global Diagnostics**: Displays all saved diagnostic, CQR interval, and SHAP summary plots.
4. To stop the dashboard, press **`Ctrl + C`** in PowerShell.

---

## Step 4: Launch the FastAPI REST Microservice

Run:
```powershell
uvicorn app.api:app --reload --port 8000
```

**How to view and test the API:**
1. Open **`http://127.0.0.1:8000/docs`** in your browser for interactive Swagger UI.
2. Test these endpoints:
   - `GET /health` — Check API status and loaded champion model.
   - `GET /model-info` — View training metadata, 5-fold CV metrics, test metrics, and slice errors.
   - `POST /predict` — Click **Try it out** → **Execute** to get JSON valuation + 90% CQR interval + top 5 SHAP drivers.
   - `POST /predict/batch` — Score multiple property records in a single call (up to 500 records).

### Test the API from Another PowerShell Window
```powershell
$body = @{
    MedInc = 8.3252
    HouseAge = 41.0
    AveRooms = 6.984127
    AveBedrms = 1.02381
    Population = 322.0
    AveOccup = 2.555556
    Latitude = 37.88
    Longitude = -122.23
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 5
```
To stop the API server, press **`Ctrl + C`** in the PowerShell window running `uvicorn`.

---

## Step 5: View Jupyter Notebooks & Saved Visualizations

### Option A: Open Directly in Your IDE
Click and open these pre-executed files in VS Code / Antigravity / Cursor:
- `notebooks/01_EDA.ipynb` — Dataset audit, censoring inspection, correlations, coastal plots.
- `notebooks/02_Modeling.ipynb` — 7-model benchmarking, CQR intervals, slice errors, and SHAP explanations.
- `figures/` — Contains all generated `.png` charts:
  - `model_benchmark_comparison.png`
  - `actual_vs_predicted.png`
  - `conformal_prediction_intervals.png`
  - `slice_error_analysis.png`
  - `shap_summary.png`
  - `feature_importance.png`
  - `target_distribution.png`
  - `correlation_heatmap.png`
  - `geographic_distribution.png`
  - `medinc_vs_medhouseval.png`

### Option B: Launch Jupyter in Your Browser
```powershell
jupyter notebook
```
Then navigate to the `notebooks/` folder in the browser tab.

---

## Step 6: Run the Automated Test Suite (`pytest`)

Run all 9 unit and integration tests (feature engineering, pipeline serialization, FastAPI endpoints, and Streamlit UI rendering):

```powershell
pytest tests -v
```

Expected result: **`9 passed`**.

---

## Step 7: Re-Train the Full Pipeline From Scratch (Optional)

If you modify features, models, or tuning settings and want to regenerate `artifacts/` and `figures/`:

```powershell
python -u train.py
```

Optional flags:
- `--trials 15` — Number of Optuna hyperparameter search trials per boosted model (default: `15`).
- `--bootstrap 1000` — Number of bootstrap resamples for 95% confidence intervals (default: `1000`).

Example:
```powershell
python -u train.py --trials 25 --bootstrap 1000
```

---

## Step 8: Run Inside Docker (Optional)

If you have Docker Desktop installed, you can containerize and run the FastAPI inference service:

```powershell
docker build -t california-house-price-api .
docker run -p 8000:8000 california-house-price-api
```
Then open `http://127.0.0.1:8000/docs`.
