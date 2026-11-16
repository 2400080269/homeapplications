# California House Price Predictor

An end-to-end Machine Learning regression project predicting California median home values using **Multiple Linear Regression**, **Random Forest**, and **XGBoost**.

---

## 📊 Summary of Model Performance

All models were evaluated on an identical 20% test split ($N = 4,128$) using features scaled strictly with `StandardScaler` fitted on the training split ($N = 16,512$):

| Model | RMSE ($100k) | MAE ($100k) | $R^2$ Score | Rank |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost Regressor** | **0.4718** (~$47,180) | **0.3096** (~$30,960) | **0.8301** | 🥇 **Best (Champion)** |
| **Random Forest Regressor** | **0.5059** (~$50,590) | **0.3278** (~$32,780) | **0.8047** | 🥈 Runner-up |
| **Multiple Linear Regression (Baseline)** | **0.7456** (~$74,560) | **0.5332** (~$53,320) | **0.5758** | 🥉 Baseline |

---

## 📁 Project Structure

```
house-price-predictor/
├── data/                                # Cached California housing dataset
├── figures/                             # Generated plots & visual charts
│   ├── actual_vs_predicted.png          # Comparison of model predictions vs actual values
│   ├── feature_importance.png           # Feature importance for Random Forest & XGBoost
│   ├── target_distribution.png          # Histogram and boxplot of MedHouseVal
│   ├── correlation_heatmap.png          # Feature correlation matrix
│   ├── medinc_vs_medhouseval.png        # Scatter plot of Median Income vs House Value
│   └── geographic_distribution.png      # California price distribution by lat/long coordinates
├── models/                              # Saved models backup directory
├── venv/                                # Isolated Python 3.11 virtual environment
├── 01_EDA.ipynb                         # Jupyter notebook for Exploratory Data Analysis
├── 02_Modeling.ipynb                    # Jupyter notebook for Model Benchmarking & Evaluation
├── train.py                             # Complete training, evaluation, and export pipeline
├── predict.py                           # Inference script for making predictions on new data
├── generate_notebooks.py                # Script to regenerate notebooks
├── run_eda.py                           # Script to generate EDA figures
├── requirements.txt                     # Pinned project dependencies
├── house_price_model.pkl                # Serialized champion XGBoost model
├── scaler.pkl                           # Serialized StandardScaler
└── .gitignore                           # Git ignore rules for venv and checkpoints
```

---

## 🚀 How to Access and View Outputs

### 1. Directly in the IDE / File Explorer
You can open and view the generated figures and notebooks directly:
- **Plots**: Open any image in `figures/` (e.g. `actual_vs_predicted.png`, `correlation_heatmap.png`).
- **Notebooks**: Open `01_EDA.ipynb` or `02_Modeling.ipynb` in your IDE notebook viewer.

### 2. View Training & Evaluation Output in the Terminal
To re-run training and view the live benchmark table in the terminal:
```powershell
cd "house-price-predictor"
.\venv\Scripts\activate
python train.py
```

### 3. Run Inference on Custom Inputs
To generate house price predictions for new property profiles using the saved model:
```powershell
python predict.py
```

### 4. Interactive Jupyter Notebook in Browser
To launch the Jupyter web interface:
```powershell
cd "house-price-predictor"
.\venv\Scripts\activate
jupyter notebook
```
Then click on `01_EDA.ipynb` or `02_Modeling.ipynb` in the browser tab.
