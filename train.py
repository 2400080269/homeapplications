import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb


def save_figure(fig, filename):
    fig.tight_layout()
    file_path = os.path.join("figures", filename)
    fig.savefig(file_path, dpi=200)
    plt.close(fig)
    print(f"  --> Saved: {file_path}")


def main():
    print("=" * 60)
    print("California House Price Prediction Pipeline")
    print("=" * 60)

    # 1. Load Data
    print("\n[Step 1] Loading California Housing Dataset...")
    data_dir = os.path.join(os.path.dirname(__file__), "data") if "__file__" in locals() else "./data"
    os.makedirs(data_dir, exist_ok=True)
    housing = fetch_california_housing(data_home=data_dir, as_frame=True)
    df = housing.frame
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Features: {list(housing.feature_names)}")
    print(f"Target: {housing.target_names[0]} (Median House Value in $100k)")

    # Check for missing values
    missing_counts = df.isnull().sum()
    print(f"Missing values count across features:\n{missing_counts[missing_counts > 0] if missing_counts.sum() > 0 else 'No missing values found.'}")

    # Create directory for figures if it doesn't exist
    os.makedirs("figures", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 2. Preprocessing
    print("\n[Step 2] Splitting Data & Preprocessing...")
    X = df[housing.feature_names]
    y = df[housing.target_names[0]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("StandardScaler fitted on training data and applied to train and test sets.")

    # 3. Model Training & Evaluation
    print("\n[Step 3] Training Models...")

    models = {
        "Multiple Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost Regressor": xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
    }

    results = []
    predictions = {}

    for name, model in models.items():
        print(f"  --> Training {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        predictions[name] = y_pred

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Model": name,
            "RMSE": rmse,
            "MAE": mae,
            "R2 Score": r2
        })

    # 4. Results Comparison
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("Model Evaluation Summary:")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("=" * 60)

    # 5. Determine Best Model
    best_row = results_df.sort_values(by="R2 Score", ascending=False).iloc[0]
    best_name = best_row["Model"]
    best_model = models[best_name]
    print(f"\nBest Performing Model: {best_name} (R2: {best_row['R2 Score']:.4f}, RMSE: {best_row['RMSE']:.4f})")

    # 6. Save Model and Scaler
    model_path = "house_price_model.pkl"
    scaler_path = "scaler.pkl"
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\n[Saved] Best model saved to: {model_path}")
    print(f"[Saved] Scaler saved to: {scaler_path}")

    # Also save inside models/ directory for cleanliness
    joblib.dump(best_model, os.path.join("models", model_path))
    joblib.dump(scaler, os.path.join("models", scaler_path))

    # 7. Generate Evaluation Figures
    print("\n[Step 4] Generating Evaluation Figures in './figures/'...")

    # Residuals and Actual vs Predicted Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, y_pred) in zip(axes, predictions.items()):
        ax.scatter(y_test, y_pred, alpha=0.3, edgecolors='none', color='teal')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        r2_val = results_df.loc[results_df["Model"] == name, "R2 Score"].values[0]
        rmse_val = results_df.loc[results_df["Model"] == name, "RMSE"].values[0]
        ax.set_title(f"{name}\nR2: {r2_val:.3f} | RMSE: {rmse_val:.3f}")
        ax.set_xlabel("Actual Values ($100k)")
        ax.set_ylabel("Predicted Values ($100k)")
        ax.grid(True, linestyle="--", alpha=0.6)
    save_figure(fig, "actual_vs_predicted.png")

    # Feature Importance for Random Forest & XGBoost
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    rf_feat_imp = pd.Series(models["Random Forest Regressor"].feature_importances_, index=housing.feature_names).sort_values(ascending=True)
    rf_feat_imp.plot(kind="barh", ax=axes[0], color="skyblue")
    axes[0].set_title("Random Forest - Feature Importance")
    axes[0].set_xlabel("Importance")

    xgb_feat_imp = pd.Series(models["XGBoost Regressor"].feature_importances_, index=housing.feature_names).sort_values(ascending=True)
    xgb_feat_imp.plot(kind="barh", ax=axes[1], color="salmon")
    axes[1].set_title("XGBoost - Feature Importance")
    axes[1].set_xlabel("Importance")

    save_figure(fig, "feature_importance.png")

    print("\nPipeline execution completed successfully!")

if __name__ == "__main__":
    main()
