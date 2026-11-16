"""
Inference script demonstrating how to load the trained XGBoost model and StandardScaler
to predict California median house prices for custom input features.
"""

import os
import joblib
import pandas as pd
import numpy as np

def predict_house_price(custom_inputs=None):
    # 1. Paths
    model_path = "house_price_model.pkl"
    scaler_path = "scaler.pkl"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model or scaler not found. Please run train.py first.")

    # 2. Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # 3. Default sample input if none provided
    # Features: ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
    if custom_inputs is None:
        custom_inputs = pd.DataFrame([
            {
                "MedInc": 8.3252,       # $83,252 median income
                "HouseAge": 41.0,       # 41 years old
                "AveRooms": 6.984127,   # ~7 rooms avg
                "AveBedrms": 1.023810,  # ~1 bedroom avg
                "Population": 322.0,    # 322 residents
                "AveOccup": 2.555556,   # ~2.6 people per household
                "Latitude": 37.88,      # Bay Area coordinates
                "Longitude": -122.23
            },
            {
                "MedInc": 3.8462,       # $38,462 median income
                "HouseAge": 52.0,       # 52 years old
                "AveRooms": 6.281853,
                "AveBedrms": 1.081081,
                "Population": 565.0,
                "AveOccup": 2.181467,
                "Latitude": 37.85,
                "Longitude": -122.25
            },
            {
                "MedInc": 1.6875,       # $16,875 median income (lower income area)
                "HouseAge": 25.0,
                "AveRooms": 4.120000,
                "AveBedrms": 1.050000,
                "Population": 1150.0,
                "AveOccup": 3.200000,
                "Latitude": 36.50,
                "Longitude": -119.50
            }
        ])

    print("=" * 70)
    print("California House Price Prediction Inference")
    print("=" * 70)
    print("\nInput Housing Profiles:")
    print(custom_inputs.to_string(index=True))

    # 4. Scale inputs using the pre-fitted scaler
    scaled_inputs = scaler.transform(custom_inputs)

    # 5. Predict using the trained model
    predicted_val_100k = model.predict(scaled_inputs)

    # Convert from $100k units to actual USD
    predicted_usd = predicted_val_100k * 100000

    results = custom_inputs.copy()
    results["Predicted_MedHouseVal ($100k)"] = np.round(predicted_val_100k, 3)
    results["Estimated_Price ($USD)"] = [f"${val:,.2f}" for val in predicted_usd]

    print("\n" + "=" * 70)
    print("Prediction Results:")
    print("=" * 70)
    print(results[["MedInc", "HouseAge", "AveRooms", "Predicted_MedHouseVal ($100k)", "Estimated_Price ($USD)"]].to_string(index=True))
    print("=" * 70)

    return results

if __name__ == "__main__":
    predict_house_price()
