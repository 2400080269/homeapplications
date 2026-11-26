"""CLI inference script showing point predictions, 90% CQR intervals, and local SHAP drivers."""

import argparse
import json
from typing import Any, Dict, List

from src.inference import HousePricePredictor

DEFAULT_SAMPLES: List[Dict[str, Any]] = [
    {
        "Label": "San Francisco (High Income Coastal)",
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984127,
        "AveBedrms": 1.023810,
        "Population": 322.0,
        "AveOccup": 2.555556,
        "Latitude": 37.88,
        "Longitude": -122.23,
    },
    {
        "Label": "Los Angeles (Urban Middle Income)",
        "MedInc": 4.2500,
        "HouseAge": 38.0,
        "AveRooms": 5.300000,
        "AveBedrms": 1.060000,
        "Population": 1200.0,
        "AveOccup": 2.850000,
        "Latitude": 34.05,
        "Longitude": -118.24,
    },
    {
        "Label": "Fresno (Central Valley Inland)",
        "MedInc": 1.8500,
        "HouseAge": 25.0,
        "AveRooms": 4.450000,
        "AveBedrms": 1.050000,
        "Population": 1450.0,
        "AveOccup": 3.400000,
        "Latitude": 36.74,
        "Longitude": -119.79,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run house price inference with CQR uncertainty and TreeSHAP attributions."
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional JSON string representing one property dict or list of dicts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = HousePricePredictor()

    if args.json:
        records = json.loads(args.json)
        if isinstance(records, dict):
            records = [records]
        labels = [f"Custom Input #{i + 1}" for i in range(len(records))]
    else:
        labels = [item["Label"] for item in DEFAULT_SAMPLES]
        records = [{k: v for k, v in item.items() if k != "Label"} for item in DEFAULT_SAMPLES]

    results = predictor.predict(records, include_shap=True, top_k_drivers=5)

    print("=" * 78)
    print(f"California House Price Predictor — Inference ({predictor.champion_name})")
    print("=" * 78)

    for label, res in zip(labels, results):
        print(f"\n[{label}]")
        print(
            f"  Estimated Value:   ${res['predicted_price_usd']:,.2f} "
            f"({res['predicted_value_100k']:.3f} in $100k)"
        )
        print(
            f"  90% CQR Interval:  [${res['interval_90_low_usd']:,.2f}  --  "
            f"${res['interval_90_high_usd']:,.2f}]"
        )
        if res.get("top_drivers"):
            print("  Top 5 TreeSHAP Drivers:")
            for drv in res["top_drivers"]:
                val = drv["shap_impact_usd"]
                sign = "+" if val >= 0 else "-"
                print(
                    f"    - {drv['feature']:<22s} (val={drv['feature_value']:>9.3f}) -> "
                    f"{sign}${abs(val):,.2f}"
                )
    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
