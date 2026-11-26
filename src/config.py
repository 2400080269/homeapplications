"""Central configuration and domain constants for the California House Price Predictor."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

MODEL_PIPELINE_PATH = ARTIFACTS_DIR / "house_price_pipeline.joblib"
QUANTILE_LOW_PATH = ARTIFACTS_DIR / "quantile_low_pipeline.joblib"
QUANTILE_HIGH_PATH = ARTIFACTS_DIR / "quantile_high_pipeline.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"

RAW_FEATURES: List[str] = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

TARGET_COLUMN: str = "MedHouseVal"
TARGET_CAP_VALUE: float = 5.0  # California Census right-censoring ceiling (~$500k)
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
CV_FOLDS: int = 5
ALPHA_CONFIDENCE: float = 0.90  # 90% Prediction Interval (5th to 95th percentile)

# Major California Economic Hubs (Latitude, Longitude)
CALIFORNIA_HUBS: Dict[str, Tuple[float, float]] = {
    "SF": (37.7749, -122.4194),        # San Francisco
    "SJ": (37.3382, -121.8863),        # San Jose / Silicon Valley
    "LA": (34.0522, -118.2437),        # Los Angeles
    "SD": (32.7157, -117.1611),        # San Diego
    "SAC": (38.5816, -121.4944),       # Sacramento
}

# Reference points along the California Pacific Coastline (North to South)
CALIFORNIA_COASTLINE_POINTS: List[Tuple[float, float]] = [
    (41.7558, -124.2026),  # Crescent City
    (40.8021, -124.1637),  # Eureka
    (39.4457, -123.8053),  # Fort Bragg
    (38.3333, -123.0481),  # Bodega Bay
    (37.7749, -122.5100),  # SF Ocean Beach
    (37.4636, -122.4286),  # Half Moon Bay
    (36.9741, -122.0308),  # Santa Cruz
    (36.6002, -121.8947),  # Monterey
    (35.3658, -120.8499),  # Morro Bay
    (34.4208, -119.6982),  # Santa Barbara
    (34.0195, -118.4912),  # Santa Monica
    (33.7701, -118.1937),  # Long Beach
    (33.5427, -117.7854),  # Laguna Beach
    (33.1959, -117.3795),  # Oceanside
    (32.7157, -117.2500),  # San Diego Coast
]

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "MedInc": "Median income in block group (tens of thousands of USD)",
    "HouseAge": "Median house age in block group (years)",
    "AveRooms": "Average number of rooms per household",
    "AveBedrms": "Average number of bedrooms per household",
    "Population": "Block group population",
    "AveOccup": "Average number of household members",
    "Latitude": "Block group latitude",
    "Longitude": "Block group longitude",
}


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for model training, tuning, and evaluation."""

    test_size: float = TEST_SIZE
    random_state: int = RANDOM_STATE
    cv_folds: int = CV_FOLDS
    optuna_trials: int = 15
    n_bootstrap: int = 1000
    spatial_knn_neighbors: int = 15
    geo_clusters: int = 20
    include_spatial_comps: bool = True
    feature_bounds: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "MedInc": (0.1, 20.0),
            "HouseAge": (1.0, 100.0),
            "AveRooms": (0.5, 50.0),
            "AveBedrms": (0.2, 15.0),
            "Population": (1.0, 50000.0),
            "AveOccup": (0.5, 25.0),
            "Latitude": (32.0, 42.5),
            "Longitude": (-125.0, -114.0),
        }
    )
