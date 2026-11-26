"""Unit tests for geospatial and domain feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.config import CALIFORNIA_HUBS, RAW_FEATURES
from src.features import CaliforniaHousingFeatureEngineer, haversine_distance_km


def _sample_df(n: int = 40) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(123)
    df = pd.DataFrame(
        {
            "MedInc": rng.uniform(1.5, 10.0, n),
            "HouseAge": rng.uniform(5.0, 50.0, n),
            "AveRooms": rng.uniform(3.5, 8.5, n),
            "AveBedrms": rng.uniform(0.9, 1.5, n),
            "Population": rng.uniform(200.0, 3000.0, n),
            "AveOccup": rng.uniform(1.8, 4.5, n),
            "Latitude": rng.uniform(32.8, 38.5, n),
            "Longitude": rng.uniform(-122.5, -117.0, n),
        }
    )
    y = 0.45 * df["MedInc"].to_numpy() + rng.normal(0.0, 0.2, n)
    return df, y


def test_haversine_sf_to_la():
    sf_lat, sf_lon = CALIFORNIA_HUBS["SF"]
    la_lat, la_lon = CALIFORNIA_HUBS["LA"]
    dist = float(haversine_distance_km(sf_lat, sf_lon, la_lat, la_lon))
    # Actual great-circle distance between SF and LA is ~559 km
    assert 550.0 < dist < 570.0


def test_feature_engineer_no_mutation_and_no_nans():
    df, y = _sample_df(40)
    original_copy = df.copy()

    fe = CaliforniaHousingFeatureEngineer(
        include_spatial_comps=True,
        spatial_knn_neighbors=5,
        geo_clusters=5,
        oof_folds=4,
    )
    out = fe.fit_transform(df, y)

    pd.testing.assert_frame_equal(df, original_copy)
    assert list(out.columns) == list(fe.get_feature_names_out())
    assert len(out) == len(df)
    assert out.shape[1] > len(RAW_FEATURES)
    assert not out.isnull().any().any()
    assert np.isfinite(out.to_numpy()).all()


def test_feature_engineer_prevents_manual_fit_then_transform_leakage():
    df, y = _sample_df(50)
    fe = CaliforniaHousingFeatureEngineer(
        include_spatial_comps=True,
        spatial_knn_neighbors=5,
        geo_clusters=5,
        oof_folds=5,
    )
    out_fit_transform = fe.fit_transform(df, y)
    out_manual_transform = fe.transform(df)
    pd.testing.assert_frame_equal(out_fit_transform, out_manual_transform)


def test_feature_engineer_raises_on_missing_columns():
    df, y = _sample_df(20)
    fe = CaliforniaHousingFeatureEngineer(geo_clusters=4, oof_folds=3)
    with pytest.raises(ValueError, match="Missing required input columns"):
        fe.fit(df.drop(columns=["Latitude"]), y)
