"""Domain-driven & geospatial feature engineering transformer (leak-free)."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

from src.config import (
    CALIFORNIA_COASTLINE_POINTS,
    CALIFORNIA_HUBS,
    RANDOM_STATE,
    RAW_FEATURES,
)


def haversine_distance_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Compute vectorized great-circle Haversine distance in kilometers."""
    earth_radius_km = 6371.0
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return earth_radius_km * c


class CaliforniaHousingFeatureEngineer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer for domain & geospatial feature engineering.

    Combines:
    1. Structural housing ratios and log-compressed right-tail volume features.
    2. Haversine distances to California's primary economic hubs and Pacific coastline.
    3. Rotated spatial coordinate projections aligned with California's NW-SE geography.
    4. Unsupervised spatial K-Means cluster proximity (fitted strictly on X_train).
    5. Out-of-fold Haversine KNN comparable neighborhood price & volatility proxies
       (computed strictly out-of-fold on training rows to prevent target leakage).
    """

    def __init__(
        self,
        include_spatial_comps: bool = True,
        spatial_knn_neighbors: int = 15,
        geo_clusters: int = 20,
        oof_folds: int = 5,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.include_spatial_comps = include_spatial_comps
        self.spatial_knn_neighbors = spatial_knn_neighbors
        self.geo_clusters = geo_clusters
        self.oof_folds = oof_folds
        self.random_state = random_state

    def _validate_input_df(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            missing = [col for col in RAW_FEATURES if col not in X.columns]
            if missing:
                raise ValueError(f"Missing required input columns: {missing}")
            return X[RAW_FEATURES].astype(np.float64).copy()
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != len(RAW_FEATURES):
            raise ValueError(
                f"Expected array of shape (n_samples, {len(RAW_FEATURES)}), got {arr.shape}"
            )
        return pd.DataFrame(arr, columns=RAW_FEATURES)

    @staticmethod
    def _compute_weighted_knn_stats(
        distances_rad: np.ndarray,
        neighbor_targets: np.ndarray,
        global_fallback: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute inverse-distance weighted mean and dispersion of neighbor prices."""
        eps = 1e-5
        weights = 1.0 / (distances_rad + eps)
        weight_sums = weights.sum(axis=1, keepdims=True)
        norm_weights = np.where(weight_sums > 0, weights / weight_sums, 1.0 / weights.shape[1])

        weighted_mean = np.sum(norm_weights * neighbor_targets, axis=1)
        # Replace any numerical edge case with global fallback
        weighted_mean = np.where(np.isfinite(weighted_mean), weighted_mean, global_fallback)

        diff_sq = (neighbor_targets - weighted_mean[:, None]) ** 2
        weighted_std = np.sqrt(np.sum(norm_weights * diff_sq, axis=1))
        weighted_std = np.where(np.isfinite(weighted_std), weighted_std, 0.0)
        return weighted_mean, weighted_std

    def _fit_spatial_comps(self, coords_rad: np.ndarray, y_arr: np.ndarray) -> None:
        """Fit full training spatial index and precompute strict OOF training comps."""
        n_samples = len(coords_rad)
        n_neighbors = min(self.spatial_knn_neighbors, max(1, n_samples - 1))
        self.global_target_mean_ = float(np.mean(y_arr))
        self.y_train_ref_ = y_arr.copy()

        # Fit full neighbor index for unseen validation/test/production data
        self.spatial_nn_ = NearestNeighbors(n_neighbors=n_neighbors, metric="haversine")
        self.spatial_nn_.fit(coords_rad)

        # Compute strict Out-Of-Fold spatial comps for training rows so no row uses itself
        oof_mean = np.full(n_samples, self.global_target_mean_, dtype=np.float64)
        oof_std = np.zeros(n_samples, dtype=np.float64)

        effective_folds = min(self.oof_folds, n_samples)
        if effective_folds >= 2:
            kf = KFold(n_splits=effective_folds, shuffle=True, random_state=self.random_state)
            for tr_idx, val_idx in kf.split(coords_rad):
                fold_k = min(self.spatial_knn_neighbors, len(tr_idx))
                fold_nn = NearestNeighbors(n_neighbors=fold_k, metric="haversine")
                fold_nn.fit(coords_rad[tr_idx])
                dists, nn_idx = fold_nn.kneighbors(coords_rad[val_idx])
                neighbor_y = y_arr[tr_idx][nn_idx]
                f_mean, f_std = self._compute_weighted_knn_stats(
                    dists, neighbor_y, self.global_target_mean_
                )
                oof_mean[val_idx] = f_mean
                oof_std[val_idx] = f_std

        self.train_oof_comps_mean_ = oof_mean
        self.train_oof_comps_std_ = oof_std

    def fit(self, X: pd.DataFrame | np.ndarray, y: Optional[pd.Series | np.ndarray] = None):
        """Fit spatial clustering and optional out-of-fold spatial comps on training data."""
        df = self._validate_input_df(X)
        coords_deg = df[["Latitude", "Longitude"]].to_numpy()
        coords_rad = np.radians(coords_deg)

        effective_clusters = min(self.geo_clusters, len(df))
        self.kmeans_ = KMeans(
            n_clusters=effective_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        self.kmeans_.fit(coords_deg)

        self._fit_coords_rad_ = coords_rad.copy()
        self._n_fit_samples_ = len(df)
        self.has_spatial_comps_ = False

        if self.include_spatial_comps and y is not None:
            y_arr = np.asarray(y, dtype=np.float64)
            self._fit_spatial_comps(coords_rad, y_arr)
            self.has_spatial_comps_ = True

        self.feature_names_out_ = self._build_feature_names()
        return self

    def _build_feature_names(self) -> List[str]:
        names = list(RAW_FEATURES) + [
            "BedroomsPerRoom",
            "PeoplePerRoom",
            "HouseholdsEst",
            "IncomePerRoom",
            "IncomePerPerson",
            "LogPopulation",
            "LogAveOccup",
            "DistSFKm",
            "DistSJKm",
            "DistLAKm",
            "DistSDKm",
            "DistSacKm",
            "MinDistToHubKm",
            "MinDistToCoastKm",
            "GeoRot30X",
            "GeoRot30Y",
            "GeoRot45X",
            "GeoRot45Y",
            "GeoClusterMinDistKm",
        ]
        if self.has_spatial_comps_:
            names.extend(["SpatialCompsPrice", "SpatialCompsPriceStd"])
        return names

    def _transform_core(
        self,
        df: pd.DataFrame,
        use_precomputed_oof: bool,
    ) -> pd.DataFrame:
        out = df.copy()

        safe_rooms = np.clip(out["AveRooms"].to_numpy(), 0.5, None)
        safe_occup = np.clip(out["AveOccup"].to_numpy(), 0.5, None)
        med_inc = out["MedInc"].to_numpy()
        pop = np.maximum(out["Population"].to_numpy(), 0.0)

        # 1. Structural & Economic Ratios
        out["BedroomsPerRoom"] = out["AveBedrms"].to_numpy() / safe_rooms
        out["PeoplePerRoom"] = out["AveOccup"].to_numpy() / safe_rooms
        out["HouseholdsEst"] = pop / safe_occup
        out["IncomePerRoom"] = med_inc / safe_rooms
        out["IncomePerPerson"] = med_inc / safe_occup
        out["LogPopulation"] = np.log1p(pop)
        out["LogAveOccup"] = np.log1p(np.maximum(out["AveOccup"].to_numpy(), 0.0))

        # 2. Haversine Distances to Major Economic Hubs
        lat = out["Latitude"].to_numpy()
        lon = out["Longitude"].to_numpy()

        hub_dists = []
        for hub_key, (hub_lat, hub_lon) in CALIFORNIA_HUBS.items():
            col_name = f"Dist{hub_key.capitalize()}Km" if hub_key != "SAC" else "DistSacKm"
            if hub_key in {"SF", "SJ", "LA", "SD"}:
                col_name = f"Dist{hub_key}Km"
            dist_km = haversine_distance_km(lat, lon, hub_lat, hub_lon)
            out[col_name] = dist_km
            hub_dists.append(dist_km)

        out["MinDistToHubKm"] = np.min(np.vstack(hub_dists), axis=0)

        # 3. Minimum Haversine Distance to Pacific Coastline
        coast_dists = [
            haversine_distance_km(lat, lon, c_lat, c_lon)
            for c_lat, c_lon in CALIFORNIA_COASTLINE_POINTS
        ]
        out["MinDistToCoastKm"] = np.min(np.vstack(coast_dists), axis=0)

        # 4. Rotated Geospatial Coordinate Projections
        theta30 = np.radians(30.0)
        theta45 = np.radians(45.0)
        out["GeoRot30X"] = lon * np.cos(theta30) + lat * np.sin(theta30)
        out["GeoRot30Y"] = -lon * np.sin(theta30) + lat * np.cos(theta30)
        out["GeoRot45X"] = lon * np.cos(theta45) + lat * np.sin(theta45)
        out["GeoRot45Y"] = -lon * np.sin(theta45) + lat * np.cos(theta45)

        # 5. Spatial Micro-Cluster Proximity (km to nearest KMeans centroid)
        coords_deg = np.column_stack([lat, lon])
        nearest_cluster = self.kmeans_.predict(coords_deg)
        centroids = self.kmeans_.cluster_centers_[nearest_cluster]
        out["GeoClusterMinDistKm"] = haversine_distance_km(
            lat, lon, centroids[:, 0], centroids[:, 1]
        )

        # 6. Spatial Comparable Neighborhood Price Features
        if self.has_spatial_comps_:
            if use_precomputed_oof:
                out["SpatialCompsPrice"] = self.train_oof_comps_mean_
                out["SpatialCompsPriceStd"] = self.train_oof_comps_std_
            else:
                coords_rad = np.radians(coords_deg)
                dists, nn_idx = self.spatial_nn_.kneighbors(coords_rad)
                neighbor_y = self.y_train_ref_[nn_idx]
                c_mean, c_std = self._compute_weighted_knn_stats(
                    dists, neighbor_y, self.global_target_mean_
                )
                out["SpatialCompsPrice"] = c_mean
                out["SpatialCompsPriceStd"] = c_std

        return out[self.feature_names_out_]

    def fit_transform(
        self, X: pd.DataFrame | np.ndarray, y: Optional[pd.Series | np.ndarray] = None
    ) -> pd.DataFrame:
        """Fit transformer and return leak-free OOF transformed training features."""
        self.fit(X, y)
        df = self._validate_input_df(X)
        return self._transform_core(df, use_precomputed_oof=self.has_spatial_comps_)

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Transform features on unseen data (or exact training data without leakage)."""
        if not hasattr(self, "kmeans_"):
            raise RuntimeError("Transformer must be fitted before calling transform().")

        df = self._validate_input_df(X)
        use_oof = False
        if self.has_spatial_comps_ and len(df) == self._n_fit_samples_:
            coords_rad = np.radians(df[["Latitude", "Longitude"]].to_numpy())
            if np.array_equal(coords_rad, self._fit_coords_rad_):
                use_oof = True

        return self._transform_core(df, use_precomputed_oof=use_oof)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return output feature names for scikit-learn Pipeline compatibility."""
        if not hasattr(self, "feature_names_out_"):
            raise RuntimeError("Transformer must be fitted first.")
        return np.asarray(self.feature_names_out_, dtype=object)
