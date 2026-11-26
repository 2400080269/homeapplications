"""FastAPI REST microservice for California House Price Prediction."""

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src import __version__
from src.inference import HousePricePredictor


class HousingFeatures(BaseModel):
    """Input schema matching California Census block group features."""

    MedInc: float = Field(..., gt=0.0, le=30.0, description="Median income ($10k units)", examples=[8.3252])
    HouseAge: float = Field(..., ge=1.0, le=120.0, description="Median house age (years)", examples=[41.0])
    AveRooms: float = Field(..., gt=0.0, le=200.0, description="Average rooms per household", examples=[6.9841])
    AveBedrms: float = Field(..., gt=0.0, le=50.0, description="Average bedrooms per household", examples=[1.0238])
    Population: float = Field(..., ge=1.0, le=50000.0, description="Block group population", examples=[322.0])
    AveOccup: float = Field(..., gt=0.0, le=2000.0, description="Average occupancy per household", examples=[2.5556])
    Latitude: float = Field(..., ge=32.0, le=42.5, description="Block group latitude", examples=[37.88])
    Longitude: float = Field(..., ge=-125.0, le=-114.0, description="Block group longitude", examples=[-122.23])


class BatchPredictionRequest(BaseModel):
    """Batch prediction request payload."""

    records: List[HousingFeatures] = Field(..., min_length=1, max_length=500)
    include_shap: bool = Field(default=False, description="Include per-sample SHAP feature drivers")


class FeatureDriver(BaseModel):
    """Local SHAP attribution for one feature."""

    feature: str
    feature_value: float
    shap_impact_100k: float
    shap_impact_usd: float


class PredictionResponse(BaseModel):
    """Prediction payload with calibrated 90% prediction interval and SHAP drivers."""

    predicted_value_100k: float
    predicted_price_usd: float
    interval_90_low_100k: float
    interval_90_high_100k: float
    interval_90_low_usd: float
    interval_90_high_usd: float
    model_name: str
    shap_base_value_100k: Optional[float] = None
    top_drivers: Optional[List[FeatureDriver]] = None


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    count: int
    predictions: List[PredictionResponse]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts once during FastAPI startup."""
    try:
        app.state.predictor = HousePricePredictor()
    except FileNotFoundError:
        app.state.predictor = None
    yield


app = FastAPI(
    title="California House Price Predictor API",
    description=(
        "Production ML inference API powered by leak-free geospatial feature engineering, "
        "Optuna-tuned gradient boosting/stacking, Conformalized Quantile Regression (90% CQR intervals), "
        "and real-time TreeSHAP explanations."
    ),
    version=__version__,
    lifespan=lifespan,
)


def get_predictor(request: Request) -> HousePricePredictor:
    """Dependency providing the loaded predictor or raising HTTP 503 if uninitialized."""
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        try:
            predictor = HousePricePredictor()
            request.app.state.predictor = predictor
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return predictor


@app.get("/health")
def health_check(request: Request) -> Dict[str, Any]:
    """Liveness and readiness check."""
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        try:
            predictor = HousePricePredictor()
            request.app.state.predictor = predictor
        except FileNotFoundError:
            return {"status": "degraded", "model_loaded": False, "version": __version__}
    return {
        "status": "healthy",
        "model_loaded": True,
        "champion_model": predictor.champion_name,
        "version": __version__,
    }


@app.get("/model-info")
def model_info(predictor: HousePricePredictor = Depends(get_predictor)) -> Dict[str, Any]:
    """Expose training metadata, CV benchmarks, holdout metrics, and slice evaluations."""
    return predictor.metadata


@app.post("/predict", response_model=PredictionResponse)
def predict_single(
    payload: HousingFeatures,
    include_shap: bool = True,
    predictor: HousePricePredictor = Depends(get_predictor),
) -> PredictionResponse:
    """Predict median house value with 90% CQR uncertainty interval and local SHAP drivers."""
    results = predictor.predict(payload.model_dump(), include_shap=include_shap, top_k_drivers=5)
    return PredictionResponse(**results[0])


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(
    payload: BatchPredictionRequest,
    predictor: HousePricePredictor = Depends(get_predictor),
) -> BatchPredictionResponse:
    """Score up to 500 housing block groups in a single batch call."""
    records = [item.model_dump() for item in payload.records]
    results = predictor.predict(records, include_shap=payload.include_shap, top_k_drivers=5)
    return BatchPredictionResponse(
        count=len(results),
        predictions=[PredictionResponse(**r) for r in results],
    )
