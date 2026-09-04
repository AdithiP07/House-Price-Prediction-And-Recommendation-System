import os
import json
from fastapi import APIRouter, HTTPException, status
from app.core.config import settings
from app.core.logger import logger

router = APIRouter()


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Service Health Check"
)
def health_check():
    model_loaded = os.path.exists(settings.MODEL_PATH)
    catalog_loaded = os.path.exists(settings.CATALOG_PATH)
    return {
        "status": "healthy" if (model_loaded and catalog_loaded) else "degraded",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "artifacts": {
            "house_price_model": model_loaded,
            "property_catalog": catalog_loaded,
        }
    }


@router.get(
    "/analytics",
    status_code=status.HTTP_200_OK,
    summary="Model Performance & Market Analytics",
    description="Returns cross-model benchmark metrics, feature importance rankings, and dataset distribution statistics."
)
def get_analytics():
    if not os.path.exists(settings.METRICS_PATH):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analytics metrics file not found. Please train models first."
        )
    try:
        with open(settings.METRICS_PATH, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Failed to read analytics metrics: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get(
    "/locations",
    status_code=status.HTTP_200_OK,
    summary="Available Locations List"
)
def get_locations():
    if os.path.exists(settings.METRICS_PATH):
        try:
            with open(settings.METRICS_PATH, "r") as f:
                data = json.load(f)
            return {"locations": data.get("locations", [])}
        except Exception:
            pass

    # Default fallback locations
    return {
        "locations": [
            "Banashankari",
            "Bellandur",
            "Electronic City",
            "Hebbal",
            "HSR Layout",
            "Indiranagar",
            "Jayanagar",
            "Koramangala",
            "Malleshwaram",
            "Marathahalli",
            "Rajajinagar",
            "Sarjapur Road",
            "Thanisandra",
            "Whitefield",
            "Yelahanka"
        ]
    }
