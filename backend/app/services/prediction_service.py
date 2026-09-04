import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

from app.core.config import settings
from app.core.logger import logger
from app.schemas.prediction import PredictionRequest, PredictionResponse


PREMIUM_AMENITY_WEIGHTS = {
    "Swimming Pool": 1.5,
    "Clubhouse": 1.2,
    "Tennis Court": 1.4,
    "Gym": 1.0,
    "High-Speed Elevators": 1.0,
    "Landscaped Garden": 0.9,
    "24/7 Security": 0.8,
    "Power Backup": 0.8,
    "Covered Parking": 0.8,
    "Children's Play Area": 0.7,
    "Intercom": 0.5,
    "Jogging Track": 0.6,
}


def calculate_luxury_score(amenities_str: str) -> float:
    if not amenities_str or not str(amenities_str).strip():
        return 1.0
    items = [a.strip() for a in str(amenities_str).split(",")]
    raw_score = sum(PREMIUM_AMENITY_WEIGHTS.get(item, 0.5) for item in items)
    normalized = min(10.0, max(1.0, (raw_score / 11.2) * 10.0))
    return round(normalized, 2)


def format_inr(amount: float) -> str:
    if amount >= 10000000:
        return f"₹{amount / 10000000:.2f} Cr"
    else:
        return f"₹{amount / 100000:.2f} Lakhs"


class PredictionService:
    def __init__(self):
        self.model = None
        self.kmeans_bundle = None
        self._load_artifacts()

    def _load_artifacts(self):
        try:
            if os.path.exists(settings.MODEL_PATH):
                self.model = joblib.load(settings.MODEL_PATH)
                logger.info(f"Loaded house price model successfully from {settings.MODEL_PATH}")
            else:
                logger.warning(f"Model artifact not found at {settings.MODEL_PATH}")

            if os.path.exists(settings.KMEANS_PATH):
                self.kmeans_bundle = joblib.load(settings.KMEANS_PATH)
                logger.info("Loaded K-Means clustering artifacts successfully")
        except Exception as e:
            logger.error(f"Error loading model artifacts: {e}", exc_info=True)

    def predict(self, req: PredictionRequest) -> PredictionResponse:
        if self.model is None:
            self._load_artifacts()
            if self.model is None:
                raise RuntimeError("Prediction model artifact is not available. Please train models first.")

        # Compute engineered features
        lux_score = calculate_luxury_score(req.amenities)

        # Prepare single-row DataFrame for pipeline input
        input_data = {
            "Area": float(req.area),
            "BHK": int(req.bhk),
            "Bathrooms": int(req.bathrooms),
            "Property Age": int(req.property_age),
            "luxury_score": float(lux_score),
            "Location": req.location,
            "Furnishing Status": req.furnishing_status or "Semi-Furnished",
        }
        df_input = pd.DataFrame([input_data])

        # Inference
        raw_pred = float(self.model.predict(df_input)[0])
        predicted_price = max(500000.0, round(raw_pred, -3))  # Round to nearest 1000

        # Calculate price per sqft
        price_per_sqft = round(predicted_price / req.area, 2)

        # Segment into property category using K-Means if available, else fallback
        category = "Mid-Range"
        if self.kmeans_bundle is not None:
            try:
                scaler = self.kmeans_bundle["scaler"]
                kmeans = self.kmeans_bundle["kmeans"]
                labels_map = self.kmeans_bundle["labels_map"]
                feat = np.array([[req.area, predicted_price, lux_score]])
                feat_scaled = scaler.transform(feat)
                c_id = kmeans.predict(feat_scaled)[0]
                category = labels_map.get(c_id, "Mid-Range")
            except Exception as e:
                logger.warning(f"Could not compute K-Means cluster: {e}")
                if predicted_price > 15000000:
                    category = "Luxury"
                elif predicted_price < 6000000:
                    category = "Budget"

        # Model confidence score
        confidence_score = 0.94 if req.bhk <= 4 and req.area <= 4000 else 0.88

        return PredictionResponse(
            predicted_price=predicted_price,
            predicted_price_formatted=format_inr(predicted_price),
            price_per_sqft=price_per_sqft,
            property_category=category,
            confidence_score=confidence_score,
            inputs={
                "area": req.area,
                "bhk": req.bhk,
                "bathrooms": req.bathrooms,
                "property_age": req.property_age,
                "location": req.location,
                "furnishing_status": req.furnishing_status,
                "luxury_score": lux_score,
            }
        )


prediction_service = PredictionService()
