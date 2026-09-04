from fastapi import APIRouter, HTTPException, status
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.prediction_service import prediction_service
from app.core.logger import logger

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict House Price",
    description="Calculates estimated property valuation based on location, area, BHK, bathrooms, age, and amenities."
)
def predict_house_price(request: PredictionRequest):
    try:
        logger.info(f"Incoming prediction request: area={request.area}, bhk={request.bhk}, location={request.location}")
        response = prediction_service.predict(request)
        logger.info(f"Prediction result: {response.predicted_price_formatted} ({response.property_category})")
        return response
    except ValueError as ve:
        logger.warning(f"Validation error in prediction: {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Prediction inference failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while estimating property price: {str(e)}"
        )
