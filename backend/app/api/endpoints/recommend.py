from fastapi import APIRouter, HTTPException, status
from app.schemas.recommendation import RecommendationRequest, RecommendationResponse
from app.services.recommendation_service import recommendation_service
from app.core.logger import logger

router = APIRouter()


@router.post(
    "/recommend",
    response_model=RecommendationResponse,
    status_code=status.HTTP_200_OK,
    summary="Recommend Similar Properties",
    description="Recommends the top 5 most similar and desirable properties matched against budget, location, BHK, and amenities."
)
def recommend_properties(request: RecommendationRequest):
    try:
        logger.info(
            f"Incoming recommendation query: budget={request.budget}, "
            f"loc={request.location}, bhk={request.bhk}, amenities={request.amenities}"
        )
        response = recommendation_service.recommend(request)
        logger.info(f"Generated {response.total_recommended} recommendations successfully.")
        return response
    except Exception as e:
        logger.error(f"Recommendation generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while generating property recommendations: {str(e)}"
        )
