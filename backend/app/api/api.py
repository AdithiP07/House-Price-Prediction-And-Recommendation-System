from fastapi import APIRouter
from app.api.endpoints import predict, recommend, analytics

api_router = APIRouter()

# Prediction router
api_router.include_router(predict.router, tags=["Prediction"])

# Recommendation router
api_router.include_router(recommend.router, tags=["Recommendation"])

# Analytics & Utility router
api_router.include_router(analytics.router, tags=["Analytics & Health"])
