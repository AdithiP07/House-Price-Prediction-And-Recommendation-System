from typing import List, Optional
from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    budget: float = Field(..., gt=100000, description="Target budget in INR")
    location: Optional[str] = Field(default=None, description="Preferred location or locality")
    bhk: Optional[int] = Field(default=None, ge=1, le=10, description="Preferred BHK configuration")
    amenities: Optional[List[str]] = Field(
        default_factory=list,
        description="List of desirable amenities (e.g. ['Gym', 'Swimming Pool', '24/7 Security'])"
    )
    top_n: Optional[int] = Field(default=5, ge=1, le=20, description="Number of properties to recommend")

    model_config = {
        "json_schema_extra": {
            "example": {
                "budget": 8500000,
                "location": "Whitefield",
                "bhk": 3,
                "amenities": ["Gym", "Swimming Pool", "24/7 Security", "Clubhouse"],
                "top_n": 5
            }
        }
    }


class RecommendedProperty(BaseModel):
    property_id: str = Field(..., description="Unique property identifier")
    location: str = Field(..., description="Property locality")
    area: float = Field(..., description="Property area in square feet")
    bhk: int = Field(..., description="Number of bedrooms")
    bathrooms: int = Field(..., description="Number of bathrooms")
    property_age: int = Field(..., description="Age in years")
    furnishing_status: str = Field(..., description="Furnishing status")
    amenities: str = Field(..., description="List of amenities")
    price: float = Field(..., description="Property price in INR")
    price_formatted: str = Field(..., description="Formatted price (Lakhs/Crores)")
    price_per_sqft: float = Field(..., description="Price per square foot")
    property_category: str = Field(..., description="Market segment: Budget, Mid-Range, Luxury")
    similarity_score: float = Field(..., description="Matching similarity percentage (0-100%)")
    match_reasons: List[str] = Field(..., description="Key drivers for recommendation match")


class RecommendationResponse(BaseModel):
    total_recommended: int = Field(..., description="Number of recommended properties returned")
    query_criteria: dict = Field(..., description="Echo of user search preferences")
    recommendations: List[RecommendedProperty] = Field(..., description="Top ranked similar properties")
