from typing import Optional
from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    area: float = Field(..., gt=150, le=15000, description="Property area in square feet")
    bhk: int = Field(..., ge=1, le=10, description="Number of bedrooms (BHK)")
    bathrooms: int = Field(..., ge=1, le=10, description="Number of bathrooms")
    property_age: int = Field(..., ge=0, le=100, description="Age of the property in years")
    location: str = Field(..., min_length=2, max_length=100, description="Geographical location or locality")
    furnishing_status: Optional[str] = Field(
        default="Semi-Furnished",
        description="Furnishing status: 'Furnished', 'Semi-Furnished', or 'Unfurnished'"
    )
    amenities: Optional[str] = Field(
        default="Gym, Swimming Pool, 24/7 Security, Power Backup, Covered Parking",
        description="Comma-separated list of amenities"
    )

    @field_validator("location")
    @classmethod
    def clean_location(cls, v: str) -> str:
        return v.strip().title()

    @field_validator("furnishing_status")
    @classmethod
    def validate_furnishing(cls, v: Optional[str]) -> str:
        valid_options = ["Furnished", "Semi-Furnished", "Unfurnished"]
        if not v or v.strip().title() not in valid_options:
            return "Semi-Furnished"
        return v.strip().title()

    model_config = {
        "json_schema_extra": {
            "example": {
                "area": 1500,
                "bhk": 3,
                "bathrooms": 2,
                "property_age": 5,
                "location": "Whitefield"
            }
        }
    }


class PredictionResponse(BaseModel):
    predicted_price: float = Field(..., description="Estimated house price in INR")
    predicted_price_formatted: str = Field(..., description="Human-readable price in Lakhs/Crores")
    price_per_sqft: float = Field(..., description="Price per square foot")
    property_category: str = Field(..., description="Category: Budget, Mid-Range, or Luxury")
    confidence_score: float = Field(..., description="Model confidence indicator (0.0 to 1.0)")
    inputs: dict = Field(default_factory=dict, description="Echo of evaluated input features")
