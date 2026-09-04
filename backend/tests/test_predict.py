import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data


def test_locations_endpoint():
    response = client.get("/locations")
    assert response.status_code == 200
    data = response.json()
    assert "locations" in data
    assert len(data["locations"]) > 0
    assert "Whitefield" in data["locations"]


def test_predict_endpoint_valid():
    payload = {
        "area": 1500,
        "bhk": 3,
        "bathrooms": 2,
        "property_age": 5,
        "location": "Whitefield"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Contract validation matching project requirements
    assert "predicted_price" in data
    assert isinstance(data["predicted_price"], (int, float))
    assert data["predicted_price"] > 0
    
    # Metadata fields
    assert "predicted_price_formatted" in data
    assert "price_per_sqft" in data
    assert "property_category" in data
    assert data["property_category"] in ["Budget", "Mid-Range", "Luxury"]
    assert "confidence_score" in data


def test_predict_endpoint_with_amenities():
    payload = {
        "area": 2500,
        "bhk": 4,
        "bathrooms": 4,
        "property_age": 2,
        "location": "Indiranagar",
        "furnishing_status": "Furnished",
        "amenities": "Gym, Swimming Pool, Clubhouse, 24/7 Security, Tennis Court"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_price"] > 0


def test_predict_validation_error():
    # Negative area and invalid bhk
    payload = {
        "area": -500,
        "bhk": 0,
        "bathrooms": 1,
        "property_age": 5,
        "location": "Whitefield"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Pydantic validation failure
