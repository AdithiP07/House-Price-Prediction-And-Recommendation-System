import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app

client = TestClient(app)


def test_recommend_endpoint_valid():
    payload = {
        "budget": 8500000,
        "location": "Whitefield",
        "bhk": 3,
        "amenities": ["Gym", "Swimming Pool", "24/7 Security"],
        "top_n": 5
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "total_recommended" in data
    assert "recommendations" in data
    assert len(data["recommendations"]) == 5
    
    first = data["recommendations"][0]
    assert "property_id" in first
    assert "location" in first
    assert "price" in first
    assert "similarity_score" in first
    assert 0 <= first["similarity_score"] <= 100
    assert "match_reasons" in first
    assert len(first["match_reasons"]) > 0


def test_recommend_any_criteria():
    payload = {
        "budget": 12000000,
        "top_n": 3
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["recommendations"]) == 3


def test_recommend_validation_error():
    payload = {
        "budget": 50  # Too small (< 100000)
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 422
