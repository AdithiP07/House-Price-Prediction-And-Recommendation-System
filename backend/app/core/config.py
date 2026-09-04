import os
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT_DIR = os.path.dirname(BASE_DIR)

class Settings(BaseModel):
    PROJECT_NAME: str = "AI-Powered House Price Prediction & Recommendation System"
    VERSION: str = "1.0.0"
    API_V1_STR: str = ""
    
    # File Paths
    MODELS_DIR: str = os.path.join(ROOT_DIR, "models")
    DATA_DIR: str = os.path.join(ROOT_DIR, "data")
    FRONTEND_DIR: str = os.path.join(ROOT_DIR, "frontend")
    REPORTS_DIR: str = os.path.join(ROOT_DIR, "reports")
    
    MODEL_PATH: str = os.path.join(ROOT_DIR, "models", "house_price_model.pkl")
    PREPROCESSOR_PATH: str = os.path.join(ROOT_DIR, "models", "preprocessor.pkl")
    KMEANS_PATH: str = os.path.join(ROOT_DIR, "models", "kmeans_model.pkl")
    PCA_PATH: str = os.path.join(ROOT_DIR, "models", "pca_model.pkl")
    CATALOG_PATH: str = os.path.join(ROOT_DIR, "models", "property_catalog.pkl")
    METRICS_PATH: str = os.path.join(ROOT_DIR, "models", "metrics_summary.json")
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: list[str] = ["*"]

settings = Settings()
