import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


class Config:
    """Application configuration loaded from environment variables with safe defaults."""

    BASE_DIR = BASE_DIR
    SECRET_KEY = os.getenv("SECRET_KEY", "crowdcast-ai-dev-secret-key-change-in-production")
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    DEBUG = os.getenv("FLASK_DEBUG", "1") in ("1", "true", "True")
    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", "5000"))

    # Database
    DATABASE_PATH = str(BASE_DIR / os.getenv("DATABASE_PATH", "database/database.db"))

    # ML Model Paths
    ML_DIR = BASE_DIR / "ml"
    DATASET_PATH = str(ML_DIR / "dataset.csv")
    SAVED_MODEL_DIR = ML_DIR / "saved_model"
    CROWD_MODEL_PATH = str(SAVED_MODEL_DIR / "crowd_model.joblib")
    MODEL_METADATA_PATH = str(SAVED_MODEL_DIR / "model_metadata.json")

    # Traffic Provider Settings
    TRAFFIC_PROVIDER = os.getenv("TRAFFIC_PROVIDER", "tomtom").lower()
    TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY", "").strip()
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "").strip()

    # OpenStreetMap / Geospatial Services
    NOMINATIM_USER_AGENT = os.getenv(
        "NOMINATIM_USER_AGENT", "CrowdCastAI-LocationIntelligence/1.0 (Academic Mini-Project)"
    )
    NOMINATIM_BASE_URL = os.getenv("NOMINATIM_BASE_URL", "https://nominatim.openstreetmap.org")
    OVERPASS_API_URL = os.getenv("OVERPASS_API_URL", "https://overpass-api.de/api/interpreter")
    OSRM_BASE_URL = os.getenv("OSRM_BASE_URL", "https://router.project-osrm.org")

    # Allowed Search Radii (in metres)
    ALLOWED_RADII = [500, 1000, 2000, 5000, 10000]
    DEFAULT_RADIUS = 1000

    # Crowd-Traffic Fusion Engine Initial Weights (Demonstration defaults, configurable)
    FUSION_WEIGHTS = {
        "traffic_congestion": float(os.getenv("WEIGHT_TRAFFIC_CONGESTION", "0.35")),
        "expected_crowd": float(os.getenv("WEIGHT_EXPECTED_CROWD", "0.30")),
        "historical_patterns": float(os.getenv("WEIGHT_HISTORICAL_PATTERNS", "0.20")),
        "event_impact": float(os.getenv("WEIGHT_EVENT_IMPACT", "0.15")),
    }

    # Admin defaults
    ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
    ADMIN_DEFAULT_PASSWORD = os.getenv("ADMIN_DEFAULT_PASSWORD", "crowdcast_admin_2026")

