"""
Configuration management for PPE Detection System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration"""
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = DATA_DIR / "models"
    LOG_DIR = BASE_DIR / "logs"
    
    # Model settings
    MODEL_PATH = os.getenv("MODEL_PATH", str(MODEL_DIR / "best.pt"))
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
    DEVICE = os.getenv("DEVICE", "cpu")
    IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))
    
    # Camera settings
    CAMERA_SOURCE = int(os.getenv("CAMERA_SOURCE", "0"))
    CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "640"))
    CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "480"))
    CAMERA_FPS = int(os.getenv("CAMERA_FPS", "30"))
    
    # API settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "5000"))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # PPE Detection classes
    REQUIRED_PPE = ["helmet", "safety_glasses", "gloves"]
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        required_dirs = [cls.DATA_DIR, cls.MODEL_DIR, cls.LOG_DIR]
        for directory in required_dirs:
            directory.mkdir(parents=True, exist_ok=True)

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    
class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    
class TestConfig(Config):
    """Test configuration"""
    TESTING = True

config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "test": TestConfig,
    "default": DevelopmentConfig
}
