"""
Main application entry point
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.config import config_by_name
from app.utils.logger import setup_logger

def main():
    """Main function"""
    # Load configuration
    config = config_by_name["development"]
    config.validate()
    
    # Setup logger
    logger = setup_logger(
        "ppe-detection",
        log_file=str(config.LOG_DIR / "app.log"),
        level=config.LOG_LEVEL
    )
    
    logger.info("PPE Detection System starting...")
    logger.info(f"Model path: {config.MODEL_PATH}")
    logger.info(f"Device: {config.DEVICE}")
    
    # TODO: Start web application or detection service
    print("PPE Detection System")
    print(f"Configuration: {config.__name__}")
    print(f"Model: {config.MODEL_PATH}")
    print("Ready to start...")

if __name__ == "__main__":
    main()
