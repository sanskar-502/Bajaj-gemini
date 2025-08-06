#!/usr/bin/env python3
"""
LLM-powered Intelligent Query–Retrieval System
Main entry point for the application
"""

import os
import sys
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from config import Config
from api import app
import uvicorn

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def check_environment():
    """Check if required environment variables are set"""
    required_vars = ['OPENAI_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not getattr(Config, var, None):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please set these variables in your .env file or environment")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        Config.UPLOAD_DIR,
        Config.VECTOR_STORE_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def main():
    """Main function to run the application"""
    try:
        logger.info("Starting LLM-powered Intelligent Query–Retrieval System")
        
        # Check environment
        if not check_environment():
            sys.exit(1)
        
        # Create directories
        create_directories()
        
        # Log configuration
        logger.info(f"Vector DB Type: {Config.VECTOR_DB_TYPE}")
        logger.info(f"Embedding Model: {Config.EMBEDDING_MODEL}")
        logger.info(f"API Host: {Config.API_HOST}")
        logger.info(f"API Port: {Config.API_PORT}")
        
        # Start the server
        logger.info("Starting FastAPI server...")
        uvicorn.run(
            app,
            host=Config.API_HOST,
            port=Config.API_PORT,
            log_level=Config.LOG_LEVEL.lower(),
            access_log=True
        )
        
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Application failed to start: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 