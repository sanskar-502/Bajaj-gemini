#!/usr/bin/env python3
"""
LLM-powered Intelligent Query Retrieval System
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
    # Check which LLM provider is configured
    provider = Config.LLM_PROVIDER.lower()
    
    if provider == "openai":
        if not Config.OPENAI_API_KEY:
            logger.error("OpenAI API key is required when using OpenAI provider")
            logger.error("Please set OPENAI_API_KEY in your .env file or environment")
            return False
    elif provider == "gemini":
        if not Config.GEMINI_API_KEY:
            logger.error("Gemini API key is required when using Gemini provider")
            logger.error("Please set GEMINI_API_KEY in your .env file or environment")
            return False
    else:
        # Check if at least one provider is available
        if not Config.OPENAI_API_KEY and not Config.GEMINI_API_KEY:
            logger.error("No LLM provider configured. Please set either OPENAI_API_KEY or GEMINI_API_KEY")
            logger.error("You can also set LLM_PROVIDER to specify which one to use")
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
        logger.info(f"LLM Provider: {Config.LLM_PROVIDER}")
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