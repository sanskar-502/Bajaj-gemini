# main.py

#!/usr/bin/env python3
"""
LLM-powered Intelligent Query Retrieval System
Main entry point for starting the FastAPI application.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path to ensure all modules are found
sys.path.append(str(Path(__file__).parent))

import uvicorn
from api import app
from config import Config

def setup_logging(log_level: str):
    """Configures application-wide logging."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("app.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

def create_directories(config: Config):
    """Creates necessary directories specified in the configuration."""
    logger = logging.getLogger(__name__)
    directories_to_create = [config.UPLOAD_DIR]
    
    if config.VECTOR_DB_TYPE == "faiss":
        directories_to_create.append(config.VECTOR_STORE_DIR)
    
    for directory in directories_to_create:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")
        except OSError as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            raise

def main():
    """
    Main function to initialize configuration, set up the environment,
    and run the FastAPI application server.
    """
    try:
        config = Config()
        setup_logging(config.LOG_LEVEL)
        logger = logging.getLogger(__name__)
        
        logger.info("🚀 Starting Intelligent Query–Retrieval System...")
        
        create_directories(config)
        
        logger.info("=" * 50)
        logger.info(f"LLM Provider       : {config.LLM_PROVIDER}")
        logger.info(f"Vector DB Type     : {config.VECTOR_DB_TYPE}")
        logger.info(f"Embedding Model    : {config.EMBEDDING_MODEL}")
        logger.info(f"API Server listening on http://{config.API_HOST}:{config.API_PORT}")
        logger.info("=" * 50)
        
        uvicorn.run(
            app,
            host=config.API_HOST,
            port=config.API_PORT,
            log_level=config.LOG_LEVEL.lower(),
            access_log=True
        )
        
    except ValueError as e:
        logging.getLogger(__name__).error(f"❌ Configuration Error: {str(e)}")
        logging.getLogger(__name__).error("Please correct your .env file and try again.")
        sys.exit(1)
        
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("👋 Shutting down gracefully...")
        
    except Exception as e:
        logging.getLogger(__name__).critical(f"💥 Application failed to start: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()