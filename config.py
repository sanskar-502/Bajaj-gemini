import os
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the Intelligent Query-Retrieval System"""
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
    
    # Vector Database Configuration
    VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "faiss")  # faiss or pinecone
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "legal-docs")
    
    # Embedding Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Retrieval Configuration
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    
    # Document Processing
    SUPPORTED_FORMATS = [".pdf", ".docx", ".txt", ".pptx"]
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "50"))  # MB
    
    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    
    # Storage Configuration
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
    VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "vector_store")
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def get_embedding_config(cls) -> Dict[str, Any]:
        """Get embedding configuration"""
        return {
            "model": cls.EMBEDDING_MODEL,
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP
        }
    
    @classmethod
    def get_retrieval_config(cls) -> Dict[str, Any]:
        """Get retrieval configuration"""
        return {
            "top_k": cls.TOP_K_RESULTS,
            "similarity_threshold": cls.SIMILARITY_THRESHOLD
        } 