import os
import shutil
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from config import Config
from models import (
    QueryRequest, QueryResponse, UploadResponse, ErrorResponse,
    DocumentMetadata, SearchResult
)
from document_processor import DocumentProcessor
from vector_store import VectorStore
from query_engine import QueryEngine

# Initialize FastAPI app
app = FastAPI(
    title="LLM-powered Intelligent Query–Retrieval System",
    description="An intelligent legal-insurance-HR-compliance document analyst assistant",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
config = Config()
document_processor = DocumentProcessor()
vector_store = VectorStore()
query_engine = QueryEngine(vector_store)

# Create necessary directories
os.makedirs(config.UPLOAD_DIR, exist_ok=True)
os.makedirs(config.VECTOR_STORE_DIR, exist_ok=True)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LLM-powered Intelligent Query–Retrieval System",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check vector store
        stats = vector_store.get_stats()
        
        return {
            "status": "healthy",
            "vector_store": stats,
            "config": {
                "vector_db_type": config.VECTOR_DB_TYPE,
                "embedding_model": config.EMBEDDING_MODEL,
                "supported_formats": config.SUPPORTED_FORMATS
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload and process a document
    
    Args:
        file: Document file to upload
        
    Returns:
        Upload response with document metadata
    """
    try:
        # Validate file format
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in config.SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported formats: {config.SUPPORTED_FORMATS}"
            )
        
        # Validate file size
        if file.size > config.MAX_FILE_SIZE * 1024 * 1024:  # Convert MB to bytes
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {config.MAX_FILE_SIZE}MB"
            )
        
        # Save file temporarily
        temp_file_path = os.path.join(config.UPLOAD_DIR, file.filename)
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process document in background
        background_tasks.add_task(process_document_background, temp_file_path)
        
        return UploadResponse(
            success=True,
            document_id=file.filename,
            metadata=DocumentMetadata(
                filename=file.filename,
                document_type="unknown",  # Will be updated after processing
                upload_timestamp="",  # Will be updated after processing
                file_size=file.size,
                page_count=None,
                company_name=None,
                document_version=None
            ),
            chunks_created=0,  # Will be updated after processing
            message="Document uploaded successfully. Processing in background."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

async def process_document_background(file_path: str):
    """Process document in background"""
    try:
        # Process document
        result = document_processor.process_document(file_path)
        
        # Add to vector store
        vector_store.add_documents(result["chunks"])
        
        # Clean up temporary file
        os.remove(file_path)
        
        print(f"Document processed successfully: {result['document_id']}")
        
    except Exception as e:
        print(f"Error processing document {file_path}: {str(e)}")
        # Clean up temporary file even if processing failed
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query documents with a natural language question
    
    Args:
        request: Query request object
        
    Returns:
        Structured query response
    """
    try:
        # Validate query
        is_valid, validation_message = query_engine.validate_query(request.question)
        if not is_valid:
            raise HTTPException(status_code=400, detail=validation_message)
        
        # Process query
        response = query_engine.process_query(request)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/documents", response_model=List[DocumentMetadata])
async def list_documents():
    """List all uploaded documents"""
    try:
        documents = vector_store.get_all_documents()
        
        # Group by document and extract metadata
        doc_metadata = {}
        for doc in documents:
            doc_id = doc["metadata"]["document_id"]
            if doc_id not in doc_metadata:
                doc_metadata[doc_id] = {
                    "filename": doc_id,
                    "document_type": doc["metadata"]["document_type"],
                    "upload_timestamp": doc["metadata"]["upload_timestamp"],
                    "file_size": 0,  # Not stored in vector store
                    "page_count": doc["metadata"].get("page_count"),
                    "company_name": doc["metadata"].get("company_name"),
                    "document_version": None
                }
        
        return list(doc_metadata.values())
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document from the system"""
    try:
        success = vector_store.delete_document(document_id)
        
        if success:
            return {"message": f"Document {document_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@app.get("/search", response_model=List[SearchResult])
async def search_documents(
    query: str,
    top_k: Optional[int] = None,
    document_ids: Optional[str] = None
):
    """
    Search documents with a query
    
    Args:
        query: Search query
        top_k: Number of results to return
        document_ids: Comma-separated list of document IDs to search in
        
    Returns:
        List of search results
    """
    try:
        # Parse document IDs
        doc_ids = None
        if document_ids:
            doc_ids = [doc_id.strip() for doc_id in document_ids.split(",")]
        
        # Search
        results = vector_store.search(query, top_k, doc_ids)
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    try:
        vector_stats = vector_store.get_stats()
        
        return {
            "vector_store": vector_stats,
            "config": {
                "vector_db_type": config.VECTOR_DB_TYPE,
                "embedding_model": config.EMBEDDING_MODEL,
                "chunk_size": config.CHUNK_SIZE,
                "chunk_overlap": config.CHUNK_OVERLAP,
                "top_k_results": config.TOP_K_RESULTS,
                "similarity_threshold": config.SIMILARITY_THRESHOLD
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            details=str(exc),
            error_code="INTERNAL_ERROR"
        ).dict()
    )

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True
    ) 