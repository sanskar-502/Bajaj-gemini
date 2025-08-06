from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from enum import Enum

class DocumentType(str, Enum):
    """Supported document types"""
    INSURANCE_POLICY = "insurance_policy"
    LEGAL_CONTRACT = "legal_contract"
    HR_POLICY = "hr_policy"
    COMPLIANCE_DOC = "compliance_doc"
    UNKNOWN = "unknown"

class LogicTreeType(str, Enum):
    """Logic tree operation types"""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    CONDITIONAL = "CONDITIONAL"

class ClauseInfo(BaseModel):
    """Information about a specific clause or section"""
    title: str = Field(..., description="Title or name of the clause")
    text: str = Field(..., description="Full text content of the clause")
    document: str = Field(..., description="Source document filename")
    page: Optional[int] = Field(None, description="Page number where clause appears")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score (0-1)")
    clause_id: Optional[str] = Field(None, description="Unique identifier for the clause")
    section: Optional[str] = Field(None, description="Section or subsection identifier")

class LogicCondition(BaseModel):
    """Individual logic condition"""
    condition: str = Field(..., description="Description of the condition")
    met: bool = Field(..., description="Whether the condition is met")
    source_clause: Optional[str] = Field(None, description="Source clause for this condition")

class LogicTree(BaseModel):
    """Logic tree structure for reasoning"""
    type: LogicTreeType = Field(..., description="Type of logic operation")
    conditions: List[Union[str, 'LogicTree']] = Field(..., description="List of conditions or nested logic trees")
    result: Optional[bool] = Field(None, description="Result of the logic evaluation")

class QueryResponse(BaseModel):
    """Structured response for document queries"""
    answer: str = Field(..., description="Natural language answer to the query")
    clauses_used: List[ClauseInfo] = Field(..., description="List of relevant clauses used")
    logic_tree: LogicTree = Field(..., description="Logic tree showing reasoning process")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    query_intent: Optional[str] = Field(None, description="Detected intent of the query")
    entities: Optional[Dict[str, Any]] = Field(None, description="Extracted entities from query")

class DocumentMetadata(BaseModel):
    """Metadata for uploaded documents"""
    filename: str = Field(..., description="Original filename")
    document_type: DocumentType = Field(..., description="Type of document")
    upload_timestamp: str = Field(..., description="Upload timestamp")
    file_size: int = Field(..., description="File size in bytes")
    page_count: Optional[int] = Field(None, description="Number of pages")
    company_name: Optional[str] = Field(None, description="Company name if applicable")
    document_version: Optional[str] = Field(None, description="Document version")

class QueryRequest(BaseModel):
    """Request model for document queries"""
    question: str = Field(..., description="User's question about the documents")
    document_ids: Optional[List[str]] = Field(None, description="Specific documents to search in")
    include_logic: bool = Field(True, description="Whether to include logic tree in response")
    max_results: Optional[int] = Field(None, description="Maximum number of clauses to return")

class UploadResponse(BaseModel):
    """Response for document upload"""
    success: bool = Field(..., description="Whether upload was successful")
    document_id: str = Field(..., description="Unique identifier for the uploaded document")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    chunks_created: int = Field(..., description="Number of text chunks created")
    message: str = Field(..., description="Status message")

class SearchResult(BaseModel):
    """Individual search result"""
    content: str = Field(..., description="Text content of the result")
    metadata: Dict[str, Any] = Field(..., description="Metadata about the result")
    score: float = Field(..., description="Similarity score")
    source_document: str = Field(..., description="Source document")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")
    error_code: Optional[str] = Field(None, description="Error code")

# Update forward references
LogicTree.model_rebuild() 