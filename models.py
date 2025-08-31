"""
Pydantic models for request/response schemas
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class ChatMessage(BaseModel):
    """Individual chat message model"""
    role: str = Field(..., description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Content of the message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., description="User message", min_length=1)
    session_id: Optional[str] = Field(None, description="Chat session identifier")
    use_rag: bool = Field(True, description="Whether to use RAG for context")

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str = Field(..., description="AI assistant response")
    session_id: str = Field(..., description="Chat session identifier")
    sources: List[Dict[str, Any]] = Field(default=[], description="RAG sources used")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

class DocumentUploadResponse(BaseModel):
    """Response model for document upload"""
    message: str = Field(..., description="Upload status message")
    document_id: str = Field(..., description="Unique document identifier")
    chunks_created: int = Field(..., description="Number of text chunks created")
    filename: str = Field(..., description="Original filename")

class DocumentChunk(BaseModel):
    """Model for document chunks"""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    content: str = Field(..., description="Chunk text content")
    metadata: Dict[str, Any] = Field(default={}, description="Chunk metadata")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")

class SearchResult(BaseModel):
    """Model for vector search results"""
    chunk_id: str = Field(..., description="Chunk identifier")
    content: str = Field(..., description="Chunk content")
    score: float = Field(..., description="Similarity score")
    metadata: Dict[str, Any] = Field(default={}, description="Chunk metadata")

class ConversationHistory(BaseModel):
    """Model for conversation history"""
    session_id: str = Field(..., description="Session identifier")
    messages: List[ChatMessage] = Field(default=[], description="List of messages")
    created_at: datetime = Field(default_factory=datetime.now, description="Session creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update time")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
