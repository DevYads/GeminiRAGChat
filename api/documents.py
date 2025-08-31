"""
Document management API endpoints
"""
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from typing import List

from models import DocumentUploadResponse, DocumentChunk
from services.document_service import DocumentService
from services.vector_store import VectorStore

router = APIRouter()

# Initialize document service
document_service = DocumentService()

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(app_request: Request, file: UploadFile = File(...)):
    """
    Upload and process a document for RAG functionality
    
    Args:
        file: Uploaded document file
        app_request: FastAPI request object to access app state
        
    Returns:
        DocumentUploadResponse with processing results
        
    Raises:
        HTTPException: If document processing fails
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Process document into chunks
        chunks = await document_service.process_document(file)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No content extracted from document")
        
        # Get vector store from app state and add chunks
        vector_store: VectorStore = app_request.app.state.vector_store
        vector_store.add_document_chunks(chunks)
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        return DocumentUploadResponse(
            message="Document uploaded and processed successfully",
            document_id=document_id,
            chunks_created=len(chunks),
            filename=file.filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error uploading document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload document: {str(e)}"
        )

@router.get("/search")
async def search_documents(query: str, app_request: Request, top_k: int = 5):
    """
    Search documents using vector similarity
    
    Args:
        query: Search query
        top_k: Number of top results to return
        app_request: FastAPI request object to access app state
        
    Returns:
        Search results
    """
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Get vector store from app state
        vector_store: VectorStore = app_request.app.state.vector_store
        
        # Perform search
        results = vector_store.search(query, top_k=min(top_k, 20))  # Limit to max 20 results
        
        return {
            "query": query,
            "results": results,
            "total_results": len(results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error searching documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to search documents: {str(e)}"
        )

@router.get("/stats")
async def get_document_stats(app_request: Request):
    """
    Get document store statistics
    
    Args:
        app_request: FastAPI request object to access app state
        
    Returns:
        Document store statistics
    """
    try:
        # Get vector store from app state
        vector_store: VectorStore = app_request.app.state.vector_store
        
        stats = vector_store.get_stats()
        
        return {
            "vector_store_stats": stats,
            "status": "operational"
        }
        
    except Exception as e:
        print(f"Error getting document stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get document statistics: {str(e)}"
        )

@router.delete("/clear")
async def clear_document_store(app_request: Request):
    """
    Clear all documents from the vector store
    
    Args:
        app_request: FastAPI request object to access app state
        
    Returns:
        Success message
    """
    try:
        # Get vector store from app state
        vector_store: VectorStore = app_request.app.state.vector_store
        
        # Get stats before clearing
        stats_before = vector_store.get_stats()
        
        # Clear the store
        vector_store.clear()
        
        return {
            "message": "Document store cleared successfully",
            "documents_removed": stats_before["total_chunks"]
        }
        
    except Exception as e:
        print(f"Error clearing document store: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear document store: {str(e)}"
        )

@router.get("/supported-formats")
async def get_supported_formats():
    """
    Get list of supported document formats
    
    Returns:
        List of supported file formats
    """
    return {
        "supported_formats": list(document_service.supported_formats),
        "max_file_size": "No explicit limit set",
        "processing_info": {
            "chunk_size": document_service.chunk_size,
            "chunk_overlap": document_service.chunk_overlap
        }
    }
