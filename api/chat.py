"""
Chat API endpoints
"""
from fastapi import APIRouter, HTTPException, Request
from typing import Dict, Any

from models import ChatRequest, ChatResponse, ErrorResponse
from services.chat_service import ChatService
from services.gemini_service import GeminiService
from services.vector_store import VectorStore

router = APIRouter()

# Initialize services
chat_service = ChatService()
gemini_service = GeminiService()

@router.post("/message", response_model=ChatResponse)
async def send_message(request: ChatRequest, app_request: Request):
    """
    Send a message to the chatbot and get a response
    
    Args:
        request: Chat request containing message and optional session_id
        app_request: FastAPI request object to access app state
        
    Returns:
        ChatResponse with AI assistant response
        
    Raises:
        HTTPException: If message processing fails
    """
    try:
        # Get or create session
        session_id = chat_service.create_session(request.session_id)
        
        # Add user message to conversation history
        chat_service.add_message(session_id, "user", request.message)
        
        # Get conversation context
        conversation_history = chat_service.get_context_for_llm(session_id)
        
        # Get RAG context if enabled
        rag_context = []
        sources = []
        
        if request.use_rag:
            try:
                # Get vector store from app state
                vector_store: VectorStore = app_request.app.state.vector_store
                
                # Search for relevant context
                search_results = vector_store.search(request.message, top_k=3)
                
                if search_results:
                    rag_context = search_results
                    sources = [
                        {
                            "chunk_id": result.chunk_id,
                            "filename": result.metadata.get("filename", "Unknown"),
                            "score": result.score,
                            "content_preview": result.content[:200] + "..." if len(result.content) > 200 else result.content
                        }
                        for result in search_results
                    ]
                    
            except Exception as e:
                print(f"RAG search failed: {str(e)}")
                # Continue without RAG if search fails
        
        # Generate response using Gemini
        response_text = await gemini_service.generate_response(
            user_message=request.message,
            conversation_history=conversation_history,
            rag_context=rag_context
        )
        
        # Add assistant response to conversation history
        chat_service.add_message(session_id, "assistant", response_text)
        
        return ChatResponse(
            response=response_text,
            session_id=session_id,
            sources=sources
        )
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process chat message: {str(e)}"
        )

@router.get("/history/{session_id}")
async def get_chat_history(session_id: str):
    """
    Get chat history for a session
    
    Args:
        session_id: Session identifier
        
    Returns:
        Chat history for the session
    """
    try:
        if not chat_service.session_exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        
        messages = chat_service.get_conversation_history(session_id)
        
        return {
            "session_id": session_id,
            "messages": messages,
            "message_count": len(messages)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve chat history: {str(e)}"
        )

@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """
    Clear chat history for a session
    
    Args:
        session_id: Session identifier
        
    Returns:
        Success message
    """
    try:
        success = chat_service.clear_session(session_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {"message": f"Session {session_id} cleared successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear session: {str(e)}"
        )

@router.get("/sessions")
async def list_sessions():
    """
    List all active chat sessions
    
    Returns:
        List of active session IDs
    """
    try:
        sessions = chat_service.get_all_sessions()
        return {
            "sessions": sessions,
            "total_sessions": len(sessions)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list sessions: {str(e)}"
        )

@router.get("/test")
async def test_chat_service():
    """
    Test chat service connectivity
    
    Returns:
        Service status
    """
    try:
        # Test Gemini connection
        gemini_status = gemini_service.test_connection()
        
        return {
            "chat_service": "operational",
            "gemini_ai": "connected" if gemini_status else "disconnected",
            "status": "healthy" if gemini_status else "degraded"
        }
        
    except Exception as e:
        return {
            "chat_service": "error",
            "gemini_ai": "error", 
            "status": "unhealthy",
            "error": str(e)
        }
