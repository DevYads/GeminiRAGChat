"""
FastAPI Chatbot Service with Google Gemini AI and RAG functionality
"""
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

from api.chat import router as chat_router
from api.documents import router as documents_router
from services.vector_store import VectorStore

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="AI Chatbot with RAG",
    description="A FastAPI-based chatbot service with Google Gemini AI integration and RAG functionality",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize vector store on startup
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    # Initialize vector store singleton
    vector_store = VectorStore()
    app.state.vector_store = vector_store
    print("✅ Vector store initialized")
    
    # Verify Gemini API key
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("⚠️  Warning: GEMINI_API_KEY not found in environment variables")
    else:
        print("✅ Gemini API key found")

# Include routers
app.include_router(chat_router, prefix="/api/chat", tags=["Chat"])
app.include_router(documents_router, prefix="/api/documents", tags=["Documents"])

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML page"""
    return FileResponse("static/index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AI Chatbot with RAG",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5000,
        reload=True
    )
