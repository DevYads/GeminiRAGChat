# Overview

This is a FastAPI-based AI chatbot service that combines Google Gemini AI with Retrieval-Augmented Generation (RAG) functionality. The application allows users to upload documents that get processed into searchable chunks, and then chat with an AI assistant that can reference this uploaded content to provide more informed responses. The system includes both a REST API backend and a web-based chat interface.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Backend Framework
- **FastAPI**: Chosen as the web framework for its automatic API documentation, type safety, and async support
- **Python 3.x**: Programming language with strong AI/ML ecosystem support
- **Uvicorn**: ASGI server for running the FastAPI application

## AI Integration
- **Google Gemini AI**: Primary language model (gemini-2.5-flash) for generating chat responses
- **RAG (Retrieval-Augmented Generation)**: Enhances AI responses by providing relevant document context
- **Conversation Memory**: In-memory storage of chat sessions with configurable context window (10 messages)

## Document Processing Pipeline
- **Multi-format Support**: Handles PDF and TXT file uploads
- **Text Chunking**: Splits documents into 1000-character chunks with 200-character overlap for better retrieval
- **Vector Embeddings**: Uses SentenceTransformers (all-MiniLM-L6-v2 model) for semantic search capabilities

## Vector Store
- **In-memory Storage**: Simple vector store implementation using NumPy arrays
- **Semantic Search**: Cosine similarity-based retrieval of relevant document chunks
- **Real-time Processing**: Immediate availability of uploaded documents for chat context

## API Architecture
- **RESTful Design**: Structured endpoints for chat and document management
- **Pydantic Models**: Type-safe request/response schemas with validation
- **Router Pattern**: Modular endpoint organization (chat, documents)
- **CORS Support**: Configured for frontend integration

## Frontend Interface
- **Static Web App**: Bootstrap-based responsive chat interface
- **Real-time Chat**: JavaScript-powered messaging with typing indicators
- **Document Upload**: Drag-and-drop file upload functionality
- **Session Management**: Persistent chat sessions across page refreshes

## Service Layer
- **Chat Service**: Manages conversation history and session state
- **Document Service**: Handles file processing and text extraction
- **Gemini Service**: Interfaces with Google AI APIs
- **Vector Store Service**: Manages document embeddings and similarity search

## Data Models
- **Conversation History**: Structured chat message storage with timestamps
- **Document Chunks**: Processed text segments with metadata and embeddings
- **Search Results**: Ranked document chunks with similarity scores

# External Dependencies

## AI Services
- **Google Gemini AI**: Primary language model requiring GEMINI_API_KEY
- **SentenceTransformers**: Local embedding model for vector search (all-MiniLM-L6-v2)

## Python Libraries
- **FastAPI**: Web framework and API server
- **Pydantic**: Data validation and serialization
- **PyPDF2**: PDF text extraction
- **NumPy**: Vector operations and similarity calculations
- **python-multipart**: File upload handling
- **python-dotenv**: Environment variable management

## Frontend Dependencies
- **Bootstrap 5.1.3**: CSS framework for responsive design
- **Font Awesome 6.0**: Icon library for UI elements
- **Vanilla JavaScript**: Client-side functionality without additional frameworks

## Development Tools
- **Uvicorn**: ASGI server for development and production
- **CORS Middleware**: Cross-origin request handling for frontend integration

## File Processing
- **PDF Support**: PyPDF2 for extracting text from PDF documents
- **Text Files**: Native Python support for plain text processing