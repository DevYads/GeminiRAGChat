"""
Simple in-memory vector store for RAG functionality
"""
import os
import numpy as np
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types

from models import DocumentChunk, SearchResult

class VectorStore:
    """Simple in-memory vector store using Gemini embeddings"""
    
    def __init__(self):
        """
        Initialize vector store with Gemini embedding client
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        self.client = genai.Client(api_key=api_key)
        self.embedding_model = "text-embedding-004"  # Latest Gemini embedding model
        self.chunks: Dict[str, DocumentChunk] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.dimension = None
    
    def add_document_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Add document chunks to the vector store
        
        Args:
            chunks: List of document chunks to add
        """
        if not chunks:
            return
        
        # Store chunks and generate embeddings
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk
            
            try:
                # Generate embedding using Gemini
                response = self.client.models.embed_content(
                    model=self.embedding_model,
                    contents=chunk.content
                )
                
                if response and hasattr(response, 'embeddings') and response.embeddings:
                    # Get the first embedding from the response
                    embedding_obj = response.embeddings[0]
                    if embedding_obj and hasattr(embedding_obj, 'values'):
                        embedding = np.array(embedding_obj.values)
                        self.embeddings[chunk.chunk_id] = embedding
                        
                        # Set dimension on first embedding
                        if self.dimension is None:
                            self.dimension = len(embedding)
                        
                        # Also store embedding in chunk model for potential serialization
                        chunk.embedding = embedding.tolist()
                
            except Exception as e:
                print(f"Error generating embedding for chunk {chunk.chunk_id}: {str(e)}")
                # Continue with other chunks even if one fails
    
    def search(self, query: str, top_k: int = 5, similarity_threshold: float = 0.3) -> List[SearchResult]:
        """
        Search for similar chunks using vector similarity
        
        Args:
            query: Search query
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score threshold
            
        Returns:
            List of search results sorted by similarity score
        """
        if not self.chunks:
            return []
        
        try:
            # Generate query embedding using Gemini
            response = self.client.models.embed_content(
                model=self.embedding_model,
                contents=query
            )
            
            if not response or not hasattr(response, 'embeddings') or not response.embeddings:
                return []
            
            # Get the first embedding from the response
            embedding_obj = response.embeddings[0] if response.embeddings else None
            if not embedding_obj or not hasattr(embedding_obj, 'values'):
                return []
            
            query_embedding = np.array(embedding_obj.values)
            
            # Calculate similarities
            similarities = []
            for chunk_id, chunk_embedding in self.embeddings.items():
                # Cosine similarity
                similarity = self._cosine_similarity(query_embedding, chunk_embedding)
                
                if similarity >= similarity_threshold:
                    similarities.append((chunk_id, similarity))
            
            # Sort by similarity score (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Convert to SearchResult objects
            results = []
            for chunk_id, score in similarities[:top_k]:
                chunk = self.chunks[chunk_id]
                result = SearchResult(
                    chunk_id=chunk_id,
                    content=chunk.content,
                    score=float(score),
                    metadata=chunk.metadata
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error during vector search: {str(e)}")
            return []
    
    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """
        Get a specific chunk by ID
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            DocumentChunk if found, None otherwise
        """
        return self.chunks.get(chunk_id)
    
    def remove_chunk(self, chunk_id: str) -> bool:
        """
        Remove a chunk from the vector store
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            True if chunk was removed, False if not found
        """
        if chunk_id in self.chunks:
            del self.chunks[chunk_id]
            del self.embeddings[chunk_id]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all chunks and embeddings"""
        self.chunks.clear()
        self.embeddings.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics
        
        Returns:
            Dictionary with store statistics
        """
        return {
            "total_chunks": len(self.chunks),
            "embedding_dimension": self.dimension,
            "model_name": "Gemini text-embedding-004",
            "memory_usage_mb": self._estimate_memory_usage()
        }
    
    
    def _estimate_memory_usage(self) -> float:
        """
        Estimate memory usage in MB
        
        Returns:
            Estimated memory usage in megabytes
        """
        if not self.embeddings:
            return 0.0
        
        # Estimate based on embedding size and count
        embedding_size = len(next(iter(self.embeddings.values()))) * 4  # 4 bytes per float32
        total_embeddings_size = len(self.embeddings) * embedding_size
        
        # Add chunk text size (rough estimate)
        total_text_size = sum(len(chunk.content.encode('utf-8')) for chunk in self.chunks.values())
        
        return (total_embeddings_size + total_text_size) / (1024 * 1024)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity score
        """
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
