"""
Google Gemini AI service for generating chat responses
"""
import os
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types

from models import SearchResult

class GeminiService:
    """Service for interacting with Google Gemini AI"""
    
    def __init__(self):
        """Initialize Gemini client"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"  # Using the newest model series
        
        # System prompt for the chatbot
        self.system_prompt = """You are a helpful AI assistant with access to a knowledge base through RAG (Retrieval-Augmented Generation). 

When responding to users:
1. Use the provided context from the knowledge base when relevant
2. If the context contains relevant information, reference it naturally in your response
3. If the context doesn't contain relevant information, respond based on your general knowledge
4. Be concise but helpful
5. If you're unsure about something, acknowledge it honestly
6. Always maintain a friendly and professional tone

When context is provided, integrate it naturally into your response without explicitly mentioning "based on the provided context" unless specifically asked about your sources."""
    
    async def generate_response(
        self, 
        user_message: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None,
        rag_context: Optional[List[SearchResult]] = None
    ) -> str:
        """
        Generate response using Gemini AI with optional RAG context
        
        Args:
            user_message: User's message
            conversation_history: Previous conversation messages
            rag_context: RAG search results for context
            
        Returns:
            Generated response text
            
        Raises:
            Exception: If AI generation fails
        """
        try:
            # Prepare the context from RAG results
            context_text = ""
            if rag_context:
                context_parts = []
                for result in rag_context:
                    context_parts.append(f"Source: {result.metadata.get('filename', 'Unknown')}\n{result.content}")
                
                if context_parts:
                    context_text = "\n\n=== KNOWLEDGE BASE CONTEXT ===\n" + "\n\n---\n".join(context_parts) + "\n=== END CONTEXT ===\n\n"
            
            # Build the conversation
            messages = []
            
            # Add conversation history
            if conversation_history:
                for msg in conversation_history:
                    messages.append(
                        types.Content(
                            role="user" if msg["role"] == "user" else "model",
                            parts=[types.Part(text=msg["content"])]
                        )
                    )
            
            # Add current user message with context
            current_message = context_text + user_message
            messages.append(
                types.Content(
                    role="user",
                    parts=[types.Part(text=current_message)]
                )
            )
            
            # Generate response
            response = self.client.models.generate_content(
                model=self.model,
                contents=messages,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    temperature=0.7,
                    max_output_tokens=1000
                )
            )
            
            if response and response.text:
                return response.text
            else:
                return "I apologize, but I'm having trouble generating a response right now. Please try again."
                
        except Exception as e:
            print(f"Error generating response with Gemini: {str(e)}")
            raise Exception(f"Failed to generate AI response: {str(e)}")
    
    async def generate_simple_response(self, user_message: str) -> str:
        """
        Generate a simple response without conversation history or RAG
        
        Args:
            user_message: User's message
            
        Returns:
            Generated response text
        """
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text=user_message)]
                    )
                ],
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    temperature=0.7,
                    max_output_tokens=1000
                )
            )
            
            if response and response.text:
                return response.text
            else:
                return "I apologize, but I'm having trouble generating a response right now. Please try again."
                
        except Exception as e:
            print(f"Error generating simple response with Gemini: {str(e)}")
            raise Exception(f"Failed to generate AI response: {str(e)}")
    
    def test_connection(self) -> bool:
        """
        Test connection to Gemini AI
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents="Hello"
            )
            return response is not None and response.text is not None
        except Exception as e:
            print(f"Gemini connection test failed: {str(e)}")
            return False
