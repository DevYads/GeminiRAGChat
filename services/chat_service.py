"""
Chat service for managing conversation history and context
"""
import uuid
from typing import Dict, List, Optional
from datetime import datetime

from models import ChatMessage, ConversationHistory

class ChatService:
    """Service for managing chat conversations and history"""
    
    def __init__(self):
        # In-memory storage for conversations (in production, use a database)
        self.conversations: Dict[str, ConversationHistory] = {}
        self.max_context_messages = 10  # Limit context to last 10 messages
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """
        Create a new chat session
        
        Args:
            session_id: Optional session ID, generates new one if not provided
            
        Returns:
            Session ID
        """
        if not session_id:
            session_id = str(uuid.uuid4())
        
        if session_id not in self.conversations:
            self.conversations[session_id] = ConversationHistory(
                session_id=session_id,
                messages=[],
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        
        return session_id
    
    def add_message(self, session_id: str, role: str, content: str) -> None:
        """
        Add a message to the conversation history
        
        Args:
            session_id: Session identifier
            role: Message role (user/assistant)
            content: Message content
        """
        if session_id not in self.conversations:
            self.create_session(session_id)
        
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.now()
        )
        
        self.conversations[session_id].messages.append(message)
        self.conversations[session_id].updated_at = datetime.now()
        
        # Keep only the last N messages to manage context size
        if len(self.conversations[session_id].messages) > self.max_context_messages * 2:
            # Keep last max_context_messages pairs (user + assistant)
            self.conversations[session_id].messages = \
                self.conversations[session_id].messages[-self.max_context_messages * 2:]
    
    def get_conversation_history(self, session_id: str) -> List[ChatMessage]:
        """
        Get conversation history for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of chat messages
        """
        if session_id not in self.conversations:
            return []
        
        return self.conversations[session_id].messages
    
    def get_context_for_llm(self, session_id: str) -> List[Dict[str, str]]:
        """
        Get conversation context formatted for LLM
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of message dictionaries formatted for LLM
        """
        messages = self.get_conversation_history(session_id)
        
        # Convert to format expected by LLM
        context = []
        for message in messages[-self.max_context_messages:]:  # Last N messages
            context.append({
                "role": message.role,
                "content": message.content
            })
        
        return context
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear conversation history for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was cleared, False if session didn't exist
        """
        if session_id in self.conversations:
            del self.conversations[session_id]
            return True
        return False
    
    def get_all_sessions(self) -> List[str]:
        """
        Get all active session IDs
        
        Returns:
            List of session IDs
        """
        return list(self.conversations.keys())
    
    def session_exists(self, session_id: str) -> bool:
        """
        Check if a session exists
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session exists, False otherwise
        """
        return session_id in self.conversations
