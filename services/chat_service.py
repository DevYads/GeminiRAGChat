"""
Chat service for managing conversation history and context
"""
import uuid
from typing import Dict, List, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from models import ChatMessage, ConversationHistory
from database import get_db, ConversationMessage, ConversationSession, SessionLocal

class ChatService:
    """Service for managing chat conversations and history"""
    
    def __init__(self):
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
        
        db = SessionLocal()
        try:
            # Check if session exists
            existing_session = db.query(ConversationSession).filter(
                ConversationSession.session_id == session_id
            ).first()
            
            if not existing_session:
                # Create new session
                new_session = ConversationSession(
                    session_id=session_id,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    is_active=True
                )
                db.add(new_session)
                db.commit()
                
        finally:
            db.close()
        
        return session_id
    
    def add_message(self, session_id: str, role: str, content: str, use_rag: bool = True) -> None:
        """
        Add a message to the conversation history
        
        Args:
            session_id: Session identifier
            role: Message role (user/assistant)
            content: Message content
            use_rag: Whether RAG was used for this message
        """
        db = SessionLocal()
        try:
            # Ensure session exists
            self.create_session(session_id)
            
            # Add message to database
            message = ConversationMessage(
                session_id=session_id,
                role=role,
                content=content,
                timestamp=datetime.utcnow(),
                use_rag=use_rag
            )
            db.add(message)
            
            # Update session timestamp
            session = db.query(ConversationSession).filter(
                ConversationSession.session_id == session_id
            ).first()
            if session:
                session.updated_at = datetime.utcnow()
            
            db.commit()
            
        finally:
            db.close()
    
    def get_conversation_history(self, session_id: str) -> List[ChatMessage]:
        """
        Get conversation history for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of chat messages
        """
        db = SessionLocal()
        try:
            # Get messages from database
            db_messages = db.query(ConversationMessage).filter(
                ConversationMessage.session_id == session_id
            ).order_by(ConversationMessage.timestamp).all()
            
            # Convert to ChatMessage objects
            messages = []
            for db_msg in db_messages:
                message = ChatMessage(
                    role=db_msg.role,
                    content=db_msg.content,
                    timestamp=db_msg.timestamp
                )
                messages.append(message)
            
            return messages
            
        finally:
            db.close()
    
    def get_context_for_llm(self, session_id: str) -> List[Dict[str, str]]:
        """
        Get conversation context formatted for LLM
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of message dictionaries formatted for LLM
        """
        db = SessionLocal()
        try:
            # Get recent messages from database
            db_messages = db.query(ConversationMessage).filter(
                ConversationMessage.session_id == session_id
            ).order_by(ConversationMessage.timestamp.desc()).limit(self.max_context_messages).all()
            
            # Reverse to get chronological order
            db_messages.reverse()
            
            # Convert to format expected by LLM
            context = []
            for db_msg in db_messages:
                context.append({
                    "role": db_msg.role,
                    "content": db_msg.content
                })
            
            return context
            
        finally:
            db.close()
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear conversation history for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was cleared, False if session didn't exist
        """
        db = SessionLocal()
        try:
            # Check if session exists
            session = db.query(ConversationSession).filter(
                ConversationSession.session_id == session_id
            ).first()
            
            if not session:
                return False
            
            # Delete all messages for this session
            db.query(ConversationMessage).filter(
                ConversationMessage.session_id == session_id
            ).delete()
            
            # Mark session as inactive (or delete it)
            session.is_active = False
            session.updated_at = datetime.utcnow()
            
            db.commit()
            return True
            
        finally:
            db.close()
    
    def get_all_sessions(self) -> List[str]:
        """
        Get all active session IDs
        
        Returns:
            List of session IDs
        """
        db = SessionLocal()
        try:
            sessions = db.query(ConversationSession).filter(
                ConversationSession.is_active == True
            ).all()
            return [session.session_id for session in sessions]
            
        finally:
            db.close()
    
    def session_exists(self, session_id: str) -> bool:
        """
        Check if a session exists
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session exists, False otherwise
        """
        db = SessionLocal()
        try:
            session = db.query(ConversationSession).filter(
                ConversationSession.session_id == session_id,
                ConversationSession.is_active == True
            ).first()
            return session is not None
            
        finally:
            db.close()
