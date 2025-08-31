"""
Database configuration and models for persistent storage
"""
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Text, DateTime, Boolean, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")

# Create engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ConversationMessage(Base):
    """Database model for conversation messages"""
    __tablename__ = "conversation_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, nullable=False)
    role = Column(String, nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    use_rag = Column(Boolean, default=True, nullable=False)

class ConversationSession(Base):
    """Database model for conversation sessions"""
    __tablename__ = "conversation_sessions"
    
    session_id = Column(String, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """Create database tables"""
    Base.metadata.create_all(bind=engine)