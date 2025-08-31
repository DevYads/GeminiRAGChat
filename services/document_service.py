"""
Document processing service for text extraction and chunking
"""
import io
import uuid
import PyPDF2
from typing import List, Dict, Any
from fastapi import UploadFile, HTTPException

from models import DocumentChunk

class DocumentService:
    """Service for handling document processing and text extraction"""
    
    def __init__(self):
        self.supported_formats = {'.pdf', '.txt'}
        self.chunk_size = 1000
        self.chunk_overlap = 200
    
    async def process_document(self, file: UploadFile) -> List[DocumentChunk]:
        """
        Process uploaded document and return text chunks
        
        Args:
            file: Uploaded file object
            
        Returns:
            List of DocumentChunk objects
            
        Raises:
            HTTPException: If file format is not supported or processing fails
        """
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Check file format
        file_extension = self._get_file_extension(file.filename)
        if file_extension not in self.supported_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Supported formats: {', '.join(self.supported_formats)}"
            )
        
        try:
            # Extract text based on file type
            if file_extension == '.pdf':
                text = await self._extract_pdf_text(file)
            elif file_extension == '.txt':
                text = await self._extract_text_file(file)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file format")
            
            if not text.strip():
                raise HTTPException(status_code=400, detail="No text content found in document")
            
            # Create text chunks
            chunks = self._create_chunks(text, file.filename)
            
            return chunks
            
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    
    async def _extract_pdf_text(self, file: UploadFile) -> str:
        """Extract text from PDF file"""
        try:
            content = await file.read()
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading PDF file: {str(e)}")
    
    async def _extract_text_file(self, file: UploadFile) -> str:
        """Extract text from text file"""
        try:
            content = await file.read()
            return content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                content = await file.read()  # Re-read the file
                return content.decode('latin-1')
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error decoding text file: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading text file: {str(e)}")
    
    def _create_chunks(self, text: str, filename: str) -> List[DocumentChunk]:
        """
        Split text into chunks with overlap
        
        Args:
            text: Full text content
            filename: Original filename for metadata
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        
        # Simple chunking by character count with overlap
        start = 0
        chunk_number = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Try to break at sentence or word boundaries
            if end < len(text):
                # Look for sentence boundary
                last_period = chunk_text.rfind('.')
                last_exclamation = chunk_text.rfind('!')
                last_question = chunk_text.rfind('?')
                
                sentence_end = max(last_period, last_exclamation, last_question)
                
                if sentence_end > self.chunk_size * 0.7:  # If we found a good sentence break
                    chunk_text = chunk_text[:sentence_end + 1]
                    end = start + sentence_end + 1
                else:
                    # Fall back to word boundary
                    last_space = chunk_text.rfind(' ')
                    if last_space > self.chunk_size * 0.7:
                        chunk_text = chunk_text[:last_space]
                        end = start + last_space
            
            # Create chunk with metadata
            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                content=chunk_text.strip(),
                metadata={
                    'filename': filename,
                    'chunk_number': chunk_number,
                    'start_char': start,
                    'end_char': end
                },
                embedding=None
            )
            
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            start = end - self.chunk_overlap
            chunk_number += 1
            
            # Prevent infinite loop
            if start >= len(text):
                break
        
        return chunks
    
    def _get_file_extension(self, filename: str) -> str:
        """Get file extension from filename"""
        return '.' + filename.split('.')[-1].lower() if '.' in filename else ''
