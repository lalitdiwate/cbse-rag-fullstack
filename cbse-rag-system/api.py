"""
FastAPI Backend for CBSE RAG System
Integrates with simple_rag.py pipeline
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import os
from datetime import datetime
import asyncio
from pathlib import Path
import shutil

# Import the RAG pipeline
from simple_rag import RAGPipeline, DocumentProcessor, print_result

# Initialize FastAPI
app = FastAPI(
    title="CBSE RAG API",
    description="Advanced RAG system for CBSE educational content",
    version="1.0.0"
)

# CORS configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    //allow_origins=CORS_ORIGINS,
    allow_origins=["*"],  # You can restrict this to your domain for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG pipeline instance
rag_pipeline = None
documents_metadata = []
query_history = []

# Create upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# Pydantic Models
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    grade: int = Field(..., ge=1, le=12)
    subject: str
    llm_model: str = Field(default="openai/gpt-4-turbo")
    temperature: float = Field(default=0.7, ge=0, le=2)
    use_reranking: bool = True
    use_hyde: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
    metadata: Dict


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    environment: str
    services: Dict
    stats: Optional[Dict] = None


# Initialize RAG pipeline on startup
@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline on startup"""
    global rag_pipeline
    
    print("=" * 60)
    print("Starting CBSE RAG System Backend")
    print("=" * 60)
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("WARNING: OPENROUTER_API_KEY not found in environment")
        print("Please set it in .env file or environment variables")
        return
    
    try:
        rag_pipeline = RAGPipeline(
            openrouter_api_key=api_key,
            llm_model=os.getenv("DEFAULT_MODEL", "openai/gpt-4-turbo"),
            use_reranking=True,
            use_hyde=True
        )
        print("✓ RAG Pipeline initialized successfully")
        print("=" * 60)
    except Exception as e:
        print(f"✗ Failed to initialize RAG pipeline: {e}")
        print("=" * 60)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "CBSE RAG System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    
    services = {
        "vector_db": "ready" if rag_pipeline else "not_initialized",
        "llm": "connected" if rag_pipeline and rag_pipeline.llm else "not_configured",
        "embeddings": "ready" if rag_pipeline and rag_pipeline.embedding_service else "not_initialized",
        "processing": "ready" if rag_pipeline and rag_pipeline.doc_processor else "not_initialized"
    }
    
    stats = {
        "total_documents": len(documents_metadata),
        "total_queries": len(query_history),
        "avg_response_time": calculate_avg_response_time()
    }
    
    return HealthResponse(
        status="healthy" if rag_pipeline else "degraded",
        timestamp=datetime.utcnow().isoformat(),
        environment=os.getenv("APP_ENV", "development"),
        services=services,
        stats=stats
    )


@app.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    subject: str = "Mathematics",
    grade: str = "10",
    chapter: str = "",
    topic: str = ""
):
    """Upload and process a PDF document"""
    
    if not rag_pipeline:
        raise HTTPException(503, "RAG pipeline not initialized. Check OPENROUTER_API_KEY.")
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are supported")
    
    if not chapter:
        raise HTTPException(400, "Chapter name is required")
    
    try:
        # Save file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Metadata
        metadata = {
            'filename': file.filename,
            'subject': subject,
            'grade': int(grade),
            'chapter': chapter,
            'topic': topic,
            'upload_date': datetime.utcnow().isoformat(),
            'status': 'processing'
        }
        
        documents_metadata.append(metadata)
        
        # Process in background
        background_tasks.add_task(
            process_document_background,
            str(file_path),
            metadata
        )
        
        return {
            "status": "success",
            "message": f"Document '{file.filename}' uploaded and queued for processing",
            "metadata": metadata
        }
        
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {str(e)}")


async def process_document_background(file_path: str, metadata: Dict):
    """Process document in background"""
    try:
        print(f"\nProcessing document: {metadata['filename']}")
        
        rag_pipeline.add_document(
            file_path,
            {
                'subject': metadata['subject'],
                'grade': metadata['grade'],
                'chapter': metadata['chapter'],
                'topic': metadata.get('topic', '')
            }
        )
        
        # Update status
        for doc in documents_metadata:
            if doc['filename'] == metadata['filename']:
                doc['status'] = 'processed'
                doc['processed_date'] = datetime.utcnow().isoformat()
                break
        
        print(f"✓ Document processed: {metadata['filename']}")
        
    except Exception as e:
        print(f"✗ Failed to process document: {e}")
        
        # Update status
        for doc in documents_metadata:
            if doc['filename'] == metadata['filename']:
                doc['status'] = 'failed'
                doc['error'] = str(e)
                break


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG system"""
    
    if not rag_pipeline:
        raise HTTPException(503, "RAG pipeline not initialized. Check OPENROUTER_API_KEY.")
    
    start_time = datetime.utcnow()
    
    try:
        # Update pipeline settings
        rag_pipeline.use_reranking = request.use_reranking
        rag_pipeline.use_hyde = request.use_hyde
        rag_pipeline.llm.model = request.llm_model
        
        # Query
        result = rag_pipeline.query(
            question=request.question,
            grade=request.grade,
            subject=request.subject,
            top_k=10,
            rerank_k=5
        )
        
        # Calculate response time
        elapsed = (datetime.utcnow() - start_time).total_seconds()
        
        # Log query
        query_history.append({
            'question': request.question,
            'grade': request.grade,
            'subject': request.subject,
            'model': request.llm_model,
            'response_time': elapsed,
            'timestamp': start_time.isoformat(),
            'num_sources': len(result['sources'])
        })
        
        return QueryResponse(
            answer=result['answer'],
            sources=result['sources'],
            metadata={
                **result['metadata'],
                'response_time': elapsed,
                'llm_model': request.llm_model
            }
        )
        
    except Exception as e:
        raise HTTPException(500, f"Query failed: {str(e)}")


@app.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    return {
        "total": len(documents_metadata),
        "documents": documents_metadata
    }


@app.get("/queries")
async def list_queries(limit: int = 10):
    """Get recent query history"""
    return {
        "total": len(query_history),
        "queries": query_history[-limit:]
    }


@app.get("/stats")
async def get_statistics():
    """Get system statistics"""
    
    # Group by subject
    subjects = {}
    for doc in documents_metadata:
        subject = doc.get('subject', 'Unknown')
        subjects[subject] = subjects.get(subject, 0) + 1
    
    # Group by grade
    grades = {}
    for doc in documents_metadata:
        grade = doc.get('grade', 0)
        grades[f"Grade {grade}"] = grades.get(f"Grade {grade}", 0) + 1
    
    # Query stats
    total_queries = len(query_history)
    avg_response_time = calculate_avg_response_time()
    
    return {
        "documents": {
            "total": len(documents_metadata),
            "by_subject": subjects,
            "by_grade": grades,
            "processed": len([d for d in documents_metadata if d.get('status') == 'processed']),
            "processing": len([d for d in documents_metadata if d.get('status') == 'processing']),
            "failed": len([d for d in documents_metadata if d.get('status') == 'failed'])
        },
        "queries": {
            "total": total_queries,
            "avg_response_time": avg_response_time,
            "recent": query_history[-5:] if query_history else []
        }
    }


@app.delete("/documents/clear")
async def clear_documents():
    """Clear all documents (for testing)"""
    global documents_metadata
    documents_metadata = []
    
    # Clear uploaded files
    for file in UPLOAD_DIR.glob("*.pdf"):
        file.unlink()
    
    return {"status": "success", "message": "All documents cleared"}


@app.delete("/queries/clear")
async def clear_queries():
    """Clear query history (for testing)"""
    global query_history
    query_history = []
    return {"status": "success", "message": "Query history cleared"}


def calculate_avg_response_time() -> float:
    """Calculate average response time from query history"""
    if not query_history:
        return 0.0
    
    total_time = sum(q.get('response_time', 0) for q in query_history)
    return round(total_time / len(query_history), 2)


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Not Found",
        "message": "The requested resource was not found",
        "path": str(request.url)
    }


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {
        "error": "Internal Server Error",
        "message": "An unexpected error occurred",
        "details": str(exc)
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    
    print("\n" + "=" * 60)
    print("CBSE RAG System - FastAPI Backend")
    print("=" * 60)
    print(f"Starting server on http://0.0.0.0:{port}")
    print(f"API Documentation: http://localhost:{port}/docs")
    print(f"Health Check: http://localhost:{port}/health")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=True

    )
