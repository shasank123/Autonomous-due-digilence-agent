# src/api/main.py (Enhanced with Agent Integration)
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional
import uuid
import logging
import time
from datetime import datetime
import asyncio
import json

from src.core.config import settings
from src.agents.orchestrator import DueDiligenceOrchestrator

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enhanced Request/Response Models
class DueDiligenceRequest(BaseModel):
    company_ticker: str = Field(..., min_length=1, max_length=10, regex="^[A-Z]+$")
    analysis_type: str = Field(default="comprehensive", regex="^(comprehensive|financial|legal|market)$")
    questions: List[str] = Field(default_factory=list)
    user_id: Optional[str] = None
    priority: str = Field(default="normal", regex="^(low|normal|high)$")
    
    @validator('company_ticker')
    def ticker_uppercase(cls, v):
        return v.upper()

class AnalysisResponse(BaseModel):
    request_id: str
    status: str = Field(..., regex="^(initialized|running|completed|failed)$")
    progress: float = Field(..., ge=0.0, le=1.0)
    estimated_completion: Optional[str] = None
    results: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    timestamp: str
    current_step: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    ongoing_analyses: int
    components: Dict[str, Any]

# FastAPI App
app = FastAPI(
    title="Autonomous Due Diligence API",
    description="AI-powered multi-agent due diligence system with AutoGen",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State with Enhanced Tracking
class AnalysisState:
    def __init__(self):
        self.ongoing_analyses: Dict[str, Dict] = {}
        self.completed_analyses: Dict[str, Dict] = {}
        self.orchestrator = DueDiligenceOrchestrator()
        self.start_time = datetime.utcnow()

state = AnalysisState()

# Dependency Injection
async def get_orchestrator():
    return state.orchestrator

# Enhanced Middleware
@app.middleware("http")
async def log_requests(request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(f"Request {request_id}: {request.method} {request.url.path}")
    
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.info(f"Request {request_id} completed in {process_time:.2f}s - Status: {response.status_code}")
    
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# Enhanced Routes
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_company(
    request: DueDiligenceRequest,
    background_tasks: BackgroundTasks,
    orchestrator: DueDiligenceOrchestrator = Depends(get_orchestrator)
):
    """Main endpoint for due diligence analysis with enhanced validation"""
    try:
        request_id = str(uuid.uuid4())
        
        # Enhanced validation
        validation_result = await _validate_analysis_request(request)
        if not validation_result["valid"]:
            raise HTTPException(status_code=400, detail=validation_result["error"])
        
        # Initialize analysis state with enhanced tracking
        state.ongoing_analyses[request_id] = {
            "status": "initialized",
            "progress": 0.0,
            "start_time": datetime.utcnow(),
            "request": request.dict(),
            "current_step": "initialization",
            "last_update": datetime.utcnow(),
            "retry_count": 0
        }
        
        # Start background analysis with error handling
        background_tasks.add_task(
            _run_analysis_pipeline_with_retry,
            request_id,
            request,
            orchestrator
        )
        
        logger.info(f"Analysis initialized: {request_id} for {request.company_ticker}")
        
        return AnalysisResponse(
            request_id=request_id,
            status="initialized",
            progress=0.0,
            estimated_completion=_estimate_completion_time(request.analysis_type),
            timestamp=datetime.utcnow().isoformat(),
            current_step="initialization"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis initialization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during analysis initialization")

@app.get("/analysis/{request_id}", response_model=AnalysisResponse)
async def get_analysis_status(request_id: str):
    """Get status of ongoing analysis with enhanced error handling"""
    try:
        if request_id in state.ongoing_analyses:
            analysis = state.ongoing_analyses[request_id]
        elif request_id in state.completed_analyses:
            analysis = state.completed_analyses[request_id]
        else:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        return AnalysisResponse(
            request_id=request_id,
            status=analysis["status"],
            progress=analysis["progress"],
            results=analysis.get("results", {}),
            error=analysis.get("error"),
            timestamp=analysis.get("timestamp", datetime.utcnow().isoformat()),
            current_step=analysis.get("current_step")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status retrieval failed for {request_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check with component status"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "2.0.0",
            "ongoing_analyses": len(state.ongoing_analyses),
            "components": {
                "orchestrator": state.orchestrator.is_healthy(),
                "uptime": str(datetime.utcnow() - state.start_time),
                "memory_usage": _get_memory_usage(),
                "system_load": _get_system_load()
            }
        }
        
        # Check if any component is unhealthy
        if not all(health_status["components"].values()):
            health_status["status"] = "degraded"
            
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow().isoformat(),
            version="2.0.0",
            ongoing_analyses=len(state.ongoing_analyses),
            components={"error": str(e)}
        )

@app.get("/analyses")
async def list_analyses(
    status: Optional[str] = Query(None, regex="^(initialized|running|completed|failed)$"),
    limit: int = Query(50, ge=1, le=1000)
):
    """List analyses with filtering and pagination"""
    try:
        all_analyses = {**state.ongoing_analyses, **state.completed_analyses}
        
        if status:
            filtered = {k: v for k, v in all_analyses.items() if v.get("status") == status}
        else:
            filtered = all_analyses
        
        # Sort by start time (most recent first)
        sorted_analyses = sorted(
            filtered.items(),
            key=lambda x: x[1].get("start_time", datetime.min),
            reverse=True
        )[:limit]
        
        return {
            "analyses": [
                {
                    "request_id": k,
                    "status": v.get("status"),
                    "company": v.get("request", {}).get("company_ticker"),
                    "start_time": v.get("start_time"),
                    "progress": v.get("progress", 0.0)
                }
                for k, v in sorted_analyses
            ],
            "total": len(sorted_analyses)
        }
        
    except Exception as e:
        logger.error(f"List analyses failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Enhanced Background Tasks
async def _run_analysis_pipeline_with_retry(
    request_id: str, 
    request: DueDiligenceRequest, 
    orchestrator: DueDiligenceOrchestrator,
    max_retries: int = 3
):
    """Run analysis pipeline with retry logic"""
    for attempt in range(max_retries):
        try:
            await _run_analysis_pipeline(request_id, request, orchestrator)
            break  # Success, break out of retry loop
        except Exception as e:
            state.ongoing_analyses[request_id]["retry_count"] = attempt + 1
            logger.warning(f"Analysis {request_id} attempt {attempt + 1} failed: {e}")
            
            if attempt == max_retries - 1:  # Final attempt failed
                logger.error(f"Analysis {request_id} failed after {max_retries} attempts")
                state.ongoing_analyses[request_id].update({
                    "status": "failed",
                    "error": f"Analysis failed after {max_retries} attempts: {str(e)}",
                    "completion_time": datetime.utcnow()
                })
                # Move to completed analyses
                state.completed_analyses[request_id] = state.ongoing_analyses.pop(request_id)
            else:
                # Wait before retry
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

async def _run_analysis_pipeline(request_id: str, request: DueDiligenceRequest, orchestrator: DueDiligenceOrchestrator):
    """Enhanced analysis pipeline with progress tracking"""
    try:
        # Update status
        state.ongoing_analyses[request_id].update({
            "status": "running",
            "progress": 0.1,
            "current_step": "orchestration"
        })
        
        # Execute analysis with progress callback
        def progress_callback(progress: float, step: str = None):
            state.ongoing_analyses[request_id].update({
                "progress": progress,
                "current_step": step or state.ongoing_analyses[request_id].get("current_step"),
                "last_update": datetime.utcnow()
            })
        
        results = await orchestrator.execute_analysis(
            request_id=request_id,
            company_ticker=request.company_ticker,
            analysis_type=request.analysis_type,
            questions=request.questions,
            user_id=request.user_id,
            update_progress_callback=progress_callback
        )
        
        # Mark completion
        state.ongoing_analyses[request_id].update({
            "status": "completed",
            "progress": 1.0,
            "results": results,
            "completion_time": datetime.utcnow(),
            "current_step": "completed"
        })
        
        logger.info(f"Analysis {request_id} completed successfully")
        
        # Move to completed analyses (keep for 24 hours)
        state.completed_analyses[request_id] = state.ongoing_analyses.pop(request_id)
        
    except Exception as e:
        logger.error(f"Analysis pipeline failed for {request_id}: {e}", exc_info=True)
        state.ongoing_analyses[request_id].update({
            "status": "failed",
            "error": str(e),
            "completion_time": datetime.utcnow()
        })
        raise

# Enhanced Helper Functions
async def _validate_analysis_request(request: DueDiligenceRequest) -> Dict[str, Any]:
    """Enhanced request validation"""
    try:
        # Check ticker format
        if not request.company_ticker.isalpha():
            return {"valid": False, "error": "Ticker must contain only letters"}
        
        # Check rate limiting (simplified)
        recent_analyses = [
            a for a in state.ongoing_analyses.values() 
            if a.get("start_time") and (datetime.utcnow() - a["start_time"]).total_seconds() < 300
        ]
        if len(recent_analyses) > 10:  # Max 10 analyses per 5 minutes
            return {"valid": False, "error": "Rate limit exceeded"}
        
        # Check question length
        for question in request.questions:
            if len(question) > 1000:
                return {"valid": False, "error": "Questions too long (max 1000 characters)"}
        
        return {"valid": True}
        
    except Exception as e:
        return {"valid": False, "error": f"Validation error: {str(e)}"}

def _estimate_completion_time(analysis_type: str) -> str:
    """Enhanced completion time estimation"""
    estimates = {
        "financial": "2-3 minutes",
        "legal": "3-4 minutes", 
        "market": "2-3 minutes",
        "comprehensive": "5-7 minutes"
    }
    return estimates.get(analysis_type, "3-5 minutes")

def _get_memory_usage() -> Dict[str, Any]:
    """Get memory usage information"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent()
        }
    except ImportError:
        return {"error": "psutil not available"}

def _get_system_load() -> Dict[str, Any]:
    """Get system load information"""
    try:
        import psutil
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "load_avg": psutil.getloadavg() if hasattr(psutil, "getloadavg") else "N/A",
            "disk_usage": psutil.disk_usage('/').percent
        }
    except ImportError:
        return {"error": "psutil not available"}

# Enhanced Error Handlers
@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    error_id = str(uuid.uuid4())
    logger.error(f"Internal server error {error_id}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error", 
            "error_id": error_id,
            "support_contact": "support@duediligence.ai"
        }
    )

@app.exception_handler(429)
async def rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Rate limit exceeded", 
            "retry_after": "60 seconds",
            "limit": "10 requests per 5 minutes"
        }
    )

@app.exception_handler(422)
async def validation_error_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation error",
            "errors": exc.errors() if hasattr(exc, 'errors') else str(exc)
        }
    )

# Startup and Shutdown Events
@app.on_event("startup")
async def startup_event():
    logger.info("Due Diligence API starting up...")
    # Initialize components
    await state.orchestrator.initialize()

@app.on_event("shutdown") 
async def shutdown_event():
    logger.info("Due Diligence API shutting down...")
    # Cleanup resources
    for request_id in list(state.ongoing_analyses.keys()):
        state.ongoing_analyses[request_id]["status"] = "cancelled"
        state.completed_analyses[request_id] = state.ongoing_analyses.pop(request_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
        reload=settings.DEBUG
    )