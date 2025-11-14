# src/api/main.py (Enhanced with Agent Integration)
import asyncio
import os
import uuid
import logging
import time
from typing import Dict, List, Optional, Any, Annotated
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from enum import Enum
import regex
import json
import dotenv

from fastapi import FastAPI, BackgroundTasks, Depends, Request, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, field_validator
import uvicorn
from dotenv import load_dotenv
import redis.asyncio as redis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import prometheus_client
from prometheus_client import Histogram, Counter, Gauge

# Load environment variables
load_dotenv()

log_dir = 'logs'
os.makedirs(log_dir, exist_ok = True)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/api.log')
    ]
)
logger = logging.getLogger("due_digilence_api")

# Import our project components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from agents.financial_analyst import FinancialAgentTeam, create_financial_team
    from rag.core import ProductionRAGSystem
    from data.collectors.sec_edgar import SECDataCollector
    from data.collectors.company_resolver import CompanyResolver
    from data.processors.document_parser import DocumentProcessor
    from agents.orchestrator import DueDigillenceOrchestator

except ImportError as e:
    logger.error(f"Failed to import project modules {e}")
    raise

app_start_time = time.time()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Global components with proper error handling
class ComponentStatus(Enum):
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"

class SystemComponents:
    def __init__(self):
        self.financial_agent_team : Optional[FinancialAgentTeam] = None
        self.rag_system : Optional[ProductionRAGSystem] = None
        self.orchestator : Optional[DueDigillenceOrchestator] = None
        self.sec_collector: Optional[SECDataCollector] = None
        self.company_resolver: Optional[CompanyResolver] = None
        self.document_processor: Optional[DocumentProcessor] = None
        self.status : Dict[str, ComponentStatus] = {}
        self.redis_client : Optional[redis.Redis] = None

components = SystemComponents()

# Redis connection for session storage
async def init_redis():
    try:
        components.redis_client = await redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),  # Fixed default Redis port
        password=os.getenv('REDIS_PASSWORD'),
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
        retry_on_timeout=True
        )
        await components.redis_client.ping()
        logger.info("✅ Redis connection established")
        components.status['redis'] = ComponentStatus.HEALTHY
    
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        components.status['redis'] = ComponentStatus.FAILED
        components.redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown events with comprehensive error handling"""
    global session_manager
    global app_start_time
    app_start_time = time.time() # Reset on each startup
    startup_success = False

    try:
        logger.info("🚀 Initializing Financial Due Diligence API...")
        # Initialize Redis first
        await init_redis()

        # Initialize session manager
        if components.redis_client and components.status.get('redis') == ComponentStatus.HEALTHY:
            session_manager = RedisSessionManager(components.redis_client)
            logger.info("✅ Redis Session Manager initialized")
        else:
            logger.warning("⚠️ Redis unavailable - session management disabled")
            session_manager = None

        # Initialize RAG system
        try:
            components.rag_system = ProductionRAGSystem()
            components.status['rag_system'] = ComponentStatus.HEALTHY
            logger.info("✅ RAG System initialized")
        
        except Exception as e:
            logger.error(f"❌ RAG System initialization failed: {e}")
            components.status['rag_system'] = ComponentStatus.FAILED
            raise

        # Initialize financial agent team
        try:
            components.financial_agent_team = await create_financial_team(
                rag_system=components.rag_system
            )
            components.status['financial_agent'] = ComponentStatus.HEALTHY
            logger.info("✅ Financial Agent Team initialized")
            
        except Exception as e:
            logger.error(f"❌ Financial Agent Team initialization failed: {e}")
            components.status['financial_agent'] = ComponentStatus.FAILED
            raise

        # Initialize other components
        try:
            components.company_resolver = CompanyResolver()
            components.sec_collector = SECDataCollector()
            components.document_processor = DocumentProcessor()
            components.status.update({
                'sec_collector': ComponentStatus.HEALTHY,
                'company_resolver': ComponentStatus.HEALTHY,
                'document_processor': ComponentStatus.HEALTHY
            })
            logger.info("✅ Support components initialized")
        
        except Exception as e:
            logger.error(f"⚠️ Some support components failed: {e}")
        
        # Initialize orchestrator
        try:
            components.orchestator = DueDigillenceOrchestator()
            components.status['orchestator'] = ComponentStatus.HEALTHY
            logger.info("✅ Analysis Orchestrator initialized")
        
        except Exception as e:
            logger.error(f"⚠️ Orchestrator not available: {e}")
            components.status['orchestator'] = ComponentStatus.DEGRADED
        
        startup_success = True
        logger.info("🎉 Financial Due Diligence API startup completed successfully!")

    except Exception as e:
        logger.critical(f"💥 API startup failed: {e}")
        startup_success = False
        raise

    yield

    # Shutdown
    try:
        logger.info("🛑 Shutting down Financial Due Diligence API...")
        
        cleanup_tasks = []

        if components.financial_agent_team:
            cleanup_tasks.append(components.financial_agent_team.close())
        
        if components.redis_client:
            cleanup_tasks.append(components.redis_client.close())

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        logger.info("✅ API shutdown completed")
        
    except Exception as e:
        logger.error(f"⚠️ Cleanup warning: {e}")

# Create FastAPI app
app = FastAPI(
    title = "Autonomous Due Diligence API",
    description="AI-powered multi-agent system for comprehensive company financial analysis",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Prometheus metrics with duplicate protection
"""
_metrics_initialized = False

if not hasattr(app, 'metrics_initialized'):
    REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
    REQUEST_DURATION = Histogram('api_request_duration_seconds', 'API request duration')
    ACTIVE_ANALYSES = Gauge('active_analyses', 'Number of active analyses')
    SESSION_COUNT = Gauge('analysis_sessions_total', 'Total analysis sessions')
    app.metrics_initialized = True"""

# Add rate limiting to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded,_rate_limit_exceeded_handler)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins = os.getenv('allowed_origins', '*').split(','),
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts = os.getenv('allowed_hosts', '*').split(',')
)

# Pydantic models with comprehensive validation
class AnalysisType(str, Enum):
    COMPREHENSIVE = "comprehensive"
    FINANCIAL = "financial"
    QUICK = "quick"
    CUSTOM = "custom"

class AnalysisRequest(BaseModel):
    company_ticker: str = Field(..., min_length=1, max_length=10, description="Company stock ticker symbol")
    analysis_type: AnalysisType = Field(default=AnalysisType.COMPREHENSIVE)
    additional_context: Optional[str] = Field(default= None, max_length=1000)
    priority: Annotated[str, Field(pattern="^(low|normal|high|urgent)$")] = "normal"
    timeout_seconds: int = Field(default=300, ge=60, le=1000)

    @field_validator('company_ticker')
    def validate_ticker(cls, v):
        if not v.isalphanum():
            raise ValueError('Ticker must be alphanumeric')
        return v.upper()
    
class AnalysisResponse(BaseModel):
    company: str
    session_id: str
    status: str
    progress: int = Field(ge=0,le=100)
    estimated_completion: Optional[str] = None
    message: str
    created_at: str
    queue_position: Optional[int] = None

class AnalysisResult(BaseModel):
    session_id: str
    company: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    warnings: List[str] = []
    timestamp: str
    processing_time: Optional[float] = None
    data_quality_score: Optional[float] = Field(None, ge=0, le=1)

class CompanySearchRequest(BaseModel):
    query : str = Field(..., min_length=0, max_length=50, description="Company name or ticker to search")
    max_results: int = Field(default=10,ge=1, le=50)

class CompanySearchResponse(BaseModel):
    results: List[Dict[str, str]]
    total_count: int
    search_duration: float

class SystemHealth(BaseModel):
    status: str
    components: Dict[str, str]
    timestamp: str
    version: str = "1.0.0"
    uptime: float
    active_sessions: int

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: str

# Custom exception handlers
class AnalysisError(Exception):
    def __init__(self, message: str, session_id: str = None, status_code: int = 500):
        self.message = message
        self.session_id = session_id
        self.status_code = status_code
        super().__init__(self.message)

class CompanyNotFoundError(AnalysisError):
    def __init__(self, company: str):
        super().__init__(f" Company not found: {company}", status_code=404)

class RateLimitError(AnalysisError):
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, status_code=429)

# Session management
class AnalysisSession:
    def __init__(self, session_id: str, company: str, analysis_type: str):
        self.company = company
        self.session_id = session_id
        self.analysis_type = analysis_type
        self.status = "pending"
        self.progress = 0
        self.created_at = datetime.now(timezone.utc)
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.result: Optional[Dict] = None 
        self.error: Optional[str] = None
        self.warnings: List[str] = []
        self.progress_updates: List[Dict] = []
    
    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "company": self.company,
            "status": self.status,
            "progress": self.progress,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
            "warnings": self.warnings
        }

class RedisSessionManager:

    """Production-grade session management using Redis"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.session_prefix = "analysis_session:"
        self.default_ttl = 86400 # 24 hours in seconds

    async def create_session(self, session_id: str, company: str, analysis_type: str):
        """Create a new analysis session in Redis"""

        session = AnalysisSession(session_id, company, analysis_type)
        await self.store_session(session)
        return session
    
    async def get_session(self, session_id: str) -> Optional[AnalysisSession]:
         """Retrieve session from Redis"""
         try:
             session_data = await self.redis.get(f"{self.session_prefix}{session_id}")
             if not session_data:
                 return None
             
             data_dict = json.loads(session_data)
             return self._dict_to_session(data_dict)
         
         except (json.JSONDecodeError, KeyError) as e:
             logger.error(f"Failed to decode session {session_id}: {e}")
             return None
         
         except Exception as e:
             logger.error(f"Redis error retrieving session {session_id}: {e}")
             return None
         
    async def update_session(self, session: AnalysisSession) -> bool:
        """Update session in Redis with error handling"""
        try:
            await self.store_session(session)
            return True
        
        except Exception as e:
            logger.error(f"Failed to update session {session.session_id}: {e}")
            return False
        
    async def update_progress(self, session_id: str, progress: int, message: str) -> bool:
        """Update session progress atomically"""
        try:
            # Use Redis transaction for atomic update
            async with self.redis.pipeline(transaction=True) as pipe:
                session_data = await pipe.get(f"{self.session_prefix}{session_id}").execute()
                if not session_data[0]:
                    return  False
                
                data_dict = json.loads(session_data)
                data_dict['progress'] = progress
                data_dict['progress_updates'].append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "progress": progress,
                    "message": message
                })

                await pipe.setex(
                    f"{self.session_prefix}{session_id}",
                    self.default_ttl,
                    json.dumps(data_dict)
                ).execute()

                logger.info(f"Updated progress for {session_id}: {progress}% - {message}")
                return True
            
        except Exception as e:
            logger.error(f"Progress update failed for {session_id}: {e}")
            return False

    async def delete_session(self, session_id: str) -> bool:
        """Delete session from Redis"""
        try:
            await self.redis.delete(f"{self.session_prefix}{session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
        
    async def cleanup_expired_sessions(self) -> int:
        """Clean up sessions older than TTL (Redis handles automatically)"""
        # Redis automatically expires sessions due to TTL
        # This is just for manual cleanup if needed
        return 0
    
    async def get_active_sessions_count(self) -> int:
        """Get count of active analysis sessions"""
        try:
            # Use SCAN instead of KEYS for production (non-blocking)
            count = 0
            async for key in self.redis.scan_iter(f"{self.session_prefix}*"):
                count+=1
            return count
        
        except Exception as e:
            logger.error(f"Failed to count active sessions: {e}")
            return 0
    
    # Private methods
    async def store_session(self, session: AnalysisSession):
        """Store session with proper error handling"""
        try:
            session_data = json.dumps(session.to_dict(), default=str)
            await self.redis.setex(
                f"{self.session_prefix}{session.session_id}",
                self.default_ttl,
                session_data
            )
            return True       

        except Exception as e:
            logger.error(f"Failed to store session {session.session_id}: {e}")
            return False
             
    def _dict_to_session(self, data_dict: Dict) -> AnalysisSession:
        """Convert dictionary back to AnalysisSession object"""

        session = AnalysisSession(
            data_dict['session_id'],
            data_dict['company'],
            data_dict['analysis_type']
        )

        # Restore all attributes
        for key, value in data_dict.items():
            if hasattr(session, key):
                setattr(session, key, value)

        return session
    
# Global session manager instance
session_manager: Optional[RedisSessionManager] = None

async def get_session_manager() -> RedisSessionManager:
    """Dependency injection for session manager"""
    if not session_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Session management service unavailable"
        )
    return session_manager

# Dependency injections with error handling
async def get_financial_agent() -> FinancialAgentTeam:
    if (not components.financial_agent_team or components.status.get('financial_agent') != ComponentStatus.HEALTHY):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Financial agent service is currently unavailable"
        )
    return components.financial_agent_team

async def get_rag_system() -> ProductionRAGSystem:
    if (not components.rag_system or components.status.get('rag_system') != ComponentStatus.HEALTHY):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system is currently unavailable"
        )
    
    return components.rag_system

async def get_company_resolver() -> CompanyResolver:
    if not components.company_resolver:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Company resolver service is currently unavailable"
        )
    return components.company_resolver

# Exception handlers
@app.exception_handler(AnalysisError)
async def analysis_error_handler(request: Request, exc: AnalysisError):
    logger.warning(f"Analysis error:{exc.message} (session: {exc.session_id})")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            message=exc.message,
            session_id=exc.session_id,
            timestamp=datetime.now(timezone.utc).isoformat()
        ).model_dump()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP error {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            timestamp=datetime.now(timezone.utc).isoformat()
        ).model_dump()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.warning(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail="An unexpected error occurred",
            timestamp=datetime.now(timezone.utc).isoformat()
        ).model_dump()
    )

#Background Task with Production Session Management
async def process_analysis(session_id: str, company: str, analysis_type: str, additional_context: str):
    """Production-grade background task with proper session management"""
    session_mgr = await get_session_manager()

    # Create session in Redis
    session = await session_mgr.create_session(session_id, company, analysis_type)
    session.started_at = datetime.now(timezone.utc).isoformat()
    session.status = "processing"
    await session_mgr.update_session(session)

    try:
        ACTIVE_ANALYSES.inc()
        SESSION_COUNT.inc()

        logger.info(f"Starting analysis for {company} (session: {session_id})")

        # Get dependencies with proper error handling
        financial_agent = await get_financial_agent()
        rag_system = await get_rag_system()

        # Progress updates with Redis
        await session_mgr.update_progress(session_id, 10, "Initializing analysis...")

        # Data validation
        await session_mgr.update_progress(session_id, 30, "Fetching company data...")
        data_available = await financial_agent.ensure_company_data(company)

        if not data_available:
            raise AnalysisError(
                f"Insufficient data available for {company}",
                session_id=session_id,
                status_code=422
            )
        
        # Core analysis
        await session_mgr.update_progress(session_id, 60, "Running financial analysis...")
        result = await financial_agent.analyze_company(
            company_ticker=company,
            additional_context=additional_context
        )

        # Finalize
        await session_mgr.update_progress(session_id, 90, "Finalizing results...")
        session.result=result
        session.status="completed"
        session.progress=100
        session.completed_at=datetime.now(timezone.utc).isoformat()

        processing_time = 0.0
        if session.completed_at and session.started_at:
            completed_dt = datetime.fromisoformat(session.completed_at.replace('Z', '+00:00'))
            started_dt = datetime.fromisoformat(session.started_at.replace('Z', '+00:00'))
            processing_time = (completed_dt - started_dt).total_seconds()

        session.result['processing_time'] = processing_time

        await session_mgr.update_session(session)
        logger.info(f"Analysis completed for {company} (Session: {session_id}) in {processing_time:.2f}s")

    except Exception as e:
        logger.error(f"Analysis failed for {company} (Session: {session_id}): {e}")

        # Update session with error
        session.status= "failed"
        session.error= str(e)
        session.completed_at= datetime.now(timezone.utc).isoformat()
        await session_mgr.update_session(session)

        raise AnalysisError(
            f"Analysis processing failed: {str(e)}",
            session_id=session_id
        )
    
    finally:
        ACTIVE_ANALYSES.dec()

# API Routes
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Autonomous Due Diligence API", "status": "healthy"}

@app.get("/health", response_model=SystemHealth)
async def health_check():
    """Comprehensive health check endpoint"""
    component_statuses = {name: status.value for name, status in components.status.items()}
    active_sessions = await session_manager.get_active_sessions_count()

    if all(status == ComponentStatus.HEALTHY for status in components.status.values()):
        overall_status = "healthy"
    elif any(status == ComponentStatus.FAILED for status in components.status.values()):
        overall_status = "unhealthy"
    else:
        overall_status = "degraded"

    return SystemHealth(
        status= overall_status,
        components= component_statuses,
        timestamp= datetime.now(timezone.utc).isoformat(),
        uptime= time.time() - app_start_time,
        active_sessions=active_sessions
    )

@app.post("/analyze", response_model= AnalysisResponse)
@limiter.limit("10/minute")
async def start_analysis(
    request: Request,
    analysis_request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    session_mgr: RedisSessionManager = Depends(get_session_manager),
    financial_agent: FinancialAgentTeam = Depends(get_financial_agent),
    company_resolver: CompanyResolver = Depends(get_company_resolver)
):
    #Start a new financial analysis for a company
    start_time = time.time()

    try:
        # Validate company exists
        cik = company_resolver.get_cik(analysis_request.company_ticker)
        if not cik:
            raise CompanyNotFoundError(analysis_request.company_ticker)
        
        # Create analysis session using Redis session manager
        session_id = str(uuid.uuid4())
        session = await session_mgr.create_session(
            session_id,
            analysis_request.company_ticker,
            analysis_request.analysis_type.value
        )

        # Start background processing
        background_tasks.add_task(
            process_analysis,
            session_id,
            analysis_request.company_ticker,
            analysis_request.analysis_type.value,
            analysis_request.additional_context or ""
        )

        # Calculate estimated completion
        estimated_completion = (datetime.now(timezone.utc) + 
                              timedelta(seconds=analysis_request.timeout_seconds)).isoformat()
        
        response = AnalysisResponse(
            session_id=session_id,
            company = analysis_request.company_ticker,
            status = "queued",
            progress = 0,
            estimated_completion=estimated_completion,
            message="Analysis queued for processing",
            created_at=datetime.now(timezone.utc).isoformat(),
            queue_position=1
        )

        REQUEST_COUNT.labels(method= "POST", endpoint= "/analyze", status= "200").inc()
        REQUEST_DURATION.observe(time.time() - start_time)

        return response
    
    except CompanyNotFoundError as e:
        REQUEST_COUNT.labels(method= "POST", endpoint = "/analyze", status = "404").inc()
        raise

    except Exception as e:
        REQUEST_COUNT.labels(method= "POST", endpoint= "/analyze", status= "500").inc()
        logger.error(f"Failed to start analysis for {analysis_request.company_ticker}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start analysis"
        )

@app.get("/analysis/{session_id}", response_model=AnalysisResult)
@limiter.limit("60/minute")
async def get_analysis_result(
    request: Request,
    session_id: str,
    session_mgr: RedisSessionManager = Depends(get_session_manager)
):
    """Get analysis result from Redis"""
    start_time = time.time()

    try:
        session = await session_mgr.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Analysis session {session_id} not found"
            )
        
        if session.status == "failed" and session.error:
            raise AnalysisError(
                f"Analysis failed: {session.error}",
                session_id=session_id,
                status_code=422
            )
        
        processing_time = None
        if session.started_at and session.completed_at:
            started = datetime.fromisoformat(session.started_at.replace('Z', '+00:00'))
            completed = datetime.fromisoformat(session.completed_at.replace('Z', '+00:00'))
            processing_time = (completed - started).total_seconds()

        result = AnalysisResult(
            session_id=session.session_id,
            company=session.company,
            status=session.status,
            result=session.result,
            error=session.error,
            warnings=session.warnings,
            timestamp=session.completed_at.isoformat() if session.completed_at else datetime.now(timezone.utc).isoformat(),
            processing_time=processing_time
        )

        REQUEST_COUNT.labels(method="GET", endpoint= "/analysis/{session_id}", status= "200").inc()
        REQUEST_DURATION.observe(time.time() - start_time)

        return result
    
    except HTTPException:
        REQUEST_COUNT.labels(method= "GET", endpoint= "/analysis/{session_id}", status= "404").inc()
        raise

    except Exception as e:
        REQUEST_COUNT.labels(method= "GET", endpoint= "/analysis/{session_id}", status= "500").inc()
        logger.error(f"Failed to get analysis result for {session_id}: {e}")
        raise

@app.post("/company/search", response_model=CompanySearchResponse)
@limiter.limit("30/minute")
async def search_companies(
    request: Request,
    search_request: CompanySearchRequest,
    company_resolver: CompanyResolver = Depends(get_company_resolver)
):
    #Search for companies by name or ticker
    start_time = time.time()

    try:
        results = company_resolver.search_companies(search_request.query)

        limited_results =  results[:search_request.max_results]

        response = CompanySearchResponse(
            results=limited_results,
            total_count=len(results),
            search_duration=time.time() - start_time
        )

        REQUEST_COUNT.labels(method= "POST", endpoint= "/company/search", status= "200").inc()
        REQUEST_DURATION.observe(time.time() - start_time)

        return response
    
    except Exception as e:
        REQUEST_COUNT.labels(method= "POST", endpoint= "/company/search", status= "500").inc()
        logger.error(f"Company search failed for query '{search_request.query}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Company search service temporarily unavailable"
        )

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return prometheus_client.generate_latest()

# Store app start time for uptime calculation
app_start_time = time.time()

load_dotenv()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        access_log=True

    )




        

        





    
    





















                
        

                



        
                 


 
        

















