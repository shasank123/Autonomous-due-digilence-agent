# src/core/config.py
from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # API Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    
    # CORS Settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8501"
    ]
    
    # Database Settings
    DATABASE_URL: str = "sqlite:///./due_diligence.db"
    
    # LLM Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    
    # RAG Settings
    VECTOR_STORE_PATH: str = "./data/vector_stores/financial_data"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Agent Settings
    MAX_AGENT_RETRIES: int = 3
    AGENT_TIMEOUT: int = 300  # 5 minutes
    
    # MCP Settings
    MCP_SERVERS: List[str] = [
        "financial_mcp",
        "legal_mcp", 
        "market_mcp"
    ]
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()