# src/core/config.py
from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List, Union, Optional
import os

class Settings(BaseSettings):
    # --- API Settings ---
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    
    # --- CORS Settings ---
    ALLOWED_ORIGINS: Union[List[str], str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8501"
    ]
    
    @field_validator('ALLOWED_ORIGINS', mode='before')
    @classmethod
    def parse_allowed_origins(cls, v):
        """Parse ALLOWED_ORIGINS from string or list"""
        if isinstance(v, str):
            # Split comma-separated string into list
            return [origin.strip() for origin in v.split(',')]
        return v
    
    # --- Database / Redis Settings (Required for main.py) ---
    DATABASE_URL: str = "sqlite:///./due_diligence.db"
    
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    
    # --- LLM Settings ---
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4-turbo"  # Default model used by Orchestrator
    
    # --- SEC Data Settings (Required for sec_edgar.py) ---
    SEC_EDGAR_EMAIL: str = "admin@example.com" # Critical for SEC User-Agent compliance
    
    # --- RAG Settings ---
    VECTOR_STORE_PATH: str = "./data/vector_stores/financial_data"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # --- Agent Settings ---
    MAX_AGENT_RETRIES: int = 3
    AGENT_TIMEOUT: int = 300  # 5 minutes
    
    # --- MCP Settings ---
    MCP_SERVERS: List[str] = [
        "financial_mcp",
        "legal_mcp", 
        "market_mcp"
    ]
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore" # Allows extra env vars without crashing

# Instantiate singleton
settings = Settings()