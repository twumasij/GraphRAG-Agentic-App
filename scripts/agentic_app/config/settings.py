from dataclasses import dataclass
import os
from dotenv import load_dotenv

@dataclass
class Config:
    """Configuration settings for the application."""
    ARANGO_HOST: str = "http://localhost:8529"
    ARANGO_DB: str = "flights_db"
    ARANGO_USER: str = "root"
    ARANGO_PASSWORD: str = ""
    
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama2"
    OLLAMA_TEMPERATURE: float = 0.1
    OLLAMA_TOP_P: float = 0.95
    OLLAMA_MAX_TOKENS: int = 2048
    
    CACHE_SIZE: int = 1000
    MAX_WORKERS: int = 4
    
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 1
    
    LOG_LEVEL: str = "DEBUG"
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    GRAPH_FILE: str = 'data/graph_networkx.pkl'
    
    @classmethod
    def from_env(cls) -> 'Config':
        load_dotenv()
        
        return cls(
            ARANGO_HOST=os.getenv('ARANGO_HOST', cls.ARANGO_HOST),
            ARANGO_DB=os.getenv('ARANGO_DB', cls.ARANGO_DB),
            ARANGO_USER=os.getenv('ARANGO_USER', cls.ARANGO_USER),
            ARANGO_PASSWORD=os.getenv('ARANGO_PASSWORD', cls.ARANGO_PASSWORD),
            OLLAMA_BASE_URL=os.getenv('OLLAMA_BASE_URL', cls.OLLAMA_BASE_URL),
            OLLAMA_MODEL=os.getenv('OLLAMA_MODEL', cls.OLLAMA_MODEL),
            OLLAMA_TEMPERATURE=float(os.getenv('OLLAMA_TEMPERATURE', cls.OLLAMA_TEMPERATURE)),
            OLLAMA_TOP_P=float(os.getenv('OLLAMA_TOP_P', cls.OLLAMA_TOP_P)),
            OLLAMA_MAX_TOKENS=int(os.getenv('OLLAMA_MAX_TOKENS', cls.OLLAMA_MAX_TOKENS)),
            CACHE_SIZE=int(os.getenv('CACHE_SIZE', cls.CACHE_SIZE)),
            MAX_WORKERS=int(os.getenv('MAX_WORKERS', cls.MAX_WORKERS)),
            MAX_RETRIES=int(os.getenv('MAX_RETRIES', cls.MAX_RETRIES)),
            RETRY_DELAY=int(os.getenv('RETRY_DELAY', cls.RETRY_DELAY)),
            LOG_LEVEL=os.getenv('LOG_LEVEL', cls.LOG_LEVEL),
            LOG_FORMAT=os.getenv('LOG_FORMAT', cls.LOG_FORMAT),
            GRAPH_FILE=os.getenv('GRAPH_FILE', cls.GRAPH_FILE)
        )

config = Config.from_env() 