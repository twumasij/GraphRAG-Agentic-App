import re
import logging
import pickle
from functools import wraps
import time
from typing import Any, Callable
from ..config.settings import config

# Constants
MAX_RETRIES = config.MAX_RETRIES
RETRY_DELAY = config.RETRY_DELAY
IATA_PATTERN = re.compile(r'^[A-Z]{3}$')

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

file_handler = logging.FileHandler("llm_backend.log")
file_handler.setLevel(logging.DEBUG) 
file_formatter = logging.Formatter(config.LOG_FORMAT)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

def retry_on_failure(max_retries: int = MAX_RETRIES, delay: int = RETRY_DELAY):
    """Decorator to retry a function on failure."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

def validate_iata_code(code: str) -> bool:
    """Validate IATA airport code format."""
    return bool(IATA_PATTERN.match(code))

def print_status(message: str):
    """Print status message with timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}") 