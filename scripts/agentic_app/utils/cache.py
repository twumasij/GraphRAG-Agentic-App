import threading
from typing import Any, Optional
from ..config.settings import config

class ThreadSafeCache:
    def __init__(self, maxsize: int = config.CACHE_SIZE):
        self.cache = {}
        self.lock = threading.Lock()
        self.maxsize = maxsize

    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            return self.cache.get(key)

    def set(self, key: str, value: Any) -> None:
        with self.lock:
            if len(self.cache) >= self.maxsize:
                # Remove oldest item (Python 3.7+ dicts maintain insertion order)
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = value

    def clear(self) -> None:
        with self.lock:
            self.cache.clear()

    def __len__(self) -> int:
        with self.lock:
            return len(self.cache)

db_cache = ThreadSafeCache() 