import threading
import time
import requests
import pickle
from arango.client import ArangoClient
from typing import Dict, Any
from ..config.settings import config
from .helpers import logger

class MetricsCollector:
    def __init__(self):
        self.lock = threading.Lock()
        self.metrics = {
            'queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'db_queries': 0,
            'errors': 0,
            'response_times': [],
            'tool_usage': {},
            'llm_calls': 0
        }

    def record_query(self):
        with self.lock:
            self.metrics['queries'] += 1

    def record_cache_hit(self):
        with self.lock:
            self.metrics['cache_hits'] += 1

    def record_cache_miss(self):
        with self.lock:
            self.metrics['cache_misses'] += 1

    def record_db_query(self):
        with self.lock:
            self.metrics['db_queries'] += 1

    def record_error(self):
        with self.lock:
            self.metrics['errors'] += 1

    def record_response_time(self, time_ms: float):
        with self.lock:
            self.metrics['response_times'].append(time_ms)

    def record_tool_usage(self, tool_name: str):
        with self.lock:
            self.metrics['tool_usage'][tool_name] = self.metrics['tool_usage'].get(tool_name, 0) + 1
            logger.debug(f"Recorded tool usage: {tool_name}, total: {self.metrics['tool_usage'][tool_name]}")

    def record_llm_call(self):
        with self.lock:
            self.metrics['llm_calls'] += 1

    def get_metrics(self) -> Dict[str, Any]:
        with self.lock:
            m = self.metrics.copy()
            # average response time
            if m['response_times']:
                m['avg_response_time'] = sum(m['response_times']) / len(m['response_times'])
            # cache hit rate
            hits, misses = m['cache_hits'], m['cache_misses']
            if hits + misses > 0:
                m['cache_hit_rate'] = hits / (hits + misses)
            else:
                m['cache_hit_rate'] = 0
            return m

    def reset(self):
        with self.lock:
            self.metrics = {
                'queries': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'db_queries': 0,
                'errors': 0,
                'response_times': [],
                'tool_usage': {},
                'llm_calls': 0
            }

metrics = MetricsCollector()

class HealthCheck:
    def __init__(self):
        self.lock = threading.Lock()
        self.status = {
            'database': False,
            'llm': False,
            'graph': False,
            'last_check': None,
            'errors': []
        }

    def check_database(self) -> bool:
        try:
            client = ArangoClient(hosts=config.ARANGO_HOST)
            try:
                db = client.db(config.ARANGO_DB)
            except Exception:
                db = client.db(config.ARANGO_DB, username=config.ARANGO_USER, password=config.ARANGO_PASSWORD)
            db.properties()
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return False

    def check_llm(self) -> bool:
        try:
            r = requests.get(f"{config.OLLAMA_BASE_URL}/api/version")
            r.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"LLM health check failed: {str(e)}")
            return False

    def check_graph(self) -> bool:
        try:
            with open(config.GRAPH_FILE, 'rb') as f:
                pickle.load(f)
            return True
        except Exception as e:
            logger.error(f"Graph health check failed: {str(e)}")
            return False

    def run_health_check(self) -> Dict[str, Any]:
        with self.lock:
            self.status['database'] = self.check_database()
            self.status['llm'] = self.check_llm()
            self.status['graph'] = self.check_graph()
            self.status['last_check'] = time.time()
            self.status['healthy'] = all([
                self.status['database'],
                self.status['llm'],
                self.status['graph']
            ])
            return self.status

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            return self.status.copy()

    def is_healthy(self) -> bool:
        with self.lock:
            return self.status.get('healthy', False)

health_check = HealthCheck() 