from langchain.agents import Tool
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from typing import Annotated, Dict, List, Optional, TypedDict, Union, Any
from dotenv import load_dotenv
from arango.client import ArangoClient
import networkx as nx
import pandas as pd
import pickle
import os
import sys
import time
import warnings
import requests
import json
import logging
import re
from functools import wraps, lru_cache
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

# LangChain's base LLM
from langchain.llms.base import LLM
from pydantic import Field

# Local imports
from scripts.agentic_app.tools.flight_graph import FlightGraphTools
from scripts.agentic_app.tools.graph_query_tools import GraphQueryTools

# ---------------------------
# 1) Configuration
# ---------------------------
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
    
    LOG_LEVEL: str = "INFO"
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

# ---------------------------
# 2) Helpers / Classes
# ---------------------------
MAX_RETRIES = config.MAX_RETRIES
RETRY_DELAY = config.RETRY_DELAY
IATA_PATTERN = re.compile(r'^[A-Z]{3}$')
CACHE_SIZE = config.CACHE_SIZE
MAX_WORKERS = config.MAX_WORKERS

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

    def record_llm_call(self):
        with self.lock:
            self.metrics['llm_calls'] += 1

    def get_metrics(self) -> dict:
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

    def run_health_check(self) -> dict:
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

    def get_status(self) -> dict:
        with self.lock:
            return self.status.copy()

    def is_healthy(self) -> bool:
        with self.lock:
            return self.status.get('healthy', False)

health_check = HealthCheck()

class ThreadSafeCache:
    def __init__(self, maxsize=1000):
        self.cache = {}
        self.lock = threading.Lock()
        self.maxsize = maxsize

    def get(self, key):
        with self.lock:
            return self.cache.get(key)

    def set(self, key, value):
        with self.lock:
            if len(self.cache) >= self.maxsize:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = value

db_cache = ThreadSafeCache(maxsize=CACHE_SIZE)
thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)

def retry_on_failure(max_retries: int = MAX_RETRIES, delay: int = RETRY_DELAY):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                        raise
                    logger.warning(f"Attempt {attempt+1} failed: {str(e)}")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

@lru_cache(maxsize=CACHE_SIZE)
def validate_iata_code(code: str) -> bool:
    return bool(IATA_PATTERN.match(code))

def print_status(message, end='\n'):
    sys.stdout.write(f"\r{message}")
    sys.stdout.flush()
    if end == '\n':
        print()

def print_spinner(duration):
    for _ in range(duration):
        for char in '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏':
            print_status(f"Processing {char}", end='')
            time.sleep(0.1)

# ---------------------------
# 3) GraphState definition
# ---------------------------
class GraphState(TypedDict):
    messages: Annotated[List[dict], ...]
    tools: Dict[str, Any]
    tool_results: Dict[str, Any]
    current_tool: Optional[str]

# ---------------------------
# 4) Arango / NetworkX setup
# ---------------------------
@retry_on_failure()
def get_arango_db():
    try:
        client = ArangoClient(hosts=config.ARANGO_HOST)
        try:
            db = client.db(config.ARANGO_DB)
        except Exception:
            db = client.db(config.ARANGO_DB, username=config.ARANGO_USER, password=config.ARANGO_PASSWORD)
        db.properties()
        logger.info("Successfully connected to ArangoDB")
        return db
    except Exception as e:
        logger.error(f"Failed to connect to ArangoDB: {str(e)}")
        raise

@lru_cache(maxsize=1)
def get_networkx_graph():
    try:
        with open(config.GRAPH_FILE, 'rb') as f:
            graph = pickle.load(f)
            logger.info("Successfully loaded NetworkX graph")
            return graph
    except FileNotFoundError:
        logger.error("NetworkX graph file not found!")
        raise
    except Exception as e:
        logger.error(f"Error loading graph data: {str(e)}")
        raise

# ---------------------------
# 5) FlightGraphTools
# ---------------------------
class FlightGraphTools:
    def __init__(self):
        logger.info("Initializing FlightGraphTools...")
        self.db = get_arango_db()
        self.G = get_networkx_graph()
        self._precompute_metrics()
        logger.info("FlightGraphTools initialized successfully")

    def _precompute_metrics(self):
        logger.info("Precomputing network metrics...")
        self.pagerank = nx.pagerank(self.G, weight='distance')
        self.betweenness = nx.betweenness_centrality(self.G, weight='distance')
        logger.info("Network metrics precomputed")

    def _get_cached_result(self, key: str) -> Optional[str]:
        return db_cache.get(key)

    def _cache_result(self, key: str, result: str):
        db_cache.set(key, result)

    def find_direct_flights(self, airport_code: str) -> str:
        if not validate_iata_code(airport_code):
            raise ValueError(f"Invalid IATA code format: {airport_code}")
        cache_key = f"direct_flights_{airport_code}"
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result

        try:
            aql = """
            FOR airport IN airports
                FILTER airport.iata == @airport_code
                LET flights = (
                    FOR v, e IN 1..1 ANY airport GRAPH 'flights_graph'
                        RETURN {
                            destination: v.iata,
                            airport_name: v.name,
                            city: v.city,
                            country: v.country,
                            distance: e.distance
                        }
                )
                RETURN {
                    source: airport.name,
                    flights: flights
                }
            """
            cursor = self.db.aql.execute(aql, bind_vars={'airport_code': airport_code})
            results = list(cursor)

            if not results or not results[0]['flights']:
                return f"No direct flights found from {airport_code}"

            flights_df = pd.DataFrame(results[0]['flights'])
            flights_df = flights_df.sort_values('distance')

            response = [f"Direct flights from {results[0]['source']}:"]
            response.append(f"\nTotal direct flights: {len(flights_df)}")
            response.append("\nTop 5 closest destinations:")
            response.append(flights_df[['destination', 'city', 'country', 'distance']].head().to_string())
            response.append("\nTop 5 furthest destinations:")
            response.append(flights_df[['destination', 'city', 'country', 'distance']].tail().to_string())

            result = "\n".join(response)
            self._cache_result(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"Error finding direct flights: {str(e)}")
            raise

    def count_direct_flights(self, airport_code: str) -> str:
        aql = """
        FOR airport IN airports
            FILTER airport.iata == @airport_code
            LET count = LENGTH(
                FOR v, e IN 1..1 ANY airport GRAPH 'flights_graph'
                    RETURN 1
            )
            RETURN { airport: airport.iata, direct_flights: count }
        """
        cursor = self.db.aql.execute(aql, bind_vars={'airport_code': airport_code})
        result = list(cursor)
        if result:
            return f"Airport {result[0]['airport']} has {result[0]['direct_flights']} direct flights."
        return f"No data found for {airport_code}"

    def check_direct_flight(self, source: str, destination: str) -> str:
        aql = """
        FOR airport IN airports
            FILTER airport.iata == @source
            LET flightExists = LENGTH(
                FOR v, e IN 1..1 ANY airport GRAPH 'flights_graph'
                    FILTER v.iata == @destination
                    RETURN 1
            )
            RETURN { source: airport.iata, direct_to_destination: flightExists > 0 }
        """
        cursor = self.db.aql.execute(aql, bind_vars={'source': source, 'destination': destination})
        result = list(cursor)
        if result and result[0]['direct_to_destination']:
            return f"There is a direct flight from {source} to {destination}."
        return f"There is no direct flight from {source} to {destination}."

    def find_shortest_path(self, source: str, target: str) -> str:
        if not all(validate_iata_code(code) for code in [source, target]):
            raise ValueError("Invalid IATA code format")

        cache_key = f"shortest_path_{source}_{target}"
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result

        try:
            path = nx.shortest_path(self.G, source, target, weight='distance')
            total_distance = sum(self.G[path[i]][path[i+1]]['distance'] for i in range(len(path)-1))
            stops = [f"{iata} ({self.G.nodes[iata]['city']}, {self.G.nodes[iata]['country']})" for iata in path]
            response = [
                f"Shortest path from {path[0]} to {path[-1]}:",
                f"Number of stops: {len(path)-2}",
                f"Total distance: {total_distance:.2f} km",
                f"Route: {' → '.join(stops)}"
            ]
            result = "\n".join(response)
            self._cache_result(cache_key, result)
            return result
        except nx.NetworkXNoPath:
            return f"No path found between {source} and {target}"
        except Exception as e:
            logger.error(f"Error finding shortest path: {str(e)}")
            raise

    def find_minimum_stop_route(self, source: str, target: str) -> str:
        try:
            path = nx.shortest_path(self.G, source, target)  # Unweighted => fewest stops
            stops = [f"{iata} ({self.G.nodes[iata]['city']}, {self.G.nodes[iata]['country']})" for iata in path]
            response = [
                f"Route with the fewest stops from {source} to {target}:",
                f"Number of stops: {len(path)-2}",
                f"Route: {' → '.join(stops)}"
            ]
            return "\n".join(response)
        except nx.NetworkXNoPath:
            return f"No route found between {source} and {target}"

    def compare_routes(self, source: str, target: str) -> str:
        shortest = self.find_shortest_path(source, target)
        fewest_stops = self.find_minimum_stop_route(source, target)
        response = [
            "Comparison of Routes:",
            "\nShortest (by distance):",
            shortest,
            "\nRoute with Fewest Stops:",
            fewest_stops
        ]
        return "\n".join(response)

    def analyze_airport_importance(self, airport_code: str) -> str:
        if not validate_iata_code(airport_code):
            raise ValueError(f"Invalid IATA code format: {airport_code}")

        if airport_code not in self.G:
            return f"Airport {airport_code} not found in the network"

        cache_key = f"airport_importance_{airport_code}"
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result

        try:
            aql = """
            FOR airport IN airports
                FILTER airport.iata == @airport_code
                RETURN airport
            """
            cursor = self.db.aql.execute(aql, bind_vars={'airport_code': airport_code})
            airport_info = list(cursor)[0]

            response = [
                f"Airport Analysis for {airport_code}:",
                f"Name: {airport_info['name']}",
                f"Location: {airport_info['city']}, {airport_info['country']}",
                "\nNetwork Metrics:",
                f"PageRank Score: {self.pagerank[airport_code]:.6f}",
                f"Betweenness Centrality: {self.betweenness[airport_code]:.6f}",
                f"Total Connections: {self.G.degree(airport_code)}",
                f"Incoming Flights: {self.G.in_degree(airport_code)}",
                f"Outgoing Flights: {self.G.out_degree(airport_code)}"
            ]
            result = "\n".join(response)
            self._cache_result(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"Error analyzing airport importance: {str(e)}")
            raise

    def compare_airport_importance(self, airport1: str, airport2: str) -> str:
        analysis1 = self.analyze_airport_importance(airport1)
        analysis2 = self.analyze_airport_importance(airport2)
        response = [
            f"Comparison of {airport1} and {airport2}:",
            "\n--- Analysis for " + airport1 + " ---",
            analysis1,
            "\n--- Analysis for " + airport2 + " ---",
            analysis2
        ]
        return "\n".join(response)

    def get_top_incoming_airports(self, top_n: int = 5) -> str:
        incoming = sorted(self.G.nodes, key=lambda n: self.G.in_degree(n), reverse=True)
        response = ["Top Incoming Airports:"]
        for airport in incoming[:top_n]:
            response.append(f"{airport}: {self.G.in_degree(airport)} incoming flights")
        return "\n".join(response)

    def get_top_outgoing_airports(self, top_n: int = 5) -> str:
        outgoing = sorted(self.G.nodes, key=lambda n: self.G.out_degree(n), reverse=True)
        response = ["Top Outgoing Airports:"]
        for airport in outgoing[:top_n]:
            response.append(f"{airport}: {self.G.out_degree(airport)} outgoing flights")
        return "\n".join(response)

    def average_flight_distance(self, airport_code: str) -> str:
        edges = list(self.G.out_edges(airport_code, data=True))
        if not edges:
            return f"No outgoing flights from {airport_code}."
        distances = [edge[2]['distance'] for edge in edges]
        avg_distance = sum(distances) / len(distances)
        return f"Average flight distance from {airport_code}: {avg_distance:.2f} km"

    def flight_distance_distribution(self) -> str:
        import statistics
        distances = [data['distance'] for _, _, data in self.G.edges(data=True)]
        if not distances:
            return "No flight distance data available."
        min_d = min(distances)
        max_d = max(distances)
        mean_d = statistics.mean(distances)
        median_d = statistics.median(distances)
        response = [
            "Flight Distance Distribution:",
            f"Minimum distance: {min_d:.2f} km",
            f"Maximum distance: {max_d:.2f} km",
            f"Mean distance: {mean_d:.2f} km",
            f"Median distance: {median_d:.2f} km"
        ]
        return "\n".join(response)

    def get_isolated_airports(self) -> str:
        isolated = [node for node in self.G.nodes if self.G.degree(node) == 0]
        if not isolated:
            return "No isolated airports found."
        return "Isolated Airports:\n" + ", ".join(isolated)

    def list_airports_by_country(self, country: str) -> str:
        aql = """
        FOR airport IN airports
            FILTER airport.country == @country
            RETURN { iata: airport.iata, name: airport.name, city: airport.city }
        """
        cursor = self.db.aql.execute(aql, bind_vars={'country': country})
        results = list(cursor)
        if not results:
            return f"No airports found in {country}"
        response = [f"Airports in {country}:"]
        for airport in results:
            response.append(f"{airport['iata']}: {airport['name']} ({airport['city']})")
        return "\n".join(response)
    
    def find_flights_within_distance(self, airport_code: str, max_distance: float) -> str:
        """
        Find all direct flights from the specified airport that are within a given maximum distance.

        :param airport_code: IATA code for the source airport, e.g. 'LAX'
        :param max_distance: Maximum distance (in km) to filter destinations
        :return: A text summary of all destinations within that distance
        """
        if not validate_iata_code(airport_code):
            raise ValueError(f"Invalid IATA code format: {airport_code}")

        # Optional caching key
        cache_key = f"flights_within_distance_{airport_code}_{max_distance}"
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result

        try:
            # AQL to get direct flights, then filter by distance <= max_distance
            aql = """
            FOR airport IN airports
                FILTER airport.iata == @airport_code
                LET flights = (
                    FOR v, e IN 1..1 ANY airport GRAPH 'flights_graph'
                        FILTER e.distance <= @max_distance
                        RETURN {
                            destination: v.iata,
                            airport_name: v.name,
                            city: v.city,
                            country: v.country,
                            distance: e.distance
                        }
                )
                RETURN { source: airport.name, flights: flights }
            """
            cursor = self.db.aql.execute(
                aql,
                bind_vars={'airport_code': airport_code, 'max_distance': max_distance}
            )
            results = list(cursor)

            if not results or not results[0]['flights']:
                return f"No flights found from {airport_code} within {max_distance} km."

            flights_df = pd.DataFrame(results[0]['flights']).sort_values('distance')
            response = [
                f"Flights within {max_distance} km from {airport_code} ({results[0]['source']}):",
                f"\nNumber of destinations: {len(flights_df)}",
                flights_df[['destination', 'city', 'country', 'distance']].to_string(index=False)
            ]
            result = "\n".join(response)
            self._cache_result(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"Error in find_flights_within_distance: {str(e)}")
            raise


    def get_top_airports_by_pagerank(self, top_n: int = 5) -> str:
        """
        List the top N airports based on their PageRank scores in the flight network.

        :param top_n: Number of airports to show
        :return: A text summary of the top N airports by PageRank
        """
        
        if not hasattr(self, 'pagerank') or not self.pagerank:
            return "PageRank data not available. Ensure metrics have been precomputed."

        # Sort by PageRank descending
        sorted_airports = sorted(self.pagerank, key=lambda a: self.pagerank[a], reverse=True)
        top_airports = sorted_airports[:top_n]

        response = [f"Top {top_n} Airports by PageRank:"]
        for rank, airport in enumerate(top_airports, start=1):
            response.append(f"{rank}. {airport} (score: {self.pagerank[airport]:.6f})")
        return "\n".join(response)


    def get_country_connectivity(self, country: str) -> str:
        """
        Show how many airports in the given country, and how they connect to the rest of the network.
        Returns a list of airports in that country and the number of direct flights from each.

        :param country: Name of the country (e.g. 'United States', 'Germany')
        :return: A text summary describing connectivity of that country's airports
        """
        # Query for all airports in that country
        aql = """
        FOR a IN airports
            FILTER a.country == @country
            LET numDirect = LENGTH(
                FOR v, e IN 1..1 ANY a GRAPH 'flights_graph'
                    RETURN 1
            )
            RETURN { iata: a.iata, name: a.name, city: a.city, direct_flights: numDirect }
        """
        try:
            results = list(self.db.aql.execute(aql, bind_vars={'country': country}))
            if not results:
                return f"No airports found in {country}."

            response = [
                f"Found {len(results)} airport(s) in {country}:"
            ]
            df = pd.DataFrame(results)
            df = df.sort_values('direct_flights', ascending=False)
            for _, row in df.iterrows():
                line = f"{row['iata']} ({row['name']} in {row['city']}) => {row['direct_flights']} direct flights"
                response.append(line)

            return "\n".join(response)

        except Exception as e:
            logger.error(f"Error in get_country_connectivity: {str(e)}")
            raise


    def find_multi_stop_route(self, source: str, target: str, max_stops: int = 2) -> str:
        """
        Find a route between two airports that uses up to 'max_stops' stops (i.e., up to max_stops+1 edges).
        This may ignore distance if you do not supply a weight to the path search. 
        (Essentially a BFS limited by path length.)

        :param source: IATA code for the origin airport
        :param target: IATA code for the destination airport
        :param max_stops: The maximum number of stops (not segments). 2 means up to 3 flight segments.
        :return: A summary of the found route or a 'No route found' message
        """
        if not all(validate_iata_code(code) for code in [source, target]):
            raise ValueError("Invalid IATA code format")

        if source not in self.G:
            return f"Source airport {source} not in the graph."
        if target not in self.G:
            return f"Target airport {target} not in the graph."

        # BFS or simple path enumeration up to a certain length
        # We'll use networkx's all_simple_paths with a cutoff
        cutoff = max_stops + 1  # edges, so stops is edges-1
        try:
            paths = nx.all_simple_paths(self.G, source=source, target=target, cutoff=cutoff)
            # Return the first path found, or a message if none
            found_any = False
            for path in paths:
                found_any = True
                stops = [f"{p} ({self.G.nodes[p]['city']}, {self.G.nodes[p]['country']})" for p in path]
                return " → ".join(stops)
            if not found_any:
                return f"No route found from {source} to {target} within {max_stops} stops."
        except Exception as e:
            logger.error(f"Error in find_multi_stop_route: {str(e)}")
            return f"No route found from {source} to {target} within {max_stops} stops."


    def get_busiest_route(self) -> str:
        """
        Find the single route (edge) in the graph that has the highest 'traffic' measure if stored.
        Note: This method assumes there's some 'traffic' attribute on edges in the graph, e.g. 'flights_per_day'.
        If no such attribute exists, it returns a fallback message.

        :return: A text summary of the busiest route or a fallback if attribute missing
        """
        # Check if there's an attribute like 'traffic' or 'flights_per_day' on edges
        # This is just an example; adapt to your dataset
        busiest_edge = None
        max_traffic = -1

        for u, v, data in self.G.edges(data=True):
            traffic = data.get('flights_per_day', None)  # data.get('traffic', None)  
            if traffic is not None and traffic > max_traffic:
                max_traffic = traffic
                busiest_edge = (u, v, traffic)

        if busiest_edge is None:
            return ("No 'traffic' attribute found on edges, or no edges have a valid traffic value. "
                    "Cannot determine the busiest route.")

        u, v, traffic_val = busiest_edge
        return (f"Busiest route found: {u} → {v} with a traffic value of {traffic_val}.\n"
                "Note: 'traffic' is a placeholder attribute and may need to be updated in your dataset.")
    
    def get_largest_airport_in_country(self, country: str) -> str:
        """
        Find the airport with the greatest number of direct connections in the specified country.
        Returns its IATA code, name, and total connections.
        """
        aql = """
        FOR a IN airports
            FILTER a.country == @country
            LET degree = LENGTH(
                FOR v, e IN 1..1 ANY a GRAPH 'flights_graph'
                    RETURN 1
            )
            SORT degree DESC
            LIMIT 1
            RETURN {
                iata: a.iata,
                name: a.name,
                city: a.city,
                connections: degree
            }
        """
        try:
            results = list(self.db.aql.execute(aql, bind_vars={'country': country}))
            if not results:
                return f"No airports found in {country}."
            airport = results[0]
            return (f"The largest airport in {country} by direct connections is "
                    f"{airport['iata']} ({airport['name']}, {airport['city']}) "
                    f"with {airport['connections']} direct connections.")
        except Exception as e:
            logger.error(f"Error in get_largest_airport_in_country: {str(e)}")
            raise

    def get_longest_flight_in_network(self) -> str:
        """
        Find the single route/edge in the graph with the greatest distance.
        Returns the origin, destination, and distance.
        """
        longest_edge = None
        max_distance = -1

        for u, v, data in self.G.edges(data=True):
            dist = data.get('distance', None)
            if dist is not None and dist > max_distance:
                max_distance = dist
                longest_edge = (u, v)

        if not longest_edge:
            return "No flight distance data found or graph is empty."

        source, target = longest_edge
        return (f"The longest flight in the network is {source} → {target}, "
                f"with a distance of {max_distance:.2f} km.")


    def get_shortest_flight_in_network(self) -> str:
        """
        Find the single route/edge in the graph with the smallest distance.
        Returns the origin, destination, and distance.
        """
        shortest_edge = None
        min_distance = float('inf')

        for u, v, data in self.G.edges(data=True):
            dist = data.get('distance', None)
            if dist is not None and dist < min_distance:
                min_distance = dist
                shortest_edge = (u, v)

        if not shortest_edge or min_distance == float('inf'):
            return "No flight distance data found or graph is empty."

        source, target = shortest_edge
        return (f"The shortest flight in the network is {source} → {target}, "
                f"with a distance of {min_distance:.2f} km.")


    def get_local_clustering_coefficient(self, airport_code: str) -> str:
        """
        Compute the local clustering coefficient for a given airport (node) in the graph.
        Measures how connected the airport's neighbors are to each other.
        """
        if airport_code not in self.G:
            return f"Airport {airport_code} not found in the network."
        coeffs = nx.clustering(self.G)  # unweighted clustering
        if airport_code in coeffs:
            return f"Local clustering coefficient for {airport_code} is {coeffs[airport_code]:.4f}"
        return f"Could not compute clustering for {airport_code}."


    def get_highest_betweenness_airports(self, top_n: int = 5) -> str:
        """
        List the top N airports by betweenness centrality.
        (Note: self.betweenness is presumably precomputed in _precompute_metrics().)
        """
        if not hasattr(self, 'betweenness') or not self.betweenness:
            return "Betweenness centrality data not available. Ensure metrics are precomputed."

        sorted_airports = sorted(
            self.betweenness,
            key=lambda a: self.betweenness[a],
            reverse=True
        )
        top_airports = sorted_airports[:top_n]
        lines = [f"Top {top_n} airports by betweenness centrality:"]
        for rank, airport in enumerate(top_airports, 1):
            lines.append(f"{rank}. {airport} => {self.betweenness[airport]:.6f}")
        return "\n".join(lines)


    def get_average_degree_of_network(self) -> str:
        """
        Calculate the average node degree across the entire network.
        Since this is a directed graph, use average of in_degree + out_degree or something similar.
        """
        degrees = [self.G.degree(n) for n in self.G.nodes]
        if not degrees:
            return "The network is empty or no nodes found."
        avg_degree = sum(degrees) / len(degrees)
        return f"The average degree (total connections per airport) in the network is {avg_degree:.2f}."


    def get_top_hub_airports(self, top_n: int = 5) -> str:
        """
        Identify the top N hub airports by total connections (in_degree + out_degree).
        """
        # For directed graphs, self.G.degree(...) is total connections
        nodes_by_degree = sorted(self.G.nodes, key=lambda n: self.G.degree(n), reverse=True)
        top_nodes = nodes_by_degree[:top_n]
        results = [f"Top {top_n} hub airports by total connections:"]
        for rank, n in enumerate(top_nodes, 1):
            deg = self.G.degree(n)
            results.append(f"{rank}. {n} => {deg} total connections")
        return "\n".join(results)


    def find_longest_route_within_k_stops(self, source: str, max_stops: int) -> str:
        """
        Explore all simple paths from 'source' up to 'max_stops' edges, 
        find the single path with the greatest total distance. 
        (Potentially large computational cost for bigger graphs.)
        """
        if source not in self.G:
            return f"Airport {source} not found in the network."
        best_path = None
        best_dist = -1
        # For each simple path up to cutoff
        paths = nx.all_simple_paths(self.G, source=source, cutoff=max_stops)
        for path in paths:
            # Sum distance of edges in path
            dist_sum = 0
            valid = True
            for i in range(len(path) - 1):
                edge_data = self.G.get_edge_data(path[i], path[i+1])
                if not edge_data or 'distance' not in edge_data:
                    valid = False
                    break
                dist_sum += edge_data['distance']
            if valid and dist_sum > best_dist:
                best_dist = dist_sum
                best_path = path

        if not best_path:
            return (f"No valid routes found from {source} within {max_stops} stops "
                    "or no distance data on edges.")
        route_str = " → ".join(best_path)
        return (f"Longest route from {source} within {max_stops} stops: {route_str} "
                f"({best_dist:.2f} km).")


    def list_countries_in_network(self) -> str:
        """
        Return a sorted list of unique countries present in the 'airports' collection.
        """
        aql = """
        FOR a IN airports
            COLLECT c = a.country
            RETURN c
        """
        try:
            countries = list(self.db.aql.execute(aql))
            if not countries:
                return "No countries found in the network."
            sorted_countries = sorted([c for c in countries if c])
            return "Countries in the network:\n" + "\n".join(sorted_countries)
        except Exception as e:
            logger.error(f"Error in list_countries_in_network: {str(e)}")
            raise


    def find_airports_near_geolocation(self, lat: float, lon: float, radius_km: float = 100.0) -> str:
        """
        Find airports within 'radius_km' of a given latitude/longitude.
        Assumes each airport doc has 'latitude' and 'longitude' fields in the DB.
        """
        # Basic approach: use Haversine formula in AQL or Python.
        # We'll do a simple AQL approach if the DB supports geo indexes.
        # If not, we can implement a local Haversine check in Python.
        # For brevity, let's do a local python approach:
        from math import radians, sin, cos, sqrt, atan2

        def haversine(lat1, lon1, lat2, lon2):
            R = 6371.0  # Earth radius in km
            dlat = radians(lat2 - lat1)
            dlon = radians(lon2 - lon1)
            a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            return R * c

        aql = """
        FOR a IN airports
            RETURN { iata: a.iata, name: a.name, lat: a.latitude, lon: a.longitude }
        """
        try:
            results = list(self.db.aql.execute(aql))
            nearby = []
            for r in results:
                if r['lat'] is not None and r['lon'] is not None:
                    distance = haversine(lat, lon, r['lat'], r['lon'])
                    if distance <= radius_km:
                        nearby.append((r['iata'], r['name'], distance))

            if not nearby:
                return (f"No airports found within {radius_km} km of ({lat}, {lon}).")

            # Sort by distance ascending
            nearby.sort(key=lambda x: x[2])
            lines = [f"Airports within {radius_km} km of ({lat:.4f}, {lon:.4f}):"]
            for iata, name, dist in nearby:
                lines.append(f"{iata} ({name}) => {dist:.2f} km")
            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Error in find_airports_near_geolocation: {str(e)}")
            raise



# ---------------------------
# 6) OllamaWrapper
# ---------------------------
class OllamaWrapper(LLM):
    """
    Simple wrapper for local Ollama server, ensuring we handle a single string prompt.
    """
    base_url: str = Field(default="http://localhost:11434")
    model: str = Field(default="llama2")

    def __init__(self, base_url: str = None, model: str = None, **kwargs):
        super().__init__(**kwargs)
        if base_url:
            self.base_url = base_url
        if model:
            self.model = model

    def _llm_type(self) -> str:
        return "ollama"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Called by LLM.invoke() or LLM.generate() with a single string prompt.
        """
        # The prompt must be a string (LangChain >0.1.7).
        if not isinstance(prompt, str):
            raise ValueError(
                f"OllamaWrapper _call expects a single string. Got {type(prompt)}."
            )

        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": config.OLLAMA_TEMPERATURE,
                "top_p": config.OLLAMA_TOP_P,
                "num_predict": config.OLLAMA_MAX_TOKENS
            }
        }
        if stop:
            data["stop"] = stop

        try:
            metrics.record_llm_call()
            logger.info(f"Ollama request → {self.base_url} / model={self.model}")
            resp = requests.post(f"{self.base_url}/api/generate", json=data)
            resp.raise_for_status()
            result = resp.json()
            if "response" not in result:
                raise ValueError(f"Unexpected Ollama response format: {result}")
            return result["response"]
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Ollama request failed: {str(e)}")

# ---------------------------
# 7) Initialize Ollama
# ---------------------------
def initialize_llm():
    print("Using Ollama for local LLaMA inference...")
    try:
        wrapper = OllamaWrapper(model=config.OLLAMA_MODEL, base_url=config.OLLAMA_BASE_URL)
        r = requests.get(f"{wrapper.base_url}/api/version")
        r.raise_for_status()
        return wrapper
    except Exception as e:
        print(f"Warning: Could not initialize with model {config.OLLAMA_MODEL}: {str(e)}")
        print("Falling back to 'llama2' ...")
        return OllamaWrapper(model="llama2", base_url=config.OLLAMA_BASE_URL)

# ---------------------------
# 8) Agent node function
# ---------------------------
def agent_node_fn(state: GraphState, llm: LLM) -> GraphState:
    """
    Combine the list of messages into a single string, then call llm.invoke(...)
    and append the result as an AIMessage.
    """
    conversation = state["messages"]
    if not conversation:
        return state

    # Build one big prompt from the messages
    prompt_parts = []
    for msg in conversation:
        if isinstance(msg, SystemMessage):
            prompt_parts.append(f"System: {msg.content}")
        elif isinstance(msg, HumanMessage):
            prompt_parts.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            prompt_parts.append(f"Assistant: {msg.content}")
        else:
            # In case any other message
            role = msg.get("role", "Unknown")
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")
    # Join them with newlines
    prompt_text = "\n".join(prompt_parts)

    # LOG the prompt before calling the LLM
    logger.debug("Agent node: sending this prompt to the LLM:\n%s", prompt_text)

    # Call LLM
    llm_output = llm.invoke(prompt_text) 


    # LOG the response from the LLM
    logger.debug("Agent node: received this response from the LLM:\n%s", llm_output)

    # Append as AIMessage
    state["messages"].append(AIMessage(content=llm_output))
    return state

# ---------------------------
# 9) Create agent graph
# ---------------------------
def create_agent_graph(llm: LLM, tools: List[BaseTool]) -> StateGraph:
    tool_node = ToolNode(tools)
    workflow = StateGraph(GraphState)

    def agent_node(state: GraphState) -> GraphState:
        return agent_node_fn(state, llm)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tool_executor", tool_node)

    def route(state: GraphState) -> Union[str, List[str]]:
        messages = state["messages"]
        if not messages:
            return END
        last_message = messages[-1]
        if isinstance(last_message, AIMessage):
            # If AI message contains "TOOL:", route to tool_executor
            # (You can adjust your own logic or use a structured approach to parse out tools)
            if "TOOL:" in last_message.content:
                logger.debug("Routing - LLM has chosen to call a tool: %s", last_message.content)
                return "tool_executor"
            else:
                return END
        return END

    workflow.add_conditional_edges(
        "agent",
        route,
        {
            "tool_executor": "tool_executor",
            END: END
        }
    )
    # after tool execution, we return to agent to incorporate the tool result
    workflow.add_edge("tool_executor", "agent")
    workflow.set_entry_point("agent")
    return workflow.compile()

# ---------------------------
# 10) Main program
# ---------------------------
def main():
    health_status = health_check.run_health_check()
    if not health_status['healthy']:
        logger.error("Application failed health check. Please check logs for details.")
        return

    # Initialize FlightGraphTools and GraphQueryTools
    flight_tools = FlightGraphTools()
    graph_tools = GraphQueryTools(flight_tools.db, flight_tools.G)

    tool_list = [
        Tool(
            name="TextToAQL",
            func=graph_tools.text_to_aql_to_text,
            description="Convert natural language to AQL query, execute it, and return results in natural language. Use for questions about direct flights, routes, and general flight information."
        ),
        Tool(
            name="TextToNetworkX",
            func=graph_tools.text_to_nx_algorithm_to_text,
            description="Execute NetworkX algorithms based on natural language query. Use for complex graph analysis like centrality, shortest paths, and connectivity."
        ),
        Tool(
            name="FindDirectFlights",
            func=flight_tools.find_direct_flights,
            description="Find all direct flights from a given airport. Input: IATA code, e.g. 'LAX'."
        ),
        Tool(
            name="CountDirectFlights",
            func=flight_tools.count_direct_flights,
            description="Return the number of direct flights from the specified airport. Input: 'LAX'."
        ),
        Tool(
            name="CheckDirectFlight",
            func=flight_tools.check_direct_flight,
            description="Check if there's a direct flight between two airports. Input: 'LAX,JFK'."
        ),
        Tool(
            name="FindShortestPath",
            func=flight_tools.find_shortest_path,
            description="Find the shortest path (by distance) between two airports. Input: 'LAX,JFK'."
        ),
        Tool(
            name="FindMinimumStopRoute",
            func=flight_tools.find_minimum_stop_route,
            description="Find a route with the fewest stops between two airports. Input: 'LAX,JFK'."
        ),
        Tool(
            name="CompareRoutes",
            func=flight_tools.compare_routes,
            description="Compare the distance-based shortest route vs. fewest-stop route. Input: 'LAX,JFK'."
        ),
        Tool(
            name="AnalyzeAirportImportance",
            func=flight_tools.analyze_airport_importance,
            description="Analyze a single airport's importance (PageRank, etc.). Input: 'LAX'."
        ),
        Tool(
            name="CompareAirportImportance",
            func=flight_tools.compare_airport_importance,
            description="Compare importance metrics for two airports. Input: 'LAX,JFK'."
        ),
        Tool(
            name="GetTopIncomingAirports",
            func=flight_tools.get_top_incoming_airports,
            description="List top N airports by incoming flights. Input: integer or blank (defaults=5)."
        ),
        Tool(
            name="GetTopOutgoingAirports",
            func=flight_tools.get_top_outgoing_airports,
            description="List top N airports by outgoing flights. Input: integer or blank (defaults=5)."
        ),
        Tool(
            name="AverageFlightDistance",
            func=flight_tools.average_flight_distance,
            description="Calculate average distance of all direct flights from an airport. Input: 'LAX'."
        ),
        Tool(
            name="FlightDistanceDistribution",
            func=flight_tools.flight_distance_distribution,
            description="Summary stats (min, max, mean, median) for all flight distances. No input."
        ),
        Tool(
            name="GetIsolatedAirports",
            func=flight_tools.get_isolated_airports,
            description="List airports with no connections. No input."
        ),
        Tool(
            name="ListAirportsByCountry",
            func=flight_tools.list_airports_by_country,
            description="List all airports located in a specified country. Input: 'Germany'."
        ),
        Tool(
            name="FindFlightsWithinDistance",
            func=flight_tools.find_flights_within_distance,
            description="List direct flights from an airport that are within a certain max distance. Input: 'LAX,1000' (IATA, distance)."
        ),
        Tool(
            name="GetTopAirportsByPageRank",
            func=flight_tools.get_top_airports_by_pagerank,
            description="List top N airports by PageRank. Input: integer or blank (defaults=5)."
        ),
        Tool(
            name="GetCountryConnectivity",
            func=flight_tools.get_country_connectivity,
            description="Show how many airports in a country and how many direct flights each has. Input: 'Germany'."
        ),
        Tool(
            name="FindMultiStopRoute",
            func=flight_tools.find_multi_stop_route,
            description="Find a route between two airports with up to 'max_stops' stops. Input: 'LAX,JFK,2'."
        ),
        Tool(
            name="GetBusiestRoute",
            func=flight_tools.get_busiest_route,
            description="Find the single route with the highest 'traffic' attribute in the network. No input."
        ),
        Tool(
            name="GetLargestAirportInCountry",
            func=flight_tools.get_largest_airport_in_country,
            description="Get the airport with the greatest number of direct connections in a country. Input: 'Germany'."
        ),
        Tool(
            name="GetLongestFlightInNetwork",
            func=flight_tools.get_longest_flight_in_network,
            description="Find the single longest (by distance) flight/edge in the entire network. No input."
        ),
        Tool(
            name="GetShortestFlightInNetwork",
            func=flight_tools.get_shortest_flight_in_network,
            description="Find the single shortest (by distance) flight/edge in the entire network. No input."
        ),
        Tool(
            name="GetLocalClusteringCoefficient",
            func=flight_tools.get_local_clustering_coefficient,
            description="Compute the local clustering coefficient for a given airport. Input: 'LAX'."
        ),
        Tool(
            name="GetHighestBetweennessAirports",
            func=flight_tools.get_highest_betweenness_airports,
            description="List top N airports by betweenness centrality. Input: integer or blank (defaults=5)."
        ),
        Tool(
            name="GetAverageDegreeOfNetwork",
            func=flight_tools.get_average_degree_of_network,
            description="Compute average node degree across the entire network. No input."
        ),
        Tool(
            name="GetTopHubAirports",
            func=flight_tools.get_top_hub_airports,
            description="List top N hub airports by total connections (in+out). Input: integer or blank (defaults=5)."
        ),
        Tool(
            name="FindLongestRouteWithinKStops",
            func=flight_tools.find_longest_route_within_k_stops,
            description="From a source airport, find the single route with greatest total distance within N stops. Input: 'LAX,3'."
        ),
        Tool(
            name="ListCountriesInNetwork",
            func=flight_tools.list_countries_in_network,
            description="List all unique countries present in the network. No input."
        ),
        Tool(
            name="FindAirportsNearGeolocation",
            func=flight_tools.find_airports_near_geolocation,
            description="Find airports within a certain radius of a latitude/longitude. Input: '34.0522,-118.2437,100'."
        ),
    ]


    llm = initialize_llm()
    agent_executor = create_agent_graph(llm, tool_list)

    print("\nFlight Network Analysis System Ready!")
    print("You can ask questions about:")
    print("1. Direct flights from an airport (e.g., 'Show me all direct flights from LAX')")
    print("2. Shortest path between airports (e.g., 'What's the best route from JFK to LHR?')")
    print("3. Airport analysis (e.g., 'Analyze the importance of DXB in the network')")
    print("\nType 'quit' to exit")
    print("Type 'metrics' to see system metrics")
    print("Type 'health' to check system health")

    while True:
        try:
            query = input("\nWhat would you like to know? ").strip()
            if query.lower() == 'quit':
                break
            elif query.lower() == 'metrics':
                current_metrics = metrics.get_metrics()
                print("\nSystem Metrics:")
                print(f"Total Queries: {current_metrics['queries']}")
                print(f"Cache Hit Rate: {current_metrics.get('cache_hit_rate', 0):.2%}")
                print(f"Average Response Time: {current_metrics.get('avg_response_time', 0):.2f}ms")
                print(f"LLM Calls: {current_metrics['llm_calls']}")
                print("\nTool Usage:")
                for tool_name, count in current_metrics['tool_usage'].items():
                    print(f" - {tool_name}: {count}")
                continue
            elif query.lower() == 'health':
                status = health_check.get_status()
                print("\nSystem Health:")
                print(f"Database: {'✓' if status['database'] else '✗'}")
                print(f"LLM: {'✓' if status['llm'] else '✗'}")
                print(f"Graph: {'✓' if status['graph'] else '✗'}")
                print(f"Overall Status: {'✓' if status['healthy'] else '✗'}")
                if status['errors']:
                    print("\nErrors:")
                    for err in status['errors']:
                        print(f" - {err}")
                continue

            metrics.record_query()
            start_time = time.time()

            print_status("Processing your request...")
            try:
                state = {
                    "messages": [
                        SystemMessage(content="You are a helpful assistant that analyzes flight networks. "
                                              "Use the provided tools to answer questions about flights, routes, "
                                              "and airport importance. Use IATA codes in your answers."),
                        HumanMessage(content=query)
                    ],
                    "tools": {tool.name: tool for tool in tool_list},
                    "tool_results": {},
                    "current_tool": None
                }

                result = agent_executor.invoke(state)
                response_time = (time.time() - start_time) * 1000
                metrics.record_response_time(response_time)

                # Grab the last AIMessage from result
                messages = result["messages"]
                ai_msgs = [m for m in messages if isinstance(m, AIMessage)]
                if ai_msgs:
                    print("\nResponse:", ai_msgs[-1].content)
                else:
                    print("\n(No AI response generated.)")

            except Exception as e:
                metrics.record_error()
                print(f"\nError processing query: {str(e)}")
                print("Please try rephrasing or use specific IATA codes (e.g., 'LAX').")

            if metrics.metrics['queries'] % 10 == 0:
                health_check.run_health_check()

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            metrics.record_error()
            print(f"\nError processing request: {str(e)}")
            print("Please try rephrasing or use specific IATA codes (e.g., 'LAX').")

    print("\nFinal System Metrics:")
    final_metrics = metrics.get_metrics()
    print(f"Total Queries: {final_metrics['queries']}")
    print(f"Cache Hit Rate: {final_metrics['cache_hit_rate']:.2%}")
    print(f"Average Response Time: {final_metrics.get('avg_response_time', 0):.2f}ms")
    print(f"Total Errors: {final_metrics['errors']}")

if __name__ == "__main__":
    main()
