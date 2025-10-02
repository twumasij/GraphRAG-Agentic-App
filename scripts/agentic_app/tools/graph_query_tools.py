import networkx as nx
from typing import Dict, Any, Optional
from ..utils.helpers import logger
from ..config.settings import config
from arango.client import ArangoClient
import pickle

class GraphQueryTools:
    def __init__(self, db, G):
        self.db = db
        self.G = G
        self.schema = self._get_schema()

    def _get_schema(self) -> Dict[str, Any]:
        """Get the schema of the graph from ArangoDB."""
        try:
            collections = self.db.collections()
            schema = {
                "vertices": [],
                "edges": []
            }
            for col in collections:
                if col['type'] == 3:  # Edge collection
                    schema["edges"].append(col['name'])
                elif col['type'] == 2:  # Document collection
                    schema["vertices"].append(col['name'])
            return schema
        except Exception as e:
            logger.error(f"Error getting schema: {str(e)}")
            return {"vertices": [], "edges": []}

    def text_to_aql_to_text(self, query: str) -> str:
        """
        Convert natural language query to AQL, execute it, and return results in natural language.
        
        Args:
            query: Natural language query about the flight network
            
        Returns:
            Natural language response based on AQL query results
        """
        try:
            # Convert query to AQL based on common patterns
            aql = self._natural_to_aql(query)
            if not aql:
                return "I couldn't convert your query to AQL. Please try rephrasing."

            # Execute AQL query
            cursor = self.db.aql.execute(aql)
            results = list(cursor)

            # Convert results to natural language
            return self._results_to_text(query, results)

        except Exception as e:
            logger.error(f"Error in text_to_aql_to_text: {str(e)}")
            return f"Error processing query: {str(e)}"

    def _natural_to_aql(self, query: str) -> Optional[str]:
        """Convert natural language to AQL query."""
        query = query.lower()
        
        # Direct flights pattern
        if "direct flights" in query or "flights from" in query:
            airport = self._extract_airport_code(query)
            if airport:
                return f"""
                FOR airport IN airports
                    FILTER airport.iata == '{airport}'
                    LET flights = (
                        FOR v, e IN 1..1 ANY airport flights_graph
                        RETURN {{
                            destination: v.iata,
                            airport_name: v.name,
                            city: v.city,
                            country: v.country,
                            distance: e.distance
                        }}
                    )
                    RETURN {{ source: airport.name, flights: flights }}
                """

        # Route between airports pattern
        if "route between" in query or "path from" in query:
            source, target = self._extract_airport_pair(query)
            if source and target:
                return f"""
                FOR path IN OUTBOUND K_SHORTEST_PATHS
                    '{source}' TO '{target}'
                    flights_graph
                    OPTIONS {{bfs: true, uniqueVertices: 'path'}}
                    LIMIT 1
                    RETURN path
                """

        return None

    def _extract_airport_code(self, query: str) -> Optional[str]:
        """Extract IATA airport code from query."""
        import re
        # Look for 3-letter codes that appear after common phrases
        patterns = [
            r'from\s+([A-Z]{3})',
            r'to\s+([A-Z]{3})',
            r'at\s+([A-Z]{3})',
            r'for\s+([A-Z]{3})',
            r'airport\s+([A-Z]{3})',
            r'([A-Z]{3})'  # fallback to any 3-letter code
        ]
        for pattern in patterns:
            matches = re.findall(pattern, query.upper())
            if matches:
                return matches[0]
        return None

    def _extract_airport_pair(self, query: str) -> tuple[Optional[str], Optional[str]]:
        """Extract pair of IATA airport codes from query."""
        import re
        # Look for common patterns of airport pairs
        patterns = [
            r'from\s+([A-Z]{3}).*?to\s+([A-Z]{3})',
            r'between\s+([A-Z]{3}).*?and\s+([A-Z]{3})',
            r'([A-Z]{3}).*?(?:to|and)\s+([A-Z]{3})'
        ]
        for pattern in patterns:
            matches = re.findall(pattern, query.upper())
            if matches:
                return matches[0]
        # Fallback to any two 3-letter codes
        matches = re.findall(r'([A-Z]{3})', query.upper())
        return (matches[0], matches[1]) if len(matches) >= 2 else (None, None)

    def _results_to_text(self, query: str, results: list) -> str:
        """Convert AQL results to natural language response."""
        if not results:
            return "No results found for your query."

        if "direct flights" in query.lower():
            if not results[0].get('flights'):
                return f"No flights found from the specified airport."
            
            flights = results[0]['flights']
            response = [f"Found {len(flights)} direct flights from {results[0]['source']}:"]
            
            for flight in flights[:5]:  # Show first 5 flights
                response.append(
                    f"- {flight['destination']} ({flight['city']}, {flight['country']}) - {flight['distance']}km"
                )
            
            if len(flights) > 5:
                response.append(f"... and {len(flights)-5} more destinations")
                
            return "\n".join(response)

        # Generic response for other types of queries
        return f"Query results: {str(results)}"

    def text_to_nx_algorithm_to_text(self, query: str) -> str:
        """
        Execute NetworkX algorithms based on natural language query and return results.
        
        Args:
            query: Natural language query about the flight network
            
        Returns:
            Natural language response based on NetworkX algorithm results
        """
        try:
            # Identify and execute appropriate NetworkX algorithm
            result = self._execute_nx_algorithm(query)
            if not result:
                return "I couldn't determine which NetworkX algorithm to use for your query."

            # Convert results to natural language
            return self._nx_results_to_text(query, result)

        except Exception as e:
            logger.error(f"Error in text_to_nx_algorithm_to_text: {str(e)}")
            return f"Error processing query: {str(e)}"

    def _execute_nx_algorithm(self, query: str) -> Optional[Any]:
        """Execute appropriate NetworkX algorithm based on query."""
        query = query.lower()
        
        # Centrality analysis
        if "important" in query or "central" in query:
            return {
                'pagerank': nx.pagerank(self.G),
                'betweenness': nx.betweenness_centrality(self.G)
            }
            
        # Path analysis
        if "shortest path" in query or "best route" in query:
            source, target = self._extract_airport_pair(query)
            if source and target:
                try:
                    return nx.shortest_path(self.G, source, target, weight='distance')
                except nx.NetworkXNoPath:
                    return None

        # Connectivity analysis
        if "connected" in query or "connectivity" in query:
            return {
                'is_connected': nx.is_connected(self.G.to_undirected()),
                'components': list(nx.connected_components(self.G.to_undirected()))
            }

        return None

    def _nx_results_to_text(self, query: str, results: Any) -> str:
        """Convert NetworkX algorithm results to natural language response."""
        if results is None:
            return "No results found for your query."

        # Handle centrality results
        if isinstance(results, dict) and 'pagerank' in results:
            top_airports = sorted(
                results['pagerank'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            response = ["Top 5 most important airports by PageRank:"]
            for airport, score in top_airports:
                response.append(f"- {airport}: {score:.4f}")
            return "\n".join(response)

        # Handle path results
        if isinstance(results, list) and all(isinstance(x, str) for x in results):
            path_str = " → ".join(results)
            return f"Found path: {path_str}"

        # Handle connectivity results
        if isinstance(results, dict) and 'is_connected' in results:
            if results['is_connected']:
                return "The flight network is fully connected."
            else:
                return f"The flight network has {len(results['components'])} disconnected components."

        return f"Analysis results: {str(results)}" 