import networkx as nx
import pandas as pd
from typing import List, Dict, Any, Optional
from arango.client import ArangoClient
from ..utils.helpers import logger, validate_iata_code, retry_on_failure
from ..utils.cache import db_cache
from ..config.settings import config
import json
import pickle
from networkx import Graph

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

@retry_on_failure()
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