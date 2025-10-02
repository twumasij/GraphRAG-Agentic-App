import pandas as pd
import networkx as nx
from arango import ArangoClient
import pickle
from math import radians, sin, cos, sqrt, atan2

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on Earth"""
    R = 6371  # Earth's radius in kilometers

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c

    return distance

def test_queries():
    # Connect to ArangoDB
    print("Connecting to ArangoDB...")
    client = ArangoClient(hosts="http://localhost:8529")
    db = client.db("flights_db", username="root", password="")
    
    # Load NetworkX graph
    print("Loading NetworkX graph...")
    with open('data/graph_networkx.pkl', 'rb') as f:
        G = pickle.load(f)
    
    # Test Query 1: Find all direct flights from LAX
    print("\nQuery 1: Direct flights from LAX")
    print("ArangoDB:")
    aql = """
    FOR airport IN airports
        FILTER airport.iata == 'LAX'
        LET destinations = (
            FOR v, e IN 1 OUTBOUND airport._id GRAPH 'flights_graph'
                RETURN {
                    destination: v.iata,
                    airport_name: v.name,
                    city: v.city,
                    country: v.country
                }
        )
        RETURN {
            source: airport.name,
            num_flights: LENGTH(destinations),
            flights: (
                FOR dest IN destinations
                LIMIT 5
                RETURN dest
            )
        }
    """
    cursor = db.aql.execute(aql)
    results = list(cursor)
    for result in results:
        print(f"From {result['source']}:")
        print(f"Total outbound flights: {result['num_flights']}")
        print("Sample destinations:")
        for flight in result.get('flights', []):
            if flight['destination']:  # Filter out None values
                print(f"  → {flight['destination']}: {flight['airport_name']}, {flight['city']}, {flight['country']}")
    
    print("\nNetworkX:")
    lax_neighbors = list(G.successors('LAX'))  # Use successors instead of neighbors for directed graph
    print(f"Total outbound flights: {len(lax_neighbors)}")
    print("Sample destinations:")
    for dest in lax_neighbors[:5]:
        node = G.nodes[dest]
        print(f"  → {dest}: {node['name']}, {node['city']}, {node['country']}")
    
    # Test Query 2: Find shortest path between SFO and JFK
    print("\nQuery 2: Shortest path from SFO to JFK")
    print("ArangoDB:")
    aql = """
    FOR v, e IN OUTBOUND SHORTEST_PATH 'airports/SFO' TO 'airports/JFK' GRAPH 'flights_graph'
        COLLECT WITH COUNT INTO length
        RETURN length
    """
    cursor = db.aql.execute(aql)
    path_length = next(cursor, None)
    
    if path_length:
        print(f"Direct flight exists (path length: {path_length})")
    else:
        print("No direct path found")
    
    print("\nNetworkX:")
    try:
        path = nx.shortest_path(G, 'SFO', 'JFK', weight=None)  # Unweighted shortest path
        print(f"Path: {' → '.join(path)}")
        print(f"Number of hops: {len(path)-1}")
    except nx.NetworkXNoPath:
        print("No path found")
    
    # Test Query 3: Find top 5 airports by outbound routes
    print("\nQuery 3: Top 5 airports by outbound routes")
    print("ArangoDB:")
    aql = """
    FOR airport IN airports
        LET num_routes = LENGTH(
            FOR v, e IN 1 OUTBOUND airport._id GRAPH 'flights_graph'
                RETURN 1
        )
        SORT num_routes DESC
        LIMIT 5
        RETURN {
            airport: airport.name,
            iata: airport.iata,
            outbound_routes: num_routes
        }
    """
    cursor = db.aql.execute(aql)
    results = list(cursor)
    for result in results:
        print(f"{result['iata']} ({result['airport']}): {result['outbound_routes']} outbound routes")
    
    print("\nNetworkX:")
    out_degrees = sorted([(node, G.out_degree(node)) for node in G.nodes()], 
                        key=lambda x: x[1], reverse=True)[:5]
    for node, degree in out_degrees:
        print(f"{node} ({G.nodes[node]['name']}): {degree} outbound routes")

if __name__ == "__main__":
    test_queries() 