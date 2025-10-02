import pandas as pd
import networkx as nx
from arango import ArangoClient
import pickle
from collections import Counter

def analyze_datasets():
    print("Analyzing differences between ArangoDB and NetworkX datasets...")
    
    # Connect to ArangoDB
    print("\n1. Connecting to databases...")
    client = ArangoClient(hosts="http://localhost:8529")
    db = client.db("flights_db", username="root", password="")
    
    # Load NetworkX graph
    with open('data/graph_networkx.pkl', 'rb') as f:
        G = pickle.load(f)
    
    # Compare basic statistics
    print("\n2. Basic Statistics:")
    
    # ArangoDB stats
    aql = """
    RETURN {
        airports: LENGTH(FOR a IN airports RETURN 1),
        routes: LENGTH(FOR r IN routes RETURN 1),
        countries: LENGTH(UNIQUE(FOR a IN airports RETURN a.country))
    }
    """
    arango_stats = next(db.aql.execute(aql))
    
    print("\nArangoDB Statistics:")
    print(f"Number of airports: {arango_stats['airports']}")
    print(f"Number of routes: {arango_stats['routes']}")
    print(f"Number of countries: {arango_stats['countries']}")
    
    print("\nNetworkX Statistics:")
    print(f"Number of airports: {G.number_of_nodes()}")
    print(f"Number of routes: {G.number_of_edges()}")
    print(f"Number of countries: {len(set(nx.get_node_attributes(G, 'country').values()))}")
    
    # Compare specific major airports
    print("\n3. Comparing Major Airports:")
    major_airports = ['LAX', 'JFK', 'LHR', 'CDG', 'DXB']
    
    for iata in major_airports:
        print(f"\nAnalyzing {iata}:")
        
        # ArangoDB data
        aql = """
        FOR airport IN airports
            FILTER airport.iata == @iata
            LET out_routes = (
                FOR r IN routes
                    FILTER r._from == CONCAT('airports/', airport._key)
                    RETURN r
            )
            LET in_routes = (
                FOR r IN routes
                    FILTER r._to == CONCAT('airports/', airport._key)
                    RETURN r
            )
            RETURN {
                "name": airport.name,
                "outbound": LENGTH(out_routes),
                "inbound": LENGTH(in_routes)
            }
        """
        arango_data = next(db.aql.execute(aql, bind_vars={'iata': iata}))
        
        print("ArangoDB:")
        print(f"  Name: {arango_data['name']}")
        print(f"  Outbound routes: {arango_data['outbound']}")
        print(f"  Inbound routes: {arango_data['inbound']}")
        
        if iata in G:
            print("NetworkX:")
            print(f"  Name: {G.nodes[iata]['name']}")
            print(f"  Outbound routes: {G.out_degree(iata)}")
            print(f"  Inbound routes: {G.in_degree(iata)}")
        else:
            print("NetworkX: Airport not found")
    
    # Analyze route patterns
    print("\n4. Route Pattern Analysis:")
    
    # Get route patterns from ArangoDB
    aql = """
    FOR r IN routes
        COLLECT airline = r.airline WITH COUNT INTO count
        SORT count DESC
        LIMIT 5
        RETURN {airline: airline, count: count}
    """
    print("\nTop 5 Airlines by Route Count (ArangoDB):")
    for result in db.aql.execute(aql):
        print(f"{result['airline']}: {result['count']} routes")
    
    # Compare common routes
    print("\n5. Common Route Analysis:")
    
    # Get a sample of routes from both systems
    aql = """
    FOR r IN routes
        LIMIT 1000
        RETURN {
            source: PARSE_IDENTIFIER(r._from).key,
            destination: PARSE_IDENTIFIER(r._to).key
        }
    """
    arango_routes = set()
    for route in db.aql.execute(aql):
        arango_routes.add((route['source'], route['destination']))
    
    nx_routes = set(G.edges())
    
    common_routes = arango_routes.intersection(nx_routes)
    print(f"\nAnalyzed 1000 routes:")
    print(f"Routes in both datasets: {len(common_routes)}")
    print(f"Routes only in ArangoDB sample: {len(arango_routes - nx_routes)}")
    print(f"Routes only in NetworkX sample: {len(nx_routes - arango_routes)}")
    
    # Route Distance Analysis
    print("\n6. Route Distance Distribution:")
    print("\nLongest routes from LAX (ArangoDB):")
    
    longest_routes_query = """
    FOR r IN routes
    FILTER r._from == CONCAT('airports/', @airport)
    LET dest = DOCUMENT(r._to)
    SORT r.distance DESC
    LIMIT 5
    RETURN {
        destination: dest.iata_code,
        airport_name: dest.name,
        city: dest.city,
        country: dest.country,
        distance: r.distance
    }
    """
    
    longest_routes = db.aql.execute(longest_routes_query, bind_vars={'airport': 'LAX'})
    for route in longest_routes:
        print(f"LAX → {route['destination']}: {route['airport_name']}, {route['city']}, {route['country']} ({route['distance']} km)")

if __name__ == "__main__":
    analyze_datasets() 