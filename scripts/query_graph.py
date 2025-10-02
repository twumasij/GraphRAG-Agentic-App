from arango import ArangoClient
import networkx as nx
import pandas as pd
import pickle

def connect_to_arango():
    """Connect to ArangoDB flights database"""
    client = ArangoClient(hosts="http://localhost:8529")
    db = client.db("flights_db", username="root", password="")
    return db

def query_direct_flights(db, airport_code="LAX"):
    """Query direct flights from specified airport using AQL"""
    print(f"\nQuerying direct flights from {airport_code}...")
    
    aql = """
    FOR airport IN airports
        FILTER airport.iata == @airport_code
        FOR v, e IN 1..1 ANY airport GRAPH 'flights_graph'
            RETURN {
                destination: v.iata,
                airport_name: v.name,
                city: v.city,
                country: v.country,
                distance: e.distance
            }
    """
    
    cursor = db.aql.execute(aql, bind_vars={'airport_code': airport_code})
    results = list(cursor)
    
    if results:
        flights_df = pd.DataFrame(results)
        flights_df = flights_df.sort_values('distance')
        
        print(f"\nFound {len(results)} direct destinations")
        print("\nTop 10 closest destinations:")
        print(flights_df[['destination', 'city', 'country', 'distance']].head(10).to_string(index=False))
        
        print("\nTop 10 furthest destinations:")
        print(flights_df[['destination', 'city', 'country', 'distance']].tail(10).to_string(index=False))
    else:
        print(f"No flights found from {airport_code}")

def calculate_pagerank():
    """Calculate PageRank using NetworkX"""
    print("\nCalculating PageRank for all airports...")
    
    # Load the NetworkX graph
    with open('data/graph_networkx.pkl', 'rb') as f:
        G = pickle.load(f)
    
    # Calculate PageRank
    pagerank = nx.pagerank(G, alpha=0.85)
    
    # Convert to DataFrame and sort
    pr_df = pd.DataFrame.from_dict(pagerank, orient='index', columns=['pagerank'])
    pr_df = pr_df.sort_values('pagerank', ascending=False)
    
    # Add airport information
    top_airports = []
    for iata in pr_df.head(20).index:
        airport_data = G.nodes[iata]
        top_airports.append({
            'iata': iata,
            'name': airport_data['name'],
            'city': airport_data['city'],
            'country': airport_data['country'],
            'pagerank': pagerank[iata]
        })
    
    # Display results
    print("\nTop 20 airports by PageRank:")
    top_df = pd.DataFrame(top_airports)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(top_df[['iata', 'name', 'city', 'country', 'pagerank']].to_string(index=False))

def main():
    try:
        # Connect to ArangoDB
        db = connect_to_arango()
        
        # Query direct flights from LAX
        query_direct_flights(db, "LAX")
        
        # Calculate and display PageRank
        calculate_pagerank()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 