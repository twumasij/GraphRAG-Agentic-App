from arango import ArangoClient
import pandas as pd
import numpy as np

def connect_to_arango():
    """Connect to ArangoDB and create/get database"""
    print("Connecting to ArangoDB...")
    client = ArangoClient(hosts="http://localhost:8529")
    
    # Connect to "_system" database as root user
    sys_db = client.db("_system", username="root", password="")
    
    # Create a new database if it doesn't exist
    db_name = "flights_db"
    if not sys_db.has_database(db_name):
        sys_db.create_database(db_name)
    
    # Connect to "flights_db" database
    db = client.db(db_name, username="root", password="")
    return db

def create_collections(db):
    """Create or get collections for airports and routes"""
    # Create airports collection if it doesn't exist
    if db.has_collection("airports"):
        airports = db.collection("airports")
    else:
        airports = db.create_collection("airports")
    
    # Create routes edge collection if it doesn't exist
    if db.has_collection("routes"):
        routes = db.collection("routes")
    else:
        routes = db.create_collection("routes", edge=True)
    
    # Create index on airport IATA codes
    if not any(idx['fields'] == ['iata'] for idx in airports.indexes()):
        airports.add_hash_index(fields=['iata'], unique=True)
    
    return airports, routes

def load_graph_data():
    """Load the preprocessed graph data"""
    print("Loading graph data from CSV...")
    nodes_df = pd.read_csv('data/graph_nodes.csv')
    edges_df = pd.read_csv('data/graph_edges.csv')
    return nodes_df, edges_df

def insert_airports(airports_collection, nodes_df):
    """Insert airport nodes into ArangoDB"""
    print("Inserting airports...")
    airport_docs = []
    
    # Convert numeric columns to Python native types
    nodes_df['airport_id'] = nodes_df['airport_id'].astype(int)
    nodes_df['latitude'] = nodes_df['latitude'].astype(float)
    nodes_df['longitude'] = nodes_df['longitude'].astype(float)
    
    for _, airport in nodes_df.iterrows():
        doc = {
            '_key': str(airport['iata']),  # Ensure string key
            'airport_id': int(airport['airport_id']),
            'name': str(airport['name']),
            'city': str(airport['city']),
            'country': str(airport['country']),
            'iata': str(airport['iata']),
            'location': [float(airport['longitude']), float(airport['latitude'])]
        }
        airport_docs.append(doc)
    
    # Insert in smaller batches
    batch_size = 1000
    for i in range(0, len(airport_docs), batch_size):
        batch = airport_docs[i:i + batch_size]
        airports_collection.import_bulk(batch, on_duplicate='replace')
        print(f"Inserted batch of {len(batch)} airports")
    
    print(f"Inserted total of {len(airport_docs)} airports")

def insert_routes(routes_collection, edges_df):
    """Insert route edges into ArangoDB"""
    print("Inserting routes...")
    route_docs = []
    
    # Convert distance to float
    edges_df['distance'] = edges_df['distance'].astype(float)
    
    for _, route in edges_df.iterrows():
        doc = {
            '_from': f'airports/{str(route["source"])}',
            '_to': f'airports/{str(route["destination"])}',
            'distance': float(route['distance'])
        }
        route_docs.append(doc)
    
    # Insert in smaller batches
    batch_size = 1000
    for i in range(0, len(route_docs), batch_size):
        batch = route_docs[i:i + batch_size]
        routes_collection.import_bulk(batch, on_duplicate='replace')
        print(f"Inserted batch of {len(batch)} routes")
    
    print(f"Inserted total of {len(route_docs)} routes")

def main():
    try:
        # Connect to ArangoDB
        db = connect_to_arango()
        
        # Create collections
        airports, routes = create_collections(db)
        
        # Load data
        nodes_df, edges_df = load_graph_data()
        
        # Insert data
        insert_airports(airports, nodes_df)
        insert_routes(routes, edges_df)
        
        print("\nCreating graph view...")
        # Create a graph view if it doesn't exist
        if not db.has_graph("flights_graph"):
            graph = db.create_graph("flights_graph")
            graph.create_edge_definition(
                edge_collection="routes",
                from_vertex_collections=["airports"],
                to_vertex_collections=["airports"]
            )
        
        print("\nData successfully loaded into ArangoDB!")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 