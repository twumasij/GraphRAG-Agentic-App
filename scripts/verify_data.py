from arango import ArangoClient
import networkx as nx
import pickle
import os
import pandas as pd
import sys

def check_database():
    """Check the ArangoDB database and collections"""
    print("Checking ArangoDB database...")
    try:
        client = ArangoClient(hosts="http://localhost:8529")
        sys_db = client.db("_system", username="root", password="")
        
        # Check if flights_db exists
        if not sys_db.has_database("flights_db"):
            print("ERROR: flights_db database does not exist!")
            return False
        
        # Connect to flights_db
        db = client.db("flights_db", username="root", password="")
        
        # Check collections
        required_collections = ["airports", "routes"]
        for collection in required_collections:
            if not db.has_collection(collection):
                print(f"ERROR: {collection} collection does not exist!")
                return False
        
        # Check if graph exists
        if not db.has_graph("flights_graph"):
            print("ERROR: flights_graph does not exist!")
            return False
        
        # Check collection counts
        airports_count = db.collection("airports").count()
        routes_count = db.collection("routes").count()
        
        print(f"Found {airports_count} airports and {routes_count} routes in the database")
        
        if airports_count == 0 or routes_count == 0:
            print("ERROR: Empty collections found!")
            return False
        
        return True
    except Exception as e:
        print(f"Database check failed: {str(e)}")
        return False

def check_graph_file():
    """Check the NetworkX graph pickle file"""
    print("Checking NetworkX graph file...")
    graph_path = "data/graph_networkx.pkl"
    
    if not os.path.exists(graph_path):
        print(f"ERROR: Graph file {graph_path} does not exist!")
        return False
    
    try:
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)
        
        if not isinstance(G, nx.DiGraph):
            print("ERROR: Loaded object is not a NetworkX DiGraph!")
            return False
        
        nodes_count = G.number_of_nodes()
        edges_count = G.number_of_edges()
        
        print(f"Graph has {nodes_count} nodes and {edges_count} edges")
        
        if nodes_count == 0 or edges_count == 0:
            print("ERROR: Empty graph found!")
            return False
        
        # Check for major airports
        major_airports = ["LAX", "JFK", "LHR", "DXB", "SIN"]
        missing_airports = [airport for airport in major_airports if airport not in G.nodes]
        
        if missing_airports:
            print(f"WARNING: Some major airports are missing: {missing_airports}")
        
        return True
    except Exception as e:
        print(f"Graph check failed: {str(e)}")
        return False

def check_csv_files():
    """Check the CSV data files"""
    print("Checking CSV data files...")
    required_files = [
        "data/graph_nodes.csv",
        "data/graph_edges.csv",
        "data/airports.dat",
        "data/routes.dat"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"ERROR: {file_path} does not exist!")
            return False
    
    try:
        # Check nodes CSV
        nodes_df = pd.read_csv("data/graph_nodes.csv")
        if nodes_df.empty:
            print("ERROR: graph_nodes.csv is empty!")
            return False
        
        # Check edges CSV
        edges_df = pd.read_csv("data/graph_edges.csv")
        if edges_df.empty:
            print("ERROR: graph_edges.csv is empty!")
            return False
        
        print(f"Found {len(nodes_df)} nodes and {len(edges_df)} edges in CSV files")
        
        # Check for required columns
        required_node_columns = ["iata", "name", "city", "country"]
        missing_node_columns = [col for col in required_node_columns if col not in nodes_df.columns]
        
        if missing_node_columns:
            print(f"ERROR: Missing columns in graph_nodes.csv: {missing_node_columns}")
            return False
        
        required_edge_columns = ["source", "destination", "distance"]
        missing_edge_columns = [col for col in required_edge_columns if col not in edges_df.columns]
        
        if missing_edge_columns:
            print(f"ERROR: Missing columns in graph_edges.csv: {missing_edge_columns}")
            return False
        
        return True
    except Exception as e:
        print(f"CSV check failed: {str(e)}")
        return False

def main():
    print("=== Data Verification Tool ===")
    
    # Check database
    db_ok = check_database()
    print(f"Database check: {'PASSED' if db_ok else 'FAILED'}")
    
    # Check graph file
    graph_ok = check_graph_file()
    print(f"Graph file check: {'PASSED' if graph_ok else 'FAILED'}")
    
    # Check CSV files
    csv_ok = check_csv_files()
    print(f"CSV files check: {'PASSED' if csv_ok else 'FAILED'}")
    
    # Overall status
    if db_ok and graph_ok and csv_ok:
        print("\nAll checks PASSED! The system is ready to use.")
        return 0
    else:
        print("\nSome checks FAILED! Please fix the issues before using the system.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 