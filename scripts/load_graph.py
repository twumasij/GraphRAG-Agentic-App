import pandas as pd
import networkx as nx
import pickle

def load_graph_data():
    """Load the preprocessed graph data from CSV files"""
    print("Loading nodes and edges data...")
    nodes_df = pd.read_csv('data/graph_nodes.csv')
    edges_df = pd.read_csv('data/graph_edges.csv')
    return nodes_df, edges_df

def create_networkx_graph(nodes_df, edges_df):
    """Create a NetworkX directed graph from nodes and edges data"""
    print("Creating NetworkX directed graph...")
    G = nx.DiGraph()
    
    # Add nodes with attributes
    for _, node in nodes_df.iterrows():
        G.add_node(
            node['iata'],
            airport_id=node['airport_id'],
            name=node['name'],
            city=node['city'],
            country=node['country'],
            latitude=node['latitude'],
            longitude=node['longitude']
        )
    
    # Add edges with attributes
    for _, edge in edges_df.iterrows():
        G.add_edge(
            edge['source'],
            edge['destination'],
            distance=edge['distance']
        )
    
    return G

def save_graph(G, filename='data/graph_networkx.pkl'):
    """Save the NetworkX graph to a pickle file"""
    print(f"Saving graph to {filename}...")
    with open(filename, 'wb') as f:
        pickle.dump(G, f)

def main():
    # Load data
    nodes_df, edges_df = load_graph_data()
    
    # Create graph
    G = create_networkx_graph(nodes_df, edges_df)
    
    # Print graph statistics
    print("\nGraph Statistics:")
    print(f"Number of nodes (airports): {G.number_of_nodes()}")
    print(f"Number of edges (routes): {G.number_of_edges()}")
    
    # Calculate and print additional graph metrics
    print("\nAdditional Graph Metrics:")
    print(f"Is graph strongly connected: {nx.is_strongly_connected(G)}")
    print(f"Number of strongly connected components: {nx.number_strongly_connected_components(G)}")
    print(f"Average out-degree: {sum(dict(G.out_degree()).values()) / G.number_of_nodes():.2f}")
    
    # Save graph
    save_graph(G)
    print("\nGraph processing and saving complete!")

if __name__ == "__main__":
    main() 