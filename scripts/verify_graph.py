import networkx as nx
import pickle
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def load_graph():
    """Load the NetworkX graph from pickle file"""
    graph_path = "data/graph_networkx.pkl"
    
    if not os.path.exists(graph_path):
        print(f"ERROR: Graph file {graph_path} does not exist!")
        return None
    
    try:
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)
        
        if not isinstance(G, nx.DiGraph):
            print("ERROR: Loaded object is not a NetworkX DiGraph!")
            return None
        
        return G
    except Exception as e:
        print(f"Failed to load graph: {str(e)}")
        return None

def check_basic_properties(G):
    """Check basic graph properties"""
    print("\n=== Basic Graph Properties ===")
    
    # Node and edge count
    nodes_count = G.number_of_nodes()
    edges_count = G.number_of_edges()
    
    print(f"Nodes (airports): {nodes_count}")
    print(f"Edges (routes): {edges_count}")
    print(f"Average degree: {edges_count / nodes_count:.2f}")
    
    # Check for isolated nodes
    isolated_nodes = list(nx.isolates(G))
    print(f"Isolated airports: {len(isolated_nodes)}")
    if isolated_nodes:
        print(f"Sample isolated airports: {isolated_nodes[:5]}")
    
    # Check connectivity
    is_weakly_connected = nx.is_weakly_connected(G)
    print(f"Is weakly connected: {is_weakly_connected}")
    
    if not is_weakly_connected:
        components = list(nx.weakly_connected_components(G))
        print(f"Number of weakly connected components: {len(components)}")
        print(f"Largest component size: {len(max(components, key=len))}")
    
    # Check for strongly connected components
    strongly_connected_components = list(nx.strongly_connected_components(G))
    print(f"Number of strongly connected components: {len(strongly_connected_components)}")
    print(f"Largest strongly connected component size: {len(max(strongly_connected_components, key=len))}")
    
    return True

def check_node_attributes(G):
    """Check node attributes"""
    print("\n=== Node Attributes ===")
    
    # Get a sample node
    sample_node = list(G.nodes())[0]
    attributes = G.nodes[sample_node]
    
    print(f"Sample node: {sample_node}")
    print(f"Attributes: {attributes}")
    
    # Check required attributes
    required_attributes = ['name', 'city', 'country']
    missing_attributes = [attr for attr in required_attributes if attr not in attributes]
    
    if missing_attributes:
        print(f"WARNING: Missing required attributes: {missing_attributes}")
        return False
    
    # Check countries distribution
    countries = [G.nodes[node]['country'] for node in G.nodes()]
    country_counts = Counter(countries)
    
    print("\nTop 10 countries by airport count:")
    for country, count in country_counts.most_common(10):
        print(f"  {country}: {count}")
    
    return True

def check_edge_attributes(G):
    """Check edge attributes"""
    print("\n=== Edge Attributes ===")
    
    # Get a sample edge
    sample_edge = list(G.edges())[0]
    attributes = G.edges[sample_edge]
    
    print(f"Sample edge: {sample_edge}")
    print(f"Attributes: {attributes}")
    
    # Check required attributes
    if 'distance' not in attributes:
        print("WARNING: Missing required 'distance' attribute")
        return False
    
    # Analyze distances
    distances = [G.edges[edge]['distance'] for edge in G.edges()]
    
    print(f"Min distance: {min(distances):.2f} km")
    print(f"Max distance: {max(distances):.2f} km")
    print(f"Average distance: {sum(distances) / len(distances):.2f} km")
    
    # Distance distribution
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=50)
    plt.title('Flight Distance Distribution')
    plt.xlabel('Distance (km)')
    plt.ylabel('Frequency')
    plt.savefig('data/distance_distribution.png')
    print("Distance distribution saved to data/distance_distribution.png")
    
    return True

def analyze_centrality(G):
    """Analyze centrality metrics"""
    print("\n=== Centrality Analysis ===")
    
    # Calculate degree centrality
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    
    # Find top airports by degree
    top_in_degree = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:10]
    top_out_degree = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print("Top 10 airports by incoming flights:")
    for airport, degree in top_in_degree:
        print(f"  {airport} ({G.nodes[airport]['name']}): {degree}")
    
    print("\nTop 10 airports by outgoing flights:")
    for airport, degree in top_out_degree:
        print(f"  {airport} ({G.nodes[airport]['name']}): {degree}")
    
    # Calculate PageRank (approximation for large graphs)
    print("\nCalculating PageRank (this may take a while)...")
    pagerank = nx.pagerank(G, alpha=0.85, weight='distance')
    
    top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 airports by PageRank:")
    for airport, score in top_pagerank:
        print(f"  {airport} ({G.nodes[airport]['name']}): {score:.6f}")
    
    return True

def main():
    print("=== Graph Verification and Analysis Tool ===")
    
    # Load graph
    G = load_graph()
    if G is None:
        return 1
    
    # Check basic properties
    basic_ok = check_basic_properties(G)
    
    # Check node attributes
    node_ok = check_node_attributes(G)
    
    # Check edge attributes
    edge_ok = check_edge_attributes(G)
    
    # Analyze centrality
    centrality_ok = analyze_centrality(G)
    
    # Overall status
    if basic_ok and node_ok and edge_ok and centrality_ok:
        print("\nAll graph checks PASSED! The graph is ready for analysis.")
        return 0
    else:
        print("\nSome graph checks FAILED! Please fix the issues before using the system.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 