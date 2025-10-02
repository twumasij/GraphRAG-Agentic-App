import unittest
import sys
import os
import networkx as nx
import pickle

# Add the parent directory to the path so we can import from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.agentic_app import get_networkx_graph

class TestGraph(unittest.TestCase):
    """Test the NetworkX graph functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            self.G = get_networkx_graph()
        except Exception as e:
            self.fail(f"Setup failed: {str(e)}")
    
    def test_graph_loading(self):
        """Test that the graph loads correctly"""
        self.assertIsNotNone(self.G)
        self.assertIsInstance(self.G, nx.DiGraph)
        self.assertGreater(self.G.number_of_nodes(), 0, "Graph has no nodes")
        self.assertGreater(self.G.number_of_edges(), 0, "Graph has no edges")
    
    def test_graph_properties(self):
        """Test basic graph properties"""
        # Check if major airports exist
        major_airports = ["LAX", "JFK", "LHR", "DXB", "SIN"]
        for airport in major_airports:
            self.assertIn(airport, self.G.nodes, f"{airport} not found in graph")
        
        # Check node attributes
        for airport in major_airports:
            node = self.G.nodes[airport]
            self.assertIn('name', node, f"Name attribute missing for {airport}")
            self.assertIn('city', node, f"City attribute missing for {airport}")
            self.assertIn('country', node, f"Country attribute missing for {airport}")
    
    def test_edge_properties(self):
        """Test edge properties"""
        # Check if there's a route from LAX to JFK
        self.assertIn("JFK", self.G["LAX"], "No route from LAX to JFK")
        
        # Check edge attributes
        edge = self.G["LAX"]["JFK"]
        self.assertIn('distance', edge, "Distance attribute missing")
        self.assertGreater(edge['distance'], 0, "Distance is not positive")
    
    def test_shortest_path(self):
        """Test shortest path calculation"""
        # Test direct path
        path = nx.shortest_path(self.G, "LAX", "JFK", weight='distance')
        self.assertEqual(len(path), 2, "Direct path should have length 2")
        
        # Test path with stops
        # Find a path that requires at least one stop
        found_multi_stop = False
        for dest in self.G.nodes():
            if dest != "LAX":
                try:
                    path = nx.shortest_path(self.G, "LAX", dest, weight='distance')
                    if len(path) > 2:
                        found_multi_stop = True
                        self.assertGreater(len(path), 2, "Multi-stop path should have length > 2")
                        break
                except nx.NetworkXNoPath:
                    continue
        
        self.assertTrue(found_multi_stop, "No multi-stop paths found")
    
    def test_centrality_metrics(self):
        """Test centrality metrics calculation"""
        # Calculate PageRank
        pagerank = nx.pagerank(self.G, weight='distance')
        self.assertGreater(len(pagerank), 0, "PageRank calculation failed")
        
        # Check if major hubs have high PageRank
        major_hubs = ["ATL", "DXB", "LHR", "ORD", "PEK"]
        for hub in major_hubs:
            if hub in pagerank:
                self.assertGreater(pagerank[hub], 0.001, f"{hub} has unexpectedly low PageRank")
        
        # Calculate betweenness centrality
        betweenness = nx.betweenness_centrality(self.G, k=100, weight='distance')
        self.assertGreater(len(betweenness), 0, "Betweenness calculation failed")
    
    def test_graph_connectivity(self):
        """Test graph connectivity"""
        # Check if the graph is weakly connected
        self.assertTrue(nx.is_weakly_connected(self.G), "Graph is not weakly connected")
        
        # Get the largest strongly connected component
        largest_scc = max(nx.strongly_connected_components(self.G), key=len)
        self.assertGreater(len(largest_scc), 100, "Largest SCC is too small")

if __name__ == '__main__':
    unittest.main() 