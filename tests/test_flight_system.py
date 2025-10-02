import unittest
from arango import ArangoClient
import networkx as nx
import pandas as pd
import pickle
import os

class TestFlightSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests"""
        # Initialize ArangoDB connection
        client = ArangoClient(hosts="http://localhost:8529")
        cls.db = client.db("flights_db", username="root", password="")
        
        # Load NetworkX graph
        with open('data/graph_networkx.pkl', 'rb') as f:
            cls.G = pickle.load(f)
    
    def test_database_connection(self):
        """Test ArangoDB connection and basic query"""
        # Test if we can query airports collection
        cursor = self.db.aql.execute("RETURN LENGTH(airports)")
        count = next(cursor)
        self.assertGreater(count, 0, "Airports collection should not be empty")
        
        # Test if flights_graph exists
        graphs = self.db.graphs()
        self.assertIn("flights_graph", [g["name"] for g in graphs], "flights_graph should exist")
    
    def test_direct_flights_lax(self):
        """Test direct flights from LAX"""
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
        
        cursor = self.db.aql.execute(aql, bind_vars={'airport_code': 'LAX'})
        results = list(cursor)
        
        self.assertTrue(results, "Should get results for LAX")
        self.assertGreater(len(results[0]['flights']), 0, "LAX should have direct flights")
        
        # Convert to DataFrame for easier testing
        flights_df = pd.DataFrame(results[0]['flights'])
        
        # Test specific known routes
        destinations = set(flights_df['destination'])
        self.assertIn('JFK', destinations, "Should have direct flight to JFK")
        self.assertIn('SFO', destinations, "Should have direct flight to SFO")
        
        # Test distance calculations
        self.assertTrue(all(flights_df['distance'] > 0), "All distances should be positive")
    
    def test_shortest_path_lax_lhr(self):
        """Test shortest path calculation between LAX and LHR"""
        try:
            path = nx.shortest_path(self.G, 'LAX', 'LHR', weight='distance')
            
            # Test path exists
            self.assertIsNotNone(path, "Should find a path between LAX and LHR")
            self.assertGreater(len(path), 0, "Path should not be empty")
            
            # Calculate total distance
            total_distance = 0
            for i in range(len(path)-1):
                dist = self.G[path[i]][path[i+1]]['distance']
                total_distance += dist
            
            # Test distance is reasonable (should be around 8,750 km)
            self.assertGreater(total_distance, 8000, "Distance should be > 8000 km")
            self.assertLess(total_distance, 9500, "Distance should be < 9500 km")
            
        except nx.NetworkXNoPath:
            self.fail("Should find a path between LAX and LHR")
    
    def test_airport_analysis_atl(self):
        """Test airport analysis metrics for ATL"""
        airport_code = 'ATL'
        
        # Test airport exists
        self.assertIn(airport_code, self.G, "ATL should exist in the graph")
        
        # Test degree calculations
        degree = self.G.degree(airport_code)
        in_degree = self.G.in_degree(airport_code)
        out_degree = self.G.out_degree(airport_code)
        
        self.assertGreater(degree, 400, "ATL should have > 400 total connections")
        self.assertGreater(in_degree, 200, "ATL should have > 200 incoming flights")
        self.assertGreater(out_degree, 200, "ATL should have > 200 outgoing flights")
        
        # Test clustering coefficient
        clustering = nx.clustering(self.G, airport_code)
        self.assertGreater(clustering, 0, "Clustering coefficient should be positive")
        self.assertLess(clustering, 1, "Clustering coefficient should be less than 1")
    
    def test_invalid_airport(self):
        """Test handling of invalid airport codes"""
        # Test invalid airport in ArangoDB query
        aql = """
        FOR airport IN airports
            FILTER airport.iata == @airport_code
            RETURN airport
        """
        cursor = self.db.aql.execute(aql, bind_vars={'airport_code': 'XXX'})
        results = list(cursor)
        self.assertEqual(len(results), 0, "Should not find invalid airport XXX")
        
        # Test invalid airport in NetworkX
        self.assertNotIn('XXX', self.G, "Invalid airport XXX should not exist in graph")
        
        # Test shortest path with invalid airport
        with self.assertRaises(nx.NetworkXNoPath):
            nx.shortest_path(self.G, 'LAX', 'XXX', weight='distance')

def run_tests():
    """Run the tests and print results"""
    print("\nRunning Flight Network System Tests...")
    
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFlightSystem)
    
    # Run the tests
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    # Print summary
    print("\nTest Summary:")
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    return len(result.failures) + len(result.errors) == 0

if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1) 