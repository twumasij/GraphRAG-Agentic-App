import unittest
import sys
import os
from arango import ArangoClient

# Add the parent directory to the path so we can import from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.agentic_app import get_arango_db, FlightGraphTools

class TestDatabase(unittest.TestCase):
    """Test the database connections and queries"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            self.client = ArangoClient(hosts="http://localhost:8529")
            self.db = self.client.db("flights_db", username="root", password="")
            self.tools = FlightGraphTools()
        except Exception as e:
            self.fail(f"Setup failed: {str(e)}")
    
    def test_database_connection(self):
        """Test that we can connect to the database"""
        self.assertIsNotNone(self.db)
        # Check if the database has the required collections
        collections = self.db.collections()
        collection_names = [c['name'] for c in collections]
        self.assertIn('airports', collection_names)
        self.assertIn('routes', collection_names)
    
    def test_airports_collection(self):
        """Test that the airports collection has data"""
        airports = self.db.collection('airports')
        count = airports.count()
        self.assertGreater(count, 0, "Airports collection is empty")
        
        # Test a sample airport
        cursor = self.db.aql.execute("""
            FOR a IN airports
                FILTER a.iata == "LAX"
                RETURN a
        """)
        results = list(cursor)
        self.assertEqual(len(results), 1, "LAX airport not found")
        self.assertEqual(results[0]['name'], "Los Angeles International Airport")
    
    def test_routes_collection(self):
        """Test that the routes collection has data"""
        routes = self.db.collection('routes')
        count = routes.count()
        self.assertGreater(count, 0, "Routes collection is empty")
        
        # Test routes from LAX
        cursor = self.db.aql.execute("""
            FOR v, e IN 1..1 OUTBOUND 'airports/LAX' GRAPH 'flights_graph'
                RETURN e
        """)
        results = list(cursor)
        self.assertGreater(len(results), 0, "No routes from LAX found")
    
    def test_find_direct_flights(self):
        """Test the find_direct_flights method"""
        result = self.tools.find_direct_flights("LAX")
        self.assertIsNotNone(result)
        self.assertIn("Direct flights from", result)
        self.assertIn("destinations", result)
    
    def test_find_shortest_path(self):
        """Test the find_shortest_path method"""
        result = self.tools.find_shortest_path("LAX", "JFK")
        self.assertIsNotNone(result)
        self.assertIn("Shortest path from", result)
        self.assertIn("Total distance", result)
    
    def test_analyze_airport_importance(self):
        """Test the analyze_airport_importance method"""
        result = self.tools.analyze_airport_importance("LAX")
        self.assertIsNotNone(result)
        self.assertIn("Airport Analysis for", result)
        self.assertIn("PageRank Score", result)
        self.assertIn("Betweenness Centrality", result)
    
    def test_invalid_airport(self):
        """Test handling of invalid airport codes"""
        result = self.tools.find_direct_flights("XYZ")
        self.assertIn("No direct flights found", result)
        
        result = self.tools.analyze_airport_importance("XYZ")
        self.assertIn("not found in the network", result)

if __name__ == '__main__':
    unittest.main() 