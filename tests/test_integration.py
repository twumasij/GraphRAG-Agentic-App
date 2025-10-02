import unittest
import sys
import os
import time
from langchain.agents import initialize_agent, AgentType

# Add the parent directory to the path so we can import from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.agentic_app import FlightGraphTools, initialize_llm, Tool

class TestIntegration(unittest.TestCase):
    """Test the integration of all components"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            # Initialize flight tools
            self.tools = FlightGraphTools()
            
            # Initialize LLM
            self.llm = initialize_llm()
            
            # Create tool list for agent
            self.tool_list = [
                Tool(
                    name="FindDirectFlights",
                    func=self.tools.find_direct_flights,
                    description="Find all direct flights from a given airport. Input: IATA code, e.g. 'LAX'"
                ),
                Tool(
                    name="FindShortestPath",
                    func=self.tools.find_shortest_path,
                    description="Find shortest path between two airports. Input: 'LAX,JFK'"
                ),
                Tool(
                    name="AnalyzeAirport",
                    func=self.tools.analyze_airport_importance,
                    description="Analyze the importance of an airport. Input: 'LAX'"
                ),
            ]
            
            # Create agent
            self.agent = initialize_agent(
                tools=self.tool_list,
                llm=self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                max_iterations=3,
                early_stopping_method="generate",
                handle_parsing_errors=True
            )
        except Exception as e:
            self.fail(f"Setup failed: {str(e)}")
    
    def test_direct_flights_query(self):
        """Test a query about direct flights"""
        query = "Show me all direct flights from LAX"
        try:
            response = self.agent({"input": query})
            self.assertIsNotNone(response)
            self.assertIn("output", response)
            
            output = response["output"]
            self.assertIsInstance(output, str)
            self.assertGreater(len(output), 50)
            
            # Check if response contains flight-related terms
            flight_terms = ['flight', 'LAX', 'destination']
            has_flight_term = any(term.lower() in output.lower() for term in flight_terms)
            self.assertTrue(has_flight_term, "Response doesn't contain flight-related terms")
        except Exception as e:
            self.fail(f"Direct flights query failed: {str(e)}")
    
    def test_shortest_path_query(self):
        """Test a query about the shortest path"""
        query = "What's the best route from JFK to LAX?"
        try:
            response = self.agent({"input": query})
            self.assertIsNotNone(response)
            self.assertIn("output", response)
            
            output = response["output"]
            self.assertIsInstance(output, str)
            self.assertGreater(len(output), 50)
            
            # Check if response contains path-related terms
            path_terms = ['route', 'path', 'JFK', 'LAX', 'distance']
            has_path_term = any(term.lower() in output.lower() for term in path_terms)
            self.assertTrue(has_path_term, "Response doesn't contain path-related terms")
        except Exception as e:
            self.fail(f"Shortest path query failed: {str(e)}")
    
    def test_airport_analysis_query(self):
        """Test a query about airport analysis"""
        query = "Analyze the importance of ATL in the network"
        try:
            response = self.agent({"input": query})
            self.assertIsNotNone(response)
            self.assertIn("output", response)
            
            output = response["output"]
            self.assertIsInstance(output, str)
            self.assertGreater(len(output), 50)
            
            # Check if response contains analysis-related terms
            analysis_terms = ['ATL', 'importance', 'network', 'centrality', 'connections']
            has_analysis_term = any(term.lower() in output.lower() for term in analysis_terms)
            self.assertTrue(has_analysis_term, "Response doesn't contain analysis-related terms")
        except Exception as e:
            self.fail(f"Airport analysis query failed: {str(e)}")
    
    def test_ambiguous_query(self):
        """Test an ambiguous query that requires clarification"""
        query = "Tell me about airports in New York"
        try:
            response = self.agent({"input": query})
            self.assertIsNotNone(response)
            self.assertIn("output", response)
            
            output = response["output"]
            self.assertIsInstance(output, str)
            self.assertGreater(len(output), 20)
        except Exception as e:
            self.fail(f"Ambiguous query failed: {str(e)}")
    
    def test_invalid_query(self):
        """Test an invalid query"""
        query = "What's the weather like in Paris?"
        try:
            response = self.agent({"input": query})
            self.assertIsNotNone(response)
            self.assertIn("output", response)
            
            output = response["output"]
            self.assertIsInstance(output, str)
            self.assertGreater(len(output), 20)
        except Exception as e:
            self.fail(f"Invalid query failed: {str(e)}")
    
    def test_response_time(self):
        """Test the response time of the entire system"""
        query = "Show me direct flights from LAX"
        start_time = time.time()
        try:
            response = self.agent({"input": query})
            end_time = time.time()
            
            # Check if response time is reasonable (under 30 seconds)
            self.assertLess(end_time - start_time, 30, "System response time is too slow")
        except Exception as e:
            self.fail(f"Response time test failed: {str(e)}")

if __name__ == '__main__':
    unittest.main() 