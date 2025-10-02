import unittest
import sys
import os
import requests
import time

# Add the parent directory to the path so we can import from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.agentic_app import OllamaWrapper, initialize_llm

class TestLLM(unittest.TestCase):
    """Test the LLM integration with Ollama"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Check if Ollama is running
        try:
            response = requests.get("http://localhost:11434/api/version")
            if response.status_code != 200:
                self.skipTest("Ollama server is not running")
            
            # Initialize the LLM wrapper
            self.llm = initialize_llm()
        except requests.exceptions.ConnectionError:
            self.skipTest("Ollama server is not running")
        except Exception as e:
            self.fail(f"Setup failed: {str(e)}")
    
    def test_ollama_connection(self):
        """Test connection to Ollama server"""
        response = requests.get("http://localhost:11434/api/version")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('version', data)
    
    def test_available_models(self):
        """Test that required models are available"""
        response = requests.get("http://localhost:11434/api/tags")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Check if we have at least one model
        self.assertGreater(len(data.get('models', [])), 0)
        
        # Print available models for debugging
        model_names = [model.get('name') for model in data.get('models', [])]
        print(f"Available models: {model_names}")
    
    def test_llm_wrapper_initialization(self):
        """Test that the LLM wrapper initializes correctly"""
        self.assertIsNotNone(self.llm)
        self.assertIsInstance(self.llm, OllamaWrapper)
        self.assertEqual(self.llm._llm_type(), "ollama")
    
    def test_simple_query(self):
        """Test a simple query to the LLM"""
        prompt = "Hello, how are you?"
        try:
            response = self.llm._call(prompt)
            self.assertIsNotNone(response)
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 10)
        except Exception as e:
            self.fail(f"LLM query failed: {str(e)}")
    
    def test_flight_related_query(self):
        """Test a flight-related query"""
        prompt = "What are the benefits of direct flights?"
        try:
            response = self.llm._call(prompt)
            self.assertIsNotNone(response)
            self.assertGreater(len(response), 50)
            
            # Check if response contains flight-related terms
            flight_terms = ['flight', 'travel', 'airport', 'journey', 'passenger']
            has_flight_term = any(term.lower() in response.lower() for term in flight_terms)
            self.assertTrue(has_flight_term, "Response doesn't contain flight-related terms")
        except Exception as e:
            self.fail(f"Flight query failed: {str(e)}")
    
    def test_tool_selection_query(self):
        """Test a query that should trigger tool selection"""
        prompt = """
        You have these tools:
        - FindDirectFlights: Find all direct flights from an airport
        - FindShortestPath: Find shortest path between two airports
        - AnalyzeAirport: Analyze the importance of an airport
        
        User question: "What are all the direct flights from LAX?"
        
        Which tool should I use?
        """
        try:
            response = self.llm._call(prompt)
            self.assertIsNotNone(response)
            self.assertIn("FindDirectFlights", response)
        except Exception as e:
            self.fail(f"Tool selection query failed: {str(e)}")
    
    def test_response_time(self):
        """Test the response time of the LLM"""
        prompt = "What is the capital of France?"
        start_time = time.time()
        try:
            response = self.llm._call(prompt)
            end_time = time.time()
            
            # Check if response time is reasonable (under 10 seconds)
            self.assertLess(end_time - start_time, 10, "LLM response time is too slow")
        except Exception as e:
            self.fail(f"Response time test failed: {str(e)}")

if __name__ == '__main__':
    unittest.main() 