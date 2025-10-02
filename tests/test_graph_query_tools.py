import pytest
from scripts.agentic_app.tools.graph_query_tools import GraphQueryTools
from scripts.agentic_app.tools.flight_graph import FlightGraphTools

@pytest.fixture
def graph_tools():
    flight_tools = FlightGraphTools()
    return GraphQueryTools(flight_tools.db, flight_tools.G)

def test_text_to_aql_to_text(graph_tools):
    # Test direct flights query
    result = graph_tools.text_to_aql_to_text("Show me direct flights from LAX")
    assert result is not None
    assert "LAX" in result
    assert "flights" in result.lower()
    
    # Test route query
    result = graph_tools.text_to_aql_to_text("Show me the route between LAX and JFK")
    assert result is not None
    assert "LAX" in result
    assert "JFK" in result

def test_text_to_nx_algorithm_to_text(graph_tools):
    # Test centrality query
    result = graph_tools.text_to_nx_algorithm_to_text("What are the most important airports?")
    assert result is not None
    assert "important" in result.lower()
    
    # Test path query
    result = graph_tools.text_to_nx_algorithm_to_text("What's the shortest path from LAX to JFK?")
    assert result is not None
    assert "LAX" in result
    assert "JFK" in result

def test_extract_airport_code(graph_tools):
    assert graph_tools._extract_airport_code("Show flights from LAX") == "LAX"
    assert graph_tools._extract_airport_code("No airport code") is None

def test_extract_airport_pair(graph_tools):
    source, target = graph_tools._extract_airport_pair("Route from LAX to JFK")
    assert source == "LAX"
    assert target == "JFK"
    
    source, target = graph_tools._extract_airport_pair("No airports")
    assert source is None
    assert target is None

def test_natural_to_aql(graph_tools):
    # Test direct flights query
    aql = graph_tools._natural_to_aql("Show direct flights from LAX")
    assert aql is not None
    assert "airports" in aql
    assert "LAX" in aql
    
    # Test route query
    aql = graph_tools._natural_to_aql("Show route between LAX and JFK")
    assert aql is not None
    assert "OUTBOUND" in aql
    assert "LAX" in aql
    assert "JFK" in aql

def test_results_to_text(graph_tools):
    # Test direct flights results
    results = [{"source": "Los Angeles International", "flights": [
        {"destination": "JFK", "city": "New York", "country": "USA", "distance": 3000}
    ]}]
    text = graph_tools._results_to_text("Show direct flights from LAX", results)
    assert text is not None
    assert "Los Angeles" in text
    assert "JFK" in text
    
    # Test empty results
    text = graph_tools._results_to_text("Show direct flights from LAX", [])
    assert text == "No results found for your query." 