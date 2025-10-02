import pytest
from scripts.agentic_app.tools.flight_graph import FlightGraphTools
from scripts.agentic_app.utils.monitoring import metrics

@pytest.fixture
def flight_tools():
    return FlightGraphTools()

def test_find_direct_flights(flight_tools):
    # Test with a valid airport code
    result = flight_tools.find_direct_flights("LAX")
    assert result is not None
    assert "Direct flights from" in result
    assert "Total direct flights:" in result
    
    # Test with invalid airport code
    with pytest.raises(ValueError):
        flight_tools.find_direct_flights("INVALID")

def test_count_direct_flights(flight_tools):
    result = flight_tools.count_direct_flights("LAX")
    assert result is not None
    assert "LAX" in result
    assert "direct flights" in result

def test_check_direct_flight(flight_tools):
    # Test existing route
    result = flight_tools.check_direct_flight("LAX", "JFK")
    assert result is not None
    assert "LAX" in result
    assert "JFK" in result
    
    # Test non-existing route
    result = flight_tools.check_direct_flight("LAX", "XXX")
    assert "no direct flight" in result.lower()

def test_find_shortest_path(flight_tools):
    result = flight_tools.find_shortest_path("LAX", "JFK")
    assert result is not None
    assert "LAX" in result
    assert "JFK" in result
    assert "Total distance:" in result

def test_analyze_airport_importance(flight_tools):
    result = flight_tools.analyze_airport_importance("LAX")
    assert result is not None
    assert "LAX" in result
    assert "PageRank Score:" in result
    assert "Betweenness Centrality:" in result

def test_get_top_incoming_airports(flight_tools):
    result = flight_tools.get_top_incoming_airports()
    assert result is not None
    assert "Top Incoming Airports:" in result
    assert "incoming flights" in result

def test_get_top_outgoing_airports(flight_tools):
    result = flight_tools.get_top_outgoing_airports()
    assert result is not None
    assert "Top Outgoing Airports:" in result
    assert "outgoing flights" in result

def test_average_flight_distance(flight_tools):
    result = flight_tools.average_flight_distance("LAX")
    assert result is not None
    assert "LAX" in result
    assert "km" in result

def test_flight_distance_distribution(flight_tools):
    result = flight_tools.flight_distance_distribution()
    assert result is not None
    assert "Flight Distance Distribution:" in result
    assert "Minimum distance:" in result
    assert "Maximum distance:" in result 