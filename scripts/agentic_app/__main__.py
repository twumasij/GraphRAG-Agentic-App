import time
from typing import List
from langchain.agents import Tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from .core.agent import create_agent_graph
from .core.llm import initialize_llm
from .tools.flight_graph import FlightGraphTools
from .tools.graph_query_tools import GraphQueryTools
from .utils.monitoring import metrics, health_check
from .utils.helpers import print_status
import logging
import re

logger = logging.getLogger(__name__)

def main():
    health_status = health_check.run_health_check()
    if not health_status['healthy']:
        print("Application failed health check. Please check logs for details.")
        return

    # Initialize FlightGraphTools and GraphQueryTools
    flight_tools = FlightGraphTools()
    graph_tools = GraphQueryTools(flight_tools.db, flight_tools.G)

    tool_list = [
        Tool(
            name="TextToAQL",
            func=graph_tools.text_to_aql_to_text,
            description="Convert natural language to AQL query, execute it, and return results in natural language. Use for questions about direct flights, routes, and general flight information."
        ),
        Tool(
            name="TextToNetworkX",
            func=graph_tools.text_to_nx_algorithm_to_text,
            description="Execute NetworkX algorithms based on natural language query. Use for complex graph analysis like centrality, shortest paths, and connectivity."
        ),
   
    Tool(
        name="FindDirectFlights",
        func=flight_tools.find_direct_flights,
        description="Find all direct flights from a given airport. Input: IATA code, e.g. 'LAX'."
    ),
    Tool(
        name="CountDirectFlights",
        func=flight_tools.count_direct_flights,
        description="Return the number of direct flights from the specified airport. Input: 'LAX'."
    ),
    Tool(
        name="CheckDirectFlight",
        func=flight_tools.check_direct_flight,
        description="Check if there's a direct flight between two airports. Input: 'LAX,JFK'."
    ),
    Tool(
        name="FindShortestPath",
        func=flight_tools.find_shortest_path,
        description="Find the shortest path (by distance) between two airports. Input: 'LAX,JFK'."
    ),
    Tool(
        name="FindMinimumStopRoute",
        func=flight_tools.find_minimum_stop_route,
        description="Find a route with the fewest stops between two airports. Input: 'LAX,JFK'."
    ),
    Tool(
        name="CompareRoutes",
        func=flight_tools.compare_routes,
        description="Compare the distance-based shortest route vs. fewest-stop route. Input: 'LAX,JFK'."
    ),
    Tool(
        name="AnalyzeAirportImportance",
        func=flight_tools.analyze_airport_importance,
        description="Analyze a single airport's importance (PageRank, etc.). Input: 'LAX'."
    ),
    Tool(
        name="CompareAirportImportance",
        func=flight_tools.compare_airport_importance,
        description="Compare importance metrics for two airports. Input: 'LAX,JFK'."
    ),
    Tool(
        name="GetTopIncomingAirports",
        func=flight_tools.get_top_incoming_airports,
        description="List top N airports by incoming flights. Input: integer or blank (defaults=5)."
    ),
    Tool(
        name="GetTopOutgoingAirports",
        func=flight_tools.get_top_outgoing_airports,
        description="List top N airports by outgoing flights. Input: integer or blank (defaults=5)."
    ),
    Tool(
        name="AverageFlightDistance",
        func=flight_tools.average_flight_distance,
        description="Calculate average distance of all direct flights from an airport. Input: 'LAX'."
    ),
    Tool(
        name="FlightDistanceDistribution",
        func=flight_tools.flight_distance_distribution,
        description="Summary stats (min, max, mean, median) for all flight distances. No input."
    ),
    Tool(
        name="GetIsolatedAirports",
        func=flight_tools.get_isolated_airports,
        description="List airports with no connections. No input."
    ),
    Tool(
        name="ListAirportsByCountry",
        func=flight_tools.list_airports_by_country,
        description="List all airports located in a specified country. Input: 'Germany'."
    ),

    
    Tool(
        name="FindFlightsWithinDistance",
        func=flight_tools.find_flights_within_distance,
        description="List direct flights from an airport that are within a certain max distance. Input: 'LAX,1000' (IATA, distance)."
    ),
    Tool(
        name="GetTopAirportsByPageRank",
        func=flight_tools.get_top_airports_by_pagerank,
        description="List top N airports by PageRank. Input: integer or blank (defaults=5)."
    ),
    Tool(
        name="GetCountryConnectivity",
        func=flight_tools.get_country_connectivity,
        description="Show how many airports in a country and how many direct flights each has. Input: 'Germany'."
    ),
    Tool(
        name="FindMultiStopRoute",
        func=flight_tools.find_multi_stop_route,
        description="Find a route between two airports with up to 'max_stops' stops. Input: 'LAX,JFK,2'."
    ),
    Tool(
        name="GetBusiestRoute",
        func=flight_tools.get_busiest_route,
        description="Find the single route with the highest 'traffic' attribute in the network. No input."
    ),
    Tool(
        name="GetLargestAirportInCountry",
        func=flight_tools.get_largest_airport_in_country,
        description="Get the airport with the greatest number of direct connections in a country. Input: 'Germany'."
    ),
    Tool(
        name="GetLongestFlightInNetwork",
        func=flight_tools.get_longest_flight_in_network,
        description="Find the single longest (by distance) flight/edge in the entire network. No input."
    ),
    Tool(
        name="GetShortestFlightInNetwork",
        func=flight_tools.get_shortest_flight_in_network,
        description="Find the single shortest (by distance) flight/edge in the entire network. No input."
    ),
    Tool(
        name="GetLocalClusteringCoefficient",
        func=flight_tools.get_local_clustering_coefficient,
        description="Compute the local clustering coefficient for a given airport. Input: 'LAX'."
    ),
    Tool(
        name="GetHighestBetweennessAirports",
        func=flight_tools.get_highest_betweenness_airports,
        description="List top N airports by betweenness centrality. Input: integer or blank (defaults=5)."
    ),
    Tool(
        name="GetAverageDegreeOfNetwork",
        func=flight_tools.get_average_degree_of_network,
        description="Compute average node degree across the entire network. No input."
    ),
    Tool(
        name="GetTopHubAirports",
        func=flight_tools.get_top_hub_airports,
        description="List top N hub airports by total connections (in+out). Input: integer or blank (defaults=5)."
    ),
    Tool(
        name="FindLongestRouteWithinKStops",
        func=flight_tools.find_longest_route_within_k_stops,
        description="From a source airport, find the single route with greatest total distance within N stops. Input: 'LAX,3'."
    ),
    Tool(
        name="ListCountriesInNetwork",
        func=flight_tools.list_countries_in_network,
        description="List all unique countries present in the network. No input."
    ),
    Tool(
        name="FindAirportsNearGeolocation",
        func=flight_tools.find_airports_near_geolocation,
        description="Find airports within a certain radius of a latitude/longitude. Input: '34.0522,-118.2437,100'."
    )
    ]



    llm = initialize_llm()
    agent_executor = create_agent_graph(llm, tool_list)

    print("\nFlight Network Analysis System Ready!")
    print("You can ask questions about:")
    print("1. Direct flights from an airport (e.g., 'Show me all direct flights from LAX')")
    print("2. Shortest path between airports (e.g., 'What's the best route from JFK to LHR?')")
    print("3. Airport analysis (e.g., 'Analyze the importance of DXB in the network')")
    print("\nType 'quit' to exit")
    print("Type 'metrics' to see system metrics")
    print("Type 'health' to check system health")

    while True:
        try:
            query = input("\nWhat would you like to know? ").strip()
            if query.lower() == 'quit':
                break
            elif query.lower() == 'metrics':
                display_metrics()
                continue
            elif query.lower() == 'health':
                status = health_check.get_status()
                print("\nSystem Health:")
                print(f"Database: {'✓' if status['database'] else '✗'}")
                print(f"LLM: {'✓' if status['llm'] else '✗'}")
                print(f"Graph: {'✓' if status['graph'] else '✗'}")
                print(f"Overall Status: {'✓' if status['healthy'] else '✗'}")
                if status['errors']:
                    print("\nErrors:")
                    for err in status['errors']:
                        print(f" - {err}")
                continue

            metrics.record_query()
            start_time = time.time()

            print_status("Processing your request...")
            try:
                # Create the system message
                system_message = """You are a helpful assistant that analyzes flight networks. FOLLOW THESE INSTRUCTIONS EXACTLY:

Step 1 - Tool Call:
When you first receive a query, you MUST ONLY output a tool call in this EXACT format:
TOOL: <tool_name> <parameters>

DO NOT include any other text, explanations, or greetings before or after the tool call.

For the query "What's the shortest path from LAX to JFK?", you MUST respond with:
TOOL: FindShortestPath LAX,JFK

For the query "Show me direct flights from LAX", you MUST respond with:
TOOL: FindDirectFlights LAX

For general queries about airports or lists of airports, use TextToAQL with the query in quotes:
TOOL: TextToAQL "list all airports in the USA"

Here are all available tools:
- FindDirectFlights: For queries about direct flights from an airport. Input: IATA code (e.g., 'LAX').
- CountDirectFlights: For counting direct flights from an airport. Input: IATA code (e.g., 'LAX').
- CheckDirectFlight: For checking if there's a direct flight between airports. Input: Two IATA codes (e.g., 'LAX,JFK').
- FindShortestPath: For finding the shortest path between airports. Input: Two IATA codes (e.g., 'LAX,JFK').
- FindMinimumStopRoute: For finding route with fewest stops. Input: Two IATA codes (e.g., 'LAX,JFK').
- CompareRoutes: For comparing different route options. Input: Two IATA codes (e.g., 'LAX,JFK').
- AnalyzeAirportImportance: For analyzing an airport's importance. Input: IATA code (e.g., 'LAX').
- TextToAQL: For general flight queries. Input: Your query in quotes.
- TextToNetworkX: For complex network analysis. Input: Your query in quotes.

Step 2 - Summary:
After receiving the tool result, provide a natural language summary based ONLY on the information in the tool result.
Never make up data or include information not provided in the tool result.
"""

                # Initialize state
                initial_state = {
                    "messages": [
                        SystemMessage(content=system_message),
                        HumanMessage(content=query)
                    ],
                    "tools": tool_list,
                    "tool_results": {},
                    "current_tool": ""
                }
                
                # Invoke the agent using the run method instead of invoke
                start_time = time.time()
                result = agent_executor.run(initial_state)
                end_time = time.time()
                
                # Record metrics
                metrics.record_query()
                metrics.record_response_time((end_time - start_time) * 1000)  # Convert to ms
                
                # Check if we have a tool result
                if result.get("tool_results"):
                    # Tool was executed, metrics should already be recorded in the tool execution
                    for tool_name, tool_result in result["tool_results"].items():
                        if tool_name not in metrics.get_metrics()["tool_usage"]:
                            # Ensure the tool usage is recorded
                            metrics.record_tool_usage(tool_name)
                
                # Get the last message
                ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
                if ai_msgs:
                    last_msg = ai_msgs[-1].content
                    
                    # Check if it's a tool call
                    if "TOOL:" in last_msg:
                        try:
                            # Extract the tool call
                            tool_call_match = re.search(r"TOOL:\s*(\w+)(?:\s+(.+))?", last_msg)
                            if tool_call_match:
                                tool_name = tool_call_match.group(1)
                                tool_input = tool_call_match.group(2) if tool_call_match.group(2) else ""
                                
                                logger.info(f"Extracted tool call: {tool_name} with input: {tool_input}")
                                
                                # Check if the tool selection makes sense for the query
                                if "shortest path" in query.lower() and tool_name != "FindShortestPath":
                                    logger.warning(f"Query mentions 'shortest path' but tool {tool_name} was selected")
                                    print(f"\nRetrying with the correct tool for finding shortest paths...")
                                    tool_name = "FindShortestPath"
                                    # Extract airport codes from the query
                                    airports = re.findall(r'\b([A-Z]{3})\b', query.upper())
                                    if len(airports) >= 2:
                                        tool_input = f"{airports[0]},{airports[1]}"
                                        print(f"Using airports: {tool_input}")
                                
                                if "direct flights" in query.lower() and tool_name != "FindDirectFlights":
                                    logger.warning(f"Query mentions 'direct flights' but tool {tool_name} was selected")
                                    print(f"\nRetrying with the correct tool for finding direct flights...")
                                    tool_name = "FindDirectFlights"
                                    # Extract airport code from the query
                                    airports = re.findall(r'\b([A-Z]{3})\b', query.upper())
                                    if airports:
                                        tool_input = airports[0]
                                        print(f"Using airport: {tool_input}")
                                
                                if tool_name in result["tools"]:
                                    try:
                                        print(f"\nSearching for information...")
                                        # Record the tool usage in metrics
                                        metrics.record_tool_usage(tool_name)
                                        
                                        # Execute the tool
                                        tool_result = result["tools"][tool_name].func(tool_input)
                                        
                                        # Handle case where tool result is None or empty
                                        if tool_result is None or (isinstance(tool_result, str) and not tool_result.strip()):
                                            tool_result = f"No results found for {tool_name} with input '{tool_input}'."
                                            logger.warning(f"Empty result from tool {tool_name}")
                                        
                                        # Add tool result to state and messages for next iteration
                                        result["tool_results"][tool_name] = tool_result
                                        
                                        # For general queries that don't match the tool call, provide a helpful response
                                        if (("airport" in query.lower() and "list" in query.lower()) or 
                                            ("all" in query.lower() and "airport" in query.lower())):
                                            print("\nI don't have a specific tool to list all airports. I can show you direct flights from specific airports like LAX, JFK, etc.")
                                            print("Try asking about direct flights from a specific airport or the shortest path between two airports.")
                                            return
                                        
                                        # For queries about specific airports by name (not code)
                                        if "airport" in query.lower() and not re.search(r'\b[A-Z]{3}\b', query.upper()):
                                            airport_name = re.sub(r'(analyze|analyse|airport|importance|direct flights|flights from|flights to)', '', query.lower()).strip()
                                            if airport_name:
                                                print(f"\nTo query about {airport_name.title()} airport, please use its IATA code (e.g., LAX for Los Angeles, JFK for New York).")
                                                print("Try asking about direct flights from a specific airport code or the shortest path between two airport codes.")
                                                return
                                        
                                        result["messages"].append(SystemMessage(content=f"""Tool Result: {tool_result}

Please provide a natural language summary of this result. Remember:
1. Only include information from the tool result
2. Make it conversational and easy to understand
3. Do not include any tool calls or technical details
4. Do not make up any information not in the result"""))
                                        
                                        # Continue the conversation to get the summary
                                        result = agent_executor.run(result)
                                        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
                                        
                                        # Get the last non-tool-call message
                                        summary = None
                                        for msg in reversed(ai_msgs):
                                            if msg.content and not "TOOL:" in msg.content:
                                                summary = msg.content
                                                break
                                        
                                        if summary:
                                            print("\n" + summary)
                                        else:
                                            # If we couldn't get a summary, just show the tool result
                                            print("\nHere's what I found:", tool_result)
                                            
                                    except Exception as e:
                                        logger.error(f"Error executing tool {tool_name}: {str(e)}")
                                        print(f"\nI encountered an error while processing your request. Please try again or rephrase your question.")
                                        metrics.record_error()
                                else:
                                    logger.error(f"Unknown tool requested: {tool_name}")
                                    print(f"\nI don't know how to handle '{tool_name}'. Please try asking your question differently.")
                                    metrics.record_error()
                        except Exception as e:
                            logger.error(f"Error parsing tool call: {str(e)}")
                            print(f"\nI had trouble understanding your request: {str(e)}")
                            print("Please try rephrasing your question.")
                            metrics.record_error()
                    else:
                        print("\n" + last_msg)
                else:
                    print("\nI'm sorry, I couldn't generate a response. Please try asking your question differently.")

            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                print(f"\nError processing query: {str(e)}")
                print("Please try rephrasing or use specific IATA codes (e.g., 'LAX').")
                metrics.record_error()

            if metrics.metrics['queries'] % 10 == 0:
                health_check.run_health_check()

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            metrics.record_error()
            print(f"\nError processing request: {str(e)}")
            print("Please try rephrasing or use specific IATA codes (e.g., 'LAX').")

    print("\nFinal System Metrics:")
    final_metrics = metrics.get_metrics()
    print(f"Total Queries: {final_metrics['queries']}")
    print(f"Cache Hit Rate: {final_metrics['cache_hit_rate']:.2%}")
    print(f"Average Response Time: {final_metrics.get('avg_response_time', 0):.2f}ms")
    print(f"Total Errors: {final_metrics['errors']}")

def display_metrics():
    m = metrics.get_metrics()
    print("\nSystem Metrics:")
    print(f"Total Queries: {m['queries']}")
    print(f"Cache Hit Rate: {m['cache_hit_rate']:.2%}")
    print(f"Average Response Time: {m.get('avg_response_time', 0):.2f}ms")
    print(f"LLM Calls: {m['llm_calls']}")
    
    print("\nTool Usage:")
    if m['tool_usage']:
        for tool, count in m['tool_usage'].items():
            print(f"- {tool}: {count} times")
    else:
        print("No tools have been used yet.")

if __name__ == "__main__":
    main() 