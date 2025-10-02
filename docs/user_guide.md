# Flight Network Analysis System User Guide

## Introduction

Welcome to the Flight Network Analysis System! This application allows you to analyze flight networks using natural language queries. Powered by LangGraph and local LLM inference through Ollama, this system provides insights into direct flights, optimal routes, and airport importance.

## Getting Started

### Prerequisites

Before using the application, ensure you have:

1. **Python 3.8+** installed on your system
2. **ArangoDB** running on localhost:8529
3. **Ollama** installed and running on your system
4. The **llama2** model pulled in Ollama

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/twumasij/GraphRag-Agentic-App/
   cd flight-network-analysis
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure ArangoDB is running:
   ```
   # Check if ArangoDB is running
   curl http://localhost:8529
   ```

4. Ensure Ollama is running and has the llama2 model:
   ```
   # Pull the llama2 model if you haven't already
   ollama pull llama2
   ```

### Running the Application

Start the application by running:

```
python scripts/agentic_app.py
```

## Using the Application

The Flight Network Analysis System understands natural language queries about flight networks. Here are the types of questions you can ask:

### Direct Flight Queries

These queries help you find direct flights from a specific airport.

**Examples:**
- "Show me all direct flights from LAX"
- "What are the direct flights from JFK?"
- "List all destinations I can fly to directly from DXB"

### Route Analysis Queries

These queries help you find the best route between two airports.

**Examples:**
- "What's the best route from JFK to LHR?"
- "How can I get from LAX to NRT?"
- "Find the shortest path from SFO to CDG"

### Airport Importance Analysis

These queries help you understand the importance of an airport in the network.

**Examples:**
- "Analyze the importance of DXB in the network"
- "How important is ATL in the flight network?"
- "What's the PageRank score of LHR?"

## Understanding the Results

### Direct Flight Results

When you ask about direct flights, the system will return:
- The total number of destinations
- The top 5 closest destinations (with city, country, and distance)
- The top 5 furthest destinations (with city, country, and distance)

### Route Analysis Results

When you ask about routes between airports, the system will return:
- The number of stops required
- The total distance of the route
- A detailed route showing each airport with city and country

### Airport Analysis Results

When you ask about airport importance, the system will return:
- Basic information about the airport (name, city, country)
- Network metrics including:
  - PageRank score (indicating global importance)
  - Betweenness centrality (indicating importance as a connection hub)
  - Total connections
  - Incoming and outgoing flights

## Tips for Better Results

1. **Use IATA codes** when possible (e.g., "LAX" instead of "Los Angeles")
2. **Be specific** about what you're looking for
3. **Phrase questions clearly** to help the system understand your intent
4. If you get an error, try **rephrasing your question**

## Troubleshooting

### Common Issues

1. **"Error connecting to ArangoDB"**
   - Ensure ArangoDB is running on localhost:8529
   - Check that the "flights_db" database exists

2. **"Error making request to Ollama"**
   - Ensure Ollama is running
   - Check that you have pulled the llama2 model

3. **"No response generated"**
   - Try rephrasing your question
   - Use specific airport codes (e.g., "LAX" instead of "Los Angeles")

### Getting Help

If you encounter issues not covered in this guide, please:
1. Check the project's GitHub issues
2. Submit a detailed bug report including:
   - Your exact query
   - The error message received
   - Your system information

## Advanced Usage

### Adding New Data

The system uses data stored in ArangoDB and a NetworkX graph. To add new data:

1. Update the CSV files in the `data` directory
2. Run the data processing scripts to update the database and graph

### Customizing the Model

By default, the system uses the llama2 model. To use a different model:

1. Pull the model using Ollama: `ollama pull modelname`
2. Modify the `initialize_llm` function in `scripts/agentic_app.py` to use your preferred model

## Conclusion

The Flight Network Analysis System provides a powerful way to explore and analyze flight networks using natural language. By leveraging graph databases, network analysis, and LLMs, it offers insights that would otherwise require complex queries and specialized knowledge.

Happy exploring! 