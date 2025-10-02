import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    r = 6371  # Radius of earth in kilometers
    
    return c * r

def load_airports():
    """Load and process airports data"""
    # Column names for airports.dat
    airport_columns = [
        'airport_id', 'name', 'city', 'country', 'iata', 'icao',
        'latitude', 'longitude', 'altitude', 'timezone', 'dst',
        'tz_database_time_zone', 'type', 'source'
    ]
    
    # Read airports data
    airports_df = pd.read_csv(
        'data/airports.dat',
        names=airport_columns,
        na_values=['\\N']  # Handle missing values
    )
    
    # Keep only necessary columns and remove rows with missing IATA codes
    airports_df = airports_df[['airport_id', 'name', 'city', 'country', 'iata', 'latitude', 'longitude']]
    airports_df = airports_df.dropna(subset=['iata'])
    
    return airports_df

def load_routes():
    """Load and process routes data"""
    # Column names for routes.dat
    route_columns = [
        'airline', 'airline_id', 'source_airport', 'source_airport_id',
        'destination_airport', 'destination_airport_id', 'codeshare',
        'stops', 'equipment'
    ]
    
    # Read routes data
    routes_df = pd.read_csv(
        'data/routes.dat',
        names=route_columns,
        na_values=['\\N']  # Handle missing values
    )
    
    # Keep only necessary columns
    routes_df = routes_df[['source_airport', 'destination_airport']]
    routes_df = routes_df.dropna()
    
    return routes_df

def main():
    # Load data
    print("Loading airports data...")
    airports_df = load_airports()
    
    print("Loading routes data...")
    routes_df = load_routes()
    
    # Create a mapping of IATA codes to airport information
    airports_dict = airports_df.set_index('iata').to_dict('index')
    
    print("Processing routes and calculating distances...")
    # Add coordinates and calculate distances
    graph_edges = []
    for _, route in routes_df.iterrows():
        source = route['source_airport']
        dest = route['destination_airport']
        
        # Skip if either airport is not in our airports database
        if source not in airports_dict or dest not in airports_dict:
            continue
            
        source_info = airports_dict[source]
        dest_info = airports_dict[dest]
        
        # Calculate distance using Haversine formula
        distance = haversine_distance(
            source_info['latitude'], source_info['longitude'],
            dest_info['latitude'], dest_info['longitude']
        )
        
        graph_edges.append({
            'source': source,
            'destination': dest,
            'distance': distance
        })
    
    # Create edges DataFrame
    edges_df = pd.DataFrame(graph_edges)
    
    # Create nodes DataFrame (unique airports from edges)
    unique_airports = pd.concat([
        edges_df['source'],
        edges_df['destination']
    ]).unique()
    
    nodes_df = airports_df[airports_df['iata'].isin(unique_airports)].copy()
    
    print("Saving processed data...")
    # Save processed data
    edges_df.to_csv('data/graph_edges.csv', index=False)
    nodes_df.to_csv('data/graph_nodes.csv', index=False)
    
    print(f"Processing complete!")
    print(f"Number of nodes (airports): {len(nodes_df)}")
    print(f"Number of edges (routes): {len(edges_df)}")

if __name__ == "__main__":
    main() 