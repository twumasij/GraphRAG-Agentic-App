import pandas as pd
from arango import ArangoClient
import os
import numpy as np

def clean_string(s):
    if pd.isna(s) or s == '\\N':
        return ''
    return str(s).strip()

def clean_float(s):
    if pd.isna(s) or s == '\\N':
        return 0.0
    return float(s)

def clean_int(s):
    if pd.isna(s) or s == '\\N':
        return 0
    return int(s)

def load_data():
    # Initialize the ArangoDB client
    client = ArangoClient(hosts="http://localhost:8529")
    db = client.db("flights_db", username="root", password="")
    
    # Create collections if they don't exist
    if not db.has_collection("airports"):
        airports = db.create_collection("airports")
    else:
        airports = db.collection("airports")
        
    if not db.has_collection("routes"):
        routes = db.create_collection("routes", edge=True)
    else:
        routes = db.collection("routes")
    
    # Create graph if it doesn't exist
    if not db.has_graph("flights_graph"):
        graph = db.create_graph("flights_graph")
        # Add vertex collections
        graph.create_vertex_collection("airports")
        # Add edge definition
        graph.create_edge_definition(
            edge_collection="routes",
            from_vertex_collections=["airports"],
            to_vertex_collections=["airports"]
        )
    
    # Load data from DAT files with proper column names
    airports_df = pd.read_csv("data/airports.dat", header=None, 
                            names=['airport_id', 'name', 'city', 'country', 'iata', 'icao', 
                                  'latitude', 'longitude', 'altitude', 'timezone', 'dst', 'tz', 'type', 'source'])
    
    routes_df = pd.read_csv("data/routes.dat", header=None,
                           names=['airline', 'airline_id', 'source_airport', 'source_airport_id',
                                 'destination_airport', 'destination_airport_id', 'codeshare',
                                 'stops', 'equipment'])
    
    # Clean and prepare airport data
    airports_data = []
    for _, airport in airports_df.iterrows():
        if pd.notna(airport['iata']) and airport['iata'] != '\\N':
            clean_airport = {
                '_key': str(airport['iata']),
                'airport_id': clean_int(airport['airport_id']),
                'name': clean_string(airport['name']),
                'city': clean_string(airport['city']),
                'country': clean_string(airport['country']),
                'iata': clean_string(airport['iata']),
                'icao': clean_string(airport['icao']),
                'latitude': clean_float(airport['latitude']),
                'longitude': clean_float(airport['longitude']),
                'altitude': clean_float(airport['altitude']),
                'timezone': clean_float(airport['timezone']),
                'dst': clean_string(airport['dst']),
                'tz': clean_string(airport['tz']),
                'type': clean_string(airport['type'])
            }
            airports_data.append(clean_airport)
    
    # Clean and prepare route data
    routes_data = []
    for _, route in routes_df.iterrows():
        if (pd.notna(route['source_airport']) and pd.notna(route['destination_airport']) and
            route['source_airport'] != '\\N' and route['destination_airport'] != '\\N'):
            clean_route = {
                '_from': f'airports/{clean_string(route["source_airport"])}',
                '_to': f'airports/{clean_string(route["destination_airport"])}',
                'airline': clean_string(route['airline']),
                'airline_id': clean_string(route['airline_id']),
                'equipment': clean_string(route['equipment']),
                'stops': clean_int(route['stops'])
            }
            routes_data.append(clean_route)
    
    # Import data into collections
    print("Importing airports...")
    airports.import_bulk(airports_data, on_duplicate="replace")
    print(f"Imported {len(airports_data)} airports")
    
    print("Importing routes...")
    routes.import_bulk(routes_data, on_duplicate="replace")
    print(f"Imported {len(routes_data)} routes")
    
    print("Data import complete!")

if __name__ == "__main__":
    load_data() 