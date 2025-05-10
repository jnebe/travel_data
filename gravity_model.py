import math
import csv

from geopy.distance import geodesic, distance

from location import CensusLocationLoad
from gravity import GravityModel
from trip import Trip

def city_distance(city1, city2):
    coords1 = (city1["latitude"], city1["longitude"])
    coords2 = (city2["latitude"], city2["longitude"])
    return(distance(coords1, coords2).km)

def gravity_model(city1, city2, alpha=1, beta=1, gamma=1, G=1):
    # T_ij value relationship i and j 
    # D_ij distance i and j
    # P    population
    
    # alpha, beta
    # population parameter, should be high, if the citysize should influence the result more (0-1)
    #alpha = 0.9
    #beta = 0.9

    # gamma, distance parameter, should be high, if a greater distance should decrease the expected value
    gamma = 1

    # gravitation constant, scales the whole result
    G = 0.0000001

    P_i = city1["population"]
    P_j = city2["population"]
    D_ij = city_distance(city1, city2)

    T_ij = G * (P_i ** alpha) * (P_j ** beta) / (D_ij ** gamma)
    return(T_ij)

# makes a city to a circle based on its area -> can be refined later
def calculate_radius(area_km2):
    """returns the radius of a circle from its area."""
    return math.sqrt(area_km2 / math.pi)

# calculate the city distance

# Example: London and Birmingham
london = {
        "name": "London", 
        "latitude": 51.5072,
        "longitude": -0.1275,
        "population": 8866180,
        "area_km2": 1572 
}

birmingham = {
        "name": "Birmingham", 
        "latitude": 52.48, 
        "longitude": -1.9024,
        "population": 1157603, 
        "area_km2": 268
}

manchester = {
        "name": "Manchester",
        "latitude": 53.479,
        "longitude": -2.2452,
        "population": 568996, 
        "area_km2": 116
}

# Read the csv file
csv_1 = 'celltower_data/mod_part-00000-f7a38d19-bcf1-40c6-9036-cff194323e9b-c000.csv'

# function to sum up all trips between the city circles
# I: city1, city2, radius1, radius2  O: total_trips

def calc_total_trips(city1, city2):
    total_trips = 0
    coords1 = (city1["latitude"], city1["longitude"])
    coords2 = (city2["latitude"], city2["longitude"])
    radius1 = calculate_radius(city1["area_km2"])
    radius2 = calculate_radius(city2["area_km2"])
    with open(csv_1, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader: 
            # read in the coordinates and trips
            start = (float(row['home_coord_x']), float(row['home_coord_y']))
            end = (float(row['dest_coord_x']), float(row['dest_coord_y']))
            trips = int(row['frequency'])

            # check trip: city1 to city2 and city2 to city1
            start_in_city1 = geodesic(start, coords1).km <= radius1
            end_in_city2 = geodesic(end, coords2).km <= radius2

            start_in_city2 = geodesic(start, coords2).km <= radius2
            end_in_city1 = geodesic(end, coords1).km <= radius1

            if (start_in_city1 and end_in_city2) or (start_in_city2 and end_in_city1):
                total_trips += trips
        return(total_trips)

if __name__ == "__main__":
    locs = CensusLocationLoad.load_locations("census_data/uk_boundaries_2024.csv", "census_data/uk_2022.csv",
                              {"index": "LAD24CD", "name": "LAD24NM", "lat": "LAT", "long": "LONG"},
                              {"index": "Code", "population": "Population"})
    model = GravityModel(locs)
    trips = Trip.load_trips(locs, "celltower_data/merged_uk_data.csv",
                            {"start_lat": "home_coord_x", "start_long": "home_coord_y", "stop_lat": "dest_coord_x", "stop_long": "dest_coord_y", "number": "frequency"})
    
    real_trips = Trip.make_dict(trips)
    model_trips = Trip.make_dict(model.make_trips(len(trips)))

    Trip.to_dataframe(model_trips).write_csv("modeled_trips.csv", float_precision=2)

    for trip in model.all_trips:
        print(f"Trip from {trip.locations[0].name} to {trip.locations[1].name}: real {real_trips.get(trip, -1)} | sim {model_trips.get(trip, -1)} {model.matrix.get(trip)}")

    # 1 line trip calculation with 1 csv_file should take about 1.15 min
    trips_london_birmingham = calc_total_trips(london, birmingham)
    trips_london_manchester = calc_total_trips(london, manchester)
    trips_birmingham_manchester = calc_total_trips(birmingham, manchester)

    pred_trips_london_birmingham = gravity_model(london, birmingham)
    pred_trips_london_manchester = gravity_model(london, manchester)
    pred_trips_birmingham_manchester = gravity_model(birmingham, manchester)

    print("\nPredicted trips between London and Birmingham:", pred_trips_london_birmingham)
    print("Predicted trips between London and Manchester:", pred_trips_london_manchester)
    print("Predicted trips between Birmingham and Manchester:", pred_trips_birmingham_manchester)
    print("\n")

    print("Real trips between london and birmingham:", trips_london_birmingham)
    print("Real trips between london and manchester:", trips_london_manchester)
    print("Real trips between birmingham and manchester:", trips_birmingham_manchester,"\n")


# ToDo: 
# writing an automatic optimization to calibrate the gravity model parameter


