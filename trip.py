from pathlib import Path

import polars as pl
import tqdm
from sklearn.neighbors import BallTree
import numpy as np

from location import Location

class Trip:

    def __init__(self, location_a: Location, location_b: Location):
        if (not isinstance(location_a, Location)) or (not isinstance(location_b, Location)):
            raise TypeError(f"Trip locations need to be of type Location not {type(location_a)} or {type(location_b)}")
        self.locations: tuple[Location, Location] = tuple(sorted((location_a, location_b)))

    @property
    def distance(self):
        return self.locations[0].distance_to(self.locations[1])

    def __eq__(self, value):
        if not isinstance(value, Trip):
            return False
        return  self.locations == value.locations
    
    def __hash__(self):
        return hash(self.locations)
    
    def __repr__(self):
        return f"Trip(locations={self.locations})"
    
class TripResults:

    def __init__(self, results: list[Trip]):
        self.results = results
        self.dict = None
        self.df = None

    def append(self, trip: Trip):
        self.results.append(trip)
        self.dict = None
        self.df = None
    
    def extend(self, trips: list[Trip]):
        self.results.extend(trips)
        self.dict = None
        self.df = None

    def update(self, trips: list[Trip]):
        self.results = trips
        self.dict = None
        self.df = None

    def as_dict(self):
        if self.dict is None:
            self.dict = {}
            for trip in tqdm.tqdm(self.results, desc="Making Dict",total=len(self.results), unit="trips"):
                if self.dict.get(trip, None) is None:
                    self.dict[trip] = 1
                else:
                    self.dict[trip] += 1
        return self.dict
    
    def as_dataframe(self):
        if self.df is None:
            rows = []
            for trip in tqdm.tqdm(self.results, desc="Making DataFrame", total=len(self.results), unit="rows"):
                rows.append(
                    {"start_lat": trip.locations[0].coordinates[0],
                    "start_long": trip.locations[0].coordinates[1],
                    "end_lat": trip.locations[1].coordinates[0],
                    "end_long": trip.locations[1].coordinates[1],
                    "distance": float(trip.distance.km)}
                )

            self.df = pl.DataFrame(data=rows,
                schema={
                    "start_lat": pl.Float64,
                    "start_long": pl.Float64,
                    "end_lat": pl.Float64,
                    "end_long": pl.Float64,
                    "distance": pl.Float64
                }
            )
        return self.df
    
    def __len__(self):
        return len(self.results)

class TripLoader:

    @staticmethod
    def load_trips(locations: list[Location], trips: Path | str, trips_schema: dict[str, str], silent: bool = False) -> TripResults:
        if isinstance(trips, str):
            trips = Path(trips)

        coords: list[tuple[float, float]] = []
        for loc in locations:
            coords.append( tuple(np.radians(loc.coordinates)) )
        tree = BallTree(coords, leaf_size=2, metric="haversine")

        start_lat, start_long, end_lat, end_long, num = trips_schema.get("start_lat"), trips_schema.get("start_long"), trips_schema.get("stop_lat"), trips_schema.get("stop_long"), trips_schema.get("number") 
        
        tripsDf = pl.read_csv(trips, infer_schema_length=None)

        trips_list = []
        for row in tqdm.tqdm(tripsDf.iter_rows(named=True), desc="Loading trips", total=tripsDf.height, disable=silent):
            for _ in range(row[num]):
                _, start_i = tree.query(np.radians( [(row[start_lat], row[start_long])] ) , 1)
                _, end_i = tree.query(np.radians( [(row[end_lat], row[end_long])] ) , 1)
                trips_list.append(Trip(
                    locations[start_i[0][0]],
                    locations[end_i[0][0]]
                ))
        return TripResults(trips_list)
    
