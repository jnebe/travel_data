from pathlib import Path

import polars as pl
import tqdm
import numpy as np

from .location import Location, LocationContainer
from .distance import BaseLocationAssigner

class Trip:

    TRIP_SCHEMA = \
    {
        "start_name": pl.String,
        "start_id": pl.String,
        "start_area": pl.Int128,
        "start_population": pl.Int64,
        "start_lat": pl.Float64,
        "start_long": pl.Float64,
        "end_name": pl.String,
        "end_id": pl.String,
        "end_area": pl.Int128,
        "end_population": pl.Int64,
        "end_lat": pl.Float64,
        "end_long": pl.Float64,
        "distance": pl.Float64
    }

    def __init__(self, location_a: Location, location_b: Location):
        if (not isinstance(location_a, Location)) or (not isinstance(location_b, Location)):
            raise TypeError(f"Trip locations need to be of type Location not {type(location_a)} or {type(location_b)}")
        self.locations: tuple[Location, Location] = tuple(sorted((location_a, location_b)))

    @property
    def distance(self):
        return self.locations[0].distance_to(self.locations[1])
    
    @staticmethod
    def from_dict(value: dict) -> "Trip":
        return \
        Trip(
            Location(value["start_name"], value["start_id"], value["start_lat"], value["start_long"], value["start_area"], value["start_population"]),
            Location(value["end_name"], value["end_id"], value["end_lat"], value["end_long"], value["end_area"], value["end_population"])
        )
    
    def __getstate__(self):
        return self.locations

    def __setstate__(self, state: tuple[Location, Location]):
        self.locations = state

    def __eq__(self, value):
        if not isinstance(value, Trip):
            return False
        return  self.locations == value.locations
    
    def __hash__(self):
        return hash(self.locations)
    
    def __repr__(self):
        return f"Trip(locations={self.locations})"
    
class TripContainer:

    def __init__(self, results: list[Trip]):
        self.trips = results
        self._dictionary = None
        self._df = None

    def append(self, trip: Trip):
        self.trips.append(trip)
        self._dictionary = None
        self._df = None
    
    def extend(self, trips: list[Trip]):
        self.trips.extend(trips)
        self._dictionary = None
        self._df = None

    def update(self, trips: list[Trip]):
        self.trips = trips
        self._dictionary = None
        self._df = None

    @property
    def trips(self) -> list[Trip]:
        if self._trips is None:
            self._trips = []
            if self.dictionary is not None:
                for trip, count in tqdm.tqdm(self.dictionary.items(), desc="Making List of Trips", total=len(self.dictionary), unit="entry(ies)"):
                    for _ in range(count):
                        self._trips.append(trip)
            elif self.df is not None:
                for row in tqdm.tqdm(self.df.iter_rows(named=True), desc="Making List of Trips", total=self.df.height, unit="row(s)"):
                    self._trips.append(
                        Trip.from_dict(row)                        
                    )
        return self._trips
    
    @trips.setter
    def trips(self, value):
        if not isinstance(value, list):
            raise TypeError("The trips property can only be set to lists of Trips!")
        for trip in value:
            if not isinstance(trip, Trip):
                raise TypeError(f"The list contains an object of type {type(trip)}, which is not a Trip!")
        self._trips = value
        self._df = None
        self._dict = None

    @property
    def df(self) -> pl.DataFrame:
        if self._df is None:
            rows = []
            for trip in tqdm.tqdm(self.trips, desc="Making DataFrame", total=len(self.trips), unit="rows"):
                rows.append(
                    {
                    "start_name": trip.locations[0].name,
                    "start_id": trip.locations[0].lid,
                    "start_area": trip.locations[0].area,
                    "start_population": trip.locations[0].population,
                    "start_lat": trip.locations[0].coordinates[0],
                    "start_long": trip.locations[0].coordinates[1],
                    "end_name": trip.locations[1].name,
                    "end_id": trip.locations[1].lid,
                    "end_area": trip.locations[1].area,
                    "end_population": trip.locations[1].population,
                    "end_lat": trip.locations[1].coordinates[0],
                    "end_long": trip.locations[1].coordinates[1],
                    "distance": float(trip.distance.km)
                    }
                )

            self._df = pl.DataFrame(data=rows,
                schema=Trip.TRIP_SCHEMA
            )
        return self._df
    
    @df.setter
    def df(self, value) -> dict[Trip, int]:
        if not isinstance(value, pl.DataFrame):
            raise TypeError("The df property can only be set to lists of Trips!")
        self._trips = None
        self._df = value
        self._dict = None

    @property
    def dictionary(self):
        if self._dict is None:
            self._dict = {}
            for trip in tqdm.tqdm(self.trips, desc="Making Dict", total=len(self.trips), unit="trips"):
                if self._dict.get(trip, None) is None:
                    self._dict[trip] = 1
                else:
                    self._dict[trip] += 1
        return self._dict
    
    @dictionary.setter
    def dictionary(self, value):
        if not isinstance(value, pl.DataFrame):
            raise TypeError("The dictionary property can only be set to lists of Trips!")
        for _key, _ in self._dict.items():
            if not isinstance(_key, Trip):
                raise TypeError(f"The key is of type {type(_key)} and not Trip!")
        self._trips = None
        self._df = None
        self._dict = value
        
      
    def to_csv(self, filename: Path):
        self.df.write_csv(filename)
    
    @staticmethod
    def from_csv(filename: Path):
        df = pl.read_csv(filename, schema=Trip.TRIP_SCHEMA)
        trips = []
        for row in tqdm.tqdm(df.iter_rows(named=True), desc="Loading Trips", total=df.height, unit="row(s)"):
            trips.append(Trip.from_dict(row))
        tc = TripContainer(trips)
        tc._df = df
        return tc

    def __len__(self):
        return len(self.trips)

class TripLoader:

    @staticmethod
    def load_trips(loc_assigner: BaseLocationAssigner, trips: Path | str, trips_schema: dict[str, str], silent: bool = False) -> TripContainer:
        if isinstance(trips, str):
            trips = Path(trips)

        start_lat, start_long, end_lat, end_long, num = trips_schema.get("start_lat"), trips_schema.get("start_long"), trips_schema.get("stop_lat"), trips_schema.get("stop_long"), trips_schema.get("number") 
        
        tripsDf = pl.read_csv(trips, infer_schema_length=None)

        trips_list = []
        for row in tqdm.tqdm(tripsDf.iter_rows(named=True), desc="Loading trips", total=tripsDf.height, disable=silent):
            start = loc_assigner.check((row[start_lat], row[start_long]))
            end = loc_assigner.check((row[end_lat], row[end_long]))
            if start is None or end is None:
                continue
            for _ in range(row[num]):
                trips_list.append(Trip(
                    start,
                    end
                ))
        return TripContainer(trips_list)
    
