from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from os import cpu_count

import polars as pl
import tqdm
from geopy.distance import distance

from .log import logger
from .location import Location
from .distance import BaseLocationAssigner

class Trip:

    TRIP_SCHEMA = \
    {
        "start_name": pl.String,
        "start_id": pl.String,
        "start_area": pl.Float64,
        "start_population": pl.Int64,
        "start_lat": pl.Float64,
        "start_long": pl.Float64,
        "end_name": pl.String,
        "end_id": pl.String,
        "end_area": pl.Float64,
        "end_population": pl.Int64,
        "end_lat": pl.Float64,
        "end_long": pl.Float64,
        "distance": pl.Float64
    }

    def __init__(self, location_a: Location, location_b: Location):
        if (not isinstance(location_a, Location)) or (not isinstance(location_b, Location)):
            raise TypeError(f"Trip locations need to be of type Location not {type(location_a)} or {type(location_b)}")
        self.home = location_a
        self.target = location_b

    def make_copy(self):
        return Trip(self.home, self.target)

    @property
    def locations(self):
        return (self.home, self.target)

    @property
    def distance(self) -> distance:
        return self.home.distance_to(self.target)
    
    @staticmethod
    def from_dict(value: dict) -> "Trip":
        return \
        Trip(
            Location(value["start_name"], value["start_id"], value["start_lat"], value["start_long"], value["start_area"], value["start_population"]),
            Location(value["end_name"], value["end_id"], value["end_lat"], value["end_long"], value["end_area"], value["end_population"])
        )
    
    def to_dict(self) -> dict:
        return {
            "start_name": self.home.name,
            "start_id": self.home.lid,
            "start_area": self.home.area,
            "start_population": self.home.population,
            "start_lat": self.home.coordinates[0],
            "start_long": self.home.coordinates[1],
            "end_name": self.target.name,
            "end_id": self.target.lid,
            "end_area": self.target.area,
            "end_population": self.target.population,
            "end_lat": self.target.coordinates[0],
            "end_long": self.target.coordinates[1],
            "distance": float(self.distance.km)
        }
    
    def to_list(self) -> list:
        return [
            self.home.name,
            self.home.lid,
            self.home.area,
            self.home.population,
            self.home.coordinates[0],
            self.home.coordinates[1],
            self.target.name,
            self.target.lid,
            self.target.area,
            self.target.population,
            self.target.coordinates[0],
            self.target.coordinates[1],
            float(self.distance.km)
        ]
    
    def __getstate__(self):
        return self.locations

    def __setstate__(self, state: tuple[Location, Location]):
        self.home = state[0]
        self.target = state[1]

    def __eq__(self, value):
        if not isinstance(value, Trip):
            return False
        return  self.locations == value.locations
    
    def __hash__(self):
        return hash(self.locations)
    
    def __repr__(self):
        return f"Trip(locations={self.locations})"
    
class TripContainer:

    def __init__(self, results: list[Trip] | pl.DataFrame | dict[Trip, int] = None):
        if results is None:
            raise ValueError("TripContainer needs to be initialized with a list of Trips, a DataFrame or a dictionary of Trips!")
        self._trips = None
        self._dictionary = None
        self._df = None

        if isinstance(results, list):
            self._trips = results
        elif isinstance(results, pl.DataFrame):
            self._df = results
        elif isinstance(results, dict):
            self._dictionary = results

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

    @staticmethod
    def chunkify(lst, n):
        return [lst[i::n] for i in range(n)]

    @staticmethod
    def process_chunk(chunk: list[Trip]) -> pl.DataFrame:
        rows = [trip.to_list() for trip in chunk]
        return pl.DataFrame(
            data=rows,
            orient="row",
            schema=Trip.TRIP_SCHEMA
        )

    @property
    def df(self) -> pl.DataFrame:
        if self._df is None:
            num_chunks = min(cpu_count(), max(1, (len(self._trips) // 250_000)))
            chunks = self.chunkify(self.trips, num_chunks)
            logger.info(f"Creating DataFrame from Trips in {num_chunks} chunks of size {len(chunks[0])}...")
            with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
                try:
                    dfs = list(executor.map(TripContainer.process_chunk, chunks))
                except KeyboardInterrupt as e:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise e
            self._df = pl.concat(dfs, rechunk=True)
            

            logger.info(f"DataFrame created with {self._df.height} rows and {self._df.width} columns")
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
        
    def as_relative(self) -> dict[Trip, float]:
        relative_trips = { }
        for key, value in tqdm.tqdm(self.dictionary.items(), desc="Making relative trip dict", total=len(self.dictionary), unit="entry(ies)"):
            relative_trips[key] = value / len(self)
        return relative_trips
      
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
        if self._trips is not None:
            return len(self.trips)
        if self._df is not None:
            return self.df.height
        if self._dictionary is not None:
            return len(self.dictionary)
        return 0 
    
    def __getitem__(self, item):
        if not isinstance(item, int):
            raise KeyError("TripContainer elements need to be accessed using index!")
        return self.trips[item]

class TripLoader:

    @staticmethod
    def load_trips(loc_assigner: BaseLocationAssigner, trips: Path | str, trips_schema: dict[str, str], keep_distance: bool = False, min_distance: float = 0.0, silent: bool = False) -> TripContainer:
        if isinstance(trips, str):
            trips = Path(trips)

        if keep_distance:
            logger.info("Will keep start and end coordinates of trips that will be mapped")
        logger.info(f"Minimum distance set to {min_distance}")

        start_lat, start_long, end_lat, end_long, num = trips_schema.get("start_lat"), trips_schema.get("start_long"), trips_schema.get("stop_lat"), trips_schema.get("stop_long"), trips_schema.get("number") 
        
        tripsDf = pl.read_csv(trips, infer_schema_length=None)

        trips_list = []
        for row in tqdm.tqdm(tripsDf.iter_rows(named=True), desc="Loading trips", total=tripsDf.height, disable=silent):
            start = loc_assigner.check((row[start_lat], row[start_long]))
            end = loc_assigner.check((row[end_lat], row[end_long]))
            if keep_distance:
                start = start.get_copy()
                end = end.get_copy()
                start.coordinates = (row[start_lat], row[start_long])
                end.coordinates = (row[end_lat], row[end_long])
            if start is None or end is None:
                continue
            new_trip = Trip(start, end)
            if new_trip.distance.km < min_distance:
                continue
            for _ in range(row[num]):
                trips_list.append(new_trip.make_copy())
        return TripContainer(trips_list)
    
