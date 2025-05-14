from pathlib import Path
from typing import Union

import polars as pl
from geopy.distance import distance
import tqdm

class Location():

    LOCATION_SCHEMA = \
    {
        "name": pl.String,
        "id": pl.String,
        "lat": pl.Float64,
        "long": pl.Float64,
        "area": pl.Int128,
        "population": pl.Int64
    }

    def __init__(self, location_name: str, location_id: str, latitude: float, longitude: float, area: int, popoluation: int = None):
        self.name: str = location_name
        self.lid: str = location_id
        self.latitude: float = latitude
        self.longitude: float = longitude
        if area is None:
            self._area: int = -1
        if isinstance(area, (int, float)) and area >= 0:
            self.area = int(area)
        if popoluation is None:
            self._population: int = -1
        if isinstance(popoluation, (int, float)) and popoluation >= 0:
            self.population = int(popoluation)

    @property
    def area(self):
        return self._area
    
    @area.setter
    def area(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError(f"Area needs to be of type integer, not {type(value)}")
        if value < 0:
            raise ValueError(f"Area needs to be zero or higher, not {value}")
        self._area = int(value)

    @property
    def population(self):
        return self._population
    
    @population.setter
    def population(self, value: int):
        if not isinstance(value, (int, float)):
            raise TypeError(f"Population needs to be of type integer or float, not {type(value)}")
        if value < 0:
            raise ValueError(f"Population needs to be zero or higher, not {value}")
        self._population = int(value)

    @property
    def coordinates(self) -> tuple[float, float]:
        return (self.latitude, self.longitude)
    
    @coordinates.setter
    def coordinates(self, value: tuple[float, float]):
        if not isinstance(value, tuple):
            raise TypeError(f"value needs to be a tuple of floats, not {value}")
        latitude, longitude = value
        if (not isinstance(latitude, float)) or (not isinstance(longitude, float)):
            raise TypeError(f"value needs to be a tuple of floats, not ({type(latitude)}, {type(longitude)})")
        self.latitude, self.longitude = latitude, longitude

    def distance_to(self, other_location: Union["Location", tuple[float, float]]) -> distance:
        if not isinstance(other_location, (Location, tuple)):
            raise TypeError(f"{other_location} needs to be of type Location or a coordinates tuple not {type(other_location)}")
        if isinstance(other_location, Location):
            return distance(self.coordinates, other_location.coordinates)
        return distance(self.coordinates, other_location)
    
    @staticmethod
    def distance_between(location_a: "Location", location_b: "Location") -> distance:
        if (not isinstance(location_a, Location)) or (not isinstance(location_b, Location)):
            raise TypeError(f"both parameters need to be of type Location! location_a: {type(location_a)}; location_b: {type(location_b)}")
        return location_a.distance_to(location_b)
    
    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "id": self.lid,
            "lat": self.latitude,
            "long": self.longitude,
            "area": self.area,
            "population": self.population
        }
    
    def __getstate__(self):
        return self.as_dict()

    def __setstate__(self, state: dict):
        self.name = state.get("name")
        self.lid = state.get("id")
        self.latitude = state.get("lat")
        self.longitude = state.get("long")
        self.area = state.get("area")
        self.population = state.get("population")

    def __eq__(self, other):
        if not isinstance(other, Location):
            return False
        
        other: Location = other

        return (other.coordinates == self.coordinates) and (self.area == other.area)
    
    def __lt__(self, other):
        if not isinstance(other, Location):
            raise TypeError("Location object can only be compared to other locations!")
        other: Location = other
        if self.population < other.population:
            return True
        if self.population > other.population:
            return False
        
        if self.latitude < other.latitude:
            return True
        if self.latitude > other.latitude:
            return False
        
        if self.longitude < other.longitude:
            return True
        if self.longitude > other.longitude:
            return False
    
    def __hash__(self):
        return hash((self.name, self.lid, self.coordinates, self.population))
    
    def __repr__(self):
        return f"Location(name={self.name},id={self.lid},coordinates={self.coordinates},population={self.population})"
        

class LocationContainer():
    
    def __init__(self, locations: list[Location] = None, df: pl.DataFrame = None):
        if locations is None and df is None:
            raise ValueError("When instantiating a LocationContainer, at least one input must be provided!")
        self._locations = locations
        self._df = df
    
    @property
    def locations(self):
        if self._locations is None and self.df is not None:
            self._locations = []
            for row in tqdm.tqdm(self.df.iter_rows(named=True), desc="Converting DataFrame to list", total=self.df.height, unit="row(s)"):
                self._locations.append(
                    Location(
                        row["name"],
                        row["id"],
                        row["lat"],
                        row["long"],
                        row["area"],
                        row["population"]
                    )
                )
        return self._locations

    @locations.setter
    def locations(self, value):
        if not isinstance(value, list):
            raise TypeError("the locations property needs to be a list of Locations")
        for v in value:
            if not isinstance(v, Location):
                raise TypeError("the list provided can only contain Locations")
        self._locations = value
    
    @property
    def df(self) -> pl.DataFrame:
        if self._df is None and self.locations is not None:
            rows = []
            for loc in tqdm.tqdm(self.locations, desc="Making DataFrame", total=len(self.locations), unit="location(s)"):
                rows.append(
                    loc.as_dict()
                )
            self._df = pl.DataFrame(data=rows,
                schema=Location.LOCATION_SCHEMA
            )
        return self._df

    @df.setter
    def df(self, value):
        if not isinstance(value, pl.DataFrame):
            raise TypeError("the df property needs to be a polars DataFrame")

    def to_csv(self, filename: Path, precision: int = 6):
        self.df.write_csv(filename, float_precision = precision)

    @staticmethod
    def from_csv(filename: Path) -> "LocationContainer":
        return LocationContainer(df=pl.read_csv(filename, schema=Location.LOCATION_SCHEMA))
        
class LocationLoader():

    @staticmethod
    def from_csv(boundaries: Path | str, population: Path | str, boundaries_schema: dict[str, str], population_schema: dict[str, str], silent: bool = False):
        if isinstance(boundaries, str):
            boundaries = Path(boundaries)
        if isinstance(population, str):
            population = Path(population)
        boundariesDf = pl.read_csv(boundaries, infer_schema_length=None)
        populationDf = pl.read_csv(population, infer_schema_length=None)
        bIndex, bName, bLat, bLong, bArea = boundaries_schema.get("index"), boundaries_schema.get("name"), boundaries_schema.get("lat"), boundaries_schema.get("long"), boundaries_schema.get("area")
        pIndex, pPopulation = population_schema.get("index"), population_schema.get("population")

        combinedDf = boundariesDf.join(populationDf, how="left", left_on=bIndex, right_on=pIndex)
        locations = []
        for row in tqdm.tqdm(combinedDf.iter_rows(named=True), desc="Loading Locations", total=combinedDf.height, disable=silent):
            locations.append(Location(row[bName], row[bIndex], row[bLat], row[bLong], row[bArea], row[pPopulation]))
        return LocationContainer(locations)
