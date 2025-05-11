from pathlib import Path
from typing import Union

import polars as pl
from geopy.distance import distance, distance
import tqdm

class Location():

    def __init__(self, name: str, id: str, latitude: float, longitude: float, popoluation: int = None):
        self.name: str = name
        self.id: str = id
        self.latitude: float = latitude
        self.longitude: float = longitude
        if popoluation is None:
            self._population: int = -1
        if isinstance(popoluation, int) and popoluation >= 0:
            self.population = popoluation
    @property
    def population(self):
        return self._population
    
    @population.setter
    def population(self, value: int):
        if not isinstance(value, int):
            raise TypeError(f"Population needs to be of type integer, not {type(value)}")
        if value < 0:
            raise ValueError(f"Population needs to be zero or higher, not {value}")
        self._population = value

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
    
    def __eq__(self, other):
        if not isinstance(other, Location):
            return False
        
        other: Location = other

        return (other.coordinates == self.coordinates) and self.distance_to(other).km < 1
    
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
        return hash((self.name, self.id, self.coordinates, self.population))
    
    def __repr__(self):
        return f"Location(name={self.name},id={self.id},coordinates={self.coordinates},population={self.population})"
        
class CensusLocationLoad():

    @staticmethod
    def load_locations(boundaries: Path | str, population: Path | str, boundaries_schema: dict[str, str], population_schema: dict[str, str], silent: bool = False):
        if isinstance(boundaries, str):
            boundaries = Path(boundaries)
        if isinstance(population, str):
            population = Path(population)
        boundariesDf = pl.read_csv(boundaries, infer_schema_length=None)
        populationDf = pl.read_csv(population, infer_schema_length=None)
        bIndex, bName, bLat, bLong = boundaries_schema.get("index"), boundaries_schema.get("name"), boundaries_schema.get("lat"), boundaries_schema.get("long")
        pIndex, pPopulation = population_schema.get("index"), population_schema.get("population")

        combinedDf = boundariesDf.join(populationDf, how="left", left_on=bIndex, right_on=pIndex)
        locations = []
        for row in tqdm.tqdm(combinedDf.iter_rows(named=True), desc="Loading Locations", total=combinedDf.height, disable=silent):
            locations.append(Location(row[bName], row[bIndex], row[bLat], row[bLong], row[pPopulation]))
        return locations
