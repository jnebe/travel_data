from abc import ABC, abstractmethod
import math
import enum

import numpy as np
from sklearn.neighbors import BallTree
from geopy.distance import distance

from .location import LocationContainer, Location

class LATypes(enum.StrEnum):
    BALLTREE = "BALLTREE"
    BEELINE = "BEELINE"
    CIRCLE = "CIRCLE"

class BaseLocationAssigner(ABC):

    @abstractmethod
    def __init__(self, locations: LocationContainer):
        super().__init__()

    @abstractmethod
    def check(self, coordinates: tuple[float, float]) -> Location | None:
        pass

class BallTreeLocationAssigner(BaseLocationAssigner):
    def __init__(self, locations: LocationContainer):
        self.locations = locations
        coords: list[tuple[float, float]] = []
        for loc in locations.locations:
            coords.append( tuple(np.radians(loc.coordinates)) )
        self.tree = BallTree(coords, leaf_size=2, metric="haversine")

    def check(self, coordinates: tuple[float, float]):
        _, location_index = self.tree.query(np.radians( [(coordinates[0], coordinates[1])] ) , 1)
        location = self.locations.locations[location_index[0][0]]
        return location
    
class BeeLineLocationAssigner(BaseLocationAssigner):
    def __init__(self, locations: LocationContainer):
        self.locations = locations

    def check(self, coordinates):
        nearest_loc = None
        shortest_distance = -1
        for loc in self.locations.locations:
            curr_distance = distance(loc.coordinates, coordinates)
            if shortest_distance == -1 or curr_distance < shortest_distance:
                nearest_loc = loc
                shortest_distance = curr_distance
        return nearest_loc

class CircleLocationAssigner(BaseLocationAssigner):
    def __init__(self, locations: LocationContainer):
        self.locations = locations
        self.areas: dict[Location, float] = {}
        for loc in self.locations.locations:
            area = loc.area / 1,000,000 # divide to get from square meters to square kilometers
            self.areas[loc] = self.calculate_radius(area) 

    @staticmethod
    def calculate_radius(area_km2):
        """returns the radius of a circle from its area."""
        return math.sqrt(area_km2 / math.pi)

    def check(self, coordinates):
        nearest_loc = None
        shortest_distance = -1
        for key, value in self.areas.items():
            curr_distance = distance(key.coordinates, coordinates)
            if curr_distance.km < value: # Check if coordinates are within circle
                if shortest_distance == -1 or curr_distance < shortest_distance: # if within multiple circles, check if the current is nearer
                    nearest_loc = key
                    shortest_distance = curr_distance
        return nearest_loc
