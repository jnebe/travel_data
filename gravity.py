from typing import Iterable, Union
from itertools import product
from random import choices

from location import Location
from trip import Trip

class GravityModel():

    Gravity = Union[float]

    @staticmethod
    def gravity(trip: Trip):
        return (trip.locations[0].population * trip.locations[1].population) / trip.distance.kilometers

    def __init__(self, locations: Iterable[Location]):
        self.matrix: dict[Trip, GravityModel.Gravity] = {}
        self.total_gravity: GravityModel.Gravity = 0.0

        for loc_a, loc_b in product(locations, repeat=2):
            if loc_a == loc_b:
                continue
            trip = Trip(loc_a, loc_b)
            if self.matrix.get(trip, None) is not None:
                continue
            gravity = self.gravity(trip)
            self.matrix[trip] = gravity
            self.total_gravity += gravity

    @property
    def all_trips(self):
        return list(self.matrix.keys())

    def make_trips(self, n: int) -> list[Trip]:
        return choices(list(self.matrix.keys()), weights=list(self.matrix.values()), k=n)

    def __len__(self):
        return len(self.matrix)

    def __repr__(self):
        return f"GravityModel(total={self.total_gravity},matrix={self.matrix})"