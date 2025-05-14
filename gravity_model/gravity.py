from typing import Iterable, Union
from itertools import product
from random import choices
import enum
from pathlib import Path
import random

from jsonpickle import encode, decode

from .location import LocationContainer
from .trip import Trip, TripContainer
from .training import total_variation_distance
from .log import logger

class ModelType(enum.StrEnum):
    BASIC = "BASIC"
    POWER = "POWER"

class GravityModel():

    Gravity = Union[float]

    def gravity(self, trip: Trip):
        return (trip.locations[0].population * trip.locations[1].population) / trip.distance.kilometers

    def __init__(self, locations: LocationContainer):
        self.matrix: dict[Trip, GravityModel.Gravity] = {}
        self.total_gravity: GravityModel.Gravity = 0.0

        for loc_a, loc_b in product(locations.locations, repeat=2):
            if loc_a == loc_b:
                continue
            trip = Trip(loc_a, loc_b)
            if self.matrix.get(trip, None) is not None:
                continue
            gravity = self.gravity(trip)
            self.matrix[trip] = gravity
            self.total_gravity += gravity

    def recreate_matrix(self):
        self.total_gravity = 0
        for trip in self.matrix.keys():
            gravity = self.gravity(trip)
            self.matrix[trip] = gravity
            self.total_gravity += gravity

    def train(self, desired: TripContainer, iterations: int = -1, metrics: dict[str, float] = None):
        model_trips = self.make_trips(len(desired))
        tvd = total_variation_distance(desired, model_trips)
        logger.critical(f"TVD: {tvd}")

    @property
    def all_trips(self):
        return list(self.matrix.keys())

    def make_trips(self, n: int) -> TripContainer:
        return TripContainer(choices(list(self.matrix.keys()), weights=list(self.matrix.values()), k=n))
    
    def matrix_as_tuples(self) -> list[tuple[Trip, Gravity]]:
        tuples = []
        for key, value in self.matrix.items():
            tuples.append((key, value))
        return tuples

    def __getstate__(self):
        return \
        {
            "type": ModelType.BASIC,
            "total": self.total_gravity,
            "matrix": self.matrix_as_tuples()
        }

    def __setstate__(self, state: dict):
        if state.get("type", None) is None or state.get("type") is not ModelType.BASIC:
            raise ValueError(f"Type needs to be {ModelType.BASIC.__str__()}")
        matrix_tuples = state.get("matrix")
        self.matrix = {}
        for key, value in matrix_tuples:
            self.matrix[key] = value
        self.total_gravity = state.get("total")

    def to_json(self, filename: Path):
        json = encode(self)
        with filename.open("w") as f:
            f.write(json)

    @staticmethod
    def from_json(filename: Path) -> "GravityModel":
        with filename.open("r") as f:
            json = f.read()
        return decode(json)

    def __len__(self):
        return len(self.matrix)

    def __repr__(self):
        return f"GravityModel(total={self.total_gravity},matrix={self.matrix})"
    
class PowerGravityModel(GravityModel):

    def gravity(self, trip):
        return (trip.locations[0].population * trip.locations[1].population) / (trip.distance.kilometers ** self.alpha)

    def __init__(self, locations, alpha: float = 1.0):
        self.alpha = alpha
        # Call the super-constructor last, because that will start the matrix generation, for which all parameters must be set!!!
        super().__init__(locations)

    def train(self, desired: TripContainer, iterations: int = -1, metrics: dict[str, float] = None):
        iteration = 0
        best_alpha = self.alpha
        lowest_tvd = 1.0
        tvd_target = metrics.get("tvd", 1.0)
        tvd = 1.0
        while tvd > tvd_target and (iteration < iterations and iterations != -1):
            model_trips = self.make_trips(250000)
            tvd = total_variation_distance(desired, model_trips)
            logger.critical(f"Iteration {iteration} - alpha {self.alpha} - TVD: {tvd}")
            if tvd < lowest_tvd:
                best_alpha = self.alpha
                lowest_tvd = tvd
            self.alpha = random.random() * 2
            self.recreate_matrix()
            iteration += 1
        self.alpha = best_alpha
        self.recreate_matrix()
        logger.critical(f"Best results with: alpha = {self.alpha}")

    def __getstate__(self):
        return \
        {
            "type": ModelType.POWER,
            "alpha": self.alpha,
            "total": self.total_gravity,
            "matrix": self.matrix_as_tuples()
        }

    def __setstate__(self, state: dict):
        if state.get("type", None) is None or state.get("type") is not ModelType.POWER:
            raise ValueError(f"Type needs to be {ModelType.BASIC.__str__()}")
        matrix_tuples = state.get("matrix")
        self.matrix = {}
        for key, value in matrix_tuples:
            self.matrix[key] = value
        self.alpha = state.get("alpha")
        self.total_gravity = state.get("total")