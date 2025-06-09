from typing import Iterable, Union
from itertools import product
from random import choices
import enum
from pathlib import Path
import random
import time

from jsonpickle import encode, decode

from .ars import PowerRandomSearch, DoublePowerRandomSearch
from .location import LocationContainer
from .trip import Trip, TripContainer
from .training import total_variation_distance, chi_square_distance, histogram_intersection_kernel
from .log import logger

class ModelType(enum.StrEnum):
    BASIC = "BASIC"
    POWER = "POWER"
    DOUBLEPOWER = "DOUBLEPOWER"

Gravity = float

class GravityModel():

    def __init__(self, locations: LocationContainer, minimum_distance: int = 100):
        self.matrix: dict[Trip, Gravity] = {}
        self.total_gravity: Gravity = 0.0

        for loc_a, loc_b in product(locations.locations, repeat=2):
            if loc_a == loc_b:
                continue
            trip = Trip(loc_a, loc_b)
            if self.matrix.get(trip, None) is not None:
                continue
            if trip.distance < minimum_distance:
                continue
            self.matrix[trip] = 0
        self.recreate_matrix()

    def gravity(self, trip: Trip):
        return (trip.locations[0].population * trip.locations[1].population) / trip.distance.kilometers

    def recreate_matrix(self):
        self.total_gravity = 0
        for trip in self.matrix.keys():
            gravity = self.gravity(trip)
            self.matrix[trip] = gravity
            self.total_gravity += gravity

    def train(self, desired: TripContainer, parameters: dict[str, tuple[float, float]], iterations: int = -1, accuracy: float = 0.1, ):
        model_trips = self.make_trips(250000)
        tvd = total_variation_distance(desired, model_trips)
        chi = chi_square_distance(desired.get_histogram(), model_trips.get_histogram())
        hik = histogram_intersection_kernel(desired.get_histogram(), model_trips.get_histogram())
        logger.critical(f"TVD: {tvd}")
        logger.critical(f"CHI^2: {chi}")
        logger.critical(f"HIK: {hik}")

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

    def gravity(self, trip: Trip):
        return (trip.locations[0].population * trip.locations[1].population) / (trip.distance.kilometers ** self.alpha)

    def __init__(self, locations, alpha: float = 1.0, minimum_distance: int = 100):
        self.alpha = alpha
        # Call the super-constructor last, because that will start the matrix generation, for which all parameters must be set!!!
        super().__init__(locations, minimum_distance)

    def train(self, desired: TripContainer, parameters: dict[str, tuple[float, float, float]], iterations: int = 100, accuracy: float = 0.1):
        ars = PowerRandomSearch(self, desired, parameters)
        ars.train(iterations, accuracy)
        ars.apply()

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

class DoublePowerGravityModel(PowerGravityModel):

    def gravity(self, trip: Trip):
        return ((trip.locations[0].population ** self.beta) * (trip.locations[1].population ** self.beta)) / (trip.distance.kilometers ** self.alpha)

    def __init__(self, locations, alpha: float = 1.0, beta: float = 1.0, minimum_distance: int = 100):
        self.beta = beta
        # Call the super-constructor last, because that will start the matrix generation, for which all parameters must be set!!!
        super().__init__(locations, alpha, minimum_distance)

    def train(self, desired: TripContainer, parameters: dict[str, tuple[float, float, float]], iterations: int = 100, accuracy: float = 0.1):
        ars = DoublePowerRandomSearch(self, desired, parameters)
        ars.train(iterations, accuracy)
        ars.apply()

    def __getstate__(self):
        return \
        {
            "type": ModelType.POWER,
            "alpha": self.alpha,
            "beta": self.beta,
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
        self.beta = state.get("beta")
        self.total_gravity = state.get("total")