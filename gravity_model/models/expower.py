from . import ModelType
from .basic import GravityModel
from ..search.random_search import RandomSearch
from ..trip import Trip, TripContainer

from sys import float_info

from math import exp

class ExponentialPowerGravityModel(GravityModel):

    def gravity(self, trip: Trip):
        denominator = (1 / (trip.distance.kilometers ** self.alpha)) * max(exp(-self.beta * trip.distance.kilometers), float_info.min)
        return (trip.locations[0].population * trip.locations[1].population) * denominator

    def __init__(self, locations, alpha: float = 1.0, beta: float = 0.1, minimum_distance: int = 100):
        self.alpha = alpha
        self.beta = beta
        # Call the super-constructor last, because that will start the matrix generation, for which all parameters must be set!!!
        super().__init__(locations, minimum_distance)

    def train(self, desired: TripContainer, parameters: dict[str, tuple[float, float, float]], iterations: int = 100, accuracy: float = 0.1, metric: str = "chi"):
        random_search = RandomSearch(self, desired, parameters)
        random_search.train(iterations, accuracy, metric)
        random_search.apply()

    def __getstate__(self):
        return \
        {
            "type": ModelType.POWER,
            "alpha": self.alpha,
            "beta" : self.beta,
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