from . import ModelType
from .basic import GravityModel
from ..random_search.single import AlphaRandomSearch
from ..trip import Trip, TripContainer
from ..log import logger

from sys import float_info

from math import exp

class ExponentialGravityModel(GravityModel):

    def gravity(self, trip: Trip):
        denominator = exp(-self.alpha * trip.distance.kilometers)
        return (trip.locations[0].population * trip.locations[1].population) * max(denominator, float_info.min)

    def __init__(self, locations, alpha: float = 1.0, minimum_distance: int = 100):
        self.alpha = alpha
        # Call the super-constructor last, because that will start the matrix generation, for which all parameters must be set!!!
        super().__init__(locations, minimum_distance)

    def train(self, desired: TripContainer, parameters: dict[str, tuple[float, float, float]], iterations: int = 100, accuracy: float = 0.1):
        ars = AlphaRandomSearch(self, desired, parameters)
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