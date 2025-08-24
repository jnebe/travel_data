from math import exp
from sys import float_info
from . import ModelType
from .doubleexpo import DoubleExponentialGravityModel
from ..trip import Trip

class TripleExponentialGravityModel(DoubleExponentialGravityModel):

    def gravity(self, trip: Trip):
        numerator = exp(self.beta * trip.locations[0].population) * exp(self.gamma * trip.locations[1].population)
        denominator = exp(-self.alpha * trip.distance.kilometers)
        return max(numerator, float_info.min) * max(denominator, float_info.min)

    def __init__(self, locations, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0, minimum_distance: int = 100):
        self.gamma = gamma
        # Call the super-constructor last, because that will start the matrix generation, for which all parameters must be set!!!
        super().__init__(locations, alpha, beta, minimum_distance)

    def __getstate__(self):
        return \
        {
            "type": ModelType.POWER,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
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
        self.gamma = state.get("gamma")
        self.total_gravity = state.get("total")