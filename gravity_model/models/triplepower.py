from . import ModelType
from .power import PowerGravityModel
from ..ars import TriplePowerRandomSearch
from ..trip import Trip, TripContainer
from ..log import logger

class TriplePowerGravityModel(PowerGravityModel):

    def gravity(self, trip: Trip):
        return ((trip.locations[0].population ** self.beta[0]) * (trip.locations[1].population ** self.beta[0])) / (trip.distance.kilometers ** self.alpha)

    def __init__(self, locations, alpha: float = 1.0, beta: tuple[float, float] = 1.0, minimum_distance: int = 100):
        self.beta = beta
        # Call the super-constructor last, because that will start the matrix generation, for which all parameters must be set!!!
        super().__init__(locations, alpha, minimum_distance)

    def train(self, desired: TripContainer, parameters: dict[str, tuple[float, float, float]], iterations: int = 100, accuracy: float = 0.1):
        ars = TriplePowerRandomSearch(self, desired, parameters)
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