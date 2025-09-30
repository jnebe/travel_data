from . import ModelType
from .basic import GravityModel
from ..trip import Trip

class SplitGravityModel(GravityModel):

    def gravity(self, trip: Trip):
        if trip.distance.kilometers < self.gamma:
            return (trip.locations[0].population * trip.locations[1].population) / (trip.distance.kilometers ** self.alpha)
        else:
            return (trip.locations[0].population * trip.locations[1].population) / (trip.distance.kilometers ** self.beta)

    def __init__(self, locations, alpha: float = 1.0, beta: float = 1.0, gamma: int = 600, minimum_distance: int = 100):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # Call the super-constructor last, because that will start the matrix generation, for which all parameters must be set!!!
        super().__init__(locations, minimum_distance)

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