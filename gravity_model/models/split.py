from . import ModelType
from .basic import GravityModel
from ..search.random_search import RandomSearch
from ..trip import Trip, TripContainer

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

    def train(self, desired: TripContainer, parameters: dict[str, tuple[float, float, float]], iterations: int = 100, accuracy: float = 0.1, metric: str = "chi"):
        random_search = RandomSearch(self, desired, parameters)
        random_search.train(iterations, accuracy, metric)
        random_search.apply()

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