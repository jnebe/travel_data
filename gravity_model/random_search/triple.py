import time
import random

from ..training import Parameter, chi_square_distance, get_histogram, kolmogorov_smirnov_statistic, get_ccdf
from ..trip import TripContainer
from ..log import logger

from . import DEFAULT_TRAINING_TRIPS

class AlphaBetaGammaRandomSearch():

    def __init__(self, model: "PowerGravityModel", desired: TripContainer, parameters: dict[str, tuple[float, float, float]]):
        self.model = model
        self.real_data = desired
        self.parameters: dict[str, Parameter] = {}
        self.metrics: dict[str, float] = { "chi" : None }
        for name, value in parameters.items():
            self.parameters[name] = Parameter(name, value[2], value[0], value[1])

    def train(self, iterations: int = 100, accuracy: float = -1.0):
        start_time = time.time()
        iteration = 0
        alpha_fixed = False
        beta_fixed = False
        
        while (iteration < iterations and iterations != -1):
            self.model.alpha = random.uniform(self.parameters["alpha"].minimum, self.parameters["alpha"].maximum)
            self.model.beta = random.uniform(self.parameters["beta"].minimum, self.parameters["beta"].maximum)
            self.model.gamma = random.uniform(self.parameters["gamma"].minimum, self.parameters["gamma"].maximum)

            self.model.recreate_matrix()
            model_trips = self.model.make_trips(DEFAULT_TRAINING_TRIPS)
            chi = chi_square_distance(get_histogram(self.real_data), get_histogram(model_trips))
            logger.info(f"Iteration {iteration} - alpha {self.model.alpha} [{self.parameters["alpha"].minimum}, {self.parameters["alpha"].maximum}] - beta home {self.model.beta[0]} [{self.parameters["beta"].minimum}, {self.parameters["beta"].maximum}] - gamma dest {self.model.beta[1]} [{self.parameters["gamma"].minimum}, {self.parameters["gamma"].maximum}] - Chi-Squared Distance: {chi}")
            if self.metrics["chi"] is None or  chi < self.metrics["chi"]:
                self.parameters["alpha"].value = self.model.alpha
                self.parameters["beta"].value = self.model.beta[0]
                self.parameters["gamma"].value = self.model.beta[1]
                self.metrics["chi"] = chi
            if (self.parameters["alpha"].maximum - self.parameters["alpha"].minimum) < accuracy:
                alpha_fixed = True
            if (self.parameters["beta"].maximum - self.parameters["beta"].minimum) < accuracy and \
                (self.parameters["gamma"].maximum - self.parameters["gamma"].minimum) < accuracy:
                beta_fixed = True
            if accuracy != -1.0 and alpha_fixed and beta_fixed:
                break
            iteration += 1
        end_time = time.time()
        logger.critical(f"Total Training time: {end_time-start_time}s")
        logger.critical(f"Best results with: alpha, beta, gamma= {self.parameters["alpha"].value}, {self.parameters["beta"].value}, {self.parameters["gamma"].value} - Chi-Squared Distance: {self.metrics["chi"]}")

    def apply(self):
        self.model.alpha = self.parameters["alpha"].value
        self.model.beta = self.parameters["beta"].value
        self.model.gamma = self.parameters["gamma"].value
        self.model.recreate_matrix()