import time
import random

from ..training import Parameter, chi_square_distance, get_histogram, kolmogorov_smirnov_statistic, get_ccdf
from ..trip import TripContainer
from ..log import logger

from . import DEFAULT_TRAINING_TRIPS

class AlphaRandomSearch():

    def __init__(self, model: "PowerGravityModel", desired: TripContainer, parameters: dict[str, tuple[float, float, float]]):
        self.model = model
        self.real_data = desired
        self.parameters: dict[str, Parameter] = {}
        self.metrics: dict[str, float] = { "chi" : None, "kss": None }
        for name, value in parameters.items():
            self.parameters[name] = Parameter(name, value[2], value[0], value[1])

    def train(self, iterations: int = 100, accuracy: float = -1.0):
        start_time = time.time()
        iteration = 0
        
        try:
            while (iteration < iterations and iterations != -1):
                self.model.alpha = random.uniform(self.parameters["alpha"].minimum, self.parameters["alpha"].maximum)
                self.model.recreate_matrix()
                model_trips: TripContainer = self.model.make_trips(DEFAULT_TRAINING_TRIPS)
                chi = chi_square_distance(get_histogram(self.real_data), get_histogram(model_trips))
                kss = kolmogorov_smirnov_statistic(get_ccdf(self.real_data), get_ccdf(model_trips))
                logger.info(f"Iteration {iteration} - alpha {self.model.alpha} [{self.parameters["alpha"].minimum}, {self.parameters["alpha"].maximum}] - Chi-Squared Distance: {chi} - KSS: {kss}")
                if (self.metrics["kss"] is not None) and kss > self.metrics["kss"]:
                    if self.model.alpha > self.parameters["alpha"].value:
                        self.parameters["alpha"].maximum = self.model.alpha
                    if self.model.alpha < self.parameters["alpha"].value:
                        self.parameters["alpha"].minimum = self.model.alpha
                if self.metrics["kss"] is None or kss < self.metrics["kss"]:
                    self.parameters["alpha"].value = self.model.alpha
                    self.metrics["chi"] = chi
                    self.metrics["kss"] = kss
                if accuracy != -1.0 and (self.parameters["alpha"].maximum - self.parameters["alpha"].minimum) < accuracy:
                    break
                iteration += 1
        except KeyboardInterrupt:
            logger.info(f"Interrupted Training during iteration {iteration}")
        end_time = time.time()
        logger.critical(f"Total Training time: {end_time-start_time}s")
        logger.critical(f"Best results with: alpha = {self.parameters["alpha"].value} - Chi-Squared Distance: {self.metrics["chi"]} - KSS: {kss}")

    def apply(self):
        self.model.alpha = self.parameters["alpha"].value
        self.model.recreate_matrix()