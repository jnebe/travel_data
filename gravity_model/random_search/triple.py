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
        self.metrics: dict[str, float] = { "chi" : None, "kss" : None }
        for name, value in parameters.items():
            self.parameters[name] = Parameter(name, value[2], value[0], value[1])

    def train(self, iterations: int = 100, accuracy: float = -1.0, metric: str = "chi"):
        start_time = time.time()
        iteration = 0
        alpha_fixed = False
        beta_fixed = False
        gamma_fixed = False
        
        try:
            while (iteration < iterations and iterations != -1):
                self.model.alpha = random.uniform(self.parameters["alpha"].minimum, self.parameters["alpha"].maximum)
                self.model.beta = random.uniform(self.parameters["beta"].minimum, self.parameters["beta"].maximum)
                self.model.gamma = random.uniform(self.parameters["gamma"].minimum, self.parameters["gamma"].maximum)

                self.model.recreate_matrix()
                model_trips = self.model.make_trips(DEFAULT_TRAINING_TRIPS)
                chi = chi_square_distance(get_histogram(self.real_data), get_histogram(model_trips))
                kss = kolmogorov_smirnov_statistic(get_ccdf(self.real_data), get_ccdf(model_trips))
                current_metrics = { "chi" : chi, "kss" : kss }
                logger.info(f"Iteration {iteration} - alpha {self.model.alpha} [{self.parameters['alpha'].minimum}, {self.parameters['alpha'].maximum}] - beta home {self.model.beta} [{self.parameters['beta'].minimum}, {self.parameters['beta'].maximum}] - gamma dest {self.model.gamma} [{self.parameters['gamma'].minimum}, {self.parameters['gamma'].maximum}] - Chi-Squared Distance: {chi} - KSS: {kss}")
                if self.metrics[metric] is None or current_metrics[metric] < self.metrics[metric]:
                    self.parameters["alpha"].value = self.model.alpha
                    self.parameters["beta"].value = self.model.beta
                    self.parameters["gamma"].value = self.model.gamma
                    self.metrics["chi"] = chi
                    self.metrics["kss"] = kss
                if accuracy != -1.0 and alpha_fixed and beta_fixed and gamma_fixed:
                    break
                iteration += 1
        except KeyboardInterrupt:
            logger.info(f"Interrupted Training during iteration {iteration}")
        end_time = time.time()
        logger.critical(f"Total Training time: {end_time-start_time}s")
        logger.critical(f"Best results with: alpha, beta, gamma = {self.parameters['alpha'].value}, {self.parameters['beta'].value}, {self.parameters['gamma'].value} - Chi-Squared Distance: {self.metrics["chi"]} - KSS: {self.metrics["kss"]}")

    def apply(self):
        self.model.alpha = self.parameters["alpha"].value
        self.model.beta = self.parameters["beta"].value
        self.model.gamma = self.parameters["gamma"].value
        self.model.recreate_matrix()