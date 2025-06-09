import time
import random

from .training import Parameter, chi_square_distance
from .trip import TripContainer
from .log import logger

class PowerRandomSearch():

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
        
        while (iteration < iterations and iterations != -1):
            self.model.alpha = random.uniform(self.parameters["alpha"].minimum, self.parameters["alpha"].maximum)
            self.model.recreate_matrix()
            model_trips: TripContainer = self.model.make_trips(250000)
            chi = chi_square_distance(self.real_data.get_histogram(), model_trips.get_histogram())
            logger.info(f"Iteration {iteration} - alpha {self.model.alpha} [{self.parameters["alpha"].minimum}, {self.parameters["alpha"].maximum}] - Chi-Squared Distance: {chi}")
            if (self.metrics["chi"] is not None) and chi > self.metrics["chi"]:
                if self.model.alpha > self.parameters["alpha"].value:
                    self.parameters["alpha"].maximum = self.model.alpha
                if self.model.alpha < self.parameters["alpha"].value:
                    self.parameters["alpha"].minimum = self.model.alpha
            if self.metrics["chi"] is None or  chi < self.metrics["chi"]:
                self.parameters["alpha"].value = self.model.alpha
                self.metrics["chi"] = chi
            if accuracy != -1.0 and (self.parameters["alpha"].maximum - self.parameters["alpha"].minimum) < accuracy:
                break
            iteration += 1
        end_time = time.time()
        logger.critical(f"Total Training time: {end_time-start_time}s")
        logger.critical(f"Best results with: alpha = {self.parameters["alpha"].value} - Chi-Squared Distance: {self.metrics["chi"]}")

    def apply(self):
        self.model.alpha = self.parameters["alpha"].value
        self.model.recreate_matrix()

class DoublePowerRandomSearch():

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
            self.model.recreate_matrix()
            model_trips = self.model.make_trips(250000)
            chi = chi_square_distance(self.real_data.get_histogram(), model_trips.get_histogram())
            logger.info(f"Iteration {iteration} - alpha {self.model.alpha} [{self.parameters["alpha"].minimum}, {self.parameters["alpha"].maximum}] - beta {self.model.beta} [{self.parameters["beta"].minimum}, {self.parameters["beta"].maximum}] - Chi-Squared Distance: {chi}")
            # if iteration > 5 and (not alpha_fixed) and (self.metrics["chi"] is not None) and chi > self.metrics["chi"]:
            #     if self.model.alpha > self.parameters["alpha"].value:
            #         self.parameters["alpha"].maximum = self.model.alpha
            #     if self.model.alpha < self.parameters["alpha"].value:
            #         self.parameters["alpha"].minimum = self.model.alpha
            # if alpha_fixed and (self.metrics["chi"] is not None) and chi > self.metrics["chi"]:
            #     if self.model.beta > self.parameters["beta"].value:
            #         self.parameters["beta"].maximum = self.model.beta
            #     if self.model.beta < self.parameters["beta"].value:
            #         self.parameters["beta"].minimum = self.model.beta
            # TODO replace with ~rougher parameter space reduction -> map direction? and decrease slowly...
            if self.metrics["chi"] is None or  chi < self.metrics["chi"]:
                self.parameters["alpha"].value = self.model.alpha
                self.parameters["beta"].value = self.model.beta
                self.metrics["chi"] = chi
            if (self.parameters["alpha"].maximum - self.parameters["alpha"].minimum) < accuracy:
                alpha_fixed = True
            if (self.parameters["beta"].maximum - self.parameters["beta"].minimum) < accuracy:
                beta_fixed = True
            if accuracy != -1.0 and alpha_fixed and beta_fixed:
                break
            iteration += 1
        end_time = time.time()
        logger.critical(f"Total Training time: {end_time-start_time}s")
        logger.critical(f"Best results with: alpha, beta = {self.parameters["alpha"].value}, {self.parameters["beta"].value} - Chi-Squared Distance: {self.metrics["chi"]}")

    def apply(self):
        self.model.alpha = self.parameters["alpha"].value
        self.model.recreate_matrix()
        
        