import time
import random

from ..training import chi_square_distance, get_histogram, kolmogorov_smirnov_statistic, get_ccdf
from ..trip import TripContainer
from .generic import GenericSearch
from ..log import logger

from . import DEFAULT_TRAINING_TRIPS

class RandomSearch(GenericSearch):

    def train(self, iterations: int = 100, accuracy: float = -1.0, metric: str = "chi"):
        start_time = time.time()
        iteration = 0
        try:
            while (iteration < iterations and iterations != -1):
                current_params = {}
                for name, param in self.parameters.items():
                    value = random.uniform(param.minimum, param.maximum)
                    current_params[name] = value
                    setattr(self.model, name, value)

                self.model.recreate_matrix()
                model_trips: TripContainer = self.model.make_trips(DEFAULT_TRAINING_TRIPS)
                chi = chi_square_distance(get_histogram(self.real_data), get_histogram(model_trips))
                kss = kolmogorov_smirnov_statistic(get_ccdf(self.real_data), get_ccdf(model_trips))
                self.add_parameter_map_point(current_params, {"chi" : chi, "kss" : kss})
                current_metrics = { "chi" : chi, "kss" : kss }
                logger.info(f"Iteration {iteration} - Chi-Squared Distance: {chi} - KSS: {kss}")
                for name, param in self.parameters.items():
                    logger.info(f"{name} = {getattr(self.model, name)} [{param.minimum}, {param.maximum}]")

                if self.metrics[metric] is None or current_metrics[metric] < self.metrics[metric]:
                    for name, param in self.parameters.items():
                        param.value = getattr(self.model, name)
                    self.metrics["chi"] = chi
                    self.metrics["kss"] = kss

                iteration += 1
        except KeyboardInterrupt:
            logger.info(f"Interrupted Training during iteration {iteration}")
        end_time = time.time()
        logger.info(f"Total Training time: {end_time-start_time}s")
        logger.info(f"Chi-Squared Distance: {self.metrics['chi']} - KSS: {self.metrics['kss']}")
        for name, param in self.parameters.items():
            logger.info(f"{name} = {param.value} [{param.minimum}, {param.maximum}]")