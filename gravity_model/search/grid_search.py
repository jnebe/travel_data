import time
import itertools

from ..training import chi_square_distance, get_histogram, kolmogorov_smirnov_statistic, get_ccdf
from ..trip import TripContainer
from .generic import GenericSearch
from ..log import logger

from . import DEFAULT_TRAINING_TRIPS

class GridSearch(GenericSearch):

    def train(self, iterations: int = 100, accuracy: float = -1.0, metric: str = "chi"):
        start_time = time.time()
        num_parameters = len(self.parameters)
        num_steps = int(iterations ** (1 / num_parameters))
        iterations = num_steps ** num_parameters
        iteration = 0
        try:
            for steps in itertools.product(range(num_steps), repeat=num_parameters):
                current_params = {}
                for param, step in zip(self.parameters.values(), steps, strict=True):
                    current_params[param.name] = param.get_step(num_steps, step)
                    setattr(self.model, param.name, param.get_step(num_steps, step))

                self.model.recreate_matrix()
                model_trips: TripContainer = self.model.make_trips(DEFAULT_TRAINING_TRIPS)
                chi = chi_square_distance(get_histogram(self.real_data), get_histogram(model_trips))
                kss = kolmogorov_smirnov_statistic(get_ccdf(self.real_data), get_ccdf(model_trips))
                self.add_parameter_map_point(current_params, {"chi" : chi, "kss" : kss})
                current_metrics = { "chi" : chi, "kss" : kss }
                logger.info(f"Iteration {iteration} | {steps} of {num_steps} - Chi-Squared Distance: {chi} - KSS: {kss}")
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