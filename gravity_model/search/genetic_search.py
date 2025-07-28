import time
import random

from ..training import Parameter, chi_square_distance, get_histogram, kolmogorov_smirnov_statistic, get_ccdf
from ..trip import TripContainer
from ..log import logger

from . import DEFAULT_TRAINING_TRIPS

class GeneticSearch:
    def __init__(self, model, desired: TripContainer, parameters: dict[str, tuple[float, float, float]], population_size=20, mutation_rate=0.1):
        self.model = model
        self.real_data: TripContainer = desired
        self.parameters: dict[str, Parameter] = {}
        self.fitness = None
        for name, value in parameters.items():
            self.parameters[name] = Parameter(name, value[2], value[0], value[1])
        self.population_size = population_size
        self.mutation_rate = mutation_rate

    def generate_individual(self):
        return {
            name: random.uniform(p.minimum, p.maximum)
            for name, p in self.parameters.items()
        }

    def evaluate_fitness(self, individual, metric):
        for name, value in individual.items():
            setattr(self.model, name, value)
        self.model.recreate_matrix()
        trips = self.model.make_trips(min(self.real_data.df.height, DEFAULT_TRAINING_TRIPS))
        chi = chi_square_distance(get_histogram(self.real_data), get_histogram(trips))
        kss = kolmogorov_smirnov_statistic(get_ccdf(self.real_data), get_ccdf(trips))
        logger.info(f"Individual {individual} - Chi-Squared Distance: {chi} - KSS: {kss}")
        if metric == "chi":
            fitness = chi
        elif metric == "kss":
            fitness = kss
        return fitness
    
    def train(self, iterations: int = 100, accuracy: float = -1.0, metric: str = "chi"):
        start_time = time.time()
        num_generations = max(1, iterations // self.population_size)
        generation = 0

        try:
            population = [self.generate_individual() for _ in range(self.population_size)]
            for generation in range(num_generations):
                logger.info(f"Generation {generation + 1} of {num_generations} - Population size: {len(population)}")
                fitness_scores = [(self.evaluate_fitness(ind, metric=metric), ind) for ind in population]
                fitness_scores.sort(key=lambda x: x[0])
                logger.info(f"Generation {generation} - Best fitness: {fitness_scores[0][0]}")
                if generation < num_generations:
                    logger.info(f"Keeping the best {max(1, int(self.population_size/5))} individuals for the next generation")
                    new_population = fitness_scores[:max(1, int(self.population_size/5))]  # Elitism

                    while len(new_population) < self.population_size:
                        parent1 = self.tournament_select(fitness_scores)
                        parent2 = self.tournament_select(fitness_scores)
                        child = self.crossover(parent1, parent2)
                        self.mutate(child)
                        new_population.append((None, child))

                    population = [ind for _, ind in new_population]

                best = fitness_scores[0]
                if self.fitness is None or best[0] < self.fitness:
                    individual = best[1]
                    for name, param in self.parameters.items():
                        param.value = individual.get(name)
                    self.fitness = best[0]

        except KeyboardInterrupt:
            logger.info(f"Interrupted Training during generation {generation}")
        end_time = time.time()
        logger.info(f"Total Training time: {end_time-start_time}s")
        logger.info(f"Best Fitness ({metric}): {self.fitness}")
        for name, param in self.parameters.items():
            logger.info(f"{name} = {param.value} [{param.minimum}, {param.maximum}]")

    def tournament_select(self, scored_population, k=3):
        return min(random.sample(scored_population, k), key=lambda x: x[0])[1]

    def crossover(self, p1, p2):
        child = {}
        for key in p1:
            alpha = random.uniform(0.3, 0.7)  # Bias toward middle, but still random
            value = p1[key] * alpha + p2[key] * (1 - alpha)
            parameter = self.parameters[key]
            low, high = parameter.minimum, parameter.maximum
            child[key] = max(low, min(high, value))  # Clamp to valid range
        return child

    def mutate(self, individual):
        for metric, parameter in self.parameters.items():
            if random.random() < self.mutation_rate:
                individual[metric] = random.uniform(parameter.minimum, parameter.maximum)

    def apply(self):
        for name, param in self.parameters.items():
            setattr(self.model, name, param.value)
        self.model.recreate_matrix()