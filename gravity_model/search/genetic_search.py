from pathlib import Path
import time
import random

from ..training import chi_square_distance, get_histogram, kolmogorov_smirnov_statistic, get_ccdf
from ..trip import TripContainer
from .generic import GenericSearch
from ..log import logger

from . import DEFAULT_TRAINING_TRIPS

class GeneticSearch(GenericSearch):
    def __init__(self, model, desired: TripContainer, parameters: dict[str, tuple[float, float, float]], csv_path: Path | None = None, population_size=20, mutation_rate=0.2):
        super().__init__(model=model, desired=desired, parameters=parameters, csv_path=csv_path)
        
        self.fitness = None
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
        if metric == "chi":
            fitness = chi
        elif metric == "kss":
            fitness = kss
        self.add_parameter_map_point(individual, {"chi" : chi, "kss" : kss})
        return fitness, (chi, kss)
    
    def population_diversity(self, population: list[dict[str, float]]) -> float:
        if len(population) < 2:
            return 0

        total_distance = 0
        count = 0

        for name, param in self.parameters.items():
            count += 1
            maximum_pop = -1
            minimum_pop = -1
            for individual in population:
                if maximum_pop == -1 or individual[name] > maximum_pop:
                    maximum_pop = individual[name]
                if minimum_pop == -1 or individual[name] < minimum_pop:
                    minimum_pop = individual[name]
            parameter_distance = (maximum_pop - minimum_pop) / (param.maximum - param.minimum)
            if parameter_distance > 1:
                raise RuntimeError(f"Parameter {name} has a distance of {parameter_distance} which is greater than 1.")
            total_distance += parameter_distance
        if total_distance > len(self.parameters):
            raise RuntimeError(f"Total distance {total_distance} exceeds number of parameters {len(self.parameters)}")
        return total_distance / count
    
    def calculate_fitness(self, population: list[dict[str, float]], metric: str = "chi") -> list[tuple[float, dict[str, float]]]:
        fitness_scores = []
        for individual in population:
            fitness, metrics = self.evaluate_fitness(individual, metric=metric)
            logger.info(f"Individual {len(fitness_scores) + 1} of {len(population)} | {self.population_size} {individual}  - Fitness: {fitness} - Metrics (chi,  KSS): {metrics}")
            fitness_scores.append((fitness, individual))
        return fitness_scores
    
    def tournament_select(self, parents: tuple[float, dict[str, float]], k=2, p=0.7):        
        # Calculate geometric weights: p * (1 - p)^i
        weights = [p * ((1 - p) ** i) for i in range(k)]
        
        # Normalize weights to ensure they sum to 1
        total = sum(weights)
        normalized_weights = [w / total for w in weights]
        
        # Select one individual based on the weights
        selected = random.choices(parents[:k], weights=normalized_weights, k=1)[0]
        
        return selected[1]

    def crossover(self, p1, p2):
        child = {}
        for key in p1:
            alpha = random.uniform(0.3, 0.7)  # Bias toward middle, but still random
            value = p1[key] * alpha + p2[key] * (1 - alpha)
            parameter = self.parameters[key]
            low, high = parameter.minimum, parameter.maximum
            child[key] = max(low, min(high, value))  # Clamp to valid range
        return child

    def mutate(self, individual, mutation_rate: 0.2):
        for metric, parameter in self.parameters.items():
            roll = random.random()
            if roll < mutation_rate:
                individual[metric] = random.uniform(parameter.minimum, parameter.maximum)
            elif roll < mutation_rate * 3:
                jitter_range = (parameter.maximum - parameter.minimum) * 0.2
                individual[metric] = random.uniform(max(parameter.minimum, individual[metric] - jitter_range), min(parameter.maximum, individual[metric] + jitter_range))
    
    def make_children(self, fitness_scores: list[tuple[float, dict[str, float]]], elitism: int, diversity: float) -> list[dict[str, float]]:
        population = []
        for homeowner in random.sample(fitness_scores[0:-1], len(fitness_scores) - (elitism + 1)): # Skip the worst individual and sample the rest (population size - elitism) from all remaining individuals)
            # Restricted Tournament Selection
            neighborhood = []
            for possible_neighbor in fitness_scores[:-1]: # Skip the last individual (worst) for neighborhood selection
                if possible_neighbor[1] != homeowner[1]:
                    neighborhood.append((self.population_diversity([homeowner[1], possible_neighbor[1]]), possible_neighbor[1]))
            neighborhood.sort(key=lambda x: x[0])
            parent1 = self.tournament_select(neighborhood, max(2, self.population_size//5), p=0.95)
            parent2 = self.tournament_select(neighborhood, max(2, self.population_size//5), p=0.95)
            child = self.crossover(parent1, parent2)
            if diversity < 0.4:
                self.mutate(child, self.mutation_rate * 2)
            else:
                self.mutate(child, self.mutation_rate)
            population.append(child)
        # Add the one new individual, because we skipped the last one when making children
        population.append(self.generate_individual())
        return population

    def train(self, iterations: int = 100, accuracy: float = -1.0, metric: str = "chi"):
        start_time = time.time()
        elitism = max(2, int(self.population_size/10))
        num_generations = max(1, iterations // self.population_size)
        generation = 0
        try:
            fitness_scores = []
            population = [self.generate_individual() for _ in range(self.population_size)]
            for generation in range(num_generations):
                logger.info(f"Generation {generation + 1} of {num_generations} - Population size: {len(population)} Best from last Generation: {elitism if generation > 0 else 0} individuals")
                if  len(fitness_scores) < elitism:
                    fitness_scores = []
                else:
                    # Keep the best individuals from the previous generation
                    fitness_scores = fitness_scores[:elitism]
                    if len(fitness_scores) != elitism:
                        raise RuntimeError(f"Elitism size mismatch: {len(fitness_scores)} != {elitism}")
                fitness_scores.extend(self.calculate_fitness(population, metric=metric))
                if len(fitness_scores) != self.population_size:
                    raise RuntimeError(f"Population size mismatch: {len(fitness_scores)} != {self.population_size}")
                fitness_scores.sort(key=lambda x: x[0])

                diversity = self.population_diversity(population)
                logger.info(f"Generation {generation + 1} - Best fitness: {fitness_scores[0][0]} - Diversity: {diversity}")

                if generation < num_generations - 1:
                    population = self.make_children(fitness_scores, elitism, diversity)
                
                if len(population) != self.population_size - elitism:
                    raise RuntimeError(f"Population size mismatch after generation {generation + 1}: {len(population)} != {self.population_size - elitism} (- elitism {elitism})")

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