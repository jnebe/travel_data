import time
import itertools

from ..training import Parameter, chi_square_distance, get_histogram, kolmogorov_smirnov_statistic, get_ccdf
from ..trip import TripContainer
from ..log import logger

from . import DEFAULT_TRAINING_TRIPS

class NelderMeadSearch():

    REFLECTION_COEFFICIENT = 1.0
    EXPANSION_COEFFICIENT = 2.0
    CONTRACTION_COEFFICIENT = 0.5
    SHRINKAGE_COEFFICIENT = 0.5

    NEADER_MELD_STEPSIZE = 0.66

    GENERATION_SIZE = 10
    SHRINKAGE_REQUIREED = 5

    def __init__(self, model, desired: TripContainer, parameters: dict[str, tuple[float, float, float]]):
        self.model = model
        self.real_data = desired
        self.parameters: dict[str, Parameter] = {}
        self.metric: float = None
        for name, value in parameters.items():
            self.parameters[name] = Parameter(name, value[2], value[0], value[1])

    def initialize_default_simplex(self):
        simplex = []
        dimensions = len(self.parameters) + 1
        if dimensions < 2:
            raise ValueError("Nelder-Mead requires at least 1 dimension for the simplex.")
        # Start with a vertex at the minimum of all parameters
        simplex.append({name: param.maximum for name, param in self.parameters.items()})
        simplex.append({name: param.minimum for name, param in self.parameters.items()})
        for outer_vertex in self.parameters.keys():
            if len(simplex) >= dimensions:
                break
            vertex = {}
            for name, param in self.parameters.items():
                if name == outer_vertex:
                    vertex[name] = param.minimum
                else:
                    vertex[name] = param.maximum
            simplex.append(vertex)
        return simplex

    def initialize_user_simplex(self, user_guess: dict[str, float], step_size: float = NEADER_MELD_STEPSIZE, backslide_ratio: float = 0.33):
        simplex = []
        dimensions = len(self.parameters) + 1
        if dimensions < 2:
            raise ValueError("Nelder-Mead requires at least 1 dimension for the simplex.")
        for name in self.parameters.keys():
            if name not in user_guess:
                raise ValueError(f"User guess must include parameter: {name}")
        # Start with a vertex at the user guess
        # slide back user guess by a ratio of the step size
        for name, param in self.parameters.items():
            if abs(user_guess[name] - self.parameters[name].minimum) < abs(user_guess[name] - self.parameters[name].maximum):
                param_vector = (param.maximum - param.minimum)
            else:
                param_vector = (param.minimum - param.maximum)
            logger.info(f"User guess {name} = {user_guess[name]} | param_vector = {param_vector}")
            logger.info(f"User guess {name} = {user_guess[name]} | step_size = {step_size} | backslide_ratio = {backslide_ratio}")
            user_guess[name] -= (param_vector * (step_size * backslide_ratio))
            logger.info(f"User guess {name} after backslide = {user_guess[name]}")
        user_guess = self.clamp_vertex(
            user_guess
        )
        simplex.append(user_guess)
        print(f"User guess: {user_guess}")

        # Add vertices around the user guess
        for name, param in self.parameters.items():
            new_vertex = user_guess.copy()

            if abs(user_guess[name] - self.parameters[name].minimum) < abs(user_guess[name] - self.parameters[name].maximum):
                param_vector = (param.maximum - param.minimum)
            else:
                param_vector = (param.minimum - param.maximum)

            for new_vertex_keys in new_vertex.keys():
                if new_vertex_keys != name:
                    continue
                new_vertex[new_vertex_keys] += (param_vector * (step_size * (1 - backslide_ratio)))
            
            new_vertex = self.clamp_vertex(
                new_vertex
            )
            simplex.append(new_vertex)
        return simplex

    def initialize_simplex(self, user_guess: dict[str, float] = None, step_size: float = None):
        if user_guess is None:
            return self.initialize_default_simplex()
        if len(user_guess) != len(self.parameters):
            raise ValueError("User guess must match the number of parameters.")
        return self.initialize_user_simplex(user_guess) if step_size is None else self.initialize_user_simplex(user_guess, step_size)

    def evaluate_simplex(self, simplex):
        # Apply simplex parameters to the model
        for name, value in simplex.items():
            setattr(self.model, name, value)
        self.model.recreate_matrix()

        # Check simplex performance
        logger.info(f"Testing simplex: {simplex}")
        model_trips: TripContainer = self.model.make_trips(DEFAULT_TRAINING_TRIPS)
        chi = chi_square_distance(get_histogram(self.real_data), get_histogram(model_trips))
        kss = kolmogorov_smirnov_statistic(get_ccdf(self.real_data), get_ccdf(model_trips))
        return chi, kss
    
    def clamp_vertex(self, vertex):
        # Ensure all vertex values are within the parameter bounds
        return {name: max(min(value, param.maximum), param.minimum) for name, (value, param) in zip(self.parameters.keys(), zip(vertex.values(), self.parameters.values()))}

    def train(self, iterations: int = 100, accuracy: float = -1.0, metric: str = "chi"):
        start_time = time.time()
        dimensions = len(self.parameters)
        initial_simplex = self.initialize_simplex()
        logger.info(f"Initial simplex: {initial_simplex}")
        vertex_performance: list[tuple[float, dict[str, float]]] = []

        generation = 0
        current_gen_iteration = 0
        current_gen_shrinkage = 0
        best_vertex = { name: None for name in self.parameters.keys() }
        try:
            for iteration in range(iterations):
                if current_gen_iteration >= self.GENERATION_SIZE and current_gen_shrinkage >= self.SHRINKAGE_REQUIREED:
                    logger.info(f"Generation {generation} completed. Re-initializing simplex.")
                    generation += 1

                    best_vertex = vertex_performance[0][1]
                    initial_simplex = self.initialize_simplex(user_guess=best_vertex.copy(), step_size=self.NEADER_MELD_STEPSIZE - ((self.NEADER_MELD_STEPSIZE / 2) * (iteration / iterations)))
                    logger.info(f"Re-initializing simplex at iteration {iteration} with best vertex: {best_vertex}")
                    logger.info(f"Re-initialized simplex: {initial_simplex}")
                    vertex_performance = []
                    current_gen_iteration = 0
                    current_gen_shrinkage = 0

                # 1 Evaluate all vertices and sort by performance
                if len(vertex_performance) <= 0:
                    for vertex in initial_simplex:
                        chi, kss = self.evaluate_simplex(vertex)
                        if metric == "chi":
                            performance_metric = chi
                        elif metric == "kss":
                            performance_metric = kss
                        else:
                            raise ValueError(f"Unknown metric: {metric}")
                        vertex_performance.append((performance_metric, vertex))
                else:
                    current_gen_iteration += 1

                vertex_performance.sort(key=lambda x: x[0])
                logger.info(f"Iteration {iteration} | Best performance: {vertex_performance[0][0]} | {metric}")
                logger.info(f"Current generation iteration: {current_gen_iteration} | Current generation shrinkage: {current_gen_shrinkage}")

                # 1.2 Apply best vertex
                if self.metric is None or vertex_performance[0][0] < self.metric:
                    best_vertex = vertex_performance[0][1]
                    for name, value in best_vertex.items():
                        self.parameters[name].value = value
                    self.metric = vertex_performance[0][0]

                # 1.2 check termination condition
                if accuracy > 0 and self.metric < accuracy:
                    logger.info(f"Training completed with metric {self.metric} below accuracy threshold {accuracy}.")
                    break

                # 2. Calculate centroid of the simplex excluding the worst vertex
                centroid = {name: 0.0 for name in self.parameters.keys()}
                for vertex in initial_simplex[:-1]:
                    for name, value in vertex.items():
                        centroid[name] += value
                centroid = {name: value / dimensions for name, value in centroid.items()}
                logger.info(f"Centroid: {centroid}")

                # 2.1 Identify best, second worst, and worst vertices
                best_vertex = vertex_performance[0]
                second_worst_vertex = vertex_performance[-2]
                worst_vertex = vertex_performance[-1]
                logger.info(f"Best vertex: {best_vertex[1]} with performance {best_vertex[0]}")
                logger.info(f"Second worst vertex: {second_worst_vertex[1]} with performance {second_worst_vertex[0]}")
                logger.info(f"Worst vertex: {worst_vertex[1]} with performance {worst_vertex[0]}")

                # 3. Reflection
                reflection_vertex = self.clamp_vertex(
                    {name: centroid[name] + self.REFLECTION_COEFFICIENT * (centroid[name] - worst_vertex[1][name]) for name in self.parameters.keys()}
                )
                logger.info(f"Reflection vertex: {reflection_vertex}")
                reflection_chi, reflection_kss = self.evaluate_simplex(reflection_vertex)
                reflection_performance = reflection_chi if metric == "chi" else reflection_kss
                if best_vertex[0] <= reflection_performance < second_worst_vertex[0]:
                    logger.info(f"Accepting reflection vertex: {reflection_vertex} with performance {reflection_performance}")
                    vertex_performance[-1] = (reflection_performance, reflection_vertex)
                    continue

                # 4. Expansion
                if reflection_performance < best_vertex[0]:
                    expansion_vertex = self.clamp_vertex(
                        {name: centroid[name] + self.EXPANSION_COEFFICIENT * (reflection_vertex[name] - centroid[name]) for name in self.parameters.keys()}
                    )
                    logger.info(f"Expansion vertex: {expansion_vertex}")
                    expansion_chi, expansion_kss = self.evaluate_simplex(expansion_vertex)
                    expansion_performance = expansion_chi if metric == "chi" else expansion_kss
                    if expansion_performance < reflection_performance:
                        logger.info(f"Accepting expansion vertex: {expansion_vertex} with performance {expansion_performance}")
                        vertex_performance[-1] = (expansion_performance, expansion_vertex)
                        continue
                    else:
                        logger.info(f"Accepting reflection vertex: {reflection_vertex} with performance {reflection_performance}")
                        vertex_performance[-1] = (reflection_performance, reflection_vertex)
                        continue

                # 5. Contraction
                if reflection_performance < worst_vertex[0]:
                    contraction_vertex = self.clamp_vertex(
                        {name: centroid[name] + self.CONTRACTION_COEFFICIENT * (reflection_vertex[name] - centroid[name]) for name in self.parameters.keys()}
                    )
                elif reflection_performance >= worst_vertex[0]:
                    contraction_vertex = self.clamp_vertex(
                        {name: centroid[name] + self.CONTRACTION_COEFFICIENT * (worst_vertex[1][name] - centroid[name]) for name in self.parameters.keys()}
                    )
                logger.info(f"Contraction vertex: {contraction_vertex}")
                contraction_chi, contraction_kss = self.evaluate_simplex(contraction_vertex)
                contraction_performance = contraction_chi if metric == "chi" else contraction_kss
                if contraction_performance < worst_vertex[0]:
                    logger.info(f"Accepting contraction vertex: {contraction_vertex} with performance {contraction_performance}")
                    vertex_performance[-1] = (contraction_performance, contraction_vertex)
                    continue

                # 6. Shrinkage
                logger.info("Shrinking simplex")
                for i in range(1, dimensions + 1):
                    shrinked_vertex = self.clamp_vertex(
                        {name: best_vertex[1][name] + self.SHRINKAGE_COEFFICIENT * (vertex_performance[i][1][name] - best_vertex[1][name]) for name in self.parameters.keys()}
                    )
                    logger.info(f"Shrinked vertex {i}: {shrinked_vertex}")
                    shrinked_chi, shrinked_kss = self.evaluate_simplex(shrinked_vertex)
                    shrinked_performance = shrinked_chi if metric == "chi" else shrinked_kss
                    vertex_performance[i] = (shrinked_performance, shrinked_vertex)

                current_gen_shrinkage += 1                
        except KeyboardInterrupt:
            logger.info(f"Interrupted Training during iteration {iteration}")
        end_time = time.time()
        logger.info(f"Total Training time: {end_time-start_time}s")
        logger.info(f"{metric}: {self.metric}")
        for name, param in self.parameters.items():
            logger.info(f"{name} = {param.value} [{param.minimum}, {param.maximum}]")
        

    def apply(self):
        for name, param in self.parameters.items():
            setattr(self.model, name, param.value)
        self.model.recreate_matrix()