#!/bin/python
from pathlib import Path

import click

from gravity_model.search import POWER_LAW_TUPLE, EXPONENTIAL_TUPLE, POWER_LAW_DIST_TUPLE, POWER_LAW_POP_TUPLE, DISTANCE_SPLIT_TUPLE
from gravity_model.log import logger
from gravity_model.trip import TripContainer
from gravity_model.location import LocationContainer
from gravity_model.models import ModelType
from gravity_model.models.basic import GravityModel
from gravity_model.models.power import PowerGravityModel
from gravity_model.models.doublepower import DoublePowerGravityModel
from gravity_model.models.triplepower import TriplePowerGravityModel
from gravity_model.models.expo import ExponentialGravityModel
from gravity_model.models.doubleexpo import DoubleExponentialGravityModel
from gravity_model.models.tripleexpo import TripleExponentialGravityModel
from gravity_model.models.expower import ExponentialPowerGravityModel
from gravity_model.models.split import SplitGravityModel

from gravity_model.search import SearchType


@click.command()
@click.argument("location_data", metavar="[Location Data]", type=click.Path(exists=True, readable=True, dir_okay=False, path_type=Path))
@click.argument("model_output", metavar="[Model Output]", type=click.Path(readable=True, dir_okay=False, path_type=Path))
@click.argument("model_type", metavar="[Model Type]", type=click.Choice(ModelType, case_sensitive=False))
@click.option("-i", "--iterations", type=int, default=100)
@click.option("-m", "--metric", type=str, default="chi")
@click.option("-s", "--search-type", type=click.Choice(SearchType, case_sensitive=False), default=SearchType.RANDOM)
@click.option("--optimize", type=click.Path(exists=True, readable=True, dir_okay=False, path_type=Path))
@click.option("--default-parameter", type=(str, float), multiple=True)
@click.option("--training-parameter", type=(str, float, float, float), multiple=True)
@click.option("--metric-map", type=click.Path(exists=False, dir_okay=False, path_type=Path))
def main(location_data: Path, model_output: Path, model_type: ModelType, search_type: SearchType, optimize: Path, iterations: int, metric: str, default_parameter: list[tuple[str, float]], training_parameter: list[tuple[str, float, float, float]], metric_map: Path):
    logger.info(f"Loading location data from {location_data.absolute().as_posix()}")
    locs = LocationContainer.from_csv(location_data)
    default_parameter = {element[0]: element[1] for element in default_parameter}
    training_parameter = {element[0]: tuple(element[1:]) for element in training_parameter}

    logger.info(f"Instantiating model {model_type}")
    model = None
    if model_type is ModelType.BASIC:
        model = GravityModel(locs)
    elif model_type is ModelType.POWER:
        model = PowerGravityModel(locs, default_parameter.get("alpha", 1.0))
    elif model_type is ModelType.DOUBLEPOWER:
        model = DoublePowerGravityModel(locs, default_parameter.get("alpha", 1.0), default_parameter.get("beta", 1.0))
    elif model_type is ModelType.TRIPLEPOWER:
        model = TriplePowerGravityModel(locs, default_parameter.get("alpha", 1.0), default_parameter.get("beta", 1.0), default_parameter.get("gamma", 1.0))
    elif model_type is ModelType.EXPO:
        model = ExponentialGravityModel(locs, default_parameter.get("alpha", 0.01))
    elif model_type is ModelType.DOUBLEEXPO:
        model = DoubleExponentialGravityModel(locs, default_parameter.get("alpha", 0.01), default_parameter.get("beta", 0.01))
    elif model_type is ModelType.TRIPLEEXPO:
        model = TripleExponentialGravityModel(locs, default_parameter.get("alpha", 0.01), default_parameter.get("beta", 0.01), default_parameter.get("gamma", 0.01))
    elif model_type is ModelType.EXPOWER:
        model = ExponentialPowerGravityModel(locs, default_parameter.get("alpha", 1.0), default_parameter.get("beta", 0.01))
    elif model_type is ModelType.SPLIT:
        model = SplitGravityModel(locs, default_parameter.get("alpha", 1.0), default_parameter.get("beta", 1.0), default_parameter.get("gamma", 600))


    if not model:
        logger.critical(f"Failed to instantiate a model of type {model_type}")

    if model and optimize:
        logger.info(f"Starting Training...")
        logger.info(f"Loading desired output data from {optimize.absolute().as_posix()}")
        target_trips = TripContainer.from_csv(optimize)
        accuracy=0.0005
        if model_type is ModelType.BASIC:
            logger.warning("Training a basic model does not require optimization, but we will still calculate the error metrics!")
            model.train(desired=target_trips, iterations=1)
        elif model_type is ModelType.POWER:
            accuracy=0.0005
            parameters={"alpha": POWER_LAW_DIST_TUPLE}
        elif model_type is ModelType.DOUBLEPOWER:
            accuracy=0.0005
            parameters={"alpha": POWER_LAW_DIST_TUPLE, "beta": POWER_LAW_POP_TUPLE}
        elif model_type is ModelType.TRIPLEPOWER:
            accuracy=0.0005
            parameters={"alpha": POWER_LAW_DIST_TUPLE, "beta": POWER_LAW_POP_TUPLE, "gamma": POWER_LAW_POP_TUPLE}
        elif model_type is ModelType.EXPO:
            accuracy=0.00005
            parameters={"alpha": EXPONENTIAL_TUPLE}
        elif model_type is ModelType.DOUBLEEXPO:
            accuracy=0.00005
            parameters={"alpha": EXPONENTIAL_TUPLE, "beta": EXPONENTIAL_TUPLE}
        elif model_type is ModelType.TRIPLEEXPO:
            accuracy=0.00005
            parameters={"alpha": EXPONENTIAL_TUPLE, "beta": EXPONENTIAL_TUPLE, "gamma": EXPONENTIAL_TUPLE}
        elif model_type is ModelType.EXPOWER:
            accuracy=0.0005
            parameters={"alpha": POWER_LAW_DIST_TUPLE, "beta": EXPONENTIAL_TUPLE}
        elif model_type is ModelType.SPLIT:
            accuracy=0.0005
            parameters={"alpha": POWER_LAW_DIST_TUPLE, "beta": POWER_LAW_DIST_TUPLE, "gamma": DISTANCE_SPLIT_TUPLE}
        parameters.update(training_parameter)
        model.train(desired=target_trips, iterations=iterations, accuracy=accuracy, metric=metric, parameters=parameters, search_type=search_type, metric_map=metric_map)
    logger.info(f"Storing model at {model_output.absolute().as_posix()}")
    model.to_json(model_output)

if __name__ == "__main__":
    main()