#!/bin/python
from pathlib import Path

import click

from gravity_model.random_search import POWER_LAW_TUPLE, EXPONENTIAL_TUPLE, POWER_LAW_DIST_TUPLE, POWER_LAW_POP_TUPLE
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


@click.command()
@click.argument("location_data", metavar="[Location Data]", type=click.Path(exists=True, readable=True, dir_okay=False, path_type=Path))
@click.argument("model_output", metavar="[Model Output]", type=click.Path(readable=True, dir_okay=False, path_type=Path))
@click.argument("model_type", metavar="[Model Type]", type=click.Choice(ModelType, case_sensitive=False))
@click.option("-i", "--iterations", type=int, default=100)
@click.option("-m", "--metric", type=str, default="chi")
@click.option("--optimize", type=click.Path(exists=True, readable=True, dir_okay=False, path_type=Path))
@click.option("--default-parameters", type=(str, float), multiple=True)
def main(location_data: Path, model_output: Path, model_type: ModelType, optimize: Path, iterations: int, metric: str, default_parameters: dict[str, float]):
    logger.info(f"Loading location data from {location_data.absolute().as_posix()}")
    locs = LocationContainer.from_csv(location_data)
    default_parameters = dict(default_parameters)

    logger.info(f"Instantiating model {model_type}")
    model = None
    if model_type is ModelType.BASIC:
        model = GravityModel(locs)
    elif model_type is ModelType.POWER:
        model = PowerGravityModel(locs, default_parameters.get("alpha", 1.0))
    elif model_type is ModelType.DOUBLEPOWER:
        model = DoublePowerGravityModel(locs, default_parameters.get("alpha", 1.0), default_parameters.get("beta", 1.0))
    elif model_type is ModelType.TRIPLEPOWER:
        model = TriplePowerGravityModel(locs, default_parameters.get("alpha", 1.0), default_parameters.get("beta", 1.0), default_parameters.get("gamma", 1.0))
    elif model_type is ModelType.EXPO:
        model = ExponentialGravityModel(locs, default_parameters.get("alpha", 0.01))
    elif model_type is ModelType.DOUBLEEXPO:
        model = DoubleExponentialGravityModel(locs, default_parameters.get("alpha", 0.01), default_parameters.get("beta", 0.01))
    elif model_type is ModelType.TRIPLEEXPO:
        model = TripleExponentialGravityModel(locs, default_parameters.get("alpha", 0.01), default_parameters.get("beta", 0.01), default_parameters.get("gamma", 0.01))
    elif model_type is ModelType.EXPOWER:
        model = ExponentialPowerGravityModel(locs, default_parameters.get("alpha", 1.0), default_parameters.get("beta", 0.01))


    if not model:
        logger.critical(f"Failed to instantiate a model of type {model_type}")

    if model and optimize:
        logger.info(f"Starting Training...")
        logger.info(f"Loading desired output data from {optimize.absolute().as_posix()}")
        target_trips = TripContainer.from_csv(optimize)
        if model_type is ModelType.BASIC:
            model.train(target_trips)
        elif model_type is ModelType.POWER:
            model.train(desired=target_trips, iterations=iterations, accuracy=0.0005, metric=metric, parameters={"alpha": POWER_LAW_DIST_TUPLE})
        elif model_type is ModelType.DOUBLEPOWER:
            model.train(desired=target_trips, iterations=iterations, accuracy=0.0005, metric=metric, parameters={"alpha": POWER_LAW_DIST_TUPLE, "beta": POWER_LAW_POP_TUPLE})
        elif model_type is ModelType.TRIPLEPOWER:
            model.train(desired=target_trips, iterations=iterations, accuracy=0.0005, metric=metric, parameters={"alpha": POWER_LAW_DIST_TUPLE, "beta": POWER_LAW_POP_TUPLE, "gamma": POWER_LAW_POP_TUPLE})
        elif model_type is ModelType.EXPO:
            model.train(desired=target_trips, iterations=iterations, accuracy=0.00005, metric=metric, parameters={"alpha": EXPONENTIAL_TUPLE})
        elif model_type is ModelType.DOUBLEEXPO:
            model.train(desired=target_trips, iterations=iterations, accuracy=0.00005, metric=metric, parameters={"alpha": EXPONENTIAL_TUPLE, "beta": EXPONENTIAL_TUPLE})
        elif model_type is ModelType.TRIPLEEXPO:
            model.train(desired=target_trips, iterations=iterations, accuracy=0.00005, metric=metric, parameters={"alpha": EXPONENTIAL_TUPLE, "beta": EXPONENTIAL_TUPLE, "gamma": EXPONENTIAL_TUPLE})
        elif model_type is ModelType.EXPOWER:
            model.train(desired=target_trips, iterations=iterations, accuracy=0.0005, metric=metric, parameters={"alpha": POWER_LAW_DIST_TUPLE, "beta": EXPONENTIAL_TUPLE})

    logger.info(f"Storing model at {model_output.absolute().as_posix()}")
    model.to_json(model_output)

if __name__ == "__main__":
    main()