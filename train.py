#!/bin/python
from pathlib import Path

import click

from gravity_model.log import logger
from gravity_model.trip import TripContainer
from gravity_model.location import LocationContainer
from gravity_model.gravity import GravityModel, PowerGravityModel, DoublePowerGravityModel, TriplePowerGravityModel, ModelType


@click.command()
@click.argument("location_data", metavar="[Location Data]", type=click.Path(exists=True, readable=True, dir_okay=False, path_type=Path))
@click.argument("model_output", metavar="[Model Output]", type=click.Path(readable=True, dir_okay=False, path_type=Path))
@click.argument("model_type", metavar="[Model Type]", type=click.Choice(ModelType, case_sensitive=False))
@click.option("-i", "--iterations", type=int, default=100)
@click.option("--optimize", type=click.Path(exists=True, readable=True, dir_okay=False, path_type=Path))
@click.option("-d", "--default-parameter", type=(str, float), multiple=True)
def main(location_data: Path, model_output: Path, model_type: ModelType, optimize: Path, iterations: int,  default_parameter: list[tuple[str, float]]):
    logger.info(f"Loading location data from {location_data.absolute().as_posix()}")
    locs = LocationContainer.from_csv(location_data)
    default_parameter = dict(default_parameter)

    logger.info(f"Instantiating model {model_type}")
    model = None
    if model_type is ModelType.BASIC:
        model = GravityModel(locs)
    elif model_type is ModelType.POWER:
        model = PowerGravityModel(locs, default_parameter.get("alpha", 1.0))
    elif model_type is ModelType.DOUBLEPOWER:
        model = DoublePowerGravityModel(locs, default_parameter.get("alpha", 1.0), default_parameter.get("beta", 1.0))
    elif model_type is ModelType.TRIPLEPOWER:
        model = TriplePowerGravityModel(locs, default_parameter.get("alpha", 1.0), (default_parameter.get("beta", 1.0), default_parameter.get("beta", 1.0)))


    if model and optimize:
        logger.info(f"Starting Training...")
        logger.info(f"Loading desired output data from {optimize.absolute().as_posix()}")
        target_trips = TripContainer.from_csv(optimize)
        if model_type is ModelType.BASIC:
            model.train(target_trips, 1)
        if model_type is ModelType.POWER:
            model.train(desired=target_trips, iterations=iterations, accuracy=0.01, parameters={"alpha": (0.1, 5.0, 1.0)})
        if model_type is ModelType.DOUBLEPOWER:
            model.train(desired=target_trips, iterations=iterations, accuracy=0.01, parameters={"alpha": (0.1, 0.7, 0.2), "beta": (0.1, 0.7, 0.2)})
        if model_type is ModelType.TRIPLEPOWER:
            model.train(desired=target_trips, iterations=iterations, accuracy=0.01, parameters={"alpha": (0.1, 1.0, 0.2), "beta_1": (0.1, 1.0, 0.2), "beta_2": (0.1, 1.0, 0.2)})
    
    logger.info(f"Storing model at {model_output.absolute().as_posix()}")
    model.to_json(model_output)

if __name__ == "__main__":
    main()