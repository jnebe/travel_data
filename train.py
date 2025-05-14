#!/bin/python
from pathlib import Path

import click

from gravity_model.log import logger
from gravity_model.trip import TripContainer
from gravity_model.location import LocationContainer
from gravity_model.gravity import GravityModel, PowerGravityModel, ModelType
from gravity_model.training import total_variation_distance

@click.command()
@click.argument("location_data", metavar="[Location Data]", type=click.Path(exists=True, readable=True, dir_okay=False, path_type=Path))
@click.argument("model_output", metavar="[Model Output]", type=click.Path(readable=True, dir_okay=False, path_type=Path))
@click.argument("model_type", metavar="[Model Type]", type=click.Choice(ModelType, case_sensitive=False))
@click.option("--optimize", type=click.Path(exists=True, readable=True, dir_okay=False, path_type=Path))
@click.option("-d", "--default-parameter", type=(str, float), multiple=True)
def main(location_data: Path, model_output: Path, model_type: ModelType, optimize: Path, default_parameter: dict[str, float]):
    logger.info(f"Loading location data from {location_data.absolute().as_posix()}")
    locs = LocationContainer.from_csv(location_data)

    model = None
    if model_type is ModelType.BASIC:
        model = GravityModel(locs)
    elif model_type is ModelType.POWER:
        model = PowerGravityModel(locs, default_parameter.get("alpha", 1.0))

    if model and optimize:
        logger.info(f"Starting Training...")
        logger.info(f"Loading desired output data from {optimize.absolute().as_posix()}")
        target_trips = TripContainer.from_csv(optimize)
        model_trips = model.make_trips(len(target_trips))
        tvd = total_variation_distance(target_trips, model_trips)
        logger.critical(f"TVD: {tvd}")
        # TODO Do model training/hyperparameter training
        # Loop over parameter space and build models, gen trips, change parameters ....
    
    logger.info(f"Storing model at {model_output.absolute().as_posix()}")
    model.to_json(model_output)

if __name__ == "__main__":
    main()