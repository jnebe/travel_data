#!/bin/python
from pathlib import Path

import click

from gravity_model.log import logger
from gravity_model.location import LocationContainer
from gravity_model.gravity import GravityModel, PowerGravityModel, ModelType

@click.command()
@click.argument("location_data", metavar="[Location Data]", type=click.Path(exists=True, readable=True, dir_okay=False, path_type=Path))
@click.argument("model_output", metavar="[Model Output]", type=click.Path(readable=True, dir_okay=False, path_type=Path))
@click.argument("model_type", metavar="[Model Type]", type=click.Choice(ModelType, case_sensitive=False))
@click.option("--optimize", is_flag=True)
@click.option("-d", "--default-parameter", type=(str, float), multiple=True)
def main(location_data: Path, model_output: Path, model_type: ModelType, optimize: bool, default_parameter: dict[str, float]):
    logger.info(f"Loading location data from {location_data.absolute().as_posix()}")
    locs = LocationContainer.from_csv(location_data)

    if optimize:
        pass
        # TODO Do model training/hyperparameter training
    else:
        if model_type is ModelType.BASIC:
            model = GravityModel(locs)
        elif model_type is ModelType.POWER:
            model = PowerGravityModel(locs, default_parameter.get("alpha", 1.0))
    
    logger.info(f"Storing model at {model_output.absolute().as_posix()}")
    model.to_json(model_output)

if __name__ == "__main__":
    main()