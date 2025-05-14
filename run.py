#!/bin/python
from pathlib import Path

import click

from gravity_model.log import logger
from gravity_model.gravity import GravityModel

@click.command()
@click.argument("model_location", metavar="[Model]", type=click.Path(exists=True, readable=True, dir_okay=False, path_type=Path))
@click.argument("trip_output", metavar="[Trip Output]", type=click.Path(readable=True, dir_okay=False, path_type=Path))
@click.argument("number", metavar="[Number Runs]", type=int)
def main(model_location: Path, trip_output: Path, number: int):
    logger.info(f"Loading model from {model_location.absolute().as_posix()}")
    model = GravityModel.from_json(model_location)

    logger.info(f"Executing {number} trips...")
    trips = model.make_trips(number)
    
    logger.info(f"Saving results to {trip_output.absolute().as_posix()}")
    trips.to_csv(trip_output)

if __name__ == "__main__":
    main()