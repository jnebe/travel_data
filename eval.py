#!/bin/python
from pathlib import Path

import click

from gravity_model.log import logger
from gravity_model.trip import TripContainer
from gravity_model.visualize import visualize, vis_types

@click.command()
@click.argument("trip_location", metavar="[Trip Data]", type=click.Path(exists=True, readable=True, dir_okay=False, path_type=Path))
@click.argument("results_output", metavar="[Evaluation Output]", type=click.Path(readable=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("prefix", metavar="[Prefix]", type=str)
def main(trip_location: Path, results_output: Path, prefix: str):
    logger.info(f"Loading trip data from {trip_location.absolute().as_posix()}")
    trips = TripContainer.from_csv(trip_location)

    logger.info(f"Visualizing trips...")
    for type in vis_types:
        logger.info(f"Generating {type} plot")
        visualize(type, trips.df, output_directory=results_output, prefix=prefix)

if __name__ == "__main__":
    main()