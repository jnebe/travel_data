#!/bin/python
from pathlib import Path
from typing import Iterable

import click

from gravity_model.log import logger
from gravity_model.trip import TripContainer
from gravity_model.visualize_single import visualize, vis_types

@click.command()
@click.argument("trip_location", metavar="[Trip Data]", type=click.Path(exists=True, readable=True, dir_okay=False, path_type=Path))
@click.option("-c", "--compare", type=(click.Path(exists=True, readable=True, dir_okay=False, path_type=Path), str))
@click.argument("results_output", metavar="[Evaluation Output]", type=click.Path(readable=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("prefix", metavar="[Prefix]", type=str)
def main(trip_location: Path, compare: tuple[Path, str], results_output: Path, prefix: str):
    logger.info(f"Loading trip data from {trip_location.absolute().as_posix()}")
    trips = TripContainer.from_csv(trip_location)

    if compare:
        logger.info(f"Loading trip data from  {compare[0].absolute().as_posix()}")
        comparison = TripContainer.from_csv(compare[0])

    logger.info(f"Visualizing trips...")
    for current_visualization_type in vis_types:
        logger.info(f"Generating {current_visualization_type} plot")
        visualize(current_visualization_type, trips.df, output_directory=results_output, prefix=prefix)
        if compare:
            visualize(current_visualization_type, comparison.df, output_directory=results_output, prefix=compare[1])



if __name__ == "__main__":
    main()