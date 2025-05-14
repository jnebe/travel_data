#!/bin/python
from pathlib import Path

import click

from gravity_model.log import logger
from gravity_model.location import LocationLoader

@click.command()
@click.argument("boundary_data", metavar="[Boundary Data]", type=click.Path(exists=True, readable=True, dir_okay=False, path_type=Path))
@click.argument("population_data", metavar="[Population Data]", type=click.Path(exists=True, readable=True, dir_okay=False, path_type=Path))
@click.argument("location_data", metavar="[Location Data Output]", type=click.Path(exists=False, readable=True, dir_okay=False, path_type=Path))
def main(boundary_data: Path, population_data: Path, location_data: Path):
    logger.info(f"Processing {boundary_data.absolute().as_posix()} and {population_data.absolute().as_posix()}")
    locs = LocationLoader.from_csv(boundary_data, population_data,
                              {"index": "LAD24CD", "name": "LAD24NM", "lat": "LAT", "long": "LONG", "area": "Shape__Area"},
                              {"index": "Code", "population": "Population"})
    logger.info(f"Writing resulting location information to {location_data.absolute().as_posix()}")
    locs.to_csv(location_data)
    
if __name__ == "__main__":
    main()