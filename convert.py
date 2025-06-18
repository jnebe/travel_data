from pathlib import Path

import click

from gravity_model.log import logger
from gravity_model.location import LocationContainer
from gravity_model.trip import TripLoader
from gravity_model.distance import LATypes, BallTreeLocationAssigner, BeeLineLocationAssigner, CircleLocationAssigner

@click.command()
@click.argument("location_data", metavar="[Location Data]", type=click.Path(exists=True, readable=True, dir_okay=False, path_type=Path))
@click.argument("raw_data", metavar="[Unprocessed Trip Data]", type=click.Path(exists=True, readable=True, dir_okay=False, path_type=Path))
@click.argument("loc_assigner_type", metavar="[Location Assigner Type]", type=click.Choice(LATypes, case_sensitive=False))
@click.argument("results_output", metavar="[Trip Data Output]", type=click.Path(readable=True, dir_okay=False, path_type=Path))
@click.option("-k", "--keep-distance", "keep_distance", is_flag=True)
@click.option("-d", "--drop", is_flag=True)
def main(location_data: Path, raw_data: Path, loc_assigner_type:LATypes, results_output: Path, keep_distance: bool, drop: bool):
    logger.info(f"Loading location data from {location_data.absolute().as_posix()}")
    locs = LocationContainer.from_csv(location_data)
    if loc_assigner_type is LATypes.BALLTREE:
        loc_assigner = BallTreeLocationAssigner(locs)
    elif loc_assigner_type is LATypes.BEELINE:
        loc_assigner = BeeLineLocationAssigner(locs)
    elif loc_assigner_type is LATypes.CIRCLE:
        loc_assigner = CircleLocationAssigner(locs)

    logger.info(f"Loading unprocessed trip data from {raw_data.absolute().as_posix()}")
    trips = TripLoader.load_trips(
        loc_assigner, raw_data,
        {"start_lat": "home_coord_x", "start_long": "home_coord_y", "stop_lat": "dest_coord_x", "stop_long": "dest_coord_y", "number": "frequency"},
        keep_distance=keep_distance, min_distance=100 if drop else 0
    )

    logger.info(f"Saving normalized trip data to {results_output.absolute().as_posix()}")
    trips.to_csv(results_output)

if __name__ == "__main__":
    main()