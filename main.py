#!/bin/python
import click
from pathlib import Path

import polars as pl

from gravity_model import v1 as _v1

from log import logger
from location import CensusLocationLoad
from gravity import GravityModel
from trip import Trip, TripLoader
from visualize import vis_types, visualize

@click.group()
def main():
    pass

@main.command()
@click.argument("trips", type=click.Path(exists=True, readable=True, dir_okay=False, path_type=Path))
def v1(trips: Path):
    df = pl.read_csv(trips, infer_schema_length=None)
    _v1(df)

@main.command()
@click.argument("boundary_data", metavar="[Boundary Data]", type=click.Path(exists=True, readable=True, dir_okay=False, path_type=Path))
@click.argument("population_data", metavar="[Population Data]", type=click.Path(exists=True, readable=True, dir_okay=False, path_type=Path))
@click.option("-n", "--number-trips", type=int, default=100000)
@click.option("-r", "--real-trips", "trips", type=click.Path(exists=True, readable=True, dir_okay=False, path_type=Path))
@click.option("--save-model-trips", type=click.Path(writable=True, dir_okay=False, path_type=Path), help="Filename where to save mode trip data")
@click.option("--save-real-trips", type=click.Path(writable=True, dir_okay=False, path_type=Path), help="Filename where to save real trip data")
@click.option("--visualize", "dovisualize", type=click.Path(file_okay=False, path_type=Path), help="Directory where to save figures of model and real trips")
@click.option("-v", "--verbose", count=True)
def v2(boundary_data: Path, population_data: Path, number_trips: int, trips: Path, save_model_trips: Path, save_real_trips: Path, dovisualize: Path, verbose: int):
    logger.info("Loading Boundary and Population Data")
    locs = CensusLocationLoad.load_locations(boundary_data, population_data,
                              {"index": "LAD24CD", "name": "LAD24NM", "lat": "LAT", "long": "LONG"},
                              {"index": "Code", "population": "Population"})
    logger.info("Building Gravity Model")
    model = GravityModel(locs)
    
    if trips:
        logger.info("Loading real trip data and assigning it to existing locations")
        real_trips = TripLoader.load_trips(locs, trips,
                                {"start_lat": "home_coord_x", "start_long": "home_coord_y", "stop_lat": "dest_coord_x", "stop_long": "dest_coord_y", "number": "frequency"})
        number_trips = len(real_trips)
    
    logger.info(f"Simulating {number_trips} trips using the gravity model")
    model_trips = model.make_trips(number_trips)
    
    if save_model_trips:
        logger.info(f"Saving model trip data to {save_model_trips.absolute().as_posix()}")
        model_trips.as_dataframe().write_csv(save_model_trips, float_precision=2)
    if dovisualize:
        logger.info(f"Visualizing model trip data. Output in {dovisualize.absolute().as_posix()}")
        for type in vis_types:
            logger.info(f"Generating {type} plot")
            visualize(type, model_trips.as_dataframe(), output_directory=dovisualize, prefix="model")

    if trips:
        if save_real_trips:
            logger.info(f"Saving real trip data to {save_model_trips.absolute().as_posix()}")
            real_trips.as_dataframe().write_csv(save_real_trips, float_precision=2)
        if dovisualize:
            logger.info(f"Visualizing real trip data. Output in {dovisualize.absolute().as_posix()}")
            for type in vis_types:
                logger.info(f"Generating {type} plot")
                visualize(type, real_trips.as_dataframe(), output_directory=dovisualize, prefix="real")

    if verbose > 0 or ((not save_model_trips) and (not save_real_trips)):
        for trip in model.all_trips:
            if trips:
                print(f"Trip from {trip.locations[0].name} to {trip.locations[1].name}: real {real_trips.as_dict().get(trip, -1)} | sim {model_trips.as_dict().get(trip, -1)} {model.matrix.get(trip)}")
            else:
                print(f"Trip from {trip.locations[0].name} to {trip.locations[1].name}: sim {model_trips.as_dict().get(trip, -1)} {model.matrix.get(trip)}")

if __name__ == "__main__":
    main()