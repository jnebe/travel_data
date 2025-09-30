# Modelling Human Long-Range travel data

## Preqrequisites

[uv](https://github.com/astral-sh/uv) for dependency management

Use `uv venv` and then `uv sync` to create the required virtual environment and download the required packages.
After that use `uv run <your command>` to execute python scripts within the virtual environment or manually activate the virtual environment.

### [celltower_data](./celltower_data/README.md)
Uses: 
- polars
- seaborn

### [census_data](./census_data/README.md)
Uses:
- geopandas
- geoplot

## Table of Contents

- [heatmap.py README](./README-heatmap.md)
- [heatmap.py README](./README-heatmap-res.md)
- [celltower data README](./celltower_data/README.md)
- [census data README](./census_data/README.md)
- [gravity model helpers](./gravity_model/README.md)
- [gravity model implementations](./gravity_model/models/README.md)
- [gravity model search algorithms](./gravity_model/search/README.md)
- [data output formats](./FORMATS.md)

## Main Scripts

Main scripts are those that can be executed.
Other files are libraries/helper files used for those main scripts.

- [preprocess](./preprocess.py) - converts the census data into a single location dataset
- [convert](./convert.py) - converts the celltower data we have to the common format we use for trips, based on the location data created by process.py
- [train](./train.py) - uses the location data and the converted celltower data to generate a gravity model, optimising the parameters of the model to produce more similar trips based on the trip length histogram.
- [run](./run.py) - uses a gravity model to generate a certain amount of trips
- [eval](./eval.py) - produces graphs comparing the real and the model output, producing histogram, CCDF, CDF and KDE plots
- [map](./map.py) - produces heightmaps to visualize the relationship between parameters and error metrics

## Makefile

The Makefile contains a simple pipeline to produce the required data (assuming all the datasets are placed correctly), train a model and evaluate it.

We require the following files:

- `celltower_data/merged_uk_data.csv` - needs to contain the celltower data
- `census_data/uk_boundaries_merged_2024.csv` - needs to contain the boundary data
- `census_data/uk_2022.csv` - needs to contain the population data

Alternatively, if you have the location data (as `loc_data.csv`) and the converted celltower data (as `real_output.csv`) the make command will skip the initial steps.

To use the makefile use: `make full-<model name> ITERATIONS=<your iteration number> SEARCH=<your search algorithm>`
The makefile defaults to 10 iterations and a search using Nelder-Mead.

For example, to produce our triple-power model use: `make full-triplepower ITERATIONS=200`