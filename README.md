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


## Tasks

1. Extract geo-data from UK. Population and pair-wise distances between districts. We have multiple possibilites for granularity. Reasonable to me seems Local Authority Districts:

<https://geoportal.statistics.gov.uk/datasets/ons::local-authority-districts-may-2024-boundaries-uk-bfe-2/explore?location=53.279432%2C-4.166132%2C6.81>

<https://en.wikipedia.org/wiki/Districts_of_England>

We don't need Ireland but it would be good to include Scotland and Wales (everything on the mainland). Also merge big cities (e.g. London) to avoid splitting gravitational force.

1. Analyze the empirical data. Look at the files and prepare visualization toolbox. We will frequently look at CDF, CCDF and PDF of the empirical data (log-log scale and unscaled).  
   Would be cool: Visualizing a heat-map of start- and endpoints for the trips in our data-set
2. Understanding and implementing the gravity model. Basic variant measures gravity G\_(i,j) between cities i and j as P_i \* P_j / d\_(i,j). Note that G\_(i,j) is symmetric

- P_i, P_j are populations of cities i,j
- d\_(i,j) is the euclidean distance between cities i and j in kilometers

We will artificially simulate trips following this model. That is, with probability G\_(i,j) / \\sum\_(k,l) G\_(k,l) the next trip will go from city i to city j

We draw many such trips (e.g. 100 000) and then draw the CCDF. This is then compared to the CCDF of the empirical data to measure accuracy

Later on we have parameters in our model, which need to by analyzed. Additionally we might try different models. Starting with the power-law variant that has G\_(i,j) = P_i \* P_j / ( d\_(i,j) )^alpha, where alpha > 1 is a parameter to be optimized