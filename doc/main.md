# main.py: CLI Interface (Click-Based)
This is the command-line interface (CLI) entry point.

## Commands

- `v1 [trips.csv]`: Runs basic real vs. gravity comparison for a few cities.
- `v2 [boundary_data] [population_data]`:
    - Loads full dataset
    - Builds gravity model
    - Simulates trips
    - Optionally loads real trip data
    - Saves output and visualizations

## Options

- `--save-model-trips`
- `--save-real-trips`
- `--visualize`
- `--verbose`