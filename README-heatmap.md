# heatmap.py
This script provides a suite of Python functions for visualizing mobile tower locations and analyzing travel trip data across the UK. 
It enables creation of geographic plots of origin and destination towers, heatmaps showing density, visualizations of individual trip connections, 
and directional (azimuth) analyses of interregional flows. 
The code is designed to integrate with datasets containing coordinates of home and destination towers.

## Requirements
- Python 3.x
- pandas
- geopandas
- numpy
- matplotlib
- seaborn
- shapely

Install any needed packages using `pip install pandas geopandas numpy matplotlib seaborn shapely`.

## Input Data
- CSV with trip data: Must include the following columns:
  - home_coord_x (latitude of origin)
  - home_coord_y (longitude of origin)
  - dest_coord_x (latitude of destination)
  - dest_coord_y (longitude of destination)
- UK map file: A GeoJSON file, e.g., `census_data/uk_boundaries_2024_small.geojson`, containing UK boundary shapes for plot backgrounds.

## Main Functions
1. Creating GeoDataFrames
  - `create_funkturm_gdf(funkturm_data)`: `Combines origin and destination coordinates into a single GeoDataFrame (gdf) with geometry points and a type label (Home or Destination).

2. Visualizing Tower Locations
  - `plot_funkturm_map(funkturm_gdf, uk_data)`: Plots all celltower locations over the UK map, distinguishing origins (blue) and destinations (red).

3. Heatmaps
 - `plot_funkturm_heatmap(funkturm_gdf, uk_data)`: Produces a heatmap of all towers (both home and destination) using kernel density estimation (KDE).
 - `plot_home_funkturm_heatmap(funkturm_gdf, uk_data)`: Shows a heatmap for only origin towers.
 - `plot_trip_endpoints_heatmap(funkturm_data, uk_data)`: Used to plot a heatmap of all origins and destinations for trips starting in London.

4. Visualizing Trip Connections
  - `create_trip_lines(funkturm_data)`: Converts trip origin/destination data into GeoDataFrame line geometries for mapping connections.
  - `plot_trip_lines(trip_lines_gdf, uk_data)`: Plots sampled trip lines (recommended: use a sample, e.g., 10,000 rows, for clarity) over the UK map.

5. Filtering and Calculating Directions
  - `filter_trips_starting_in_london(funkturm_data)`: Selects only those trips where the origin lies within a defined bounding box for London.
  - `calculate_trip_azimuths(funkturm_data)`: Calculates the azimuth (compass direction) of each trip in degrees.
  - `plot_azimuth_windrose(azimuths, num_bins=36)`: Creates a windrose/polar histogram to visualize the directional distribution of trips.

## Example Usage

```python
# Load data
funkturm_data = pd.read_csv('path_to_csv_data')
funkturm_gdf = create_funkturm_gdf(funkturm_data)
base_map = gpd.read_file('census_data/uk_boundaries_2024_small.geojson')

# Plot locations
plot_funkturm_map(funkturm_gdf, base_map)

# Plot overall heatmap
plot_funkturm_heatmap(funkturm_gdf, base_map)

# Plot home tower heatmap
plot_home_funkturm_heatmap(funkturm_gdf, base_map)

# Plot sampled trip connections
trip_sample = funkturm_data.sample(n=10000, random_state=42)
trip_lines_gdf = create_trip_lines(trip_sample)
plot_trip_lines(trip_lines_gdf, base_map)

# Plot trip endpoints for trips starting in London
london_trips = filter_trips_starting_in_london(funkturm_data)
plot_trip_endpoints_heatmap(london_trips, base_map)

# Calculate and plot trip direction windrose
azimuths = calculate_trip_azimuths(trip_sample)
plot_azimuth_windrose(azimuths)
```

## Notes
- Adjust bounding boxes as needed for different metropolitan regions.
- Always visualize on a map background (uk_data) for geographic context.
- KDE plots may require tuning bw_adjust depending on data density.
- For large datasets, sampling is recommended for visual clarity.
