# heatmap-res.py
This scripr provides functions for analyzing and visualizing travel flows between regions using origin-destination data. 
It can be used to compare modeled and observed movement patterns, visualize start and end point densities, 
draw origin-destination lines, and analyze directional (azimuth) patterns of trips on geographic maps. 
The core focus is on UK travel data, but the approach is generic and adaptable.

## Requirements
- Python 3.x
- pandas
- geopandas
- numpy
- matplotlib
- seaborn
- shapely
- scipy

Install all packages if needed with `pip install pandas geopandas numpy matplotlib seaborn shapely scipy`.

## Input Data
- Model/Prediction Data (CSV): Must contain columns: start_lat, start_long, end_lat, end_long
- Original Data (CSV): Columns: home_coord_x (latitude), home_coord_y (longitude)
- Boundary Data (GeoJSON): UK boundaries or similar region polygons, e.g. uk_boundaries_2024_small.geojson

## Main Functions
- `load_location_data(csv_path)`: Loads trip or model CSV data as a DataFrame
- `create_point_gdf(df, prefix, label)`: Converts latitude/longitude columns into a GeoDataFrame with shapely Point geometry
- `create_trip_lines(df)`: Creates LineString geometries for each origin-destination pair
- `plot_trip_lines(trip_lines_gdf, base_map)`: Plots all trip lines over a base map
- `plot_heatmap(points_gdf, base_map, title, cmap)`: Kernel Density Estimate (KDE) point density heatmap on map
- `calculate_azimuths(df)`: Calculates azimuth (direction/bearing) for each trip
- `plot_azimuth_windrose(azimuths, num_bins)`: Visualizes directional distribution as a windrose (polar histogram)
- `compare_startpoint_density_funkturm_vs_model(original_df, model_df, base_map, grid_size)`: Compares density of origin points in model vs. observed data and visualizes differences
- `plot_startpoint_differences_histogram(original_df, model_df)`: Shows histograms of latitude/longitude differences between original and model trip origins

## Example Usage

```python
# Load modeled trip data and observed data
df = load_location_data("power_model_output.csv")
original_df = pd.read_csv("mod_part-00000.csv")
base_map = gpd.read_file("uk_boundaries_2024_small.geojson")

# Compare density of start points
compare_startpoint_density_funkturm_vs_model(original_df, df, base_map)

# Create trip lines (using a sample for clarity)
sampled_data = df.sample(n=1000, random_state=42)
trip_lines = create_trip_lines(sampled_data)
plot_trip_lines(trip_lines, base_map)

# Heatmaps of start/end points
start_gdf = create_point_gdf(df, "start", "Start")
end_gdf = create_point_gdf(df, "end", "End")
plot_heatmap(start_gdf, base_map, "Heatmap der Startregionen", cmap="Blues")
plot_heatmap(end_gdf, base_map, "Heatmap der Zielregionen", cmap="Reds")

# Azimuthal analysis
azimuths = calculate_azimuths(df)
plot_azimuth_windrose(azimuths)

# Histograms of geolocation differences
plot_startpoint_differences_histogram(original_df, df)

## Notes
- For performance and clarity, sample large datasets before plotting lines or heatmaps.
- Ensure coordinate reference system matches between data and maps ("EPSG:4326" is standard for WGS84 lat/lon).
- KDE heatmaps may require tuning bw_adjust for optimal visualization depending on the data’s spread.
- This script has similarities to `heatmap.py` but is used for comparison of modeled data and orignal data.

