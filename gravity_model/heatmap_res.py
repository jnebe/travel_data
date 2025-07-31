import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde

def load_location_data(csv_path):
    df = pd.read_csv(csv_path)
    return df

def create_point_gdf(df, prefix, label):
    gdf = df[[f"{prefix}_lat", f"{prefix}_long"]].copy()
    gdf.columns = ['lat', 'lon']
    gdf['type'] = label
    gdf['geometry'] = [Point(xy) for xy in zip(gdf['lon'], gdf['lat'])]
    return gpd.GeoDataFrame(gdf, geometry='geometry', crs="EPSG:4326")

def create_trip_lines(df):
    lines = [
        LineString([
            (row['start_long'], row['start_lat']),
            (row['end_long'], row['end_lat'])
        ])
        for _, row in df.iterrows()
    ]
    return gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")

def plot_trip_lines(trip_lines_gdf, base_map):
    fig, ax = plt.subplots(figsize=(12, 8))
    base_map.plot(ax=ax, color='lightgrey')
    trip_lines_gdf.plot(ax=ax, color='blue', linewidth=0.3, alpha=0.3)
    ax.set_xlim([-10, 4])
    ax.set_ylim([49, 61])
    plt.title("Verbindungen zwischen Start- und Zielregionen")
    plt.tight_layout()
    plt.show()

def plot_heatmap(points_gdf, base_map, title, cmap="Reds"):
    fig, ax = plt.subplots(figsize=(12, 8))
    base_map.plot(ax=ax, color='lightgrey')
    sns.kdeplot(
        x=points_gdf.geometry.x,
        y=points_gdf.geometry.y,
        ax=ax,
        cmap=cmap,
        fill=True,
        bw_adjust=0.3,
        alpha=0.6,
        thresh=0.05
    )
    ax.set_xlim([-10, 4])
    ax.set_ylim([49, 61])
    plt.title(title)
    plt.tight_layout()
    plt.show()

def calculate_azimuths(df):
    start_lat = np.radians(df['start_lat'])
    start_lon = np.radians(df['start_long'])
    end_lat = np.radians(df['end_lat'])
    end_lon = np.radians(df['end_long'])

    delta_lon = end_lon - start_lon

    x = np.sin(delta_lon) * np.cos(end_lat)
    y = np.cos(start_lat) * np.sin(end_lat) - np.sin(start_lat) * np.cos(end_lat) * np.cos(delta_lon)
    initial_bearing = np.arctan2(x, y)
    azimuths = (np.degrees(initial_bearing) + 360) % 360
    return azimuths

def plot_azimuth_windrose(azimuths, num_bins=36):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    azimuth_radians = np.radians(azimuths)
    counts, bin_edges = np.histogram(azimuth_radians, bins=num_bins, range=(0, 2*np.pi))

    bars = ax.bar(
        bin_edges[:-1],
        counts,
        width=(2 * np.pi) / num_bins,
        bottom=0.0,
        align='edge',
        color='crimson',
        alpha=0.8,
        edgecolor='black'
    )

    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    angles_dir = np.linspace(0, 2 * np.pi, len(directions), endpoint=False)
    ax.set_xticks(angles_dir)
    ax.set_xticklabels(directions, fontsize=12, fontweight='bold')

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    plt.title("Richtungsanalyse der Verbindungen (Azimut)", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

def compare_startpoint_density_funkturm_vs_model(original_df, model_df, base_map, grid_size=100):

    # Repariere ungültige Geometrien der Base Map (nur einmal machen)
    base_map['geometry'] = base_map['geometry'].buffer(0)
    union_geom = base_map.geometry.values.union_all()

    # Originaldaten: Start = home_coord
    x_orig = original_df['home_coord_y']
    y_orig = original_df['home_coord_x']

    # Modellierte Daten: Start = start_long, start_lat
    x_model = model_df['start_long']
    y_model = model_df['start_lat']

    def compute_density(x, y):
        data = np.vstack([x, y])
        kde = gaussian_kde(data)
        xi = np.linspace(min(x.min(), x_model.min()), max(x.max(), x_model.max()), grid_size)
        yi = np.linspace(min(y.min(), y_model.min()), max(y.max(), y_model.max()), grid_size)
        xi, yi = np.meshgrid(xi, yi)
        zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
        return xi, yi, zi

    xi, yi, zi_orig = compute_density(x_orig, y_orig)
    _, _, zi_model = compute_density(x_model, y_model)

    diff = zi_model - zi_orig

    # Erstelle Punkte-Array zum Maskieren (x,y-Paare als shapely Points)
    points = [Point(xy) for xy in zip(xi.ravel(), yi.ravel())]

    # Erstelle Maske, die nur Punkte innerhalb der Karte True sind
    mask = np.array([union_geom.contains(pt) for pt in points]).reshape(xi.shape)

    fig, ax = plt.subplots(figsize=(12, 8))
    base_map.plot(ax=ax, color='lightgrey')

    # Nur Werte anzeigen, die innerhalb der Karte sind, außerhalb maskieren
    masked_diff = np.ma.array(diff, mask=~mask)

    pcm = ax.pcolormesh(xi, yi, masked_diff, shading='auto', cmap='RdBu_r', alpha=0.7)
    plt.colorbar(pcm, ax=ax, label="Modell - Original (Startpunktdichte)")
    ax.set_xlim([-10, 4])
    ax.set_ylim([49, 61])
    plt.title("Differenz der Startpunktdichte (Modell vs. Original)", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_startpoint_differences_histogram(original_df, model_df):
    # Annahme: original_df hat home_coord_x (lat), home_coord_y (long)
    # model_df hat start_lat (lat), start_long (long)

    # Sicherstellen, dass beide Datenframes gleich lang sind (oder matched)
    n = min(len(original_df), len(model_df))
    original_df = original_df.iloc[:n].reset_index(drop=True)
    model_df = model_df.iloc[:n].reset_index(drop=True)

    # Differenzen berechnen
    diff_lat = model_df['start_lat'] - original_df['home_coord_x']
    diff_long = model_df['start_long'] - original_df['home_coord_y']

    # Histogramme plotten
    fig, axs = plt.subplots(1, 2, figsize=(14,5))

    axs[0].hist(diff_lat, bins=50, color='skyblue', edgecolor='black')
    axs[0].set_title('Histogramm der Breitengrad-Differenzen (Modell - Original)')
    axs[0].set_xlabel('Differenz Latitude (Grad)')
    axs[0].set_ylabel('Anzahl')

    axs[1].hist(diff_long, bins=50, color='salmon', edgecolor='black')
    axs[1].set_title('Histogramm der Längengrad-Differenzen (Modell - Original)')
    axs[1].set_xlabel('Differenz Longitude (Grad)')
    axs[1].set_ylabel('Anzahl')

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    df = load_location_data("/Users/janawusten/Desktop/graphs/tripleexpo,alpha=0.00179,beta=0.386,gamma=0.0552/tripleexpo_model_output.csv")

    base_map = gpd.read_file("/Users/janawusten/Desktop/travel_data/census_data/uk_boundaries_2024_small.geojson")

    original_df = pd.read_csv("/Users/janawusten/Downloads/mod_part-00000-f7a38d19-bcf1-40c6-9036-cff194323e9b-c000.csv")

    # plot_startpoint_differences_histogram(original_df, df)
    # start_gdf = create_point_gdf(df, "start", "Start")

    # end_gdf = create_point_gdf(df, "end", "Ende")


    # Plotten
    # compare_startpoint_density_funkturm_vs_model(original_df, df, base_map)
    #sampled_data = df.sample(n=1000, random_state=42)
    #trip_lines = create_trip_lines(sampled_data)
    #plot_trip_lines(trip_lines, base_map)
    # plot_heatmap(start_gdf, base_map, "Heatmap der Startregionen", cmap="Blues")

    # plot_heatmap(end_gdf, base_map, "Heatmap der Zielregionen", cmap="Reds")

    # azimuths = calculate_azimuths(df)
    # plot_azimuth_windrose(azimuths)
