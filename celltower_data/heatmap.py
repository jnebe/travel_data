import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import seaborn as sns
import numpy as np

def create_funkturm_gdf(funkturm_data):
    # DataFrames für die Funktürme
    # Home
    funkturm_home = funkturm_data[['home_coord_y', 'home_coord_x']].copy()
    funkturm_home.columns = ['lon', 'lat']
    funkturm_home['type'] = 'Home'

    # Destination
    funkturm_dest = funkturm_data[['dest_coord_y', 'dest_coord_x']].copy()
    funkturm_dest.columns = ['lon', 'lat']
    funkturm_dest['type'] = 'Destination'

    # Beide Sets zusammenführen
    all_towers = pd.concat([funkturm_home, funkturm_dest], ignore_index=True)

    # Konvertieren zu GeoDataFrame
    # Erstelle Point-Geometrien aus den Koordinaten
    geometry = [Point(xy) for xy in zip(all_towers['lon'], all_towers['lat'])]
    funkturm_gdf = gpd.GeoDataFrame(all_towers, geometry=geometry, crs="EPSG:4326")

    return funkturm_gdf


def plot_funkturm_map(funkturm_gdf, uk_data):
    
    # Karte erstellen und Funktürme plotten
    fig, ax = plt.subplots(figsize=(12, 8))

    # UK-Karte plotten
    uk_data.plot(ax=ax, color='lightgrey')

    # Funktürme nach Typ unterschiedlich plotten
    home_towers = funkturm_gdf[funkturm_gdf['type'] == 'Home']
    dest_towers = funkturm_gdf[funkturm_gdf['type'] == 'Destination']

    home_towers.plot(ax=ax, color='blue', markersize=0.2, marker='o', label='Home-Funktürme')
    dest_towers.plot(ax=ax, color='red', markersize=0.2, marker='^', label='Destination-Funktürme')

    # Kartenbeschriftung hinzufügen
    plt.title('Standorte der Funktürme')
    plt.legend()
    plt.tight_layout()

    # UK-Grenzen hinzufügen
    ax.set_xlim([-10, 4])   
    ax.set_ylim([49, 61]) 

    # Karte anzeigen
    plt.show()

def plot_funkturm_heatmap(funkturm_gdf, uk_data):
    fig, ax = plt.subplots(figsize=(12, 8))
    uk_data.plot(ax=ax, color='lightgrey')

    # Heatmap plotten
    sns.kdeplot(
        x=funkturm_gdf['lon'],
        y=funkturm_gdf['lat'],
        ax=ax,
        cmap="Reds",
        shade=True,
        bw_adjust=0.2,
        alpha=0.6,
        thresh=0.05
    )

    plt.title('Heatmap der Funktürme (Seaborn KDE)')

    # UK-Grenzen hinzufügen
    ax.set_xlim([-10, 4])
    ax.set_ylim([49, 61])
    plt.tight_layout()

    # Karte anzeigen
    plt.show()

def plot_home_funkturm_heatmap(funkturm_gdf, uk_data):
    home_towers = funkturm_gdf[funkturm_gdf['type'] == 'Home']

    fig, ax = plt.subplots(figsize=(12, 8))
    uk_data.plot(ax=ax, color='lightgrey')

    sns.kdeplot(
        x=home_towers['lon'],
        y=home_towers['lat'],
        ax=ax,
        cmap="Blues",
        fill=True,
        bw_adjust=0.2,
        alpha=0.6,
        thresh=0.05
    )

    plt.title('Heatmap der Start-Funktürme (Home)')
    ax.set_xlim([-10, 4])
    ax.set_ylim([49, 61])
    plt.tight_layout()
    plt.show()

def create_trip_lines(funkturm_data):
    lines = [
        LineString([
            (row['home_coord_y'], row['home_coord_x']),
            (row['dest_coord_y'], row['dest_coord_x'])
        ])
        for _, row in funkturm_data.iterrows()
    ]
    return gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")

def plot_trip_lines(trip_lines_gdf, uk_data):
    fig, ax = plt.subplots(figsize=(12, 8))

    # Hintergrundkarte
    uk_data.plot(ax=ax, color='lightgrey')

    # Trips zeichnen
    trip_lines_gdf.plot(ax=ax, color='blue', linewidth=0.1, alpha=0.05)

    plt.title('Verbindungen zwischen Funktürmen (Trips)')
    ax.set_xlim([-10, 4])
    ax.set_ylim([49, 61])
    plt.tight_layout()
    plt.show()

def calculate_trip_azimuths(funkturm_data):
    # Koordinaten extrahieren
    start_lat = np.radians(funkturm_data['home_coord_x'])
    start_lon = np.radians(funkturm_data['home_coord_y'])
    end_lat = np.radians(funkturm_data['dest_coord_x'])
    end_lon = np.radians(funkturm_data['dest_coord_y'])

    # Azimut-Berechnung
    delta_lon = end_lon - start_lon

    x = np.sin(delta_lon) * np.cos(end_lat)
    y = np.cos(start_lat) * np.sin(end_lat) - np.sin(start_lat) * np.cos(end_lat) * np.cos(delta_lon)

    initial_bearing = np.arctan2(x, y)

    # In Grad umwandeln
    azimuths = (np.degrees(initial_bearing) + 360) % 360


    return azimuths

def plot_azimuth_windrose(azimuths, num_bins=36):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    # In Radiant umwandeln
    azimuth_radians = np.radians(azimuths)

    # Histogramm
    counts, bin_edges = np.histogram(azimuth_radians, bins=num_bins, range=(0, 2 * np.pi))
    max_count = counts.max()

    # Balken zeichnen
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

    # Himmelsrichtungen statt Winkelbeschriftung
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    angles_dir = np.linspace(0, 2 * np.pi, len(directions), endpoint=False)
    ax.set_xticks(angles_dir)
    ax.set_xticklabels(directions, fontsize=12, fontweight='bold')

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    plt.title("Richtungsanalyse der Trips (Azimut)", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # CSV-Datei mit Funkturmdaten einlesen
    funkturm_data = pd.read_csv('path_to_csv_with_data')
    funkturm_gdf = create_funkturm_gdf(funkturm_data)

    # GeoJSON-Datei einlesen
    base_map = gpd.read_file('census_data/uk_boundaries_2024_small.geojson')
    
    # plot_funkturm_map(funkturm_gdf, base_map)
    # plot_funkturm_heatmap(funkturm_gdf, base_map)

    # # random Sample, da sonst zu voll
    sampled_data = funkturm_data.sample(n=10000, random_state=42)
    # trip_lines_gdf = create_trip_lines(sampled_data)
    # plot_trip_lines(trip_lines_gdf, base_map)

    azimuths = calculate_trip_azimuths(sampled_data)

    plot_azimuth_windrose(azimuths)

