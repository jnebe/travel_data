import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import seaborn as sns

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


if __name__ == "__main__":
    # CSV-Datei mit Funkturmdaten einlesen
    funkturm_data = pd.read_csv('path_to_csv_with_data')
    funkturm_gdf = create_funkturm_gdf(funkturm_data)

    # GeoJSON-Datei einlesen
    base_map = gpd.read_file('census_data/uk_boundaries_2024_small.geojson')
    
    # plot_funkturm_map(funkturm_gdf, base_map)
    plot_funkturm_heatmap(funkturm_gdf, base_map)
