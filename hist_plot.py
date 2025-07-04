import polars as pl
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import os

# Verzeichnispfad -> In den Ordner müssen alle CSV-Dateien rein, die geplottet werden sollen
csv_dir = "histogram_data/"
ground_truth_file = os.path.join(csv_dir, "real_output.csv")
model_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv") and f != "real_output.csv"]

# Ground Truth laden
df_gt = pl.read_csv(ground_truth_file)
distances_gt = df_gt["distance"].to_list()

# Modell-Ergebnisse laden
models = []
for file in model_files:
    df = pl.read_csv(os.path.join(csv_dir, file))
    distances = df["distance"].to_list()
    models.append((file, distances))


all_distances = distances_gt.copy()
for _, distances in models:
    all_distances.extend(distances)

_, bins = np.histogram(all_distances, bins=100)  # Feinere Bins für besseres Aussehen

# Plot
plt.figure(figsize=(10, 6))

plt.hist(distances_gt, bins=bins, density=True, alpha=0.6,
         label="Ground Truth", color='steelblue', edgecolor='black', linewidth=0.3)

colors = ['darkorange', 'green', 'purple']
for i, (name, distances) in enumerate(models):
    plt.hist(distances, bins=bins, density=True, alpha=0.4,
             label=f"Model {i+1}", color=colors[i % len(colors)], edgecolor='black', linewidth=0.3)

# Styling
plt.xlabel("distance")
plt.ylabel("probability")
plt.title("Distance Distribution")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

# Ground Truth CCDF
plt.ecdf(distances_gt, complementary=True,
         label="Ground Truth", color='steelblue')

colors = ['darkorange', 'green', 'purple']
for i, (name, distances) in enumerate(models):
    plt.ecdf(distances, complementary=True,
             label=f"Model {i+1}", color=colors[i % len(colors)])

# Set log-log scaling
plt.xscale('log')
plt.yscale('log')

# Styling
plt.xlabel("Distance (log scale)")
plt.ylabel("CCDF (log scale)")
plt.title("Complementary CDF (Log-Log)")
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.show()