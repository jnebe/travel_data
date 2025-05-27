import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# CSV-Datei laden
df = pd.read_csv("trip_summary.csv")

# Filtern: Nur gültige (positive) Werte für log()
df_filtered = df[(df['trips'] > 0) & (df['distance'] > 0)].copy()

# Log-Transformation
df_filtered['log_trips'] = np.log(df_filtered['trips'])
df_filtered['log_distance'] = np.log(df_filtered['distance'])

# Regression vorbereiten
X = df_filtered[['log_distance']].values
y = df_filtered['log_trips'].values

# Lineare Regression auf log-log-Daten
model = LinearRegression()
model.fit(X, y)

# Koeffizienten extrahieren
alpha_opt = -model.coef_[0]               # negatives Vorzeichen, da Modell: log(Trips) = log(G) - α * log(Dist)
log_G_opt = model.intercept_
G_opt = np.exp(log_G_opt)


# Prognosen berechnen für ALLE Datenpunkte (auch z. B. Trips = 0)
df['predicted_trips'] = G_opt / (df['distance'] ** alpha_opt)

# Re-Skalierung auf Gesamtwertniveau
scaling_factor = df['trips'].sum() / df['predicted_trips'].sum()
df['predicted_trips_scaled'] = df['predicted_trips'] * scaling_factor

# Speichern
df.to_csv("trip_predictions_scaled.csv", index=False)

print(f"Skalierungsfaktor: {scaling_factor:.4f}")
print("Skalierte Prognosen gespeichert in trip_predictions_scaled.csv")



# Ergebnisse speichern
df.to_csv("trip_predictions.csv", index=False)

# Ausgabe
print(f"Optimiertes G:     {G_opt:.4f}")
print(f"Optimiertes α:     {alpha_opt:.4f}")
print(f"Parameter (log-G): {log_G_opt:.4f}")
print("Neue Datei gespeichert: trip_predictions.csv")
