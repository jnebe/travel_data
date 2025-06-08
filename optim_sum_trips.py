import pandas as pd

# Originaldaten einlesen
df = pd.read_csv("real_output.csv")

# Filtere Strecken mit Distanz >= 10 km
df = df[df["distance"] >= 100]

# Normierung (z.B. Population in Millionen, Distanz in 100 km)
# df["start_population"] = df["start_population"] / 1_000_000  # von Personen auf Millionen
# df["end_population"] = df["end_population"] / 1_000_000      # von Personen auf Millionen
# df["distance"] = df["distance"] / 1000                          # von km auf 100 km

# Erstelle normierte Stadtpaare (sortiert, damit A-B = B-A)
df["city_pair"] = df.apply(lambda row: tuple(sorted([row["start_area"], row["end_area"]])), axis=1)

# Gruppiere nach city_pair
trip_summary = (
    df.groupby("city_pair")
    .agg(
        trips=("city_pair", "size"),
        distance=("distance", "mean"),
        start_name=("start_name", "first"),  # erste Population übernehmen
        start_population=("start_population", "first"),
        end_name=("end_name", "first"),
        end_population=("end_population", "first")
    )
    .reset_index()
)

# Entferne city_pair-Spalte, wenn nicht benötigt
# trip_summary = trip_summary.drop(columns=["city_pair"])

# Speichere Ergebnis
trip_summary.to_csv("trip_summary.csv", index=False)

print("trip_summary.csv gespeichert.")
print(trip_summary.head())
