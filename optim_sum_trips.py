import pandas as pd

# Originaldaten einlesen
df = pd.read_csv("real_output.csv")

# Normierung wird sehr wichtig für Optimierung
# -> richtige Werte müssen noch gefunden werden
df["start_population"] = df["start_population"]/100000
df["end_population"] = df["end_population"]/100000
df["distance"] = df["distance"]/1000
                            

# Erstelle normierte Stadtpaare
df["city_pair"] = df.apply(lambda row: tuple(sorted([row["start_area"], row["end_area"]])), axis=1)

# Gruppiere nach city_pair
trip_summary = (
    df.groupby("city_pair")
    .agg(
        trips=("city_pair", "size"),
        distance=("distance", "mean"),
        start_name=("start_name", "first"),
        start_population=("start_population", "mean"), #but should be the first
        end_name=("end_name", "first"),
        end_population=("end_population", "mean") #but should be the first
    )
    .reset_index()
)

# Teile city_pair wieder in zwei Spalten
#trip_summary[["start_name", "end_name"]] = pd.DataFrame(trip_summary["city_pair"].tolist(), index=trip_summary.index)

# Entferne city_pair-Spalte
#trip_summary = trip_summary.drop(columns=["city_pair"])

# Neue CSV speichern
trip_summary.to_csv("trip_summary.csv", index=False)

print("trip_summary.csv gespeichert.")
print(trip_summary.head())

