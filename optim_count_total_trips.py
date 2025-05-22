import pandas as pd

df = pd.read_csv("trip_summary.csv", usecols=["trips"])

total_trips = df["trips"].sum()

print("Total number of trips:", total_trips)

