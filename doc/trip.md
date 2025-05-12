# trip.py

Handles the representation of trips between Locations and loading/transforming trip data.

## Classes

`Trip`:

- Immutable pair of Location instances
- Automatically sorted to ensure consistent hashing/equality

`TripResults`:

- Wraps a list of Trip objects
- Can convert trips to:
    - A frequency dictionary (`as_dict()`).
    - A polars.DataFrame (`as_dataframe()`).

`TripLoader`:
- Uses BallTree (from scikit-learn) for fast spatial nearest-neighbor lookup
- Loads CSV data with origin-destination coordinates and maps them to nearby Location object
- Returns a TripResults instance
