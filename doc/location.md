# location.py

This file models geographic entities like cities or zones.

## Classes

`Location`

- Stores attributes like: name, id, latitude, longitude and population.
- Can compute: Distance to another `Location` or coordinate pair
- Supports: 
    - Custom equality based on coordinates
    - Hashing for use as dictionary keys


`CensusLocationLoad`

- Static method `load_locations(...)`:
    - Reads boundary and population CSVs
    - Merges and returns a list of Location instances