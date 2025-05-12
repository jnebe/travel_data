# gravity_model.py
This script provides an alternative gravity function and performs evaluation using real-world trip data.

## Key Features

- `gravity_model(city1, city2)`: computes predicted trips using a parameterized gravity formula:
````
T_ij = G * (P_i^α) * (P_j^β) / (D_ij^γ)
````

- `calc_total_trips()`: counts real trips between two cities based on origin/destination coordinates and radius
- `v1(df)`: prints real vs. predicted trip counts for London, Birmingham and Manchester.

