# gravity.py
This file defines classes for simulating movement between cities based on population and distance, inspired by the gravity equation from physics.

## Classes

`GravityModel`:

- Builds a matrix of pairwise "trip gravity values" for all city pairs
- The default formula is:
````
gravity = (population_A * population_B) / distance
````

- Generates synthetic trip samples based on the gravity matrix using weighted random choice

`PowerGravityModel(GravityModel)`:

- A subclass with a tunable alpha exponent:

```
gravity = (population_A * population_B) / (distance ^ alpha)
```

- Useful for fine-tuning how strongly distance affects the trip probability


## Key Methods

- `make_trips(n: int)`: returns n randomly sampled trips based on gravity weights
- `all_trips`: returns all possible trips (pairs)
- `__len__, __repr__`: utility methods