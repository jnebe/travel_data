
from .trip import Trip, TripContainer

def total_variation_distance(test: TripContainer, model: TripContainer) -> int:
    tvd_sum = 0
    model_rel = model.as_relative()
    for trip, test_amount in test.as_relative().items():
        model_amount = model_rel.get(trip, 0.0)
        tvd_sum += abs(test_amount - model_amount)
    return tvd_sum / 2