
from .trip import Trip, TripContainer

def fix_hist(histogram: list[tuple[int, int]], length) -> list[tuple[int, int]]:
    if len(histogram) >= length:
        return histogram[:length]
    diff = length - len(histogram)
    for i in range(diff):
        histogram.append((
            (len(histogram) * 50) + ((i+1) * 50),
            0
        ))
    return histogram

def make_hist_similar(histogram_a: list[tuple[int, int]], histogram_b: list[tuple[int, int]]):
    if len(histogram_a) == len(histogram_b):
        return histogram_a, histogram_b
    if len(histogram_a) > len(histogram_b):
        return histogram_a, fix_hist(histogram_b, len(histogram_a))
    if len(histogram_a) < len(histogram_b):
        return fix_hist(histogram_a, len(histogram_b)), histogram_b

def total_variation_distance(test: TripContainer, model: TripContainer) -> int:
    tvd_sum = 0
    model_rel = model.as_relative()
    for trip, test_amount in test.as_relative().items():
        model_amount = model_rel.get(trip, 0.0)
        tvd_sum += abs(test_amount - model_amount)
    return tvd_sum / 2

def chi_square_distance(target: list[tuple[int, int]], actual: list[tuple[int, int]]):
    target, actual = make_hist_similar(target, actual)
    chi = \
        0.5 * sum(
            [
                ((a[1] - b[1]) ** 2) / (a[1] + b[1])
                for (a, b) in zip(target, actual)
            ]
        )
    return chi
    
def histogram_intersection_kernel(target: list[tuple[int, int]], actual: list[tuple[int, int]]):
    target, actual = make_hist_similar(target, actual)
    hik = \
        sum(
            [
                min(a[1], b[1])
                for (a, b) in zip(target, actual)
            ]
        )
    return hik