import tqdm
import polars as pl
import logging

from .trip import Trip, TripContainer
from .log import logger

HISTROGRAM_BIN_SIZE = 10

def get_histogram(trips: TripContainer) -> list[tuple[int, int]]:
    bins = trips.df.with_columns(
        (
            (pl.col("distance") // HISTROGRAM_BIN_SIZE).alias("index")
        )
    )
    bins = bins.with_columns(
        (
            (pl.col("index") * HISTROGRAM_BIN_SIZE).alias("label")
        )
    )
    bins = bins.group_by("label").count()
    bins = bins.sort(by="label")

    total_count = bins.select(pl.col("count").sum()).item()
    bins = bins.with_columns(
        (pl.col("count") / total_count).alias("percentage")
    )

    results = []
    for bin in tqdm.tqdm(bins.iter_rows(named=True), desc="Binning", total=bins.height, unit="row(s)", disable=not logger.level < logging.INFO):
        results.append((bin["label"], bin["percentage"]))
    return results

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

def get_ccdf(trips: TripContainer) -> list[tuple[int, float]]:
    histogram = sorted(get_histogram(trips), key=lambda x: x[0])

    ccdf = []
    cumulative = 0.0

    # Traverse from the end to start
    for label, percentage in reversed(histogram):
        cumulative += percentage
        ccdf.append((label, cumulative))

    ccdf.reverse()
    return ccdf

def kolmogorov_smirnov_statistic(target: list[tuple[int, float]], actual: list[tuple[int, float]]):
    target = sorted(target, key=lambda x: x[0])
    actual = sorted(actual, key=lambda x: x[0])
    diffs = [abs(a - b) for (_, a), (_, b) in zip(target, actual)]
    return max(diffs)

class Parameter():

    def __init__(self, name: str, initial: float, minimum: float, maximum: float):
        self.name = name
        self.value = initial
        self.minimum = minimum
        self.maximum = maximum

    def get_step(self, max_steps, step: int) -> float:
        if max_steps <= 1:
            return 0.0
        step_size = (self.maximum - self.minimum) / (max_steps - 1)
        return self.minimum + step * step_size