from pathlib import Path

import math
from typing import Iterable
import polars as pl
import seaborn as sb
from matplotlib.pyplot import gca

from matplotlib.axes import Axes
import matplotlib

def set_dpi(dpi: int, save_dpi: int):
    matplotlib.rcParams["figure.dpi"] = dpi
    matplotlib.rcParams["savefig.dpi"] = save_dpi

def save_figure(plot: Axes, file: Path):
    figure = plot.get_figure()
    figure.savefig(file, bbox_inches="tight")
    figure.clear()

vis_types = ["hist", "cdf", "ccdf", "kde"]

def visualize(type: str, dataframes: Iterable[pl.DataFrame], output_directory: Path = Path("output/"), prefix: str = None):
    max_dist=100
    for df in dataframes:
        cmax_dist = df.select(pl.col("distance").max()).item()
        if cmax_dist > max_dist:
            max_dist = cmax_dist
    plot: Axes = gca()
    plot.set_xbound(0, max_dist)
    if type == "hist":
        plot = hist(plot, dataframes)
    elif type == "cdf":
        plot = cdf(plot, dataframes)
    elif type == "ccdf":
        plot = ccdf(plot, dataframes)
    elif type == "kde":
        plot = kde(plot, dataframes)
    else:
        raise ValueError("Unsupported diagram type!")
    output_directory.mkdir(parents=True, exist_ok=True)
    if prefix:
        save_figure(plot, output_directory.joinpath(f"{prefix}_{type}_plot.png"))
    else:
        save_figure(plot, output_directory.joinpath(f"{type}_plot.png"))

def hist(plot: Axes, dataframes: Iterable[pl.DataFrame]) -> Axes:
    plot_x_max = plot.get_xbound()[1] + 50
    for df in dataframes:
        sb.histplot(df, ax=plot, x="distance", stat="probability", element="bars", binwidth=50, binrange=(0, plot_x_max))
    return plot

def cdf(plot: Axes, dataframes: Iterable[pl.DataFrame]) -> Axes:
    plot.set_xscale("log")
    plot.set_yscale("log")
    for df in dataframes:
        sb.ecdfplot(df, ax=plot, x="distance")
    return plot

def ccdf(plot: Axes, dataframes: Iterable[pl.DataFrame]) -> Axes:
    plot.set_xscale("log")
    plot.set_yscale("log")
    for df in dataframes:
        sb.ecdfplot(df, ax=plot, x="distance", complementary=True)
    return plot

def kde(plot: Axes, dataframes: Iterable[pl.DataFrame]) -> Axes:
    plot.set_xscale("log")
    plot.set_yscale("log")
    for df in dataframes:
        df = df.sample(fraction=1.0, shuffle=True)
        plot = sb.kdeplot(df, ax=plot, x="distance")
    return plot