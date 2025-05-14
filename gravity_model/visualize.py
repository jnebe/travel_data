from pathlib import Path

import polars as pl
import seaborn as sb

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

def visualize(type: str, df: pl.DataFrame, output_directory: Path = Path("output/"), prefix: str = None):
    df = df.drop(["start_area", "end_area"])
    plot = None
    if type == "hist":
        plot = hist(df)
    elif type == "cdf":
        plot = cdf(df)
    elif type == "ccdf":
        plot = ccdf(df)
    elif type == "kde":
        plot = kde(df)
    else:
        raise ValueError("Unsupported diagram type!")
    output_directory.mkdir(parents=True, exist_ok=True)
    if prefix:
        save_figure(plot, output_directory.joinpath(f"{prefix}_{type}_plot.png"))
    else:
        save_figure(plot, output_directory.joinpath(f"{type}_plot.png"))

def hist(df: pl.DataFrame):
    plot = sb.histplot(df, x="distance", stat="probability", element="bars", bins=30)
    return plot

def cdf(df: pl.DataFrame):
    plot = sb.ecdfplot(df, x="distance", log_scale=(True, True))
    return plot

def ccdf(df: pl.DataFrame):
    plot = sb.ecdfplot(df, x="distance", complementary=True, log_scale=(True, True))
    return plot

def kde(df: pl.DataFrame):
    df = df.sample(fraction=1.0, shuffle=True)
    plot = sb.kdeplot(df, x="distance", log_scale=(True, True))
    return plot