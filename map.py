#!/bin/python
from pathlib import Path
from typing import Iterable
from itertools import combinations

import click
import polars as pl
import seaborn as sb
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import minimum_filter

from gravity_model.log import logger

@click.command()
@click.argument("heatmap_output", metavar="[Metric Heatmap Output]", type=click.Path(dir_okay=True, path_type=Path))
@click.argument("metric_map_data", metavar="[Metric Map Data]", nargs=-1, type=click.Path(exists=True, readable=True, dir_okay=False, file_okay=True, path_type=Path))
@click.option("-p", "--parameter", multiple=True, type=str)
@click.option("-m", "--metric", multiple=True, type=str)
def main(heatmap_output: Path, metric_map_data: Iterable[Path], parameter: Iterable[str], metric: Iterable[str]):
    columns = parameter + metric
    logger.info(f"Columns: {columns}")
    metric_map_schema = {col: pl.Float64 for col in columns}
    metric_map_df = pl.DataFrame(schema=metric_map_schema)
    
    for metric_map_file in metric_map_data:
        logger.info(f"Loading metric mapping data from {metric_map_file.absolute().as_posix()}")
        temporary_df = pl.read_csv(metric_map_file, schema=metric_map_schema, has_header=True)
        metric_map_df = pl.concat([metric_map_df, temporary_df], rechunk=True, how="vertical")
    metric_map_df = metric_map_df.group_by(parameter).mean()
    logger.info(f"Successfully loaded {len(metric_map_data)} metric mapping files.")

    logger.info(f"Creating directory {heatmap_output.absolute().as_posix()}")
    heatmap_output.mkdir(exist_ok=True, parents=True)

    for currentMetric in metric:
        if len(parameter) > 1:
            for param_i, param_c in combinations(parameter, 2):
                x = metric_map_df[param_i].to_numpy()
                y = metric_map_df[param_c].to_numpy()
                z = metric_map_df[currentMetric].to_numpy()
                xi = np.linspace(x.min(), x.max(), 100)
                yi = np.linspace(y.min(), y.max(), 100)
                X, Y = np.meshgrid(xi, yi)
                Z = griddata((x, y), z, (X, Y), method='linear')
                Z = np.where(np.isnan(Z), np.inf, Z)

                neighborhood_min = minimum_filter(Z, size=3, mode='constant', cval=np.inf)
                local_min_mask = (Z == neighborhood_min) & np.isfinite(Z)

                x_min = X[local_min_mask]
                y_min = Y[local_min_mask]

                global_idx = np.unravel_index(np.argmin(Z), Z.shape)
                x_global = X[global_idx]
                y_global = Y[global_idx]
                z_global = Z[global_idx]

                fig = go.Figure()

                fig.add_trace(
                    go.Contour(
                        x=xi,
                        y=yi,
                        z=Z,
                        contours=dict(
                            showlines=False,
                            start=Z.min(),
                            end=Z.max(),
                            size=0.001 # step between levels
                        ),
                        colorscale='RdBu',
                    )
                )

                fig.add_trace(go.Scatter(
                    x=x_min,
                    y=y_min,
                    mode='markers',
                    marker=dict(color='red', size=5, symbol='circle'),
                    # name="Local Minima"
                ))

                # Global minimum (special marker)
                fig.add_trace(go.Scatter(
                    x=[x_global],
                    y=[y_global],
                    mode='markers+text',
                    marker=dict(color='gold', size=10, symbol='circle-open-dot'),
                    text=[f"Global Min {round(z_global, 5)}<br>{param_i}:{round(x_global, 3)} {param_c}:{round(y_global, 3)}"],
                    textposition="top center",
                    textfont=dict(size=12, color='black'),
                ))

                fig.write_image(
                    heatmap_output.joinpath(f"{currentMetric}_{param_i}_{param_c}.png"),
                    format="png",
                    width=1920,
                    height=1080,
                    scale=5
                )
        else:
            param: str = parameter[0]
            sorted_by_param_metric_map_df = metric_map_df.sort(param)
            x = sorted_by_param_metric_map_df[param].to_numpy()
            z = sorted_by_param_metric_map_df[currentMetric].to_numpy()
            fig = go.Figure(data =
                go.Scatter(
                    x=x,
                    y=z,
                    mode='lines+markers',
                    line=dict(color='blue'),
                    marker=dict(size=6)
                )
            )
            fig.write_image(
                heatmap_output.joinpath(f"{currentMetric}_{param}.png"),
                    format="png",
                    width=1920,
                    height=1080,
                    scale=5
            )


if __name__ == "__main__":
    main()