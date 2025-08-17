from abc import ABC, abstractmethod
from pathlib import Path

import polars as pl

from ..training import Parameter
from ..trip import TripContainer

class GenericSearch(ABC):

    def __init__(self, model, desired: TripContainer, parameters: dict[str, tuple[float, float, float]], csv_path: Path | None = None):
        self.model = model
        self.real_data = desired
        self.parameters: dict[str, Parameter] = {}
        self.metrics: dict[str, float] = { "chi" : None, "kss": None }
        for name, value in parameters.items():
            self.parameters[name] = Parameter(name, value[2], value[0], value[1])

        self.csv_path = csv_path
        self._df = None
        if self.csv_path:
            cols = list(parameters.keys()) + list(self.metrics.keys())
            self._df = pl.DataFrame(
                {col: pl.Series(name=col, values=[], dtype=pl.Float64) for col in cols}
            )

    @abstractmethod
    def train(self, iterations: int = 100, accuracy: float = -1.0, metric: str = "chi"):
        pass

    def apply(self):
        for name, param in self.parameters.items():
            setattr(self.model, name, param.value)
        self.model.recreate_matrix()
        self.save_parameter_map()

    def add_parameter_map_point(self, params: dict[str, float], metrics: dict[str, float]):
        """
        Store a row of parameter values + metric values into the internal DataFrame.
        Will save to CSV immediately if a csv_path is provided.
        """
        if not self.csv_path:
            return

        # Ensure the row matches DataFrame columns
        row = {**params, **metrics}
        # Append to DataFrame
        new_row = pl.DataFrame([row]).cast(self._df.schema)
        self._df = pl.concat([self._df, new_row], how="vertical")

    def save_parameter_map(self):
        """Save the internal DataFrame to CSV."""
        if not self.csv_path:
            return
        self._df.write_csv(self.csv_path)