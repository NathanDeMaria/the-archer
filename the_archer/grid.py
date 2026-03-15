from collections import Counter

import pandas as pd
import numpy as np
import numpy.typing as npt

from .columns import Cols
from .constants import HOOP

_SPECIAL_TYPES = ["DunkShot", "LayUpShot", "TipShot"]


class MakeGrid:
    def __init__(self):
        # Grabbed these after running a handful
        x_endpoints = (-6, 57)
        y_endpoints = (-16, 101)

        x_length = x_endpoints[1] - x_endpoints[0] + 1
        y_length = y_endpoints[1] - y_endpoints[0] + 1
        self._make_grid = np.zeros((x_length, y_length), dtype=int)
        self._attempt_grid = np.zeros((x_length, y_length), dtype=int)
        self._x_min = x_endpoints[0]
        self._y_min = y_endpoints[0]

        # TODO: model these based on distance to the hoop.
        # I'm dropping them for now because the distance is very different than jump shots
        # TBH, I'm surprised with things like how low the dunk % is.
        self._special_makes = Counter()
        self._special_counts = Counter()

    def add_shot_df(self, shot_df: pd.DataFrame) -> None:
        for shot_type in _SPECIAL_TYPES:
            shots = shot_df[shot_df[Cols.SHOT_TYPE] == shot_type]
            self._special_makes[shot_type] += shots.scoringPlay.sum()
            self._special_counts[shot_type] += len(shots)

        for _, row in shot_df[shot_df[Cols.SHOT_TYPE] == "JumpShot"].iterrows():
            self._add_shot(row[Cols.X], row[Cols.Y], row["scoringPlay"])

    def _add_shot(self, x: int, y: int, made: bool) -> None:
        x_idx = x - self._x_min
        y_idx = y - self._y_min
        self._attempt_grid[x_idx, y_idx] += 1
        if made:
            self._make_grid[x_idx, y_idx] += 1

    @property
    def special_probs(self) -> dict[str, float]:
        return {
            shot_type: self._special_makes[shot_type] / self._special_counts[shot_type]
            for shot_type in _SPECIAL_TYPES
        }

    @property
    def distance_grid(self) -> npt.NDArray[np.float64]:
        # Create distance grid matching the shape of grids._make_grid
        shape = self._make_grid.shape
        x_coords = np.arange(self._x_min, self._x_min + shape[0])
        y_coords = np.arange(self._y_min, self._y_min + shape[1])
        bottom_left_grid = np.stack(
            np.meshgrid(x_coords, y_coords, indexing="ij"), axis=-1
        )
        return np.linalg.norm(bottom_left_grid - HOOP, axis=-1)

    @property
    def make_grid(self) -> npt.NDArray[np.int_]:
        return self._make_grid

    @property
    def attempt_grid(self) -> npt.NDArray[np.int_]:
        return self._attempt_grid
