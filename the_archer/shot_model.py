import pandas as pd
import numpy as np
import numpy.typing as npt
from endgame.ncaabb import NcaabbGender
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline, make_pipeline


from .constants import get_data_dir, HOOP_X, HOOP_Y
from .columns import Cols
from .grid import MakeGrid


def _is_all_empty(shot_df: pd.DataFrame) -> bool:
    return ((shot_df[Cols.X] == HOOP_X) & (shot_df[Cols.Y] == HOOP_Y)).all()


def _read_real_shots(path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # TODO: consider pushing this transform upward, so it can be stored as a bool
    df = df.assign(
        shootingPlay=lambda x: x.shootingPlay.fillna(False).astype(bool),
        scoringPlay=lambda x: x.scoringPlay.fillna(False).astype(bool),
        isFreeThrow=lambda x: x.isFreeThrow.fillna(False).astype(bool),
    )
    is_real_shot = (
        df["shootingPlay"]
        & df[Cols.X].notnull()
        & df[Cols.Y].notnull()
        # There's some at "min float" or something like that, lop them off
        & np.greater(df[Cols.X], -10_000)
        & np.greater(df[Cols.Y], -10_000)
        & ~df.isFreeThrow
    )
    real_shots = df[is_real_shot]

    # Drop games where all the shots locations are _HOOP, like 401600415
    # Should probably drop these before saving (since we have each game on its own)
    game_is_empty = real_shots.groupby("game_id").apply(_is_all_empty)
    game_ids_to_drop = game_is_empty[game_is_empty].index
    return real_shots[~real_shots.game_id.isin(game_ids_to_drop)]


def build_shot_model(league: NcaabbGender) -> ShotModel:
    data_root = get_data_dir(league)
    parquets = list(data_root.glob("*.parquet"))
    grids = MakeGrid()
    for path in tqdm(parquets):
        real_shots = _read_real_shots(path)
        grids.add_shot_df(real_shots)
    jump_shot_model = _build_model(grids)
    return ShotModel(jump_shot_model, grids.special_probs)


class ShotModel:
    def __init__(self, jump_shot: Pipeline, special_types: dict[str, float]):
        self._jump_shot = jump_shot
        self._special_types = special_types

    def predict(self, df: pd.DataFrame) -> npt.NDArray[np.float64]:
        outputs = np.full(len(df), fill_value=np.nan)
        is_special = df[Cols.SHOT_TYPE].isin(self._special_types.keys())

        distances = np.linalg.norm(df[[Cols.X, Cols.Y]].values, axis=-1)[~is_special]
        X = distances[:, np.newaxis]
        outputs[~is_special] = self._jump_shot.predict_proba(X)[:, 1]

        for shot_type, make_prob in self._special_types.items():
            outputs[df[Cols.SHOT_TYPE] == shot_type] = make_prob
        return outputs


def _build_model(grid: MakeGrid) -> Pipeline:
    # --- 1. Flatten and filter empty cells ---
    distance_flat = grid.distance_grid.ravel()
    makes_flat = grid.make_grid.ravel().astype(int)

    # Zero out long distances,
    # because they're either weird luck or incorrectly tagged on the opposite side.
    # Without this, we get "high" make probabilities from way deep
    makes_flat[distance_flat > 35] = 0

    attempts_flat = grid.attempt_grid.ravel().astype(int)

    mask = attempts_flat > 0
    distance_flat = distance_flat[mask]
    makes_flat = makes_flat[mask]
    attempts_flat = attempts_flat[mask]

    # --- 2. Expand counts into individual shot records ---
    # Each cell contributes `attempts` rows: `makes` ones and `misses` zeros
    misses_flat = attempts_flat - makes_flat

    X = np.repeat(distance_flat, attempts_flat).reshape(-1, 1)
    y = np.zeros(len(X), dtype=bool)
    i = 0
    for makes, misses in zip(makes_flat, misses_flat, strict=True):
        y[i : i + makes] = True
        i += makes + misses

    # Shuffle so makes/misses aren't block-ordered (matters for CV)
    rng = np.random.default_rng(1989)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    # --- 3. Fit ---
    model = make_pipeline(
        PolynomialFeatures(degree=3),
        LogisticRegression(max_iter=1000),
    )
    model.fit(X, y)
    return model
