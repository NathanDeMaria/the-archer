from pathlib import Path

import numpy as np
from endgame.ncaabb import NcaabbGender

_DATA_DIR = Path(__file__).parent / "data"


def get_data_dir(league: NcaabbGender) -> Path:
    return _DATA_DIR / league.name


HOOP_X = 25
HOOP_Y = 0
HOOP = np.array([HOOP_X, HOOP_Y])
