import pickle
from pathlib import Path

from endgame.ncaabb import NcaabbGender

from .shot_model import build_shot_model, ShotModel

_DATA = Path(__file__).parent.parent / "data"
_MODEL_FILE_NAME = "shot_model.pkl"


def save_shot_model(league: str) -> None:
    model = build_shot_model(NcaabbGender[league])
    output_path = _DATA / league / _MODEL_FILE_NAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(pickle.dumps(model))


def load_shot_model(league: NcaabbGender) -> ShotModel:
    with (_DATA / league.name / _MODEL_FILE_NAME).open("rb") as file:
        return pickle.load(file)
