from pathlib import Path
from jsonpickle import encode, decode

from .basic import GravityModel
from .power import PowerGravityModel
from .doublepower import DoublePowerGravityModel
from .triplepower import TriplePowerGravityModel

def model_from_json(filename: Path) -> GravityModel | PowerGravityModel | DoublePowerGravityModel | TriplePowerGravityModel:
    with filename.open("r") as f:
        json = f.read()
    model = decode(json)
    if not isinstance(model, (GravityModel, PowerGravityModel, DoublePowerGravityModel, TriplePowerGravityModel)):
        raise ValueError(f"{filename.as_posix()} does not contain a recognized model! ({type(model)})")
    return model