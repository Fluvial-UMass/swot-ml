from importlib import reload
import models.lstm
reload(models.lstm)
import models.transformer
reload(transformer)
# Above for testing while changing models scripts.

from .lstm import EALSTM, TEALSTM, TAPLSTM
from .transformer import EATransformer

def make(cfg: dict):
    name = cfg['model'].lower()
    
    if name == "ealstm":
        model = EALSTM(**cfg['model_args'])
    elif name == "tealstm":
        model = TEALSTM(**cfg['model_args'])
    elif name == "taplstm":
        model = TAPLSTM(**cfg['model_args'])
    elif name == "eatransformer":
        model = EATransformer(**cfg['model_args'])
    else:
        raise ValueError(f"{cfg['model']} is not a valid model name. Check /src/models/__init__.py for model config.")

    return model
    