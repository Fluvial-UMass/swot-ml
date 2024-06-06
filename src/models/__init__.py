from importlib import reload
import models.lstm
reload(models.lstm)
import models.transformer
reload(transformer)
# Above for testing while changing models scripts.

from .lstm import EALSTM, TEALSTM, TAPLSTM
from .transformer import EATransformer
from .hybrid import Hybrid

def make(cfg: dict):
    name = cfg['model'].lower()
    
    if name == "ealstm":
        model_fn = EALSTM
    elif name == "tealstm":
        model_fn = TEALSTM
    elif name == "taplstm":
        model_fn = TAPLSTM
    elif name == "eatransformer":
        model_fn = EATransformer
    elif name == "hybrid":
        model_fn = Hybrid
    else:
        raise ValueError(f"{cfg['model']} is not a valid model name. Check /src/models/__init__.py for model config.")

    return model_fn(**cfg['model_args'])
    