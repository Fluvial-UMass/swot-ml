import equinox as eqx
from jaxtyping import Array, PRNGKeyArray

from data import GraphBatch
from .layers import heads


class BaseModel(eqx.Module):
    head: eqx.nn.Linear
    target: list[str]

    def __init__(self, hidden_size: int, target: list, head: str, *, key: PRNGKeyArray):
        if not isinstance(target, list):
            raise ValueError(f"target must be a list but got: {type(target)}")
        are_strings = [isinstance(t, str) for t in target]
        if not all(are_strings):
            raise ValueError(
                f"target must be a list of string(s) but got types: {[type(t) for t in target]}"
            )
        self.target = target

        match head.lower():
            case 'linear':
                single_head = heads.Linear(hidden_size, key=key)
            case 'mlp':
                single_head = heads.MLP(hidden_size, hidden_size * 2, key=key)
            case 'gmm': 
                single_head = heads.GMM(hidden_size, hidden_size * 2, 100, key=key)
            case 'cmal': 
                single_head = heads.CMAL(hidden_size, hidden_size * 2, 100, key=key)
            case 'umal': 
                single_head = heads.UMAL(hidden_size, hidden_size * 2, 100, key=key)
            case _:
                raise NotImplementedError(f"{head} not implemented or not linked in `get_head()`")
            
        # All heads are initialized as the same but will diverge during training. 
        self.head = {t:single_head for t in target}

        

    def __call__(self, data: GraphBatch, key: PRNGKeyArray | None) -> Array:
        raise NotImplementedError

    def finetune_update(self, **kwargs):
        raise NotImplementedError
