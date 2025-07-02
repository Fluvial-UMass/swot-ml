import equinox as eqx
from jaxtyping import Array, PRNGKeyArray


class BaseModel(eqx.Module):
    head: eqx.nn.Linear
    target: list[str]

    def __init__(self, hidden_size: int, target: list, *, key: PRNGKeyArray):
        if not isinstance(target, list):
            raise ValueError(f"target must be a list of string(s) but got {type(target)}")

        self.head = eqx.nn.Linear(hidden_size, len(target), key=key)
        self.target = target

    def __call__(
        self, data: dict[str, Array | dict[str, Array]], key: PRNGKeyArray | None
    ) -> Array:
        raise NotImplementedError

    def finetune_update(self, **kwargs):
        raise NotImplementedError
