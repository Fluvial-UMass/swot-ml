import equinox as eqx
from jaxtyping import Array, PRNGKeyArray

import jax.random as jrandom

# from .gates import GatedResidualNetwork


class StaticEmbedder(eqx.Module):
    proj: eqx.nn.Linear
    # grn: GatedResidualNetwork

    def __init__(self, in_size: int, hidden_size: int, dropout: float, key: PRNGKeyArray):
        keys = jrandom.split(key, 2)

        self.proj = eqx.nn.Linear(in_size, hidden_size, key=keys[0])
        # self.grn = GatedResidualNetwork(hidden_size, None, dropout=dropout, key=keys[1])

    def __call__(self, static: Array, key: PRNGKeyArray):
        embed = self.proj(static)
        # encoded = self.grn(embed, None, key)
        return embed
