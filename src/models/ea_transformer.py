import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

from .base_model import BaseModel
from .layers.transformer import SelfAttnEncoder, CrossAttnDecoder


class EA_TRANSFORMER(BaseModel):
    static_embedder: eqx.nn.Linear
    d_encoder: SelfAttnEncoder
    i_encoder: SelfAttnEncoder
    decoder: CrossAttnDecoder

    def __init__(
        self,
        target: list,
        daily_in_size: int,
        irregular_in_size: int,
        static_in_size: int,
        seq_length: int,
        hidden_size: int,
        intermediate_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        seed: int,
    ):
        key = jrandom.PRNGKey(seed)
        keys = jrandom.split(key, num=6)

        super().__init__(hidden_size, target, key=keys[0])

        self.static_embedder = eqx.nn.Linear(static_in_size, hidden_size, key=keys[1])

        static_args = (hidden_size, intermediate_size, num_layers, num_heads, dropout)
        self.d_encoder = SelfAttnEncoder(seq_length, daily_in_size, *static_args, keys[2])
        self.i_encoder = SelfAttnEncoder(seq_length, irregular_in_size, *static_args, keys[3])
        self.decoder = CrossAttnDecoder(seq_length, *static_args, keys[4])

    def __call__(self, data: dict, key: PRNGKeyArray) -> Array:
        keys = jrandom.split(key, num=4)

        # Kluge to get it running now. Will address data loader later.
        mask = ~jnp.any(jnp.isnan(data["x_di"]), axis=1)
        data["x_di"] = jnp.where(jnp.isnan(data["x_di"]), 0, data["x_di"])
        # Kluge to get it running now. Will address data loader later.

        static = self.static_embedder(data["x_s"], keys[0])
        d_encoded = self.d_encoder(data["x_dd"], static, None, keys[1])
        i_encoded = self.i_encoder(data["x_di"], static, mask, keys[2])
        pooled_output = self.decoder(d_encoded, i_encoded, static, mask, keys[3])

        return self.head(pooled_output)
