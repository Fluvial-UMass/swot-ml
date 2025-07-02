import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

from .base_model import BaseModel
from .layers.lstm import LSTM
from .layers.bi_lstm import BiLSTM


class STACKED_LSTM(BaseModel):
    in_proj: eqx.nn.Linear
    static_proj: eqx.nn.Linear
    in_lstm: BiLSTM
    in_head: eqx.nn.Linear
    out_lstm: LSTM
    seq2seq: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        in_targets: list[str],
        out_targets: list[str],
        dynamic_size: int,
        static_size: int,
        hidden_size: int,
        seed: int,
        dropout: float,
        seq2seq: bool = False,
    ):
        self.seq2seq = seq2seq
        key = jrandom.PRNGKey(seed)
        keys = jrandom.split(key, 7)

        all_targets = in_targets + out_targets
        super().__init__(hidden_size, all_targets, key=keys[0])

        # Embedding layer for dynamic data
        self.in_proj = eqx.nn.Linear(dynamic_size, hidden_size, key=keys[1])

        # Embedding layer for static data if used.
        entity_aware = static_size > 0
        if entity_aware:
            self.static_proj = eqx.nn.Linear(static_size, hidden_size, key=keys[2])
            static_embed_size = hidden_size
        else:
            self.static_proj = None
            static_embed_size = 0

        # First LSTM layer
        self.in_lstm = BiLSTM(
            hidden_size + static_embed_size,
            hidden_size,
            dropout=dropout,
            return_all=True,
            key=keys[3],
        )
        self.in_head = eqx.nn.Linear(hidden_size * 2, len(in_targets), key=keys[4])

        # Second LSTM layer
        self.out_lstm = LSTM(
            hidden_size * 3,
            hidden_size,
            dropout=dropout,
            return_all=seq2seq,
            key=keys[5],
        )
        self.head = eqx.nn.Linear(hidden_size, len(out_targets), key=keys[6])

    def __call__(self, data: dict[str, Array | dict[str, Array]], key: PRNGKeyArray):
        keys = jrandom.split(key, 2)

        # Replace NaN values with 0s in the dynamic data
        x_d = jnp.nan_to_num(data["dynamic"]["era5"], nan=0.0)
        x = jax.vmap(self.in_proj)(x_d)

        if self.static_proj:
            x_s = self.static_proj(data["static"])
            x_s_tiled = jnp.tile(x_s, (x.shape[0], 1))
            x = jnp.concat([x, x_s_tiled], axis=1)

        ht_in = self.in_lstm(x, keys[0])
        ht_out = self.out_lstm(jnp.concat([x, ht_in], axis=-1), keys[1])

        if self.seq2seq:
            y_in = jax.vmap(self.in_head)(ht_in)
            y_out = jax.vmap(self.head)(ht_out)
        else:
            y_in = self.in_head(ht_in[-1, ...])
            y_out = self.head(ht_out[-1, ...])

        out = jnp.concat([y_in, y_out], axis=-1)
        return out
