from typing import Dict, List, Union, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from functools import partial

from models.lstm import EALSTM, TEALSTM
from models.transformer import StaticEmbedder, CrossAttnDecoder
    
class Hybrid(eqx.Module):
    static_embedder: StaticEmbedder
    ealstm_d: EALSTM
    tealstm_i: TEALSTM
    decoder: CrossAttnDecoder
    head: eqx.nn.Linear
    target: list

    def __init__(self, *, target, daily_in_size, irregular_in_size, static_in_size, hidden_size, num_layers, num_heads, seed, dropout):
        key = jax.random.PRNGKey(seed)
        keys = jrandom.split(key, 5)

        self.static_embedder = StaticEmbedder(static_in_size, hidden_size, dropout, keys[0])
        self.ealstm_d = EALSTM(daily_in_size, hidden_size, hidden_size, None, dropout, return_all=True, key=keys[1])
        self.tealstm_i = TEALSTM(irregular_in_size, hidden_size, hidden_size, None, dropout, return_all=True, key=keys[2])

        self.decoder = CrossAttnDecoder(hidden_size, hidden_size, num_layers, num_heads, dropout, keys[3])
        self.head = eqx.nn.Linear(in_features=hidden_size, out_features=len(target), key=keys[4])
        self.target = target

    # @partial(jax.checkpoint, static_argnums=(0,))
    def __call__(self, data, key):
        keys = keys = jrandom.split(key, 4)
        
        static = self.static_embedder(data['x_s'], keys[0])
        d_encoded = self.ealstm_d(data['x_dd'], static, keys[1])
        i_encoded, _ = self.tealstm_i(data['x_di'], static, keys[2])
        
        mask = ~jnp.any(jnp.isnan(data['x_di']),axis=1)
        pooled_output = self.decoder(d_encoded, i_encoded, static, mask, keys[3])
        
        return self.head(pooled_output)

