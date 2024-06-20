from typing import Dict, List, Union, Tuple

import equinox as eqx
import optax
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from importlib import reload
import models.lstm, models.transformer
reload(models.lstm)
reload(models.transformer)

from .lstm import BaseLSTM, EALSTMCell, TEALSTMCell
from .transformer import StaticEmbedder, CrossAttnDecoder

class EALSTM(BaseLSTM):
    """
    Entity-Aware LSTM (TEALSTM) model for processing time series data with dynamic and static features.
    """
    def __init__(self, dynamic_in_size, static_in_size, hidden_size, dropout, *, key):
        super().__init__(dynamic_in_size, hidden_size, None, dropout, key=key)
        self.cell = EALSTMCell(dynamic_in_size, static_in_size, hidden_size, key=key)

    def __call__(self, data, key):
        """
        Forward pass of the EALSTM module.

        Args:
            data (dict): Contains at least these two keys:
                x_d (jax.Array): Dynamic input features.
                x_s (jax.Array): Static input features.

        Returns:
            Output of the TEALSTM module and final skip count.
        """
        # Input gate is based on static watershed features
        i = jax.nn.sigmoid(self.cell.input_linear(data['x_s']))
        
        def scan_fn(state, x_d):
            new_state = self.cell(state, x_d, i)
            # Return new_state for the next step, and only the hidden state for accumulation
            return new_state, new_state[0]

        init_state = (jnp.zeros(self.hidden_size), jnp.zeros(self.hidden_size))
        _, all_states = jax.lax.scan(scan_fn, init_state, data['x_dd'])

        return all_states
    

class TEALSTM(BaseLSTM):
    """
    Time- and Entity-Aware LSTM (TEALSTM) model for processing time series data with
    dynamic and static features.
    """
    
    def __init__(self, dynamic_in_size, static_in_size, hidden_size, dropout, *, key):
        super().__init__(dynamic_in_size, hidden_size, None, dropout, key=key)
        self.cell = TEALSTMCell(dynamic_in_size, static_in_size, hidden_size, key=key)

    def __call__(self, data, key):
        """
        Forward pass of the TEALSTM module.

        Args:
            data (dict): Contains at least these two keys:
                x_d (jax.Array): Dynamic input features.
                x_s (jax.Array): Static input features.

        Returns:
            Output of the TEALSTM module and final skip count.
        """
        # Input gate is based on static watershed features
        i = jax.nn.sigmoid(self.cell.input_linear(data['x_s']))
        
        def scan_fn(state, x_d):
            skip_count = state[2]
            new_state, skip_count = self.cell(state[:2], x_d, i, skip_count)
            return (*new_state, skip_count), new_state[0]

        init_state = (jnp.zeros(self.hidden_size), jnp.zeros(self.hidden_size), int(0))
        _, all_states = jax.lax.scan(scan_fn, init_state, data['x_di'])

        return all_states


class Hybrid(eqx.Module):
    ealstm_d: EALSTM
    tealstm_i: TEALSTM
    static_embedder: StaticEmbedder
    decoder: CrossAttnDecoder
    head: eqx.nn.Linear

    def __init__(self, *, daily_in_size, irregular_in_size, static_in_size, out_size, seq_length, hidden_size, num_layers, num_heads, seed, dropout):
        key = jax.random.PRNGKey(seed)
        keys = jrandom.split(key, 5)
        self.ealstm_d = EALSTM(daily_in_size, static_in_size, hidden_size, dropout, key=keys[0])
        self.tealstm_i = TEALSTM(irregular_in_size, static_in_size, hidden_size, dropout, key=keys[1])
        
        self.static_embedder = StaticEmbedder(seq_length, static_in_size, num_heads, dropout, keys[2])
        self.decoder = CrossAttnDecoder(hidden_size, hidden_size, num_layers, num_heads, dropout, keys[3])
        
        #have to make seperate heads in a list rather than wider output dim so that we can selectively freeze them.
        self.head = eqx.nn.Linear(in_features=hidden_size, out_features=out_size, key=keys[4])

    def __call__(self, data, key):
        keys = keys = jrandom.split(key, 4)
        
        d_encoded = self.ealstm_d(data, keys[0])
        i_encoded = self.tealstm_i(data, keys[1])
        
        logit_bias = self.static_embedder(data['x_s'], keys[2])
        mask = ~jnp.any(jnp.isnan(data['x_di']),axis=1)
        pooled_output = self.decoder(d_encoded, i_encoded, logit_bias, mask, keys[3])
        
        return self.head(pooled_output)







