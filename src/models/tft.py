from typing import Dict, List, Union, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

from models.lstm import EALSTM

class GatedLinearUnit(eqx.Module):
    gates: eqx.nn.Linear
    linear: eqx.nn.Linear

    def __init__(self, input_size: int, output_size: int, *, key):
        keys = jrandom.split(key)
        self.gates = eqx.nn.Linear(input_size, output_size, key=keys[0])
        self.linear = eqx.nn.Linear(input_size, output_size, key=keys[1])

    def __call__(self, gamma):
        gates = jax.nn.sigmoid(self.gates(gamma))
        return gates * self.linear(gamma)
    
class GatedSkipLayer(eqx.Module):
    glu: GatedLinearUnit
    layer_norm: eqx.nn.LayerNorm

    def __init__(self, layer_size, *, key):
        self.glu = GatedLinearUnit(layer_size, layer_size, key=key)
        self.layer_norm = eqx.nn.LayerNorm(layer_size)

    def __call__(self, layer_input, layer_output):
        gated_output = self.glu(layer_output)
        return self.layer_norm(layer_input + gated_output)

class GatedResidualNetwork(eqx.Module):
    eta2_dynamic: eqx.nn.Linear
    eta2_static: eqx.nn.Linear
    eta1_linear: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    skip: GatedSkipLayer

    def __init__(self, grn_size, context_size=None, *, dropout=0, key):
        if isinstance(grn_size, tuple):
            input_size, hidden_size, output_size = grn_size
        elif isinstance(grn_size, int):
            input_size = hidden_size = output_size = grn_size
        else:
            raise ValueError("grn_size must either be a tuple or int for input, hidden, and output sizes")    
        keys = jax.random.split(key, 4)

        self.eta2_dynamic = eqx.nn.Linear(input_size, hidden_size, use_bias=True, key=keys[0])
        if context_size is not None:
            self.eta2_static = eqx.nn.Linear(context_size, hidden_size, use_bias=False, key=keys[1])
        else:
            self.eta2_static = None

        self.eta1_linear = eqx.nn.Linear(hidden_size, output_size, key=keys[2])
        self.dropout = eqx.nn.Dropout(dropout)
        self.skip = GatedSkipLayer(hidden_size, key=keys[3])

    def __call__(self, input:jnp.ndarray, context:jnp.ndarray, key) -> jnp.ndarray:
        if self.eta2_static and context is not None:
            context_term = self.eta2_static(context)
        elif self.eta2_static or context is not None:
            raise ValueError("Either context weights were created and no context was passed during call, " +
                             "or context was passed during call with no context weights created during init." +
                             f"\nweights:{self.eta2_static}\ncontext:{context}")
        else:
            context_term = 0

        eta2 = jax.nn.elu(self.eta2_dynamic(input) + context_term)
        eta1 = self.eta1_linear(eta2)
        eta1 = self.dropout(eta1, key=key)
        output = self.skip(input, eta1)

        return output
    

class VariableSelectionNetwork(eqx.Module):
    variable_transformers: dict
    variable_processors: dict
    weights_grn: GatedResidualNetwork
    
    def __init__(self, variable_sizes:dict, hidden_size, context_size=None, dropout=0, key=None):
        keys = jax.random.split(key, 4)   
        num_variables = len(variable_sizes)

        # Transformers project each variable into the model dimension
        transformer_keys = jax.random.split(keys[0], num_variables)
        self.variable_transformers = {
            var_name: eqx.nn.Linear(size, hidden_size, key=k)
            for (var_name, size), k in zip(variable_sizes.items(), transformer_keys)
        } # {var0: linear, ..., varn: linear}

        # Non linear processing for each variable 
        processor_keys = jax.random.split(keys[1], num_variables)
        self.variable_processors = {
            var_name: GatedResidualNetwork(hidden_size, dropout=dropout, key=k)
            for (var_name, _), k in zip(variable_sizes.items(), processor_keys)
        } # {var0: GRN, ..., varn: GRN}

        # GRN for learning per-variable hidden_size weights.
        weights_size = num_variables*hidden_size
        self.weights_grn = GatedResidualNetwork(weights_size, context_size, dropout=dropout, key=keys[2])
        
    
    def __call__(self, inputs:dict, context=None, *, key):
        keys = jrandom.split(key, 2)
        
        transformed_inputs = jnp.stack([
            jax.vmap(processor)(inputs[var_name])
            for var_name, processor in self.variable_transformers.items()
        ], axis=1) # (seq_length, num_variables, hidden_size)
        seq_length, num_variables, _ = transformed_inputs.shape

        # Process each variable
        proc_keys = jrandom.split(keys[0], num_variables)
        processed_inputs = jnp.stack([
            jax.vmap(processor, in_axes=(0, None, 0))(
                transformed_inputs[:,i], None, jrandom.split(proc_keys[i], seq_length))
            for i, (_, processor) in enumerate(self.variable_processors.items())
        ], axis=1) # array (seq_length, num_variables, hidden_size)

        # Generate variable selection weights 
        flattened = processed_inputs.reshape([seq_length,-1]) # (seq_length, num_variables * hidden_size)
        flat_weights = jax.vmap(self.weights_grn, in_axes=(0, None, 0))(
            flattened, context, jrandom.split(keys[1], seq_length))
        flat_weights = jax.nn.softmax(flat_weights, axis=-1)
        variable_weights = flat_weights.reshape(transformed_inputs.shape) # (seq_length, num_variables, hidden_size)
        
        # Weight and sum the processed inputs across the variable axis
        weighted_inputs = variable_weights * processed_inputs
        output = jnp.sum(weighted_inputs, axis=1) # (seq_length, hidden_size)
    
        return output


class TemporalFusionTransformer(eqx.Module):
    dynamic_variables: list
    missing_data_tokens: dict
    static_context_vsn: VariableSelectionNetwork
    dynamic_vsn_context_encoder: GatedResidualNetwork
    dynamic_vsn: VariableSelectionNetwork
    lstm_context_encoder: GatedResidualNetwork
    lstm_encoder: EALSTM
    lstm_skip: GatedSkipLayer
    enrichment_context_encoder: GatedResidualNetwork
    enrichment_grn: GatedResidualNetwork
    mhattention: eqx.nn.MultiheadAttention
    attention_skip: GatedSkipLayer
    feed_forward: GatedResidualNetwork
    decoder_skip: GatedSkipLayer
    dense: eqx.nn.Linear
    target: list

    def __init__(self,
                 target: list,
                 dynamic_sizes: dict, 
                 static_size: int,
                 hidden_size: int,
                 num_heads: int, 
                 dropout: float, 
                 *, 
                 key):
        keys = list(jrandom.split(key, 13))
        self.dynamic_variables = list(dynamic_sizes.keys())
        self.missing_data_tokens = {k: jnp.zeros(v) for k, v in dynamic_sizes.items()} # Learnable per-feature tokens.
        self.static_context_vsn = VariableSelectionNetwork({'static': static_size}, hidden_size, dropout=dropout, key=keys.pop())

        self.dynamic_vsn_context_encoder = GatedResidualNetwork(hidden_size, dropout=dropout, key=keys.pop())
        self.dynamic_vsn = VariableSelectionNetwork(dynamic_sizes, hidden_size, hidden_size, dropout=dropout, key=keys.pop())

        self.lstm_context_encoder = GatedResidualNetwork(hidden_size, dropout=dropout, key=keys.pop())
        self.lstm_encoder = EALSTM(hidden_size, hidden_size, hidden_size, None, dropout, return_all=True, key=keys.pop())
        self.lstm_skip = GatedSkipLayer(hidden_size, key=keys.pop()) #Shared across time
        
        self.enrichment_context_encoder = GatedResidualNetwork(hidden_size, dropout=dropout, key=keys.pop())
        self.enrichment_grn = GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout, key=keys.pop()) #Shared across time

        self.mhattention = eqx.nn.MultiheadAttention(num_heads, hidden_size, dropout_p=dropout, key=keys.pop())
        self.attention_skip = GatedSkipLayer(hidden_size, key=keys.pop())

        self.feed_forward = GatedResidualNetwork(hidden_size, dropout=dropout, key=keys.pop()) #Shared across time
        self.decoder_skip = GatedSkipLayer(hidden_size, key=keys.pop())
        
        self.dense = eqx.nn.Linear(hidden_size, len(target), key=keys.pop())
        self.target = target

    def __call__(self, 
                 data: Dict[str, jnp.ndarray], 
                 key, 
                 inspect=False) -> jnp.ndarray:
        keys = list(jrandom.split(key, 12))

        # Replace missing data with the learned missing data token
        dynamic_data = {} #{key0:(seq_len, dynamic_sizes[key0]) ... keyn:(seq_len, dynamic_sizes[keyn])}
        for k in self.dynamic_variables:
            d = data[k]
            mask = jnp.isnan(d)
            d = jnp.where(mask, self.missing_data_tokens[k], d)
            dynamic_data[k] = d
        
        # Static variable selection
        static_data = {'static': data['x_s'][jnp.newaxis,:]}
        static_vars = self.static_context_vsn(static_data, key=keys.pop()) #(1, hidden_size)

        # Dynamic variable selection
        dynamic_vsn_context = self.dynamic_vsn_context_encoder(static_vars[0,:], None, keys.pop())
        dynamic_vars = self.dynamic_vsn(dynamic_data, dynamic_vsn_context, key=keys.pop()) #(seq_len, hidden_size)
        seq_length = dynamic_vars.shape[0]

        # LSTM encoding
        lstm_context = self.lstm_context_encoder(static_vars[0,:], None, keys.pop())
        lstm_output = self.lstm_encoder(dynamic_vars, lstm_context, keys.pop())
        lstm_skip = jax.vmap(self.lstm_skip)(dynamic_vars, lstm_output)

        enrichment_context = self.enrichment_context_encoder(static_vars[0,:], None, keys.pop())
        enriched = jax.vmap(self.enrichment_grn, in_axes=(0, None, 0))(
            lstm_skip, enrichment_context, jrandom.split(keys.pop(), seq_length))

        self_attn = self.mhattention(enriched, enriched, enriched, key=keys.pop())
        attn_skip = jax.vmap(self.attention_skip)(enriched, self_attn)

        decoder_out = jax.vmap(self.feed_forward, in_axes=(0, None, 0))(
            attn_skip, None, jrandom.split(keys.pop(), seq_length))
        decoder_skip = jax.vmap(self.decoder_skip)(lstm_skip, decoder_out)

        out = self.dense(decoder_skip[-1,:])
        if inspect:
            return (
                static_vars, dynamic_vsn_context, dynamic_vars, lstm_context, 
                lstm_output, lstm_skip, enrichment_context, enriched, 
                self_attn, attn_skip, decoder_out, decoder_skip, out)
        else:
            return out

class TFT(eqx.Module):
    tft: TemporalFusionTransformer
    target: list
    
    def __init__(self, *, target, dynamic_sizes, static_size, hidden_size, num_heads, dropout, seed):
        key = jax.random.PRNGKey(seed)
        self.tft = TemporalFusionTransformer(
            target = target,
            dynamic_sizes=dynamic_sizes,
            static_size=static_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            key=key
        )
        self.target = target


    def __call__(self, data, key, inspect=False):
        return self.tft(data, key, inspect)
    





class StaticContextEncoder(eqx.Module):
    static_vsn: VariableSelectionNetwork
    dynamic_vsn_encoder: GatedResidualNetwork
    lstm_encoder: GatedResidualNetwork
    enrichment_encoder: GatedResidualNetwork

    def __init__(self, static_size, hidden_size, dropout, key):
        keys = jrandom.split(key, 4)
        self.static_vsn = VariableSelectionNetwork({'static': static_size}, hidden_size, dropout=dropout, key=keys[0])
        self.dynamic_vsn_encoder = GatedResidualNetwork(hidden_size, dropout=dropout, key=keys[1])
        self.lstm_encoder = GatedResidualNetwork(hidden_size, dropout=dropout, key=keys[2])
        self.enrichment_encoder = GatedResidualNetwork(hidden_size, dropout=dropout, key=keys[3])

    def __call__(self, static_data):
        keys = jrandom.split(key, 4)
        static_data = {'static': data['x_s'][jnp.newaxis,:]}
        static_vars = self.static_vsn(static_data, key=keys[0]) #(1, hidden_size)
        dynamic_vsn_context = self.dynamic_vsn_encoder(static_vars[0,:], None, keys[1])
        lstm_context = self.lstm_encoder(static_vars[0,:], None, keys[2])
        enrichment_context = self.enrichment_encoder(static_vars[0,:], None, keys[3])
        

    
class TemporalFusionEncoder(eqx.Module):
    dynamic_vsn: VariableSelectionNetwork
    lstm_encoder: EALSTM
    lstm_skip: GatedSkipLayer

    def __init__(self, dynamic_sizes, hidden_size, dropout, key):
        keys = jrandom.split(key, 3)
        self.dynamic_vsn = VariableSelectionNetwork(dynamic_sizes, hidden_size, hidden_size, dropout=dropout, key=keys[0])
        self.lstm_encoder = EALSTM(hidden_size, hidden_size, hidden_size, None, dropout, return_all=True, key=keys[1])
        self.lstm_skip = GatedSkipLayer(hidden_size, key=keys[2]) #Shared across time

class TemporalFusionDecoder(eqx.Module):
    enrichment_grn: GatedResidualNetwork
    mhattention: eqx.nn.MultiheadAttention
    attention_skip: GatedSkipLayer
    feed_forward: GatedResidualNetwork
    decoder_skip: GatedSkipLayer

    def __init__(self, hidden_size, num_heads, dropout, key):
        keys = jrandom.split(key, 5)
        self.enrichment_grn = GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout, key=keys[0]) #Shared across time
        self.mhattention = eqx.nn.MultiheadAttention(num_heads, hidden_size, dropout_p=dropout, key=keys[1])
        self.attention_skip = GatedSkipLayer(hidden_size, key=keys[2])
        self.feed_forward = GatedResidualNetwork(hidden_size, dropout=dropout, key=key[3]) #Shared across time
        self.decoder_skip = GatedSkipLayer(hidden_size, key=keys[4])

class TemporalFusionTransformer_take2(eqx.Module):
    """
    https://arxiv.org/pdf/1912.09363
    """
    missing_data_tokens: dict
    static_encoder: StaticContextEncoder
    encoder: TemporalFusionEncoder
    decoder: TemporalFusionDecoder
    dense: eqx.nn.Linear
    target: list

    def __init__(self,
                 target: list,
                 dynamic_sizes: dict, 
                 static_size: int,
                 hidden_size: int,
                 num_heads: int, 
                 dropout: float, 
                 *, 
                 key):
        keys = jrandom.split(key, 4)
        self.missing_data_tokens = {k: jnp.zeros(v) for k, v in dynamic_sizes.items()} # Learnable per-feature tokens.
        self.static_encoder = StaticContextEncoder(static_size, hidden_size, dropout, keys[0])
        self.encoder = TemporalFusionEncoder(dynamic_sizes,hidden_size,dropout, keys[1])
        self.decoder = TemporalFusionDecoder(hidden_size, num_heads, dropout, keys[2])
        self.dense = eqx.nn.Linear(hidden_size, len(target), key=keys[3])
        self.target = target

    def __call__(self, 
                 data: Dict[str, jnp.ndarray], 
                 key) -> jnp.ndarray:
        keys = list(jrandom.split(key, 12))

        # Replace missing data with the learned missing data token
        dynamic_data = {} #{key0:(seq_len, dynamic_sizes[key0]) ... keyn:(seq_len, dynamic_sizes[keyn])}
        for k in self.dynamic_variables:
            d = data[k]
            mask = jnp.isnan(d)
            d = jnp.where(mask, self.missing_data_tokens[k], d)
            dynamic_data[k] = d

        