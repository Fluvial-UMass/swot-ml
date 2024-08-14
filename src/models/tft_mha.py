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
    


class StaticEmbedder(eqx.Module):
    proj: eqx.nn.Linear
    grn: GatedResidualNetwork

    def __init__(self, in_size, hidden_size, dropout, key):
        keys = jrandom.split(key, 2)

        self.proj = eqx.nn.Linear(in_size, hidden_size, key=keys[0])
        self.grn = GatedResidualNetwork(hidden_size, None, dropout=dropout, key=keys[1])

    def __call__(self, static, key):
        embed = self.proj(static)
        encoded = self.grn(embed, None, key)
        return encoded
    
class StaticContextHeadBias(eqx.Module):
    out_shape: tuple
    linear: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    layernorm: eqx.nn.LayerNorm
    
    def __init__(self, hidden_size, num_heads, dropout, key):
        size_per_head = hidden_size // num_heads
        out_size = num_heads * size_per_head
        self.out_shape = (num_heads, hidden_size // num_heads)

        self.linear = eqx.nn.Linear(hidden_size, out_size, key=key)
        self.dropout = eqx.nn.Dropout(dropout)
        self.layernorm = eqx.nn.LayerNorm(self.out_shape)

    def __call__(self, data, key):
        embed = self.linear(data)
        embed = self.dropout(embed, key=key)
        embed = jnp.reshape(embed, self.out_shape)
        embed = self.layernorm(embed)
        return embed

class HybridEncoderBlock(eqx.Module):
    head_context: StaticContextHeadBias
    dynamic_proj: eqx.nn.Linear
    attn: eqx.nn.MultiheadAttention
    lstm: EALSTM
    skip: GatedSkipLayer

    def __init__(self, in_size, hidden_size, num_heads, dropout, key):
        keys = jrandom.split(key, 5)
        self.head_context = StaticContextHeadBias(hidden_size, num_heads, dropout, keys[0])
        self.dynamic_proj = eqx.nn.Linear(in_size, hidden_size, key=keys[1])
        self.attn = eqx.nn.MultiheadAttention(num_heads, hidden_size, dropout_p=dropout, key=keys[2])
        self.lstm = EALSTM(hidden_size, hidden_size, hidden_size, None, dropout, return_all=True, key=keys[3])
        self.skip = GatedSkipLayer(hidden_size, key=keys[4]) #Shared across time

    def __call__(self, dynamic, static, key):
        keys = jrandom.split(key, 3)

        embed = jax.vmap(self.dynamic_proj)(dynamic)
        mask_vec = ~jnp.any(jnp.isnan(dynamic),axis=1)
        mask = jnp.outer(mask_vec, mask_vec)
        # mask = jnp.tril(mask)

        head_context = self.head_context(static, keys[0])
        def process_heads(q_h, k_h, v_h):
            q_h += head_context
            k_h += head_context
            return q_h, k_h, v_h
        
        attn = self.attn(embed, embed, embed, mask, process_heads=process_heads, key=keys[1])
        lstm = self.lstm(attn, static, keys[2])
        skip = jax.vmap(self.skip)(attn, lstm)
        return skip
    
class HybridEncoder(eqx.Module):
    dynamic_blocks: Dict[str, HybridEncoderBlock]
    weights_grn: GatedResidualNetwork

    def __init__(self, dynamic_sizes, static_size, hidden_size, num_heads, dropout, key):
        keys = jrandom.split(key, 3)

        # Transformers project each variable into the model dimension
        block_keys = jax.random.split(keys[1], len(dynamic_sizes))
        self.dynamic_blocks = {
            var_name: HybridEncoderBlock(size, hidden_size, num_heads, dropout, key=k)
            for (var_name, size), k in zip(dynamic_sizes.items(), block_keys)
        }

        # GRN for learning per-variable hidden_size weights.
        weights_size = len(dynamic_sizes)*hidden_size
        self.weights_grn = GatedResidualNetwork(weights_size, hidden_size, dropout=dropout, key=keys[2])

    def __call__(self, dynamic_in: dict, static, key):
        keys = jrandom.split(key, 2)

        dyn_keys = jrandom.split(keys[0])
        dynamic = jnp.stack([
            encoder(dynamic_in[var_name], static, k)
            for (var_name, encoder), k in zip(self.dynamic_blocks.items(), dyn_keys)
        ], axis=1) # (seq_length, num_variables, hidden_size)
        seq_length  = dynamic.shape[0]

        # Generate variable selection weights 
        weight_keys = jrandom.split(keys[1], seq_length)
        flattened = dynamic.reshape([seq_length,-1]) # (seq_length, num_variables * hidden_size)
        flat_weights = jax.vmap(self.weights_grn, in_axes=(0, None, 0))(
            flattened, static, weight_keys)
        flat_weights = jax.nn.softmax(flat_weights, axis=-1)
        variable_weights = flat_weights.reshape(dynamic.shape) # (seq_length, num_variables, hidden_size)
        
        # Weight and sum the processed inputs across the variable axis
        weighted_inputs = variable_weights * dynamic
        dynamic = jnp.sum(weighted_inputs, axis=1) # (seq_length, hidden_size)

        return dynamic

class EnrichedDecoder(eqx.Module):
    enrichment_grn: GatedResidualNetwork
    head_context: StaticContextHeadBias
    attn: eqx.nn.MultiheadAttention
    attn_skip: GatedSkipLayer
    feed_forward: GatedResidualNetwork
    decoder_skip: GatedSkipLayer

    def __init__(self, hidden_size, num_heads, dropout, key):
        keys = jrandom.split(key, 6)
        self.enrichment_grn = GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout, key=keys[0]) #Shared across time
        self.head_context = StaticContextHeadBias(hidden_size, num_heads, dropout, keys[1])
        self.attn = eqx.nn.MultiheadAttention(num_heads, hidden_size, dropout_p=dropout, key=keys[2])
        self.attn_skip = GatedSkipLayer(hidden_size, key=keys[3])
        self.feed_forward = GatedResidualNetwork(hidden_size, dropout=dropout, key=keys[4]) #Shared across time
        self.decoder_skip = GatedSkipLayer(hidden_size, key=keys[5])

    def __call__(self, encoded, static, key):
        keys = jrandom.split(key, 4)
        seq_length = encoded.shape[0]

        enrich_keys = jrandom.split(keys[0], seq_length)
        enriched = jax.vmap(self.enrichment_grn, in_axes=(0, None, 0))(
            encoded, static, enrich_keys)
        
        head_context = self.head_context(static, keys[1])
        def process_heads(q_h, k_h, v_h):
            q_h += head_context
            k_h += head_context
            return q_h, k_h, v_h

        mask = None
        # mask = jnp.tril(jnp.zeros((seq_length, seq_length)))
        self_attn = self.attn(enriched, enriched, enriched, mask, process_heads=process_heads, key=keys[2])
        attn_skip = jax.vmap(self.attn_skip)(enriched, self_attn)

        decoder_out = jax.vmap(self.feed_forward, in_axes=(0, None, 0))(
            attn_skip, None, jrandom.split(keys[3], seq_length))
        decoder_skip = jax.vmap(self.decoder_skip)(encoded, decoder_out)

        return decoder_skip


class TFT_MHA(eqx.Module):
    static_embedder: StaticEmbedder
    dynamic_variables: list
    encoder: HybridEncoder
    decoder: EnrichedDecoder
    dense: eqx.nn.Linear
    target: list

    def __init__(self, *, target, dynamic_sizes, static_size, hidden_size, num_heads, dropout, seed):
        key = jax.random.PRNGKey(seed)
        keys = jrandom.split(key, 4)

        self.static_embedder = StaticEmbedder(static_size, hidden_size, dropout, keys[0])
        self.dynamic_variables = list(dynamic_sizes.keys())
        self.encoder = HybridEncoder(dynamic_sizes, static_size, hidden_size, num_heads, dropout, keys[1])
        self.decoder = EnrichedDecoder(hidden_size, num_heads, dropout, keys[2])
        self.dense = eqx.nn.Linear(hidden_size, len(target), key=keys[3])
        self.target = target

    def __call__(self, data, key):
        keys = jrandom.split(key, 2)

        static_embed = self.static_embedder(data['x_s'], keys[0])

        # Replace missing data with the learned missing data token
        dynamic_data = {} #{key0:(seq_len, dynamic_sizes[key0]) ... keyn:(seq_len, dynamic_sizes[keyn])}
        for k in self.dynamic_variables:
            d = data[k]
            mask = jnp.isnan(d)
            d = jnp.where(mask, 0, d)
            dynamic_data[k] = d
        
        dynamic_encoded = self.encoder(dynamic_data, static_embed, keys[0])
        decoded = self.decoder(dynamic_encoded, static_embed, keys[1])
        out = self.dense(decoded[-1,:])
        
        return out
    