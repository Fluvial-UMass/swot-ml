import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray

from .static_mlp import StaticMLP

class MultiSourceObservationFuser(eqx.Module):
    """Fuse variable observations via cross-attention using equinox MHA."""
    
    hidden_size: int
    source_names: list[str]
    
    # Per-source embedding (projects raw obs to hidden space)
    source_embedders: dict[str, StaticMLP]
    
    # Learnable source embeddings
    source_embeddings: dict[str, Array]
    
    # Staleness projection (added to key)
    staleness_proj: eqx.nn.Linear
    
    # Cross-attention: h attends to observations
    cross_attn: eqx.nn.MultiheadAttention
    
    # Output norm
    norm: eqx.nn.LayerNorm
    
    def __init__(
        self,
        hidden_size: int,
        source_configs: dict[str, int],
        static_size: int,
        num_heads: int = 4,
        dropout_p: float = 0.0,
        *,
        key: PRNGKeyArray,
    ):
        self.hidden_size = hidden_size
        self.source_names = list(source_configs.keys())
        
        keys = list(jrandom.split(key, 4 + 2 * len(source_configs)))
        
        # Per-source embedders
        self.source_embedders = {}
        for name, feat_size in source_configs.items():
            self.source_embedders[name] = StaticMLP(
                dynamic_in_size=feat_size,
                static_in_size=static_size,
                out_size=hidden_size,
                width_size=hidden_size,
                depth=2,
                key=keys.pop(),
            )
        
        # Learnable source embeddings
        self.source_embeddings = {
            name: jrandom.normal(keys.pop(), (hidden_size,)) * 0.02
            for name in self.source_names
        }
        
        # Project staleness to hidden dim (added to keys)
        self.staleness_proj = eqx.nn.Linear(1, hidden_size, key=keys.pop())
        
        # Cross-attention
        # Query: hidden state [1, hidden_size]
        # Key/Value: observations [num_obs, hidden_size]
        self.cross_attn = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=hidden_size,
            key_size=hidden_size,
            value_size=hidden_size,
            output_size=hidden_size,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            dropout_p=dropout_p,
            key=keys.pop(),
        )
        
        self.norm = eqx.nn.LayerNorm(hidden_size)
    
    def embed_observations(
        self,
        raw_obs: dict[str, Array],
        staleness: dict[str, Array],
        static: Array,
    ) -> tuple[Array, Array, Array, list[str]]:
        """
        Embed all observations into unified tensors.
        
        Args:
            raw_obs: {source: [N, features]} raw observation features
            staleness: {source: [N]} steps since observation
            static: [N, static_size] static features
        
        Returns:
            obs_emb: [N, num_obs, hidden_size]
            staleness_emb: [N, num_obs, hidden_size]  
            valid_mask: [N, num_obs] bool
            source_order: list of source names in order
        """
        all_emb = []
        all_staleness = []
        all_masks = []
        source_order = []
        
        for name in self.source_names:
            if name not in raw_obs:
                continue
            
            features = raw_obs[name]  # [N, feat]
            stale = staleness[name]   # [N]
            
            # Validity mask
            valid = ~jnp.any(jnp.isnan(features), axis=-1)  # [N]
            safe_features = jnp.nan_to_num(features)
            
            # Embed: [N, hidden]
            embedder = self.source_embedders[name]
            emb = jax.vmap(lambda f, s: embedder(f, s))(safe_features, static)
            
            # Add source embedding
            emb = emb + self.source_embeddings[name]
            
            # Staleness embedding: [N, hidden]
            stale_emb = jax.vmap(self.staleness_proj)(stale[:, None])
            
            all_emb.append(emb)
            all_staleness.append(stale_emb)
            all_masks.append(valid)
            source_order.append(name)
        
        if len(all_emb) == 0:
            return None, None, None, []
        
        # Stack: [N, num_sources, hidden]
        obs_emb = jnp.stack(all_emb, axis=1)
        staleness_emb = jnp.stack(all_staleness, axis=1)
        valid_mask = jnp.stack(all_masks, axis=1)  # [N, num_sources]
        
        return obs_emb, staleness_emb, valid_mask, source_order
    
    def __call__(
        self,
        h: Array,                       # [N, hidden]
        raw_obs: dict[str, Array],      # {source: [N, features]}
        staleness: dict[str, Array],    # {source: [N]}
        static: Array,                  # [N, static_size]
        *,
        key: PRNGKeyArray = None,
    ) -> tuple[Array, dict]:
        """
        Fuse observations into hidden state.
        
        Args:
            h: Hidden state [N, hidden_size]
            raw_obs: Raw observations per source
            staleness: Staleness per source
            static: Static features for embedding
            key: PRNG key for dropout
        
        Returns:
            h_new: Updated hidden state [N, hidden_size]
            info: Dict with attention weights for interpretability
        """
        obs_emb, staleness_emb, valid_mask, source_order = self.embed_observations(
            raw_obs, staleness, static
        )
        
        # No observations - return unchanged
        if obs_emb is None:
            return h, {"attn_weights": None, "any_valid": jnp.zeros(h.shape[0], dtype=bool)}
        
        N, num_obs, d = obs_emb.shape
        
        # Keys incorporate staleness, values are pure observation embedding
        keys_input = obs_emb + staleness_emb  # [N, num_obs, hidden]
        values_input = obs_emb                 # [N, num_obs, hidden]
        
        # Process each location independently
        def attend_single(h_i, k_i, v_i, mask_i, key_i):
            """Cross-attention for single location."""
            # Query: [1, hidden] (hidden state as single query)
            # Key/Value: [num_obs, hidden]
            query = h_i[None, :]  # [1, hidden]
            
            # Expand mask for MHA: [1, num_obs] -> attend from 1 query to num_obs keys
            attn_mask = mask_i[None, :]  # [1, num_obs]
            
            # If no valid observations, skip attention
            any_valid = mask_i.any()
            
            out = jax.lax.cond(
                any_valid,
                lambda: self.cross_attn(
                    query=query,
                    key_=k_i,
                    value=v_i,
                    mask=attn_mask,
                    key=key_i,
                ),
                lambda: jnp.zeros_like(query),
            )
            
            return out.squeeze(0), any_valid  # [hidden], bool
        
        # vmap over locations
        loc_keys = jrandom.split(key, N)
        updates, any_valid = jax.vmap(attend_single)(
            h, keys_input, values_input, valid_mask, loc_keys
        )
        
        # Residual connection + norm
        h_new = self.norm(h + updates)
        
        return h_new, {"source_order": source_order, "any_valid": any_valid}