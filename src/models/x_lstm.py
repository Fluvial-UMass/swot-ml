import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

from .base_model import BaseModel  # Assuming BaseModel is in a higher directory
from .layers.s_lstm import sLSTM
from .layers.m_lstm import mLSTM


class xLSTM(BaseModel):
    in_proj: eqx.nn.Linear
    static_proj: eqx.nn.Linear
    xlstm_block: eqx.Module  # This will be a single xLSTM block, either sLSTMBlock or mLSTMBlock
    in_head: eqx.nn.Linear
    out_head: eqx.nn.Linear
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
        block_type: str = "mLSTM",  # "sLSTM" or "mLSTM"
        seq2seq: bool = False,
    ):
        self.seq2seq = seq2seq
        key = jrandom.PRNGKey(seed)
        keys = jrandom.split(key, 5)

        all_targets = in_targets + out_targets
        super().__init__(hidden_size, all_targets, key=keys[0])

        self.in_proj = eqx.nn.Linear(dynamic_size, hidden_size, key=keys[1])

        entity_aware = static_size > 0
        if entity_aware:
            self.static_proj = eqx.nn.Linear(static_size, hidden_size, key=keys[2])
            static_embed_size = hidden_size
        else:
            self.static_proj = None
            static_embed_size = 0

        combined_input_size = hidden_size + static_embed_size

        # Choose the specific xLSTM block type
        if block_type == "sLSTM":
            self.xlstm_block = sLSTMBlock(combined_input_size, hidden_size, dropout, key=keys[3])
        elif block_type == "mLSTM":
            self.xlstm_block = mLSTMBlock(combined_input_size, hidden_size, dropout, key=keys[3])
        else:
            raise ValueError("block_type must be 'sLSTM' or 'mLSTM'")

        self.in_head = eqx.nn.Linear(hidden_size, len(in_targets), key=keys[4])
        self.out_head = eqx.nn.Linear(hidden_size, len(out_targets), key=keys[5])

    def __call__(self, data: dict[str, Array | dict[str, Array]], key: PRNGKeyArray):
        keys = jrandom.split(key, 2)

        x_d = jnp.nan_to_num(data["dynamic"]["era5"], nan=0.0)
        x = jax.vmap(self.in_proj)(x_d)

        if self.static_proj:
            x_s = self.static_proj(data["static"])
            x_s_tiled = jnp.tile(x_s, (x.shape[0], 1))
            x = jnp.concatenate([x, x_s_tiled], axis=-1)

        # Process through the single xLSTM block
        xlstm_output = self.xlstm_block(x, keys[0])

        # For non-seq2seq, take the last hidden state
        if self.seq2seq:
            y_in = jax.vmap(self.in_head)(xlstm_output)
            y_out = jax.vmap(self.out_head)(xlstm_output)
        else:
            y_in = self.in_head(xlstm_output[-1, ...])
            y_out = self.out_head(xlstm_output[-1, ...])

        out = jnp.concatenate([y_in, y_out], axis=-1)
        return out


# Helper classes for xLSTM blocks as per Figure 3, Figure 10, Figure 11
class sLSTMBlock(eqx.Module):
    lstm: sLSTM
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    proj_up: eqx.nn.Linear
    proj_down: eqx.nn.Linear
    gated_mlp_gate: eqx.nn.Linear
    gated_mlp_value: eqx.nn.Linear

    def __init__(self, in_size: int, hidden_size: int, dropout: float, *, key: PRNGKeyArray):
        keys = jrandom.split(key, 6)
        self.lstm = sLSTM(in_size, hidden_size, dropout=dropout, return_all=True, key=keys[0])
        self.norm1 = eqx.nn.LayerNorm(hidden_size)
        self.norm2 = eqx.nn.LayerNorm(hidden_size)

        # Gated MLP (Post up-projection)
        mlp_intermediate_size = int(hidden_size * 4 / 3)  # Projection factor 4/3
        self.proj_up = eqx.nn.Linear(hidden_size, mlp_intermediate_size, key=keys[1])
        self.gated_mlp_gate = eqx.nn.Linear(hidden_size, mlp_intermediate_size, key=keys[2])
        self.gated_mlp_value = eqx.nn.Linear(mlp_intermediate_size, hidden_size, key=keys[3])
        self.proj_down = eqx.nn.Linear(mlp_intermediate_size, hidden_size, key=keys[4])

    def __call__(self, x: Array, key: PRNGKeyArray):
        residual = x

        # Pre-LayerNorm
        normed_x = self.norm1(x)
        lstm_out = self.lstm(normed_x, key)  # sLSTM already includes convolution internally.

        # Gated MLP
        # The original paper's Figure 10 shows gated MLP applied to the output of the sLSTM,
        # with a separate 'gated' input to proj_up.
        # This implementation simplifies to a standard gated MLP structure.
        gate_input = lstm_out  # Gate derived from LSTM output
        gate_signal = jax.nn.gelu(self.gated_mlp_gate(gate_input))  # GeLU activation

        mlp_intermediate = self.proj_up(lstm_out)
        mlp_out = self.proj_down(gate_signal * mlp_intermediate)

        # Second LayerNorm before adding residual
        out = self.norm2(mlp_out) + residual

        return out


class mLSTMBlock(eqx.Module):
    lstm: mLSTM
    norm: eqx.nn.GroupNorm  # GroupNorm as per Figure 11
    proj_up_input: eqx.nn.Linear
    proj_up_out_gate: eqx.nn.Linear
    proj_down: eqx.nn.Linear
    lskip_proj: eqx.nn.Linear

    def __init__(self, in_size: int, hidden_size: int, dropout: float, *, key: PRNGKeyArray):
        keys = jrandom.split(key, 6)
        # mLSTM takes in_size after initial up-projection as its input
        # The internal mLSTMCell will handle the q,k,v generation.
        self.lstm = mLSTM(hidden_size, hidden_size, dropout=dropout, return_all=True, key=keys[0])
        self.norm = eqx.nn.GroupNorm(4, hidden_size)  # 4 heads, 4 groups for GroupNorm

        # Pre up-projection with factor 2
        up_proj_size = in_size * 2
        self.proj_up_input = eqx.nn.Linear(
            in_size, hidden_size, key=keys[1]
        )  # Project input to hidden_size for LSTM
        self.proj_up_out_gate = eqx.nn.Linear(
            in_size, hidden_size, key=keys[2]
        )  # For externalized output gate
        self.proj_down = eqx.nn.Linear(hidden_size, in_size, key=keys[3])  # Down-projection
        self.lskip_proj = eqx.nn.Linear(
            in_size, hidden_size, key=keys[4]
        )  # Learnable skip connection

    def __call__(self, x: Array, key: PRNGKeyArray):
        residual = x

        # Pre-LayerNorm
        normed_x_for_lstm = eqx.nn.LayerNorm(x.shape[-1])(
            x
        )  # Apply LayerNorm here if not done before block.

        # Up-projection of input
        input_for_lstm = self.proj_up_input(normed_x_for_lstm)

        # mLSTM processing (includes causal convolution internally as per mLSTM layer)
        lstm_out = self.lstm(input_for_lstm, key)

        # GroupNorm
        normed_lstm_out = self.norm(lstm_out)

        # Learnable skip connection
        lskip_output = jax.nn.swish(self.lskip_proj(normed_x_for_lstm))  # Swish activation

        # Add learnable skip input
        combined_output = normed_lstm_out + lskip_output

        # External output gate
        output_gate_signal = jax.nn.sigmoid(self.proj_up_out_gate(normed_x_for_lstm))
        gated_output = output_gate_signal * combined_output

        # Down-projection
        down_proj_out = self.proj_down(gated_output)

        # Final residual connection
        out = residual + down_proj_out

        return out
