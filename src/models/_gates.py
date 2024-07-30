import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom


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