import equinox as eqx
import optax
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

class LSTM(eqx.Module):
    """
    A simple LSTM model implemented using Equinox.

    Attributes:
        hidden_size (int): The size of the hidden state in the LSTM cell.
        cell (equinox.nn.LSTMCell): The LSTM cell used in the model.
        linear (equinox.nn.Linear): A linear layer applied to the output of the LSTM cell.
        bias (jax.Array): A bias term added to the output of the linear layer.
    """
    hidden_size: int
    cell: eqx.nn.LSTMCell
    linear: eqx.nn.Linear
    bias: jax.Array

    def __init__(self, in_size, out_size, hidden_size, *, key):
        """
        Initializes the LSTM model.

        Args:
            in_size (int): The size of the input features.
            out_size (int): The size of the output.
            hidden_size (int): The size of the hidden state in the LSTM cell.
            key (jax.random.PRNGKey): A random key for initializing the model parameters.
        """
        ckey, lkey = jrandom.split(key)
        self.hidden_size = hidden_size
        self.cell = eqx.nn.LSTMCell(in_size, hidden_size, key=ckey)
        self.linear = eqx.nn.Linear(hidden_size, out_size, key=lkey)
        self.bias = jnp.zeros(out_size)

    def __call__(self, inp):
        """
        Forward pass through the LSTM model.

        Args:
            inp (jax.Array): The input sequence with shape [sequence_length, in_size].

        Returns:
            jax.Array: The output of the model with shape [sequence_length, out_size].
        """
        scan_fn = lambda state, x: (self.cell(x, state), None)
        init_state = (jnp.zeros(self.cell.hidden_size),
                      jnp.zeros(self.cell.hidden_size))
        (out, _), _ = jax.lax.scan(scan_fn, init_state, inp)
        return self.linear(out)
        

def lr_dict_scheduler(epoch, lr_dict):
    """
    Returns the learning rate for a given epoch based on a multi-step schedule.

    Args:
        epoch (int): The current epoch.
        lr_dict (dict): A dictionary mapping epochs to learning rates.

    Returns:
        float: The learning rate for the given epoch.
    """
    current_lr = None
    for milestone_epoch in sorted(lr_dict.keys(), reverse=True):
        if epoch >= milestone_epoch:
            current_lr = lr_dict[milestone_epoch]
            break
    return current_lr if current_lr is not None else lr_dict[0]

def clip_gradients(grads, max_norm):
    """
    Clip gradients to prevent them from exceeding a maximum norm.

    Args:
        grads (jax.grad): The gradients to be clipped.
        max_norm (float): The maximum norm for clipping.

    Returns:
        jax.grad: The clipped gradients.
    """
    total_norm = jax.tree_util.tree_reduce(lambda x, y: x + y, jax.tree_util.tree_map(lambda x: jnp.sum(x ** 2), grads))
    total_norm = jnp.sqrt(total_norm)
    scale = jnp.minimum(max_norm / total_norm, 1.0)
    return jax.tree_map(lambda g: scale * g, grads)


def l2_regularization(params, l2_reg):
    """
    Computes the L2 regularization term.

    Args:
        params: The parameters of the model.
        l2_reg (float): The L2 regularization strength.

    Returns:
        float: The L2 regularization term.
    """
    reg_term = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
    return reg_term * l2_reg
    
    
@eqx.filter_value_and_grad
def compute_loss(model, x, y):
    """
    Computes the mean squared error loss between the model predictions and the targets.

    Args:
        model (LSTM): The LSTM model.
        x (jax.Array): The input data with shape [batch_size, sequence_length, in_size].
        y (jax.Array): The target data with shape [batch_size, out_size].

    Returns:
        float: The mean squared error loss.
    """
    pred_y = jax.vmap(model)(x)
    return jnp.mean(jnp.square(y - pred_y))


@eqx.filter_jit
def make_step(model, x, y, opt_state, optim, max_grad_norm=None, l2_reg=None):
    """
    Performs a single optimization step, updating the model parameters.

    Args:
        model (LSTM): The LSTM model.
        x (jax.Array): The input data with shape [batch_size, sequence_length, in_size].
        y (jax.Array): The target data with shape [batch_size, out_size].
        opt_state: The state of the optimizer.
        optim: The optimizer.
        max_grad_norm (float, optional): The maximum norm for clipping gradients. Defaults to None.
        l2_reg (float, optional): The L2 regularization strength. Defaults to None.

    Returns:
        tuple: A tuple containing the loss, updated model, and updated optimizer state.
    """
    loss, grads = compute_loss(model, x, y)
    if max_grad_norm is not None:
        grads = clip_gradients(grads, max_grad_norm)
    if l2_reg is not None:
        loss += add_regularization(model.params, l2_reg)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state
