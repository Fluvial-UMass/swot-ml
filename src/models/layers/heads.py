import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


class Linear(eqx.Module):
    layer: eqx.nn.Linear

    def __init__(self, latent_size, *, key: PRNGKeyArray):
        self.layer = eqx.nn.Linear(latent_size, 1, key=key)

    def __call__(self, x: Array) -> Array:
        return self.layer(x)


class MLP(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, latent_size: int, hidden_size: int, n_layers: int = 2, *, key: PRNGKeyArray):
        self.mlp = eqx.nn.MLP(
            in_size=latent_size, out_size=1, width_size=hidden_size, depth=n_layers, key=key
        )

    def __call__(self, x: Array):
        return self.mlp(x)


class GMM(eqx.Module):
    """Gaussian Mixture Density Network

    A mixture density network with Gaussian distribution as components. Good references are [#]_ and [#]_. The latter
    one forms the basis for our implementation. As such, we also use two layers in the head to provide it with
    additional flexibility, and exponential activation for the variance estimates and a softmax for weights.

    Parameters
    ----------
    n_in : int
        Number of input neurons.
    n_out : int
        Number of output neurons. Corresponds to 3 times the number of components.
    n_hidden : int
        Size of the hidden layer.

    References
    ----------
    .. [#] C. M. Bishop: Mixture density networks. 1994.
    .. [#] D. Ha: Mixture density networks with tensorflow. blog.otoro.net,
           URL: http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow, 2015.
    """

    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    _eps: float

    def __init__(self, latent_size: int, hidden_size: int, n_models: int, *, key: PRNGKeyArray):
        k1, k2 = jax.random.split(key)
        self.fc1 = eqx.nn.Linear(latent_size, hidden_size, key=k1)
        # Use a small gain for the final layer to prevent initial explosion
        self.fc2 = eqx.nn.Linear(hidden_size, n_models * 3, key=k2)

        # Initialize weights closer to zero for a "flat" start
        self.fc2 = eqx.tree_at(lambda l: l.weight, self.fc2, self.fc2.weight * 0.01)
        self._eps = 1e-3

    def __call__(self, x: Array) -> dict[str, Array]:
        h = jax.nn.relu(self.fc1(x))
        h = self.fc2(h)
        mu, s_latent, p_latent = jnp.split(h, 3, axis=-1)

        # clipping to prevent e^(sigma^2) explosion
        s_latent = jnp.clip(s_latent, -10.0, 4.0)

        # Use softplus for a smooth, positive variance
        sigma = jax.nn.softplus(s_latent) + self._eps
        log_pi = jax.nn.log_softmax(p_latent, axis=-1)

        return {"mu": mu, "sigma": sigma, "log_pi": log_pi}


def beta_softplus(x: Array, beta: int = 1, thresh: int = 20):
    # Mirroring the pytorch implementation https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html
    # Implementation from https://github.com/jax-ml/jax/issues/18443
    x_safe = jax.lax.select(x * beta < thresh, x, jnp.ones_like(x))
    return jax.lax.select(x * beta < thresh, 1 / beta * jnp.log(1 + jnp.exp(beta * x_safe)), x)


class CMAL(eqx.Module):
    """Countable Mixture of Asymmetric Laplacians.

    An mixture density network with Laplace distributions as components.

    The CMAL-head uses an additional hidden layer to give it more expressiveness (same as the GMM-head).
    CMAL is better suited for many hydrological settings as it handles asymmetries with more ease. However, it is also
    more brittle than GMM and can more often throw exceptions. Details for CMAL can be found in [#]_.

    Parameters
    ----------
    n_in : int
        Number of input neurons.
    n_out : int
        Number of output neurons. Corresponds to 4 times the number of components.
    n_hidden : int
        Size of the hidden layer.

    References
    ----------
    .. [#] D.Klotz, F. Kratzert, M. Gauch, A. K. Sampson, G. Klambauer, S. Hochreiter, and G. Nearing:
        Uncertainty Estimation with Deep Learning for Rainfall-Runoff Modelling. arXiv preprint arXiv:2012.14295, 2020.
    """

    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    _eps: float

    def __init__(self, latent_size: int, hidden_size: int, n_models: int, *, key: PRNGKeyArray):
        k1, k2 = jax.random.split(key)
        self.fc1 = eqx.nn.Linear(latent_size, hidden_size, key=k1)
        self.fc2 = eqx.nn.Linear(hidden_size, n_models * 4, key=k2)
        self._eps = 1e-5

    def __call__(self, x: Array):
        h = jax.nn.relu(self.fc1(x))
        h = self.fc2(h)

        m_latent, b_latent, t_latent, p_latent = jnp.split(h, 4, axis=-1)

        # enforce properties on component parameters and weights:
        m = m_latent  # no restrictions (depending on setting m>0 might be useful)
        b = beta_softplus(b_latent, 2) + self._eps  # scale > 0 (softplus was working good in tests)
        t = (1 - self._eps) * jax.nn.sigmoid(t_latent) + self._eps  # 0 < tau < 1

        p = jax.nn.softmax(p_latent, axis=-1)
        p = jnp.clip(p, self._eps, 1.0)
        p = p / jnp.sum(p, axis=-1, keepdims=True)

        return {"mu": m, "b": b, "tau": t, "pi": p}
