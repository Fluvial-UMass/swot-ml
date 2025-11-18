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
        self.fc2 = eqx.nn.Linear(hidden_size, n_models * 3, key=k2)
        self._eps = 1e-5

    def __call__(self, x: Array) -> dict[str, Array]:
        h = jax.nn.relu(self.fc1(x))
        h = self.fc2(h)

        # split output into mu, sigma and weights
        mu, s_latent, p_latent = jnp.split(h, 3, axis=-1)
        sigma = jnp.clip(jnp.exp(s_latent), 1e-5, 1e2)
        pi = jax.nn.softmax(p_latent, axis=-1)

        return {"mu": mu, "sigma": sigma, "pi": pi}


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

        m_latent, b_latent, t_latent, p_latent = jnp.split(h, 2, axis=-1)

        # enforce properties on component parameters and weights:
        m = m_latent  # no restrictions (depending on setting m>0 might be useful)
        b = beta_softplus(b_latent, 2) + self._eps  # scale > 0 (softplus was working good in tests)
        t = (1 - self._eps) * jax.nn.sigmoid(t_latent) + self._eps  # 0 > tau > 1
        p = (1 - self._eps) * jax.nn.softmax(p_latent, dim=-1) + self._eps  # sum(pi) = 1 & pi > 0

        return {"mu": m, "b": b, "tau": t, "pi": p}


class UMAL(eqx.Module):
    """Uncountable Mixture of Asymmetric Laplacians.

    An implicit approximation to the mixture density network with Laplace distributions which does not require to
    pre-specify the number of components. An additional hidden layer is used to provide the head more expressiveness.
    General details about UMAL can be found in [#]_. A major difference between their implementation
    and ours is the binding-function for the scale-parameter (b). The scale needs to be lower-bound. The original UMAL
    implementation uses an elu-based binding. In our experiment however, this produced under-confident predictions
    (too large variances). We therefore opted for a tailor-made binding-function that limits the scale from below and
    above using a sigmoid. It is very likely that this needs to be adapted for non-normalized outputs.

    Parameters
    ----------
    n_in : int
        Number of input neurons.
    n_out : int
        Number of output neurons. Corresponds to 2 times the output-size, since the scale parameters are also predicted.
    n_hidden : int
        Size of the hidden layer.

    References
    ----------
    .. [#] A. Brando, J. A. Rodriguez, J. Vitria, and A. R. Munoz: Modelling heterogeneous distributions
        with an Uncountable Mixture of Asymmetric Laplacians. Advances in Neural Information Processing Systems,
        pp. 8838-8848, 2019.
    """

    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    _eps: float

    def __init__(self, latent_size: int, hidden_size: int, n_models: int, *, key: PRNGKeyArray):
        k1, k2 = jax.random.split(key)
        self.fc1 = eqx.nn.Linear(latent_size, hidden_size, key=k1)
        self.fc2 = eqx.nn.Linear(hidden_size, n_models * 4, key=k2)
        self._eps = 1e-5
        self._upper_bound_scale = (
            0.5  # this parameter found empirical by testing UMAL for a limited set of basins
        )

    def __call__(self, x: Array) -> dict[str, Array]:
        """Perform a UMAL head forward pass.

        Parameters
        ----------
        x : Array
            Output of the previous model part. It provides the basic latent variables to compute the UMAL components.

        Returns
        -------
        dict[str, Array]
            Dictionary containing the means ('mu') and scale parameters ('b') to parametrize the asymmetric Laplacians.
        """
        h = jax.nn.relu(self.fc1(x))
        h = self.fc2(h)

        m_latent, b_latent = jnp.split(2, 4, axis=-1)

        # enforce properties on component parameters and weights:
        m = m_latent  # no restrictions (depending on setting m>0 might be useful)
        b = (
            self._upper_bound_scale * jax.nn.sigmoid(b_latent) + self._eps
        )  # bind scale from two sides.
        return {"mu": m, "b": b}
