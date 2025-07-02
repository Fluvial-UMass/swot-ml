import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

from .base_model import BaseModel
from .layers.ealstm import EALSTM


class EA_LSTM(BaseModel):
    """Model that uses LSTMs, MLPs, and attention to mix time frequencies.

    Attributes
    ----------
    active_source: dict
        Boolean indicating if this data source (and encoder) are to be used.
    encoders: dict
        Encoders, one for each dynamic data source.
    static_embedder: StaticEmbedder
        Embedder for static data.
    decoders: dict
        Decoders, one for each cross-attention or self-attention.
    head: eqx.nn.Linear
        Linear layer that maps the output of the decoders to the target
        variable(s).
    target: list
        Names of the target variables.
    """

    dynamic_key: str
    ealstm: EALSTM

    def __init__(
        self,
        *,
        target: list,
        dynamic_sizes: dict,
        static_size: int,
        hidden_size: int,
        seed: int,
        dropout: float,
    ):
        key = jrandom.PRNGKey(seed)
        keys = jrandom.split(key)

        super().__init__(hidden_size, target, key=keys[0])

        if len(dynamic_sizes) > 1:
            raise ValueError(
                "EALSTM model only supports 1 dynamic data source.\n"
                + f"{len(dynamic_sizes)} were passed"
            )
        self.dynamic_key = list(dynamic_sizes)[0]
        self.ealstm = EALSTM(
            dynamic_sizes[self.dynamic_key], static_size, hidden_size, dropout=dropout, key=keys[1]
        )

    def __call__(self, data: dict[str, Array | dict[str, Array]], key: PRNGKeyArray):
        """The forward pass of the data through the model

        Parameters
        ----------
        data: dict[str, Array | dict[str, Array]]
            The input data.
        key: PRNGKeyArray
            A PRNG key used to apply the model.

        Returns
        -------
        Array
            The output of the model.
        """
        final_state = self.ealstm(data["dynamic"][self.dynamic_key], data["static"], key)

        return self.head(final_state)
