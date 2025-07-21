from typing import Literal, TypeAlias
from pydantic import BaseModel, Field


class BaseModelArgs(BaseModel):
    hidden_size: int = Field(..., gt=0)
    dropout: float = Field(..., ge=0, lt=1.0)
    seed: int = 0

    def as_kwargs(self) -> dict:
        return self.model_dump(exclude={"name"})


class STGATArgs(BaseModelArgs):
    name: Literal["st_gat"]
    edge_feature_size: int = Field(..., gt=0)
    k_hops: int = Field(..., gt=0)
    seq2seq: bool = False
    return_weights: bool = False
    target: list[str] = None
    seq_length: int = None
    dense_sizes: dict[str, int] = None
    sparse_sizes: dict[str, int] = None
    static_size: int = None


class SeqAttnArgs(BaseModelArgs):
    name: Literal["lstm_mlp_attn", "flexible_hybrid"]
    num_layers: int = Field(..., gt=0)
    num_heads: int = Field(..., gt=0)
    target: list[str] = None
    seq_length: int = None
    dynamic_sizes: dict[str, int] = None
    static_size: int = None
    time_aware: dict[str, bool] = None


class StackArgs(BaseModelArgs):
    name: Literal["stacked_lstm"]
    in_targets: list[str]
    out_targets: list[str]
    dynamic_size: int = None
    static_size: int = None
    seq2seq: bool = True


class GraphLSTMArgs(BaseModelArgs):
    name: Literal["graph_lstm"]
    num_layers: int = Field(..., gt=0)
    num_heads: int = Field(..., gt=0)
    target: list[str] = None
    dynamic_size: int = None
    static_size: int = None


class MSAttnArgs(BaseModelArgs):
    name: Literal["ms_attn"]
    num_layers: int = Field(..., gt=0)
    num_heads: int = Field(..., gt=0)
    target: list[str] = None
    seq_length: int = None
    dynamic_sizes: dict[str, int] = None
    static_size: int = None
    active_source: dict[str, bool] = Field(default_factory=dict)


ModelArgs: TypeAlias = SeqAttnArgs | StackArgs | GraphLSTMArgs | MSAttnArgs | STGATArgs
