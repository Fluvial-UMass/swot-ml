import numpy as np
import yaml
from pydantic import BaseModel, Field


class AssimConfig(BaseModel):
    # These are used to modify the main config for datloading
    new_features: dict[str, list[str]]
    categorical_encoding: dict[str, list[str]] = Field(default_factory=dict)
    bitmask_encoding: dict[str, list[int]] = Field(default_factory=dict)

    # Passed to the model add_assimilator method.
    model_args: dict = Field(default_factory=dict)

    additional_steps: int = Field(..., gt=0)
    additional_hours: float = Field(np.inf, gt=0)
    freeze_base: bool = True

    @classmethod
    def from_file(cls, path: str) -> "AssimConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
