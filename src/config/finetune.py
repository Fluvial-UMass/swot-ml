from pydantic import BaseModel, Field
import yaml

class FinetuneConfig(BaseModel):
    reset_lr: bool = False
    additional_epochs: int = Field(..., gt=0)
    config_update: dict | None = Field(default_factory=dict)
    freeze_components: list[str] | None = None
    model_update_kwargs: dict | None = None

    @classmethod
    def from_yaml(cls, path: str) -> "FinetuneConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)