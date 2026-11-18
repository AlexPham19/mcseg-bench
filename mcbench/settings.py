"""Canonical defaults for the plain U-Net baseline; existing recipe unchanged."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BaselineSettings:
    seed: int = 42
    epochs: int = 50
    steps: int = 128
    batch: int = 4  # Training patches per optimizer update.
    patch: int = 256
    width: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-2
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    adam_amsgrad: bool = False
    normalization_percentiles: tuple[float, float] = (1.0, 99.5)
    group_norm_groups: int = 4
    positive_crop_probability: float = 0.5
    flip_probability: float = 0.5
    dice_smoothing: float = 1.0
    checkpoint_threshold: float = 0.5
    inference_batch: int = 8  # Tiles per forward pass, including dev selection.
    hann_weight_floor: float = 1e-3
    threshold_min: float = 0.05
    threshold_max: float = 0.95
    threshold_step: float = 0.05
    fp_budgets: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)

    def to_dict(self):
        return asdict(self)


DEFAULTS = BaselineSettings()
