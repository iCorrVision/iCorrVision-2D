from dataclasses import dataclass
from typing import Literal
from components.contracts import (
    CorrelationConfig,
    validate_calibration,
    validate_grid_params,
)

SessionMode = Literal["all", "stride", "explicit"]


@dataclass(frozen=True)
class SessionConfig:
    image_dir: str = "."
    mode: SessionMode = "all"
    reference_frame: str | None = (
        None  # required for "explicit"; optional override for "all"/"stride"
    )
    stride_n: int | None = None  # required for "stride"
    explicit_frame_list: str | None = None  # required for "explicit"

    def __post_init__(self) -> None:
        if self.mode == "stride" and (self.stride_n is None or self.stride_n <= 0):
            raise ValueError("stride mode requires a positive stride_n")
        if self.mode == "explicit" and (
            self.reference_frame is None or self.explicit_frame_list is None
        ):
            raise ValueError(
                "explicit mode requires reference_frame and explicit_frame_list"
            )


@dataclass(frozen=True)
class SetupConfig:
    subset_size: int
    step_size: int
    calibration_mm_per_px: float = 1.0
    roi_mask: str | None = None  # path to .tiff; None = full reference-frame bounds

    def __post_init__(self) -> None:
        validate_grid_params(self.subset_size, self.step_size)
        validate_calibration(self.calibration_mm_per_px)


@dataclass(frozen=True)
class RunConfig:
    session: SessionConfig
    setup: SetupConfig
    correlation: CorrelationConfig
