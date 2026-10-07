import difflib
from dataclasses import dataclass, field, fields
import numpy as np
from enum import Enum
from typing import Literal, get_args


# ==============================================================================
# Session
# ==============================================================================
@dataclass
class SessionOutput:
    reference_frame: str | None = None
    deformed_frames: list[str] | None = None


class CameraChannel(Enum):
    """Which camera column to read from the DICgrabber CSV."""

    LEFT = "left"
    RIGHT = "right"


@dataclass
class ImageRecord:
    """One row of the DICgrabber CSV manifest."""

    # Mono capture is assumed, so only the left frame is recorded. Date and time may
    # be missing for frames not captured with iCorrVision.
    date: str | None
    left_frame_time: str | None
    left_frame_name: str
    right_frame_time: str | None = None
    right_frame_name: str | None = None

    @property
    def has_right(self) -> bool:
        return self.right_frame_name is not None

    def filename(self, channel: CameraChannel) -> str | None:
        """Return the image filename for the camera channel, or None if absent."""
        if channel == CameraChannel.LEFT:
            return self.left_frame_name
        return self.right_frame_name

    def timestamp(self, channel: CameraChannel) -> str | None:
        """Return the capture time for the camera channel, or None if absent."""
        if channel == CameraChannel.LEFT:
            return self.left_frame_time
        return self.right_frame_time


# ==============================================================================
# Setup
# ==============================================================================
def validate_grid_params(subset_size: int, step_size: int) -> None:
    if subset_size <= 0 or subset_size % 2 == 0:
        raise ValueError("subset_size must be a positive odd integer")
    if step_size <= 0:
        raise ValueError("step_size must be positive")


def validate_calibration(calibration_mm_per_px: float) -> None:
    if calibration_mm_per_px <= 0:
        raise ValueError("calibration_mm_per_px must be positive")


@dataclass(frozen=True)
class SetupOutput:
    """Static input gathered by the setup workflow."""

    reference_frame: np.ndarray | None = None  # full-res grayscale/BGR array, HxW(xC)
    roi_mask: np.ndarray | None = None  # bool, shape (H, W), from RoiModel.to_mask()
    subset_size: int | None = None  # odd, pixels
    step_size: int | None = None  # pixels between subset centres
    deformed_list: list[str] | None = None
    calibration_mm_per_px: float = 1.0

    def __post_init__(self) -> None:
        if self.subset_size is not None and self.step_size is not None:
            validate_grid_params(self.subset_size, self.step_size)
        validate_calibration(self.calibration_mm_per_px)


# ==============================================================================
# Correlation
# ==============================================================================

CorrelationMode = Literal["spatial", "incremental"]
TrackingMode = Literal["eulerian", "lagrangian"]
InterpolationMethodName = Literal["bicubic_spline", "biquintic_spline", "8tap"]
CriterionName = Literal["zncc", "znssd"]
SearchMethodName = Literal["brute", "pyramid"]
RefinementMethodName = Literal["icgn"]
StrainTensorName = Literal["lagrange_cloud", "lagrange_q4", "lagrange_q9"]
ShapeFunctionName = Literal["affine", "quadratic"]


@dataclass(frozen=True)
class SearchConfig:
    method: SearchMethodName = "brute"
    pyramid_levels: int = 3
    pyramid_downsample_factor: int = 2
    pyramid_fine_window: int = 8

    def __post_init__(self) -> None:
        if self.method == "pyramid":
            if self.pyramid_levels <= 0:
                raise ValueError("pyramid_levels must be positive")
            if self.pyramid_downsample_factor <= 1:
                raise ValueError("pyramid_downsample_factor must be greater than 1")
            if self.pyramid_fine_window <= 0:
                raise ValueError("pyramid_fine_window must be positive")


@dataclass(frozen=True)
class RefinementConfig:
    method: RefinementMethodName = "icgn"
    shape_function: str = "affine"


@dataclass(frozen=True)
class StrainConfig:
    method: StrainTensorName = "lagrange_cloud"

    # PointCloudGreenLagrangeStrain
    min_neighbors: int = 4
    radius_px: float = 15.0

    # WindowGreenLagrangeStrain
    strain_window: int = 5

    def __post_init__(self) -> None:
        if self.method == "lagrange_cloud":
            if self.radius_px <= 0:
                raise ValueError("look up radius must be positive")
            if self.min_neighbors < 3:
                raise ValueError(
                    "at least 3 valid neighbors are needed for a plane fit of 3 dof"
                )
        else:
            if self.strain_window < 3 or self.strain_window % 2 == 0:
                raise ValueError("strain window must be odd and >=3")
            if self.method == "lagrange_q9" and self.strain_window < 5:
                raise ValueError("lagrange_q9 needs strain window >= 5")


@dataclass(frozen=True)
class InterpolationConfig:
    method: InterpolationMethodName = "biquintic_spline"


@dataclass(frozen=True)
class RecoveryConfig:
    enabled: bool = False
    max_lost_frames: int = 2
    continuity_tol_px: float = 5.0
    fit_radius: int = 3
    min_fit_points: int = 6

    def __post_init__(self) -> None:
        if self.enabled:
            if self.max_lost_frames <= 0:
                raise ValueError("number of frames lost must be positive")
            if self.continuity_tol_px <= 0:
                raise ValueError("recovery tolerance must be positive")
            if self.fit_radius <= 0:
                raise ValueError("recovery fit radius must be positive")
            if self.min_fit_points <= 0:
                raise ValueError("At least on point needed for recovery fit")


@dataclass(frozen=True)
class CorrelationConfig:
    search_size: int
    correlation_mode: CorrelationMode = "spatial"
    tracking_mode: TrackingMode = "eulerian"
    criterion_name: CriterionName = "znssd"
    match_threshold: float = 0.6
    search: SearchConfig = field(default_factory=SearchConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    strain: StrainConfig = field(default_factory=StrainConfig)
    n_workers: int = 2
    image_interpolation: InterpolationConfig = field(
        default_factory=InterpolationConfig
    )
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)

    def __post_init__(self) -> None:
        # The Literal types list the permitted values but are not enforced at run time.
        for name, kind in (
            ("correlation_mode", CorrelationMode),
            ("tracking_mode", TrackingMode),
        ):
            if getattr(self, name) not in get_args(kind):
                raise ValueError(
                    f"{name} must be one of {list(get_args(kind))}, "
                    f"got {getattr(self, name)!r}"
                )
        if self.search_size <= 0:
            raise ValueError("search_size must be positive")
        if self.search_size % 2 == 0:
            raise ValueError("search_size must be odd")
        if not 0.0 <= self.match_threshold <= 1.0:
            raise ValueError("match_threshold must be between 0 and 1")
        if self.n_workers <= 0:
            raise ValueError("n_workers must be positive")
        if (
            self.correlation_mode == "incremental"
            and self.tracking_mode != "lagrangian"
        ):
            raise ValueError(
                "incremental correlation requires tracking_mode='lagrangian'"
            )


@dataclass(frozen=True)
class StrainField:
    """Green-Lagrange strain components and principal strains, one value per grid node."""

    exx: np.ndarray
    eyy: np.ndarray
    exy: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    principal_angle_deg: np.ndarray


@dataclass(frozen=True)
class CorrelationFrame:
    """Correlation data for one frame in physical units and pixels."""

    frame_index: int
    frame_name: str

    # Pixel units
    x_px: np.ndarray
    y_px: np.ndarray
    u_px: np.ndarray
    v_px: np.ndarray
    magnitude_px: np.ndarray
    valid_mask: np.ndarray

    # Physical units, when a calibration is given
    x_mm: np.ndarray | None = None
    y_mm: np.ndarray | None = None
    u_mm: np.ndarray | None = None
    v_mm: np.ndarray | None = None
    magnitude_mm: np.ndarray | None = None

    # Strain
    strain: StrainField | None = None

    # Analysis values
    score: np.ndarray | None = None
    recovered: np.ndarray | None = None
    converged: np.ndarray | None = None


@dataclass(frozen=True)
class CorrelationResult:
    """Full output from the correlation run."""

    grid_x_px: np.ndarray
    grid_y_px: np.ndarray
    frames: list[CorrelationFrame]
    frame_times_sec: list[float] = field(default_factory=list)
    aborted: bool = False


@dataclass(frozen=True)
class HeatmapPayload:
    """Presenter-friendly payload for drawing a heatmap or vector overlay."""

    frame_index: int
    frame_name: str
    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    magnitude: np.ndarray
    valid_mask: np.ndarray
    value_label: str = "magnitude"
    value_units: str = "px"


def check_keys(table: dict, allowed, where: str) -> None:
    """Raise ValueError for a key of `table` not in `allowed`, naming the closest valid key."""
    for key in table:
        if key not in allowed:
            close = difflib.get_close_matches(key, list(allowed), n=1)
            hint = f"; did you mean {close[0]!r}?" if close else ""
            raise ValueError(f"unknown key {key!r} in [{where}]{hint}")


def from_table(cls, table: dict, where: str):
    """Build a config dataclass from a TOML table, rejecting unknown keys."""
    check_keys(table, {f.name for f in fields(cls)}, where)
    return cls(**table)


_NESTED_CONFIGS = {
    "search": SearchConfig,
    "refinement": RefinementConfig,
    "strain": StrainConfig,
    "image_interpolation": InterpolationConfig,
    "recovery": RecoveryConfig,
}


# Used by the config loader and the archive reader. The defaults are those of the
# dataclasses; only search_size is required.
def build_correlation_config(d: dict) -> CorrelationConfig:
    check_keys(d, {f.name for f in fields(CorrelationConfig)}, "correlation")
    if "search_size" not in d:
        raise ValueError("[correlation] requires search_size")
    kwargs = dict(d)
    for name, cls in _NESTED_CONFIGS.items():
        if name in kwargs:
            kwargs[name] = from_table(cls, kwargs[name], f"correlation.{name}")
    return CorrelationConfig(**kwargs)
