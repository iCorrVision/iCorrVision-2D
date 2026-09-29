from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class ErrorMetrics:
    """Summary of the difference between a computed and a known field."""

    n_compared: int  # points with both a computed value and a truth value
    n_expected: int  # points the grid could have produced
    valid_rate: float  # n_compared / n_expected, in [0, 1]

    bias: float  # signed mean error: systematic component
    mae: float  # mean absolute error: the Challenge's e_u
    # Standard deviation of the error over the points passed in: the random component when
    # they come from one frame. Pooling several frames also mixes in the frame-to-frame
    # variation of the bias.
    dispersion: float
    rmse: float  # root mean square error: both combined
    max_abs: float  # largest single deviation

    @property
    def challenge_bias(self) -> float:
        """Bias in the DIC Challenge's sign convention (commanded - measured)."""
        return -self.bias

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        """Flat dict for a tidy-CSV row, optionally prefixed (e.g. 'u_')."""
        return {f"{prefix}{k}": v for k, v in asdict(self).items()}


def _empty(n_expected: int) -> ErrorMetrics:
    nan = float("nan")
    return ErrorMetrics(
        n_compared=0,
        n_expected=int(n_expected),
        valid_rate=0.0,
        bias=nan,
        mae=nan,
        dispersion=nan,
        rmse=nan,
        max_abs=nan,
    )


def compute_metrics(
    computed: np.ndarray,
    truth: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    ddof: int = 0,
) -> ErrorMetrics:
    """Compare a computed field against a known one.

    Args:
        computed: measured values, any shape.
        truth: known values, broadcastable to `computed`. A scalar is
            accepted, which is the common case for a rigid translation.
        valid_mask: optional boolean array selecting the points to compare.
            Points that are False, or where either array is non-finite, are
            excluded from every statistic but still counted in `n_expected`,
            so they lower `valid_rate`.
        ddof: delta degrees of freedom for the dispersion. 0 (population)
            matches the Challenge's reported sigma; 1 gives the sample
            standard deviation.

    Returns:
        ErrorMetrics. If nothing is comparable, every statistic is NaN and
        `n_compared` is 0; zero would read as a perfect result.
    """
    computed = np.asarray(computed, dtype=np.float64)
    truth = np.broadcast_to(np.asarray(truth, dtype=np.float64), computed.shape)

    usable = np.isfinite(computed) & np.isfinite(truth)
    if valid_mask is not None:
        usable &= np.asarray(valid_mask, dtype=bool)  # element-wise: keep only the points the mask selects

    n_expected = computed.size if valid_mask is None else int(np.size(valid_mask))
    if not usable.any():
        return _empty(n_expected)

    error = computed[usable] - truth[usable]
    n = int(error.size)

    return ErrorMetrics(
        n_compared=n,
        n_expected=n_expected,
        valid_rate=n / n_expected if n_expected else float("nan"),
        bias=float(error.mean()),
        mae=float(np.abs(error).mean()),
        dispersion=float(error.std(ddof=ddof)) if n > ddof else float("nan"),
        rmse=float(np.sqrt((error**2).mean())),
        max_abs=float(np.abs(error).max()),
    )


def compute_vector_metrics(
    u_computed: np.ndarray,
    v_computed: np.ndarray,
    u_truth: np.ndarray,
    v_truth: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    ddof: int = 0,
) -> dict[str, ErrorMetrics]:
    """Per-component metrics plus the Euclidean error magnitude.

    The magnitude entry is computed from the vector error
    ||(du, dv)||, not from the difference of the two magnitudes, which would
    cancel a displacement that is right in size but wrong in direction. Its
    `bias` is therefore non-negative by construction: a mean error magnitude,
    not a systematic offset.
    """
    both = np.isfinite(u_computed) & np.isfinite(v_computed)
    if valid_mask is not None:
        both &= np.asarray(valid_mask, dtype=bool)

    u = compute_metrics(u_computed, u_truth, valid_mask, ddof=ddof)
    v = compute_metrics(v_computed, v_truth, valid_mask, ddof=ddof)

    du = np.asarray(u_computed, dtype=np.float64) - np.broadcast_to(
        np.asarray(u_truth, dtype=np.float64), np.shape(u_computed)
    )
    dv = np.asarray(v_computed, dtype=np.float64) - np.broadcast_to(
        np.asarray(v_truth, dtype=np.float64), np.shape(v_computed)
    )
    magnitude = compute_metrics(np.hypot(du, dv), 0.0, both, ddof=ddof)

    return {"u": u, "v": v, "magnitude": magnitude}


def virtual_strain_gauge(
    subset_size: int,
    step_size: int,
    *,
    strain_window: int | None = None,
    radius_px: float | None = None,
) -> float:
    """Physical extent, in pixels, over which a strain value is averaged.

    Two neighbourhood definitions, one number, so that a windowed and a
    meshfree estimate can be compared on equal terms:

        windowed:  VSG = (SW - 1) * ST + SS
        meshfree:  VSG = 2 * r + SS

    Exactly one of `strain_window` (in grid nodes) or `radius_px` must be
    given. `strain_window` counts nodes, not pixels: at step 10 a window of
    15 spans 140 px of node centres.
    """
    if (strain_window is None) == (radius_px is None):
        raise ValueError("pass exactly one of strain_window or radius_px")
    if strain_window is not None:
        return float((strain_window - 1) * step_size + subset_size)
    return float(2.0 * radius_px + subset_size)
