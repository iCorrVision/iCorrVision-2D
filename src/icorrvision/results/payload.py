from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi

from components.contracts import CorrelationFrame, CorrelationResult, HeatmapPayload

STRAIN_COMPONENT_LABELS: dict[str, str] = {
    "e1": "Major principal strain (e1)",
    "e2": "Minor principal strain (e2)",
    "exx": "Normal strain (exx)",
    "eyy": "Normal strain (eyy)",
    "exy": "Shear strain (exy)",
}

QUANTITY_LABELS: dict[str, str] = {
    "magnitude": "Displacement magnitude",
    **STRAIN_COMPONENT_LABELS,
}
QUANTITY_UNITS: dict[str, str] = {
    "magnitude": "px",
    **dict.fromkeys(STRAIN_COMPONENT_LABELS, "mm/mm"),
}

# ------------------------------------------------------------------------------
# Colour scale policy
# ------------------------------------------------------------------------------

MAGNITUDE_CMAP = "viridis"  # sequential: magnitude is never negative
STRAIN_CMAP = "turbo"  # sequential: one-sided strain fields
DIVERGING_CMAP = "RdBu_r"  # fields that change sign

# Percentiles bounding the colour scale. Values beyond them are clipped and the
# colour bar gains an end triangle, so that a single outlier cannot stretch the
# scale for the remaining nodes.
SCALE_PERCENTILES: tuple[float, float] = (1.0, 99.0)

# A field counts as signed only if its smaller side reaches this fraction of
# its larger side. Otherwise sub-noise negative e1 in the first frame would put a
# tensile run on a diverging scale and leave half of the colour map unused.
_DIVERGING_RATIO = 0.1

# Signed by definition rather than by the data: zero shear is physically
# significant whatever the sign of the current test.
_SIGNED_QUANTITIES = frozenset({"exy"})

# matplotlib colorbar `extend`, keyed by (data below vmin, data above vmax).
_EXTEND = {
    (False, False): "neither",
    (True, False): "min",
    (False, True): "max",
    (True, True): "both",
}


@dataclass(frozen=True)
class ColorScale:
    """Everything a view needs to colour one quantity consistently."""

    vmin: float
    vmax: float
    cmap: str
    extend: str
    label: str
    units: str

    @property
    def colorbar_label(self) -> str:
        return f"{self.label} [{self.units}]"

    def describe(self) -> str:
        return f"{self.vmin:.3g} to {self.vmax:.3g} {self.units}"


def quantity_values(frame: CorrelationFrame, quantity: str) -> np.ndarray:
    """Grid-shaped values of `quantity`: "magnitude" or a strain component."""
    if quantity == "magnitude":
        return np.asarray(frame.magnitude_px, dtype=np.float64)
    if quantity not in STRAIN_COMPONENT_LABELS:
        raise ValueError(f"unsupported quantity: {quantity!r}")
    if frame.strain is None:
        return np.full(frame.x_px.shape, np.nan)
    return np.asarray(getattr(frame.strain, quantity), dtype=np.float64)


def color_scale(
    result: CorrelationResult, quantity: str, frame_index: int | None = None
) -> ColorScale:
    """Colour scale for one frame, or for the whole run when frame_index is None."""
    frames = result.frames if frame_index is None else [result.frames[frame_index]]
    lows: list[float] = []
    highs: list[float] = []
    data_min, data_max = np.inf, -np.inf

    for frame in frames:
        values = quantity_values(frame, quantity)
        sample = values[frame.valid_mask & np.isfinite(values)]
        if sample.size == 0:
            continue
        lo, hi = np.percentile(sample, SCALE_PERCENTILES)
        lows.append(float(lo))
        highs.append(float(hi))
        data_min = min(data_min, float(sample.min()))
        data_max = max(data_max, float(sample.max()))

    label, units = QUANTITY_LABELS[quantity], QUANTITY_UNITS[quantity]
    if not lows:
        cmap = _cmap_for(quantity, 0.0, 1.0, diverging=False)
        return ColorScale(0.0, 1.0, cmap, "neither", label, units)

    lo, hi = min(lows), max(highs)
    diverging = quantity in _SIGNED_QUANTITIES or _straddles_zero(lo, hi)
    if diverging:
        half = max(abs(lo), abs(hi)) or 1.0
        lo, hi = -half, half
    elif hi - lo < 1e-12:  # flat field, e.g. a result holding only the reference
        lo, hi = lo - 0.5, hi + 0.5
    if quantity == "magnitude":
        lo = max(lo, 0.0)

    extend = _EXTEND[(data_min < lo, data_max > hi)]
    cmap = _cmap_for(quantity, lo, hi, diverging)
    return ColorScale(lo, hi, cmap, extend, label, units)


def _straddles_zero(lo: float, hi: float) -> bool:
    if not lo < 0.0 < hi:
        return False
    return min(-lo, hi) >= _DIVERGING_RATIO * max(-lo, hi)


def _cmap_for(quantity: str, lo: float, hi: float, diverging: bool) -> str:
    if quantity == "magnitude":
        return MAGNITUDE_CMAP
    if diverging:
        return DIVERGING_CMAP
    # One-sided negative fields (e2 in tension, anything in compression) use
    # the reversed map, so the hot end always means "most strained".
    return f"{STRAIN_CMAP}_r" if abs(lo) > abs(hi) else STRAIN_CMAP


# ------------------------------------------------------------------------------
# Heatmap payloads
# ------------------------------------------------------------------------------


def get_smooth_grid(
    frame_x: np.ndarray,
    frame_y: np.ndarray,
    grid_x0: np.ndarray,
    grid_y0: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill unplaced nodes with their nearest placed neighbour's displacement."""
    u = frame_x - grid_x0
    v = frame_y - grid_y0
    valid = valid_mask & np.isfinite(u) & np.isfinite(v)

    if not valid.any():
        return grid_x0.copy(), grid_y0.copy()
    if valid.all():
        return frame_x.copy(), frame_y.copy()

    nearest = ndi.distance_transform_edt(
        ~valid, return_distances=False, return_indices=True
    )
    return grid_x0 + u[tuple(nearest)], grid_y0 + v[tuple(nearest)]


def build_payload(
    result: CorrelationResult, quantity: str, frame_index: int = -1
) -> HeatmapPayload:
    """Heatmap geometry and values for one quantity on one frame."""
    frame = result.frames[frame_index]
    values = quantity_values(frame, quantity)
    placed = frame.valid_mask & np.isfinite(frame.x_px) & np.isfinite(frame.y_px)
    shown = placed & np.isfinite(values)

    x, y = get_smooth_grid(
        frame.x_px, frame.y_px, result.grid_x_px, result.grid_y_px, placed
    )
    return HeatmapPayload(
        frame_index=frame.frame_index,
        frame_name=frame.frame_name,
        x=x,
        y=y,
        u=frame.u_px,
        v=frame.v_px,
        magnitude=np.ma.masked_array(values, mask=~shown),
        valid_mask=shown,
        value_label=QUANTITY_LABELS[quantity],
        value_units=QUANTITY_UNITS[quantity],
    )
