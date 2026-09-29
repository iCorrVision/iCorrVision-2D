"""Spatial resolution, measurement resolution and the Metrological Efficiency
Indicator, following the DIC Challenge 2.0 (Reu et al., 2022).

THE IDEA
    DIC behaves as a low-pass filter. On the Star images the commanded field is
    a sinusoid whose period grows linearly across the image, and along the
    centre row it sits at its peak everywhere. So measured/commanded along that
    row, plotted against the local period, is the code's transfer function.

    Spatial resolution  l10%  = period at which the attenuation reaches 10 %
    Measurement resol.  n     = standard deviation on a noise-floor image
    MEI (displacement)        = n * l10%
    MEI (strain)              = n * l10%^2

STRAIN MEI EXPONENT
    The paper's Eq. 3 writes n^2 * l for strain, but its own strain section says
    "the squared of the spatial resolution times the measurement resolution",
    and its Fig. 12 shows a slope of -2 for log n against log l, which makes
    n * l^2 the invariant. The text and figure agree; Eq. 3 is the outlier.
    n * l^2 is used here.

THEORY
    For noise-free images, local DIC converges to the least-squares projection of
    the true field onto the shape-function basis (Reu et al. 2022, Appendix), i.e.
    a Savitzky-Golay filter across the subset. For an affine subset of width h
    the attenuation of a cosine at the subset centre is the Dirichlet kernel
    sin(pi h/lambda) / (h sin(pi/lambda)); for h = 17 it reaches 0.900 at
    lambda = 67.8 px, the paper's published theoretical value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import Polynomial

ATTENUATION = 0.10  # the Challenge's 10 % criterion


# ---------------------------------------------------------------------------
# Savitzky-Golay theory
# ---------------------------------------------------------------------------


def _sg_coefficients(
    width: int, order: int, derivative: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares polynomial fit weights over a symmetric window.

    Returns (offsets, weights) such that sum(weights * f(offsets)) is the fitted
    polynomial's `derivative`-th coefficient at the centre: the value for
    derivative=0, the slope for derivative=1.
    """
    if width % 2 == 0:
        raise ValueError("window width must be odd")
    half = width // 2
    k = np.arange(-half, half + 1, dtype=float)
    V = np.vander(k, order + 1, increasing=True)  # columns 1, k, k^2, ...
    return k, np.linalg.pinv(V)[derivative]


def attenuation_theory(
    subset: int,
    period: np.ndarray,
    order: int = 1,
    strain_window: int | None = None,
    step: float = 1.0,
) -> np.ndarray:
    """Predicted measured/commanded ratio for a cosine field of the given period.

    Displacement: the subset acts as an order-`order` Savitzky-Golay smoother of
    width `subset` (order 1 = affine, 2 = quadratic).

    Strain (strain_window given): the smoothed displacement is then differentiated
    by a first-order least-squares fit over `strain_window` nodes spaced `step`
    pixels apart. The two linear filters multiply. This models a windowed Q4 fit
    along the direction of the gradient, which is what the centre row sees.
    """
    period = np.asarray(period, dtype=float)
    omega = 2.0 * np.pi / period

    k, w = _sg_coefficients(subset, order)
    h_disp = (w[None, :] * np.cos(np.outer(omega, k))).sum(axis=1)
    if strain_window is None:
        return h_disp

    # Slope estimator applied to sin(omega * y): exact slope at 0 is omega.
    k2, d = _sg_coefficients(strain_window, 1, derivative=1)
    y = k2 * step
    slope = (d[None, :] * np.sin(np.outer(omega, y))).sum(axis=1) / step
    return h_disp * slope / omega


def theoretical_resolution(subset: int, order: int = 1, **kwargs) -> float:
    """l10% predicted by attenuation_theory, found on a fine period grid."""
    period = np.linspace(4.0, 2000.0, 400_000)
    return first_stable_crossing(
        period, attenuation_theory(subset, period, order, **kwargs)
    )


# ---------------------------------------------------------------------------
# Measured curves
# ---------------------------------------------------------------------------


def first_stable_crossing(
    period: np.ndarray, ratio: np.ndarray, level: float = 1 - ATTENUATION
) -> float:
    """Smallest period beyond which the curve stays at or above `level`.

    "Stays above" rather than "first touches": at short periods the measured
    curve is noisy, and the polynomial can cross the level before the real
    roll-off. Returns NaN if the curve never drops below the
    level (a code that over-estimates, like the Challenge's Code K) or never
    recovers above it within the measured range.
    """
    period = np.asarray(period, float)
    ratio = np.asarray(ratio, float)
    order = np.argsort(period)
    period, ratio = period[order], ratio[order]
    below = ratio < level
    if not below.any() or below[-1]:
        return float("nan")
    last_below = int(np.flatnonzero(below)[-1])
    # Linear interpolation between the last point below and the first above.
    p0, p1 = period[last_below], period[last_below + 1]
    r0, r1 = ratio[last_below], ratio[last_below + 1]
    return float(p0 + (level - r0) * (p1 - p0) / (r1 - r0))


def first_upward_crossing(
    period: np.ndarray, ratio: np.ndarray, level: float = 1 - ATTENUATION
) -> float:
    """Smallest period at which the curve rises through `level`: the Challenge's rule.

    "The spatial period at which the signal first crossed the 10 % fractional
    attenuation line" (Reu et al. 2022). A curve that starts above the level
    (no attenuation even at the shortest period, e.g. the Challenge's Code K)
    has no crossing and returns NaN.
    """
    period = np.asarray(period, float)
    ratio = np.asarray(ratio, float)
    order = np.argsort(period)
    period, ratio = period[order], ratio[order]
    up = np.flatnonzero((ratio[:-1] < level) & (ratio[1:] >= level))
    if not up.size:
        return float("nan")
    i = int(up[0])
    p0, p1, r0, r1 = period[i], period[i + 1], ratio[i], ratio[i + 1]
    return float(p0 + (level - r0) * (p1 - p0) / (r1 - r0))


@dataclass(frozen=True)
class ResolutionFit:
    spatial_resolution: float  # l10%, px, first upward crossing (the Challenge's rule)
    conservative_resolution: float  # l10%, px, beyond which the curve stays above 90 %
    polynomial: Polynomial  # attenuation ratio as a function of period
    n_points: int


def fit_attenuation(
    period: np.ndarray, ratio: np.ndarray, degree: int = 12
) -> ResolutionFit:
    """Fit the attenuation curve and locate l10%, as the Challenge does.

    Two crossing rules are reported. The Challenge's is the first upward crossing.
    The conservative one requires the curve to stay above 90 % thereafter; where
    they disagree, the code has sub-threshold dips at long periods, a systematic
    error that the Challenge's rule does not detect.

    The Challenge uses a 12th-order polynomial because pattern-induced bias makes
    the raw curve jagged, so it crosses 90 % many times. A 12th-order polynomial
    in raw pixel units is severely ill-conditioned; Polynomial.fit maps the
    domain onto [-1, 1] first and is used instead of polyfit for that reason.
    """
    period = np.asarray(period, float)
    ratio = np.asarray(ratio, float)
    ok = np.isfinite(period) & np.isfinite(ratio)
    if ok.sum() <= degree:
        return ResolutionFit(
            float("nan"), float("nan"), Polynomial([np.nan]), int(ok.sum())
        )
    poly = Polynomial.fit(period[ok], ratio[ok], degree)
    grid = np.linspace(period[ok].min(), period[ok].max(), 20_000)
    curve = poly(grid)
    return ResolutionFit(
        first_upward_crossing(grid, curve),
        first_stable_crossing(grid, curve),
        poly,
        int(ok.sum()),
    )


def measurement_resolution(noise_values: np.ndarray) -> float:
    """1-sigma of a quantity measured on the undeformed noise-floor image."""
    v = np.asarray(noise_values, float)
    v = v[np.isfinite(v)]
    return float(v.std()) if v.size > 1 else float("nan")


def mei(n: float, spatial_resolution: float, quantity: str) -> float:
    """Metrological Efficiency Indicator. Lower is better."""
    if quantity == "displacement":
        return n * spatial_resolution
    if quantity == "strain":
        return n * spatial_resolution**2
    raise ValueError("quantity must be 'displacement' or 'strain'")


def code_mei(values: np.ndarray, lowest: int = 3) -> float:
    """The Challenge summarises each code by the mean of its three lowest MEIs."""
    v = np.sort(np.asarray(values, float)[np.isfinite(values)])
    return float(v[:lowest].mean()) if v.size else float("nan")
