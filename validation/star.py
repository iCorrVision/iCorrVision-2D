"""DIC Challenge 2.0 Star image sets: file roles, period law, truth, geometry.

TRUTH NEEDS NO LAGRANGIAN CORRECTION
    The Challenge generated deformed images from a closed-form Boolean speckle
    model with I_cur(x + u_GT(x)) = I_ref(x) (Reu et al. 2022, Eq. 2), so the
    ground truth is defined at reference positions, where DIC reports.
    Contrast Challenge 1.0's Samples 14/15, whose images were made by moving
    intensities on a fixed grid and so need the Euler-to-Lagrange step.

WHAT IS COMPARED
    Along the centre row the commanded field sits at its peak for every column,
    so the truth there is a single number: the amplitude. The column position
    only determines the local period of the sinusoid, via PERIOD_LAWS.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def _period_small(x1: np.ndarray) -> np.ndarray:
    """Stars 1-4: 10 -> 150 px over 2000 columns (paper footnote 1). x1 is 1-based."""
    return 10.0 + (150.0 - 10.0) / 2000.0 * np.asarray(x1, float)


def _period_large(x1: np.ndarray) -> np.ndarray:
    """Stars 5-6: 10 -> 300 px over 4000 columns (paper Eq. 1). x1 is 1-based."""
    return 10.0 + (300.0 - 10.0) / 4000.0 * (np.asarray(x1, float) - 1.0)


# Strain amplitude. "+/-5 % Lagrangian strain" is ambiguous between a
# displacement gradient dv/dY = 0.05 and a Green-Lagrange E_yy = 0.05; at the
# peak they differ by 0.5 * 0.05^2 = 0.00125 (2.5 %). The engine reports
# Green-Lagrange, so for the gradient reading the true E_yy is 0.05125.
STRAIN_GRADIENT = 0.05
STRAIN_GREEN_LAGRANGE = STRAIN_GRADIENT + 0.5 * STRAIN_GRADIENT**2


@dataclass(frozen=True)
class StarSet:
    folder: str
    quantity: str  # "displacement" or "strain"
    reference: str
    deformed: str
    noise: str | None  # undeformed noise-floor image, None for noise-free sets
    period_law: callable

    @property
    def amplitude(self) -> float:
        return 0.5 if self.quantity == "displacement" else STRAIN_GRADIENT

    def period(self, x0: np.ndarray) -> np.ndarray:
        """Local period at 0-based column x0 (the engine's convention)."""
        return self.period_law(np.asarray(x0, float) + 1.0)


# The noise-floor file is the one that is neither *_Ref nor *_Def / *_Reference
# nor *_Deformed, following the file naming of the Challenge distribution.
STAR_SETS: dict[str, StarSet] = {
    "Star1": StarSet(
        "Star1NoNoise",
        "displacement",
        "DIC_Challenge_Wave_Reference_NoiseFree.tif",
        "DIC_Challenge_Wave_Deformed_NoiseFree.tif",
        None,
        _period_small,
    ),
    "Star2": StarSet(
        "Star2Noise",
        "displacement",
        "DIC_Challenge_Wave_Reference_Noisy.tif",
        "DIC_Challenge_Wave_Deformed_Noisy.tif",
        "DIC_Challenge_Wave_Noisy2.tif",
        _period_small,
    ),
    "Star3": StarSet(
        "Star3NoNoiseStrain",
        "strain",
        "DIC_Challenge_Wave_Reference_Strain.tif",
        "DIC_Challenge_Wave_Deformed_Strain.tif",
        None,
        _period_small,
    ),
    "Star4": StarSet(
        "Star4NoiseStrain",
        "strain",
        "DIC_Challenge_Wave_Reference_Strain_Noise.tif",
        "DIC_Challenge_Wave_Deformed_Strain_Noise.tif",
        "DIC_Challenge_Wave_Strain_Noise2.tif",
        _period_small,
    ),
    "Star5": StarSet(
        "Star5LargeNoisy",
        "displacement",
        "DIC_Challenge_Star_Noise_Ref.tif",
        "DIC_Challenge_Star_Noise_Def.tif",
        "DIC_Challenge_Star_Noise.tif",
        _period_large,
    ),
    "Star6": StarSet(
        "Star6StrainNoisy",
        "strain",
        "DIC_Challenge_Star_Strain_Noise_Ref.tif",
        "DIC_Challenge_Star_Strain_Noise_Def.tif",
        "DIC_Challenge_Star_Strain_Noise.tif",
        _period_large,
    ),
}


def image_shape(path: Path) -> tuple[int, int]:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return img.shape[:2]


def centre_row(height: int) -> int:
    """0-based centre row. Star heights are odd (501), so this is exact."""
    return height // 2


def write_band_roi(path: Path, shape: tuple[int, int], rows_needed: int) -> Path:
    """Full-width horizontal ROI band that erodes to `rows_needed` rows.

    The engine erodes the ROI by the subset half-width before placing nodes, and
    uses one isotropic step for both axes. A band exactly one subset tall
    therefore leaves a single row of nodes, the centre row, so step 1 gives
    1-px spacing along it at the cost of one row, not the whole image.
    `rows_needed` > 1 keeps extra rows above and below for a strain window.
    """
    h, w = shape
    c = centre_row(h)
    half = rows_needed // 2
    mask = np.zeros((h, w), np.uint8)
    mask[max(0, c - half) : c + half + 1, :] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask)
    return path


def band_height(subset: int, strain_window: int | None = None, step: int = 1) -> int:
    """Rows of ROI so that erosion leaves the centre row plus any strain window."""
    extra = 0 if strain_window is None else (strain_window - 1) * step
    return subset + extra


def plateau(
    star: StarSet, x0: np.ndarray, values: np.ndarray, frac: float = 0.8
) -> float:
    """Median over the longest periods, where attenuation is smallest.

    Used to read a code's sign and amplitude convention from its own output.
    The smallest length parameter is the most reliable: at lambda = 300 px an
    affine subset of 9 px is attenuated by 0.15 %, one of 49 px by 4.4 %.
    """
    lam = star.period(x0)
    v = np.asarray(values, float)
    keep = (lam >= frac * np.nanmax(lam)) & np.isfinite(v)
    return float(np.median(v[keep])) if keep.any() else float("nan")


def analyse_line_cut(
    star: StarSet,
    x0: np.ndarray,
    deformed: np.ndarray,
    noise_x0: np.ndarray | None = None,
    noise: np.ndarray | None = None,
    amplitude: float | None = None,
) -> dict:
    """n, l10%, MEI and sign for one centre-row cut.

    Engine output and participant CSVs both go through this function, so any
    quirk in fitting or crossing detection applies to every code alike.
    The sign is read from the long-period plateau and reported: a negative sign
    on the engine's output usually means reference and deformed were swapped.
    """
    from .resolution import fit_attenuation, measurement_resolution, mei

    amplitude = star.amplitude if amplitude is None else amplitude
    sign = np.sign(plateau(star, x0, deformed)) or 1.0
    ratio = sign * np.asarray(deformed, float) / amplitude
    fit = fit_attenuation(star.period(x0), ratio)
    n = measurement_resolution(noise) if noise is not None else float("nan")
    return {
        "n": n,
        "l10_px": fit.spatial_resolution,
        "l10_conservative_px": fit.conservative_resolution,
        "mei": mei(n, fit.spatial_resolution, star.quantity),
        "mei_conservative": mei(n, fit.conservative_resolution, star.quantity),
        "sign": sign,
        "plateau": plateau(star, x0, deformed),
    }
