from abc import ABC, abstractmethod

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import map_coordinates, spline_filter


class SubpixelInterpolator(ABC):
    """Interpolator of image intensities at sub-pixel positions."""

    def prepare(self, window: np.ndarray):
        """Return the interpolation context for a window; by default the window itself."""
        return window

    @abstractmethod
    def sample_at(
        self, context, row_coords: np.ndarray, col_coords: np.ndarray
    ) -> np.ndarray:
        """Evaluate the interpolant at fractional (row, col) coordinates."""
        raise NotImplementedError


class BicubicSplineInterpolator(SubpixelInterpolator):
    """Bicubic spline interpolator (scipy.interpolate.RectBivariateSpline)."""

    def prepare(self, window: np.ndarray):
        rows, cols = window.shape
        y, x = np.arange(rows, dtype=np.float64), np.arange(cols, dtype=np.float64)
        return RectBivariateSpline(y, x, window, kx=3, ky=3, s=0)

    def sample_at(self, context, row_coords, col_coords):
        """Evaluate the bicubic spline at the given coordinates."""
        return context.ev(row_coords, col_coords)


class BiquinticSplineInterpolator(SubpixelInterpolator):
    """Biquintic B-spline interpolator with prefiltering."""

    def prepare(self, window: np.ndarray):
        return spline_filter(window.astype(np.float64), order=5)

    def sample_at(
        self, context, row_coords: np.ndarray, col_coords: np.ndarray
    ) -> np.ndarray:
        return map_coordinates(
            context, [row_coords, col_coords], order=5, prefilter=False
        )


class EightTapInterpolator(SubpixelInterpolator):
    """Separable Lanczos (a = 4) windowed-sinc interpolator, 8 taps per axis.

    Four samples lie on each side of the query point. The kernel approximates
    the ideal band-limited reconstruction filter instead of enforcing
    derivative continuity as a cubic spline does, the aim of the bias-reducing
    kernels of Schreier et al. (2000); the weights are Lanczos weights, not
    their published table.
    """

    _A = 4
    _OFFSETS = np.arange(-_A + 1, _A + 1)  # (-3, -2, -1, 0, 1, 2, 3, 4)

    @classmethod
    def _weights(cls, frac: np.ndarray) -> np.ndarray:
        """frac: fractional offsets in [0, 1), any shape. Returns (..., 8)."""
        x = frac[..., None] - cls._OFFSETS
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(
                np.abs(x) < 1e-12,
                1.0,
                cls._A
                * np.sin(np.pi * x)
                * np.sin(np.pi * x / cls._A)
                / (np.pi**2 * x**2),
            )
        w = np.where(np.abs(x) >= cls._A, 0.0, w)
        return w / w.sum(axis=-1, keepdims=True)

    def sample_at(self, array, row_coords, col_coords):
        rows, cols = array.shape
        row_coords = np.asarray(row_coords, dtype=np.float64)
        col_coords = np.asarray(col_coords, dtype=np.float64)

        r0 = np.floor(row_coords)
        c0 = np.floor(col_coords)
        rw = self._weights(row_coords - r0)  # (..., 8)
        cw = self._weights(col_coords - c0)  # (..., 8)

        r_idx = np.clip(r0[..., None] + self._OFFSETS, 0, rows - 1).astype(np.int64)
        c_idx = np.clip(c0[..., None] + self._OFFSETS, 0, cols - 1).astype(np.int64)

        neighborhood = array[r_idx[..., :, None], c_idx[..., None, :]]  # (..., 8, 8)
        return np.einsum("...i,...j,...ij->...", rw, cw, neighborhood)
