"""Synthetic test images with exactly known deformations, for the API tour.

Kept out of the notebook so that the notebook shows the use of the engine and
not the generation of test images. Nothing here is part of the engine or its API.
"""

import cv2
import numpy as np
from scipy.ndimage import fourier_shift


def speckle(height: int = 256, width: int = 256, feature_px: float = 2.5, seed: int = 0) -> np.ndarray:
    """A random speckle-like pattern: Gaussian-blurred white noise, 8-bit range.

    `feature_px` is the blur sigma, roughly the speckle radius in pixels.
    Blurring makes the pattern band-limited, so that a sub-pixel shift can be
    applied exactly (see `translate`).
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((height, width)).astype(np.float32)
    ksize = int(6 * feature_px) | 1  # smallest odd kernel covering +/-3 sigma
    blurred = cv2.GaussianBlur(noise, (ksize, ksize), sigmaX=feature_px)
    blurred = (blurred - blurred.min()) / (blurred.max() - blurred.min())
    return (blurred * 200 + 25).astype(np.float32)


def translate(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Rigid sub-pixel shift by (dx, dy) px, applied in the Fourier domain.

    A Fourier shift is exact for a band-limited image and involves no
    interpolation kernel, so any error the engine reports is its own. DIC
    Challenge Sample 3 was generated in the same way. The shift is circular: a
    strip |dx| px wide wraps around at the image edge, so nodes must stay away
    from the border, as the engine's ROI erosion and search margin ensure.
    """
    spectrum = fourier_shift(np.fft.fft2(image), shift=(dy, dx))
    return np.real(np.fft.ifft2(spectrum)).astype(np.float32)


def affine_warp(image: np.ndarray, exx: float, eyy: float, exy: float) -> np.ndarray:
    """Uniform deformation about the image centre with gradient [[exx, exy], [exy, eyy]].

    Displacement of a reference point X is u = G (X - C), so the deformation
    gradient is F = I + G everywhere and the Green-Lagrange strain is known in
    closed form: E = (F^T F - I) / 2. The image is resampled with a quintic
    spline, accurate enough that the recovered strain is limited by the engine
    and not by this function.
    """
    from scipy.ndimage import map_coordinates

    h, w = image.shape
    c = np.array([(w - 1) / 2.0, (h - 1) / 2.0])
    F = np.array([[1.0 + exx, exy], [exy, 1.0 + eyy]])
    F_inv = np.linalg.inv(F)
    # For every pixel of the deformed image, find where it came from in the
    # reference: X = C + F^-1 (x - C). Sampling the reference there builds the
    # deformed image without holes.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    src = F_inv @ np.stack([xx.ravel() - c[0], yy.ravel() - c[1]])
    src_x, src_y = src[0] + c[0], src[1] + c[1]
    out = map_coordinates(image.astype(np.float64), [src_y, src_x], order=5, mode="reflect")
    return out.reshape(h, w).astype(np.float32)


def green_lagrange(exx: float, eyy: float, exy: float) -> tuple[float, float, float]:
    """Closed-form Green-Lagrange (E_xx, E_yy, E_xy) for the gradient affine_warp applies."""
    G = np.array([[exx, exy], [exy, eyy]])
    F = np.eye(2) + G
    E = 0.5 * (F.T @ F - np.eye(2))
    return float(E[0, 0]), float(E[1, 1]), float(E[0, 1])
