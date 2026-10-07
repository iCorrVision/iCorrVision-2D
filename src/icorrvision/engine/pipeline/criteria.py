from abc import ABC, abstractmethod
import numpy as np
import cv2


def _zncc_map(ref_subset: np.ndarray, search_window: np.ndarray) -> np.ndarray:
    ref_h, ref_w = ref_subset.shape
    sch_h, sch_w = search_window.shape
    if sch_h < ref_h or sch_w < ref_w:
        return np.array([[]], dtype=np.float64)

    f_zero = ref_subset - np.mean(ref_subset)
    if np.linalg.norm(f_zero) < 1e-8:
        return np.full((sch_h - ref_h + 1, sch_w - ref_w + 1), np.nan, dtype=np.float64)

    return cv2.matchTemplate(
        search_window.astype(np.float32, copy=False),
        ref_subset.astype(np.float32, copy=False),
        cv2.TM_CCOEFF_NORMED,
    ).astype(np.float64, copy=False)


def zero_mean_normalize(vec: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the zero-mean vector and its Euclidean norm."""
    vec_zero = vec - np.mean(vec)
    norm = float(np.linalg.norm(vec_zero))
    return vec_zero, norm


class CorrelationCriterion(ABC):

    @abstractmethod
    def evaluate(self, f_vec: np.ndarray, g_vec: np.ndarray) -> float:
        """Return the correlation score of two 1D intensity vectors."""
        pass

    @abstractmethod
    def evaluate_window(
        self, ref_subset: np.ndarray, search_window: np.ndarray
    ) -> np.ndarray:
        """Return the score map of the reference subset over a 2D search window."""
        pass


class ZNSSDCriterion(CorrelationCriterion):
    """Zero-normalised sum of squared differences (ZNSSD) criterion."""

    def __init__(self, return_similarity: bool = True):
        self.return_similarity = return_similarity

    def evaluate(self, f_vec: np.ndarray, g_vec: np.ndarray) -> float:
        if f_vec.shape != g_vec.shape:
            return np.nan

        f_zero, norm_f = zero_mean_normalize(f_vec)
        g_zero, norm_g = zero_mean_normalize(g_vec)

        if norm_f < 1e-8 or norm_g < 1e-8:
            return np.nan

        f_norm = f_zero / norm_f
        g_norm = g_zero / norm_g
        znssd_val = float(np.sum((f_norm - g_norm) ** 2))

        return 1.0 - 0.5 * znssd_val if self.return_similarity else znssd_val

    def evaluate_window(
        self, ref_subset: np.ndarray, search_window: np.ndarray
    ) -> np.ndarray:

        zncc_map = _zncc_map(ref_subset, search_window)
        return zncc_map if self.return_similarity else 2.0 * (1.0 - zncc_map)


class ZNCCCriterion(CorrelationCriterion):
    """Zero-normalised cross-correlation (ZNCC) criterion."""

    def evaluate(self, f_vec: np.ndarray, g_vec: np.ndarray) -> float:
        if f_vec.shape != g_vec.shape:
            return np.nan

        f_zero, norm_f = zero_mean_normalize(f_vec)
        g_zero, norm_g = zero_mean_normalize(g_vec)

        if norm_f < 1e-8 or norm_g < 1e-8:
            return np.nan
        return float(np.dot(f_zero, g_zero) / (norm_f * norm_g))

    def evaluate_window(
        self, ref_subset: np.ndarray, search_window: np.ndarray
    ) -> np.ndarray:
        return _zncc_map(ref_subset, search_window)
