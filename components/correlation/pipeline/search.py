import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from components.contracts import SearchConfig, SetupOutput
from .criteria import CorrelationCriterion

# The coarsest pyramid level must keep at least this many pixels per subset side: a
# single pixel has no variance, so a zero-normalised criterion is undefined on it.
MIN_COARSE_SUBSET_PX = 2
# Below this the pyramid still runs, but loses a growing share of the nodes.
WARN_COARSE_SUBSET_PX = 4


def coarsest_pyramid_subset(subset_size: int, levels: int, factor: int) -> int:
    """Side, in pixels, of the subset at the coarsest pyramid level."""
    return subset_size // factor ** (levels - 1)


# Results of the integer-pixel search
@dataclass
class InitialGuess:
    x: float
    y: float
    score: float


@dataclass
class SearchRange:
    center_x: float
    center_y: float
    size: int


# Interface of the integer-pixel search methods
class SearchStrategy(ABC):
    def __init__(self, criterion: CorrelationCriterion, threshold: float = 0.6):
        self.criterion = criterion
        self.threshold = threshold

    @abstractmethod
    def search(
        self,
        ref_subset: np.ndarray,
        deformed_image: np.ndarray,
        search_range: SearchRange,
    ) -> InitialGuess | None:
        pass

    @classmethod
    def from_config(
        cls, criterion: CorrelationCriterion, threshold: float, config: SearchConfig
    ) -> "SearchStrategy":
        return cls(criterion, threshold)

    @classmethod
    def check_setup(cls, config: SearchConfig, setup: SetupOutput) -> None:
        """Raise ValueError if this method cannot work with `setup`; any setup by default."""


def _find_peak_in_window(
    criterion: CorrelationCriterion,
    ref_subset: np.ndarray,
    window: np.ndarray,
) -> InitialGuess | None:
    """Correlate ref_subset over a cropped window with the given criterion."""
    correlation = criterion.evaluate_window(ref_subset, window)
    if not np.isfinite(correlation).any():
        return None

    peak_idx = np.unravel_index(np.nanargmax(correlation), correlation.shape)
    peak_r, peak_c = int(peak_idx[0]), int(peak_idx[1])
    peak_val = correlation[peak_r, peak_c]

    # Peak index to the offset of the subset centre within the window
    sub_h, sub_w = ref_subset.shape
    local_x = peak_c + (sub_w - 1) / 2.0
    local_y = peak_r + (sub_h - 1) / 2.0

    return InitialGuess(x=local_x, y=local_y, score=float(peak_val))


class BruteSearch(SearchStrategy):
    def search(
        self,
        ref_subset: np.ndarray,
        deformed_image: np.ndarray,
        search_range: SearchRange,
    ) -> InitialGuess | None:
        cx, cy, size = search_range.center_x, search_range.center_y, search_range.size
        half = size // 2
        x0, y0 = int(np.round(cx)) - half, int(np.round(cy)) - half
        x1, y1 = x0 + size, y0 + size

        # A window beyond the image boundary is rejected
        if (
            x0 < 0
            or y0 < 0
            or x1 > deformed_image.shape[1]
            or y1 > deformed_image.shape[0]
        ):
            return None

        window = deformed_image[y0:y1, x0:x1]
        local_guess = _find_peak_in_window(self.criterion, ref_subset, window)
        if local_guess is None or local_guess.score < self.threshold:
            return None

        # Window coordinates to image coordinates
        return InitialGuess(
            x=x0 + local_guess.x, y=y0 + local_guess.y, score=local_guess.score
        )


# Coarse-to-fine search over an image pyramid
class PyramidSearch(SearchStrategy):
    def __init__(
        self,
        criterion: CorrelationCriterion,
        threshold: float = 0.6,
        *,
        pyramid_levels: int = 3,
        pyramid_downsample_factor: int = 2,
        pyramid_fine_window: int = 8,
    ):
        super().__init__(criterion, threshold)
        self.levels = pyramid_levels
        self.downsample_factor = pyramid_downsample_factor
        self.fine_window = pyramid_fine_window

    @classmethod
    def check_setup(cls, config: SearchConfig, setup: SetupOutput) -> None:
        """Reject a pyramid whose coarsest level leaves the subset a single pixel."""
        coarse = coarsest_pyramid_subset(
            setup.subset_size, config.pyramid_levels, config.pyramid_downsample_factor
        )
        if coarse < MIN_COARSE_SUBSET_PX:
            raise ValueError(
                f"a pyramid of {config.pyramid_levels} levels with factor "
                f"{config.pyramid_downsample_factor} reduces the {setup.subset_size} px "
                f"subset to {coarse} px at its coarsest level, where the criterion is "
                "undefined; use fewer levels or a smaller factor"
            )
        if coarse < WARN_COARSE_SUBSET_PX:
            logging.warning(
                "The coarsest pyramid level keeps a %d px subset; expect lost nodes.",
                coarse,
            )

    @staticmethod
    def _coarse_to_fine(coord: float, scale: int) -> float:
        """Centre of a coarse-level pixel, in original-image coordinates."""
        return coord * scale + (scale - 1) / 2.0

    @staticmethod
    def _downsample(arr: np.ndarray, factor: int) -> np.ndarray:
        if factor <= 1:
            return arr
        h, w = arr.shape

        # Trim to a multiple of the downsampling factor so the block reshape is exact
        h_trim, w_trim = h - (h % factor), w - (w % factor)

        # centered
        r0, c0 = (h - h_trim) // 2, (w - w_trim) // 2
        trimmed = arr[r0 : r0 + h_trim, c0 : c0 + w_trim]
        return trimmed.reshape(h_trim // factor, factor, w_trim // factor, factor).mean(
            axis=(1, 3)
        )

    def search(
        self,
        ref_subset: np.ndarray,
        deformed_image: np.ndarray,
        search_range: SearchRange,
    ) -> InitialGuess | None:
        cx, cy = search_range.center_x, search_range.center_y
        guess = None

        # From the coarsest level down to the original scale (level 0)
        for level in range(self.levels - 1, -1, -1):
            scale = self.downsample_factor**level
            is_finest = level == 0

            # Subset and search window at the current level
            level_subset = (
                ref_subset if is_finest else self._downsample(ref_subset, scale)
            )
            level_size = (
                level_subset.shape[0] + self.fine_window
                if is_finest
                else max(search_range.size // scale, level_subset.shape[0] + 4)
            )
            crop_size = level_size if is_finest else level_size * scale

            half = crop_size // 2
            x0, y0 = int(round(cx)) - half, int(round(cy)) - half
            x1, y1 = x0 + crop_size, y0 + crop_size

            # Keep the crop inside the image
            if (
                x0 < 0
                or y0 < 0
                or x1 > deformed_image.shape[1]
                or y1 > deformed_image.shape[0]
            ):
                return None

            # Downsample the crop on coarse levels
            crop = deformed_image[y0:y1, x0:x1]
            level_window = crop if is_finest else self._downsample(crop, scale)

            guess = _find_peak_in_window(self.criterion, level_subset, level_window)
            if guess is None:
                return None

            if is_finest and guess.score < self.threshold:
                return None

            # Estimate carried to the next finer level
            cx = x0 + (guess.x if is_finest else self._coarse_to_fine(guess.x, scale))
            cy = y0 + (guess.y if is_finest else self._coarse_to_fine(guess.y, scale))

        return InitialGuess(x=cx, y=cy, score=guess.score)

    @classmethod
    def from_config(cls, criterion, threshold, config: SearchConfig) -> "PyramidSearch":
        return cls(
            criterion,
            threshold,
            pyramid_levels=config.pyramid_levels,
            pyramid_downsample_factor=config.pyramid_downsample_factor,
            pyramid_fine_window=config.pyramid_fine_window,
        )
