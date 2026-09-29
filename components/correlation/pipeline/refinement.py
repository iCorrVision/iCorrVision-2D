from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
import logging

from components.contracts import RefinementConfig

from .criteria import CorrelationCriterion, zero_mean_normalize
from .interpolation import SubpixelInterpolator
from .search import InitialGuess
from .shape_function import ShapeFunction


# Result of the sub-pixel refinement of one node
@dataclass
class DisplacementResult:
    x_peak: float
    y_peak: float
    p_vector: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    score: float = np.nan
    converged: bool = False


# Interface of the sub-pixel refinement methods
class SubpixelRefinement(ABC):
    def __init__(
        self,
        criterion: CorrelationCriterion,
        interpolator: SubpixelInterpolator,
        shape_function: ShapeFunction,
    ):
        self.criterion = criterion
        self.interpolator = interpolator
        self.shape_function = shape_function

    @abstractmethod
    def refine(
        self,
        ref_subset: np.ndarray,
        deformed_image: np.ndarray,
        initial_guess: InitialGuess,
    ) -> DisplacementResult:
        pass

    @classmethod
    def from_config(
        cls, criterion, interpolator, shape_function, config: RefinementConfig
    ) -> "SubpixelRefinement":
        return cls(criterion, interpolator, shape_function)


# Inverse compositional Gauss-Newton (IC-GN) refinement
class ICGNRefinement(SubpixelRefinement):

    # Condition number above which the Hessian is treated as singular: the subset has
    # too little texture to determine all shape-function parameters.
    _MAX_HESSIAN_COND: float = 1e12

    def __init__(
        self,
        criterion: CorrelationCriterion,
        interpolator: SubpixelInterpolator,
        shape_function: ShapeFunction,
        max_iterations: int = 20,
        tolerance: float = 1e-3,
        # Pixels around the subset kept in the window: they hold the interpolant's reach
        # (up to 4 px each side, for 8-tap) and any movement during the iterations.
        # Ideally derived from the interpolator's support rather than fixed here.
        window_margin: int = 4,
    ):
        super().__init__(criterion, interpolator, shape_function)
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.window_margin = window_margin

    def _failed(self, cx0: float, cy0: float) -> DisplacementResult:
        """Unrefined fallback: the coarse guess, no warp, NaN score."""
        return DisplacementResult(cx0, cy0, self.shape_function.identity())

    def refine(self, ref_subset, deformed_image, initial_guess):
        # Window around the initial guess: half the subset plus the margin
        sub_h, sub_w = ref_subset.shape
        half_w, half_h = (sub_w - 1) / 2.0, (sub_h - 1) / 2.0
        pad = int(np.ceil(max(half_w, half_h))) + self.window_margin

        cx0, cy0 = initial_guess.x, initial_guess.y
        wx0, wy0 = int(np.floor(cx0)) - pad, int(np.floor(cy0)) - pad
        wx1, wy1 = int(np.ceil(cx0)) + pad, int(np.ceil(cy0)) + pad

        # A window beyond the image leaves the node unrefined
        if (
            wx0 < 0
            or wy0 < 0
            or wx1 >= deformed_image.shape[1]
            or wy1 >= deformed_image.shape[0]
        ):
            return DisplacementResult(cx0, cy0)

        window = deformed_image[wy0 : wy1 + 1, wx0 : wx1 + 1]
        win_h, win_w = window.shape
        prepared = self.interpolator.prepare(window)

        def sample(def_x, def_y):
            col, row = def_x - wx0, def_y - wy0
            if (
                col.min() < 0
                or row.min() < 0
                or col.max() >= win_w - 1
                or row.max() >= win_h - 1
            ):
                return None
            return self.interpolator.sample_at(prepared, row, col)

        # Fixed over the iterations: the saving of the inverse-compositional form
        xs = np.arange(sub_w, dtype=np.float64) - half_w
        ys = np.arange(sub_h, dtype=np.float64) - half_h
        x_grid, y_grid = np.meshgrid(xs, ys)
        x_flat, y_flat = x_grid.ravel(), y_grid.ravel()

        f_vec = ref_subset.ravel()
        grad_fy, grad_fx = np.gradient(ref_subset)

        f_zero, norm_f = zero_mean_normalize(f_vec)
        if norm_f < 1e-8:
            return DisplacementResult(cx0, cy0)

        grad_f_star = np.stack(
            [grad_fx.ravel() / norm_f, grad_fy.ravel() / norm_f], axis=1
        )

        dW_dp = self.shape_function.jacobian(x_flat, y_flat)
        J_star = np.einsum("ni,nij->nj", grad_f_star, dW_dp)
        Hessian = J_star.T @ J_star

        try:
            H_inv = np.linalg.inv(Hessian)
        except np.linalg.LinAlgError:
            logging.debug("ICGN: singular Hessian, aborting subset")
            return DisplacementResult(cx0, cy0)

        if (
            np.linalg.norm(Hessian, 1) * np.linalg.norm(H_inv, 1)
            > self._MAX_HESSIAN_COND
        ):
            logging.debug("ICGN: ill-conditioned Hessian, aborting subset")
            return self._failed(cx0, cy0)

        p = self.shape_function.identity()
        converged = False
        for _ in range(self.max_iterations):
            def_x, def_y = self.shape_function.warp(cx0, cy0, p, x_flat, y_flat)
            g_vec = sample(def_x, def_y)

            if g_vec is None:
                break
            g_zero, norm_g = zero_mean_normalize(g_vec)
            if norm_g < 1e-8:
                break
            diff = (g_zero / norm_g) - (f_zero / norm_f)
            dp = H_inv @ (J_star.T @ diff)
            try:
                p = self.shape_function.compose(p, dp)
            except np.linalg.LinAlgError:
                logging.debug("ICGN: singular warp increment, aborting subset")
                break
            du, dv = self.shape_function.translation(dp)
            if np.hypot(du, dv) < self.tolerance:
                converged = True
                break

        def_x, def_y = self.shape_function.warp(cx0, cy0, p, x_flat, y_flat)
        g_final = sample(def_x, def_y)
        score = (
            float(self.criterion.evaluate(f_vec, g_final))
            if g_final is not None
            else np.nan
        )

        u, v = self.shape_function.translation(p)
        return DisplacementResult(
            x_peak=cx0 + u, y_peak=cy0 + v, p_vector=p, score=score, converged=converged
        )
