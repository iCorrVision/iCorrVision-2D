from abc import ABC, abstractmethod
import numpy as np


class ShapeFunction(ABC):

    _MIN_DET: float = 1e-6  # a physical warp has det close to 1; near 0 means numerical breakdown

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Number of shape-function parameters (6 affine, 12 quadratic)."""
        raise NotImplementedError

    @abstractmethod
    def jacobian(self, x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
        """dW/dp evaluated at each subset-local coordinate.

        x_coords, y_coords: flat arrays of subset-local offsets from the subset
        centre, shape (N,).
        Returns: array of shape (N, 2, n_params).
        """

        raise NotImplementedError

    @abstractmethod
    def warp(
        self,
        cx0: float,
        cy0: float,
        p: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map subset-local coordinates to deformed-image coordinates, given the
        subset centre (cx0, cy0) and the parameter vector p.
        """

        raise NotImplementedError

    def identity(self) -> np.ndarray:
        """Zero displacement vector, the IC-GN iteration starting point."""

        return np.zeros(self.n_params, dtype=np.float64)

    @abstractmethod
    def compose(self, p: np.ndarray, dp: np.ndarray) -> np.ndarray:
        """Inverse compositional update: W(.;p) <- W(.;p) o W(.;dp)^-1 returned
        as a new parameter vector p.

        May raise np.linalg.LinAlgError; handling it is left to ICGNRefinement.
        """

        raise NotImplementedError

    @abstractmethod
    def translation(self, p: np.ndarray) -> tuple[float, float]:
        """Return the translation components (u, v) of a parameter vector."""
        raise NotImplementedError


class AffineShapeFunction(ShapeFunction):
    """First-order (affine) shape function: translation plus the full local
    displacement gradient.

    6 parameters: p(u, v, du/dx, du/dy, dv/dx, dv/dy)^T.
    """

    @property
    def n_params(self) -> int:
        return 6

    @staticmethod
    def _to_matrix(p: np.ndarray) -> np.ndarray:
        """Convert a 6-parameter affine vector to a 3x3 transformation matrix."""
        return np.array(
            [
                [1.0 + p[2], p[3], p[0]],
                [p[4], 1.0 + p[5], p[1]],
                [0.0, 0.0, 1.0],
            ],
            np.float64,
        )

    @staticmethod
    def _to_params(M: np.ndarray) -> np.ndarray:
        """Extract the 6-parameter affine vector from a 3x3 transformation matrix."""
        return np.array(
            [M[0, 2], M[1, 2], M[0, 0] - 1.0, M[0, 1], M[1, 0], M[1, 1] - 1.0],
            dtype=np.float64,
        )

    def jacobian(self, x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
        N = x_coords.size
        dW_dp = np.zeros((N, 2, 6), np.float64)
        dW_dp[:, 0, 0] = dW_dp[:, 1, 1] = 1.0
        dW_dp[:, 0, 2] = dW_dp[:, 1, 4] = x_coords
        dW_dp[:, 0, 3] = dW_dp[:, 1, 5] = y_coords
        return dW_dp

    def warp(
        self,
        cx0: float,
        cy0: float,
        p: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        def_x = cx0 + p[0] + (1.0 + p[2]) * x_coords + p[3] * y_coords
        def_y = cy0 + p[1] + p[4] * x_coords + (1.0 + p[5]) * y_coords
        return def_x, def_y

    def compose(self, p: np.ndarray, dp: np.ndarray) -> np.ndarray:
        det = (1.0 + dp[2]) * (1.0 + dp[5]) - dp[3] * dp[4]
        if abs(det) < self._MIN_DET:
            raise np.linalg.LinAlgError(
                f"singular warp increment (det={det:.3e}) -> refinement is diverging"
            )

        M_p = self._to_matrix(p)
        M_dp = self._to_matrix(dp)
        M_new = M_p @ np.linalg.inv(M_dp)
        return self._to_params(M_new)

    def translation(self, p: np.ndarray) -> tuple[float, float]:
        return float(p[0]), float(p[1])


class QuadraticShapeFunction(ShapeFunction):
    """Second-order (quadratic) shape function, 12 parameters.

    PARAMETER ORDER
        p = (u, v, ux, uy, vx, vy, uxx, uxy, uyy, vxx, vxy, vyy)

    CONVENTION
        xi  = dx + u + ux dx + uy dy + 1/2 uxx dx^2 + uxy dx dy + 1/2 uyy dy^2
        eta = dy + v + vx dx + vy dy + 1/2 vxx dx^2 + vxy dx dy + 1/2 vyy dy^2
    """

    @property
    def n_params(self) -> int:
        return 12

    @staticmethod
    def _to_matrix(p: np.ndarray) -> np.ndarray:
        u, v, ux, uy, vx, vy, uxx, uxy, uyy, vxx, vxy, vyy = p
        a, b = 1.0 + ux, uy  # linear part of xi
        c, d = vx, 1.0 + vy  # linear part of eta
        M = np.zeros((6, 6), dtype=np.float64)
        M[0] = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        M[1] = (u, a, b, 0.5 * uxx, uxy, 0.5 * uyy)
        M[2] = (v, c, d, 0.5 * vxx, vxy, 0.5 * vyy)
        M[3] = (
            u * u,
            2.0 * u * a,
            2.0 * u * b,
            a * a + u * uxx,
            2.0 * a * b + 2.0 * u * uxy,
            b * b + u * uyy,
        )
        M[4] = (
            u * v,
            u * c + v * a,
            u * d + v * b,
            a * c + 0.5 * u * vxx + 0.5 * v * uxx,
            a * d + b * c + u * vxy + v * uxy,
            b * d + 0.5 * u * vyy + 0.5 * v * uyy,
        )
        M[5] = (
            v * v,
            2.0 * v * c,
            2.0 * v * d,
            c * c + v * vxx,
            2.0 * c * d + 2.0 * v * vxy,
            d * d + v * vyy,
        )
        return M

    @staticmethod
    def _to_params(M: np.ndarray) -> np.ndarray:
        """Read the twelve parameters back out of rows 1 and 2."""
        return np.array(
            [
                M[1, 0],
                M[2, 0],
                M[1, 1] - 1.0,
                M[1, 2],
                M[2, 1],
                M[2, 2] - 1.0,
                2.0 * M[1, 3],
                M[1, 4],
                2.0 * M[1, 5],
                2.0 * M[2, 3],
                M[2, 4],
                2.0 * M[2, 5],
            ],
            dtype=np.float64,
        )

    def jacobian(self, x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
        N = x_coords.size
        x, y = x_coords, y_coords
        dW_dp = np.zeros((N, 2, 12), np.float64)
        dW_dp[:, 0, 0] = dW_dp[:, 1, 1] = 1.0  # u,   v
        dW_dp[:, 0, 2] = dW_dp[:, 1, 4] = x  # ux,  vx
        dW_dp[:, 0, 3] = dW_dp[:, 1, 5] = y  # uy,  vy
        dW_dp[:, 0, 6] = dW_dp[:, 1, 9] = 0.5 * x * x  # uxx, vxx
        dW_dp[:, 0, 7] = dW_dp[:, 1, 10] = x * y  # uxy, vxy
        dW_dp[:, 0, 8] = dW_dp[:, 1, 11] = 0.5 * y * y  # uyy, vyy
        return dW_dp

    def warp(
        self,
        cx0: float,
        cy0: float,
        p: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        x, y = x_coords, y_coords
        xx, xy, yy = x * x, x * y, y * y
        def_x = (
            cx0
            + x
            + p[0]
            + p[2] * x
            + p[3] * y
            + 0.5 * p[6] * xx
            + p[7] * xy
            + 0.5 * p[8] * yy
        )
        def_y = (
            cy0
            + y
            + p[1]
            + p[4] * x
            + p[5] * y
            + 0.5 * p[9] * xx
            + p[10] * xy
            + 0.5 * p[11] * yy
        )
        return def_x, def_y

    def compose(self, p: np.ndarray, dp: np.ndarray) -> np.ndarray:
        det = (1.0 + dp[2]) * (1.0 + dp[5]) - dp[3] * dp[4]
        if abs(det) < self._MIN_DET:
            raise np.linalg.LinAlgError(
                f"singular warp increment (det={det:.3e}); IC-GN is diverging"
            )

        M_new = self._to_matrix(p) @ np.linalg.inv(self._to_matrix(dp))
        return self._to_params(M_new)

    def translation(self, p: np.ndarray) -> tuple[float, float]:
        return float(p[0]), float(p[1])
