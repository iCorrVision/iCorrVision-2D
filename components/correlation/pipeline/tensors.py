import math
from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial import cKDTree

from components.contracts import SetupOutput, StrainConfig, StrainField


def cloud_neighbour_count(radius_px: float, step_size: int) -> int:
    """Nodes within radius_px of an interior node, itself included, on the square grid."""
    reach = int(radius_px // step_size)
    return sum(
        1
        for i in range(-reach, reach + 1)
        for j in range(-reach, reach + 1)
        if (i * i + j * j) * step_size**2 <= radius_px**2
    )


def minimum_cloud_radius(step_size: int, min_neighbors: int) -> float:
    """Smallest radius at which an interior node reaches min_neighbors nodes."""
    reach = math.isqrt(min_neighbors) + 1
    distances = sorted(
        math.hypot(i, j) * step_size
        for i in range(-reach, reach + 1)
        for j in range(-reach, reach + 1)
    )
    return distances[min_neighbors - 1]


def _green_lagrange_from_gradients(ux, uy, vx, vy) -> StrainField:
    exx = ux + 0.5 * (ux**2 + vx**2)
    eyy = vy + 0.5 * (uy**2 + vy**2)
    exy = 0.5 * (uy + vx) + 0.5 * (ux * uy + vx * vy)

    center = (exx + eyy) / 2.0
    spread = np.sqrt(((exx - eyy) / 2.0) ** 2 + exy**2)

    return StrainField(
        exx=exx,
        eyy=eyy,
        exy=exy,
        e1=center + spread,
        e2=center - spread,
        principal_angle_deg=np.degrees(0.5 * np.arctan2(2.0 * exy, exx - eyy)),
    )


class StrainTensor(ABC):
    @abstractmethod
    def compute(self, displacement_field, reference_grid) -> StrainField:
        pass

    @classmethod
    def from_config(cls, config: StrainConfig) -> "StrainTensor":
        return cls()

    @classmethod
    def check_setup(cls, config: StrainConfig, setup: SetupOutput) -> None:
        """Raise ValueError if this estimator cannot work with `setup`; any setup by default."""


class PointCloudGreenLagrangeStrain(StrainTensor):
    """Green-Lagrange strain from a meshfree fit over neighbouring nodes."""

    def __init__(self, radius_px: float = 15.0, min_neighbors: int = 4):
        self.radius_px = radius_px
        self.min_neighbors = min_neighbors

    @classmethod
    def from_config(cls, config: StrainConfig) -> "PointCloudGreenLagrangeStrain":
        return cls(
            radius_px=config.radius_px,
            min_neighbors=config.min_neighbors,
        )

    @classmethod
    def check_setup(cls, config: StrainConfig, setup: SetupOutput) -> None:
        """Reject a radius that reaches fewer grid nodes than min_neighbors."""
        reached = cloud_neighbour_count(config.radius_px, setup.step_size)
        if reached < config.min_neighbors:
            needed = minimum_cloud_radius(setup.step_size, config.min_neighbors)
            raise ValueError(
                f"radius_px = {config.radius_px:g} reaches {reached} node(s) at step "
                f"{setup.step_size} px, fewer than min_neighbors = {config.min_neighbors}, "
                f"so no strain could be computed; use a radius of at least {needed:g} px"
            )

    def compute(self, displacement_field, reference_grid) -> StrainField:
        valid = displacement_field.valid_mask

        x_v, y_v = reference_grid.x[valid], reference_grid.y[valid]
        u_v, v_v = displacement_field.u[valid], displacement_field.v[valid]

        if len(x_v) == 0:
            empty = np.full(reference_grid.x.shape, np.nan, dtype=np.float64)
            return StrainField(
                exx=empty,
                eyy=empty,
                exy=empty,
                e1=empty,
                e2=empty,
                principal_angle_deg=empty,
            )

        pts = np.column_stack((x_v, y_v))
        neighbor_indices = cKDTree(pts).query_ball_point(pts, r=self.radius_px)

        ux = np.full(len(pts), np.nan, dtype=np.float64)
        uy = np.full(len(pts), np.nan, dtype=np.float64)
        vx = np.full(len(pts), np.nan, dtype=np.float64)
        vy = np.full(len(pts), np.nan, dtype=np.float64)

        for i, idxs in enumerate(neighbor_indices):
            if len(idxs) < self.min_neighbors:
                continue

            dx, dy = pts[idxs, 0] - pts[i, 0], pts[idxs, 1] - pts[i, 1]
            A = np.column_stack((dx, dy, np.ones_like(dx)))
            AtA = A.T @ A

            try:
                sol_u = np.linalg.solve(AtA, A.T @ u_v[idxs])
                sol_v = np.linalg.solve(AtA, A.T @ v_v[idxs])
                ux[i], uy[i], vx[i], vy[i] = sol_u[0], sol_u[1], sol_v[0], sol_v[1]
            except np.linalg.LinAlgError:
                continue

        shape = reference_grid.x.shape
        ux_full = np.full(shape, np.nan, dtype=np.float64)
        uy_full, vx_full, vy_full = ux_full.copy(), ux_full.copy(), ux_full.copy()

        ux_full[valid], uy_full[valid] = ux, uy
        vx_full[valid], vy_full[valid] = vx, vy

        return _green_lagrange_from_gradients(ux_full, uy_full, vx_full, vy_full)


class WindowGreenLagrangeStrain(StrainTensor):
    _MIN_REDUNDANCY = 1.5  # valid points required, as a multiple of the parameter count

    def __init__(self, window_size: int = 5, order: str = "q4"):
        if window_size < 3 or window_size % 2 == 0:
            raise ValueError("window_size (grid points) must be odd and >= 3")
        if order not in ("q4", "q9"):
            raise ValueError(f"unsupported strain fit order: {order!r}")
        if order == "q9" and window_size < 5:
            raise ValueError("q9 needs window_size >= 5 to have any redundancy")
        self.window_size = window_size
        self.order = order
        self._half = window_size // 2
        self._n_params = 4 if order == "q4" else 9

    @classmethod
    def from_config(cls, config: StrainConfig) -> "WindowGreenLagrangeStrain":
        order = "q9" if config.method == "lagrange_q9" else "q4"
        return cls(window_size=config.strain_window, order=order)

    def _design_matrix(self, m: np.ndarray, n: np.ndarray) -> np.ndarray:
        cols = [np.ones_like(m), m, n, m * n]
        if self.order == "q9":
            cols += [m**2, n**2, m**2 * n, m * n**2, m**2 * n**2]
        return np.column_stack(cols)

    def compute(self, displacement_field, reference_grid) -> StrainField:
        valid = displacement_field.valid_mask
        rows, cols = valid.shape
        half = self._half
        min_points = int(np.ceil(self._MIN_REDUNDANCY * self._n_params))

        x, y = reference_grid.x, reference_grid.y
        u, v = displacement_field.u, displacement_field.v

        ux = np.full(valid.shape, np.nan, dtype=np.float64)
        uy = np.full(valid.shape, np.nan, dtype=np.float64)
        vx = np.full(valid.shape, np.nan, dtype=np.float64)
        vy = np.full(valid.shape, np.nan, dtype=np.float64)

        for i in range(rows):
            r0, r1 = max(0, i - half), min(rows, i + half + 1)
            for j in range(cols):
                if not valid[i, j]:
                    continue
                c0, c1 = max(0, j - half), min(cols, j + half + 1)

                block_valid = valid[r0:r1, c0:c1]
                if block_valid.sum() < min_points:
                    continue

                m = (x[r0:r1, c0:c1] - x[i, j])[block_valid]
                n = (y[r0:r1, c0:c1] - y[i, j])[block_valid]
                A = self._design_matrix(m, n)
                AtA = A.T @ A

                try:
                    sol_u = np.linalg.solve(AtA, A.T @ u[r0:r1, c0:c1][block_valid])
                    sol_v = np.linalg.solve(AtA, A.T @ v[r0:r1, c0:c1][block_valid])
                except np.linalg.LinAlgError:
                    continue

                # columns are [1, m, n, mn, ...]; b=coeff of m, c=coeff of n
                ux[i, j], uy[i, j] = sol_u[1], sol_u[2]
                vx[i, j], vy[i, j] = sol_v[1], sol_v[2]

        return _green_lagrange_from_gradients(ux, uy, vx, vy)
