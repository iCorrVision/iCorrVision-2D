from dataclasses import dataclass
import numpy as np
from components.contracts import RecoveryConfig


@dataclass
class RecoveryGroup:
    source_frame: int
    flat_indices: np.ndarray
    lost_frames: int


class NodeHistory:
    """Per-node record of the last frame on which each node was matched."""

    def __init__(self, x0: np.ndarray, y0: np.ndarray, mask: np.ndarray) -> None:
        self.shape = x0.shape
        # Frame 0 is the reference, where every in-ROI node is by definition
        # at its grid position. -1 marks nodes outside the ROI entirely.
        self.last_good_frame = np.where(mask, 0, -1).astype(np.int64)
        self.last_good_x = x0.copy()
        self.last_good_y = y0.copy()

    def update(self, frame_index: int, x: np.ndarray, y: np.ndarray) -> None:
        """Record positions for every node that converged on this frame."""
        got = np.isfinite(x) & np.isfinite(y)
        self.last_good_frame = np.where(got, frame_index, self.last_good_frame)
        self.last_good_x = np.where(got, x, self.last_good_x)
        self.last_good_y = np.where(got, y, self.last_good_y)


def predict_from_neighbours(
    field: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    radius: int = 3,
    min_points: int = 6,
) -> np.ndarray:
    """Fit a least-squares plane to converged neighbours at each target node.

    Returns NaN where too few neighbours are available.

    Args:
        field: grid-shaped displacement component, NaN where unconverged.
        mask: nodes inside the ROI.
        targets: boolean grid marking nodes to predict at.
        radius: half-width, in grid nodes, of the fitting window.
        min_points: minimum converged neighbours required to attempt a fit.
    """
    out = np.full(field.shape, np.nan, dtype=np.float64)
    rows, cols = field.shape
    usable = mask & np.isfinite(field)

    for i, j in np.argwhere(targets):
        r0, r1 = max(0, i - radius), min(rows, i + radius + 1)
        c0, c1 = max(0, j - radius), min(cols, j + radius + 1)
        block = field[r0:r1, c0:c1]
        good = usable[r0:r1, c0:c1].copy()
        good[i - r0, j - c0] = False  # remove the lost point from the fit
        if good.sum() < min_points:
            continue

        rr, cc = np.mgrid[r0:r1, c0:c1]
        dy = (rr[good] - i).astype(np.float64)
        dx = (cc[good] - j).astype(np.float64)
        A = np.column_stack((np.ones_like(dx), dx, dy))
        try:
            coef, *_ = np.linalg.lstsq(A, block[good], rcond=None)
        except np.linalg.LinAlgError:
            continue
        out[i, j] = coef[0]  # plane at the node itself, where dx = dy = 0
    return out


class HistoryRecovery:
    """Plan and vet recovery attempts for nodes that lost correlation."""

    def __init__(
        self,
        max_lost_frames: int = 2,
        continuity_tol_px: float = 5.0,
        fit_radius: int = 3,
        min_fit_points: int = 6,
    ) -> None:
        self.max_lost_frames = max_lost_frames
        self.continuity_tol_px = continuity_tol_px
        self.fit_radius = fit_radius
        self.min_fit_points = min_fit_points

    @classmethod
    def from_config(cls, config: RecoveryConfig) -> "HistoryRecovery":
        return cls(
            max_lost_frames=config.max_lost_frames,
            continuity_tol_px=config.continuity_tol_px,
            fit_radius=config.fit_radius,
            min_fit_points=config.min_fit_points,
        )

    def plan(
        self, history: NodeHistory, converged: np.ndarray, frame_index: int
    ) -> list[RecoveryGroup]:
        """Group the currently-failed nodes by which frame they last matched.

        Nodes lost for longer than ``max_lost_frames`` are not returned and
        stay NaN for this frame. They remain eligible if they return within the
        window on a later frame, since a failed attempt leaves their history
        entry unchanged.
        """
        failed = (history.last_good_frame >= 0) & ~converged
        if not failed.any():
            return []

        lost = frame_index - history.last_good_frame
        eligible = failed & (lost >= 1) & (lost <= self.max_lost_frames)
        if not eligible.any():
            return []

        groups: list[RecoveryGroup] = []
        for source in np.unique(history.last_good_frame[eligible]):
            sel = eligible & (history.last_good_frame == source)
            groups.append(
                RecoveryGroup(
                    source_frame=int(source),
                    flat_indices=np.flatnonzero(sel.ravel()),
                    lost_frames=int(frame_index - source),
                )
            )
        return groups

    def seed(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x0: np.ndarray,
        y0: np.ndarray,
        mask: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predicted current position for each target, from its neighbours.

        Fitted in displacement space rather than position space, so the plane
        describes the deformation and not the grid geometry; this keeps the fit
        meaningful near the ROI edge.
        """
        pu = predict_from_neighbours(
            x - x0, mask, targets, self.fit_radius, self.min_fit_points
        )
        pv = predict_from_neighbours(
            y - y0, mask, targets, self.fit_radius, self.min_fit_points
        )
        return x0 + pu, y0 + pv

    def accept(
        self,
        measured_x: np.ndarray,
        measured_y: np.ndarray,
        predicted_x: np.ndarray,
        predicted_y: np.ndarray,
        lost_frames: int,
    ) -> np.ndarray:
        """Spatial-continuity check on a batch of recovered positions.

        A node that re-locked onto the wrong speckle disagrees with the
        neighbours it used to agree with; one that merely missed some motion
        does not, because its neighbours moved with it. Comparing against
        the neighbour prediction separates those two, which a threshold on
        raw jump magnitude cannot.

        The tolerance scales with the number of lost frames because the
        prediction is extrapolated further in time. This loosens the gate for
        the nodes whose seeds are least reliable, which is why
        ``max_lost_frames`` should stay small.
        """
        residual = np.hypot(measured_x - predicted_x, measured_y - predicted_y)
        tol = self.continuity_tol_px * max(1, lost_frames)
        return np.isfinite(residual) & (residual <= tol)
