from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Callable
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import binary_erosion

from components.contracts import (
    CorrelationConfig,
    CorrelationFrame,
    CorrelationResult,
    SetupOutput,
)
from components.correlation.pipeline.search import (
    SearchStrategy,
    SearchRange,
    InitialGuess,
)
from components.correlation.pipeline.refinement import SubpixelRefinement
from components.correlation.pipeline.tensors import StrainTensor
from .recovery import HistoryRecovery, NodeHistory


@dataclass
class DisplacementField:
    u: np.ndarray
    v: np.ndarray
    valid_mask: np.ndarray


@dataclass
class ReferenceGrid:
    x: np.ndarray
    y: np.ndarray
    mask: np.ndarray
    step_size: int


class GridTracker:
    """Subset grid, ROI mask and node positions tracked over the frames."""

    def __init__(self, setup: SetupOutput, tracking_mode: str, search_size: int):
        self.tracking_mode = tracking_mode
        self.x0, self.y0, self.mask = self._build_grid(setup, search_size)
        self.cur_x, self.cur_y = self.x0.copy(), self.y0.copy()

    def _build_grid(  # search_size is unused: the margin is the subset half-width
        self, setup: SetupOutput, search_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return node_grid(setup.roi_mask, setup.subset_size, setup.step_size)

    def get_centers(self, mode: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the node positions the next frame is correlated from."""
        return (self.cur_x, self.cur_y) if mode == "incremental" else (self.x0, self.y0)

    def update(self, new_x: np.ndarray, new_y: np.ndarray) -> None:
        """Move the nodes to their matched positions under Lagrangian tracking."""
        if self.tracking_mode == "lagrangian":
            self.cur_x = np.where(np.isfinite(new_x), new_x, self.cur_x)
            self.cur_y = np.where(np.isfinite(new_y), new_y, self.cur_y)


def node_grid(
    roi_mask: np.ndarray, subset_size: int, step_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference node grid: (grid_x, grid_y, mask).

    The ROI is eroded by the subset half-width, and nodes are placed every
    step_size pixels from the corner of what remains; mask marks the nodes
    inside the eroded ROI. The setup preview draws the same grid.
    """
    margin = subset_size // 2
    roi_mask = (
        binary_erosion(roi_mask, np.ones((2 * margin + 1, 2 * margin + 1), dtype=bool))
        if margin > 0
        else roi_mask
    )

    ys, xs = np.where(roi_mask)
    if ys.size == 0:
        raise ValueError("ROI mask is empty after eroding by subset margin.")

    y_min, y_max, x_min, x_max = (
        int(ys.min()),
        int(ys.max()),
        int(xs.min()),
        int(xs.max()),
    )
    x_coords = np.arange(x_min, x_max + 1, step_size, dtype=np.float64)
    y_coords = np.arange(y_min, y_max + 1, step_size, dtype=np.float64)

    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    mask = roi_mask[grid_y.astype(np.int64), grid_x.astype(np.int64)]
    return grid_x, grid_y, mask


def _to_gray(img: np.ndarray) -> np.ndarray:
    return (
        img.astype(np.float32, copy=False)
        if img.ndim == 2
        else img[..., :3].astype(np.float32, copy=False).mean(axis=2)
    )


class CorrelationEngine:

    REACQUIRE_STALE: bool = True

    def __init__(
        self,
        search: SearchStrategy,
        refinement: SubpixelRefinement,
        tensor: StrainTensor,
        setup: SetupOutput,
        config: CorrelationConfig,
        frame_loader,
        recovery: HistoryRecovery | None = None,
    ):
        self.search: SearchStrategy = search
        self.refinement: SubpixelRefinement = refinement
        self.tensor: StrainTensor = tensor
        self.setup: SetupOutput = setup
        self.config: CorrelationConfig = config
        self.frame_loader = frame_loader
        self.recovery: HistoryRecovery | None = recovery
        self._reset_state()

    def _reset_state(self) -> None:
        """Put every node back on the reference grid, with a fresh history."""
        self.tracker: GridTracker = GridTracker(
            self.setup, self.config.tracking_mode, self.config.search_size
        )
        self.history: NodeHistory = NodeHistory(
            self.tracker.x0, self.tracker.y0, self.tracker.mask
        )

    # ------------------------------------------------------------------
    # Work partitioning
    # ------------------------------------------------------------------

    def _chunks_for(self, mask_flat: np.ndarray) -> list[np.ndarray]:
        """Split the nodes to be processed across workers."""
        indices = np.flatnonzero(mask_flat)
        if indices.size == 0:
            return []
        n = max(1, self.config.n_workers)
        return [c for c in (indices[i::n] for i in range(n)) if c.size]

    def _process_point(
        self,
        cx: float,
        cy: float,
        ref_image: np.ndarray,
        def_image: np.ndarray,
        seed: tuple[float, float] | None = None,
    ) -> tuple[float, float, float, float, float, float]:
        half = self.setup.subset_size // 2
        x0_int, y0_int = int(np.round(cx)) - half, int(np.round(cy)) - half

        if (
            x0_int < 0
            or y0_int < 0
            or (x0_int + self.setup.subset_size) > ref_image.shape[1]
            or (y0_int + self.setup.subset_size) > ref_image.shape[0]
        ):
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        ref_subset = ref_image[
            y0_int : y0_int + self.setup.subset_size,
            x0_int : x0_int + self.setup.subset_size,
        ]

        if seed is None:
            guess = self.search.search(
                ref_subset, def_image, SearchRange(cx, cy, self.config.search_size)
            )
        else:
            guess = InitialGuess(x=seed[0], y=seed[1], score=np.nan)

        if guess is None:  # explicit: a dataclass instance is always truthy
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        res = self.refinement.refine(ref_subset, def_image, guess)
        if not np.isfinite(res.score) or res.score < self.config.match_threshold:
            return np.nan, np.nan, np.nan, np.nan, res.score, np.nan

        delta_x = cx - (x0_int + half)
        delta_y = cy - (y0_int + half)
        return (
            cx,
            cy,
            res.x_peak + delta_x,
            res.y_peak + delta_y,
            res.score,
            float(res.converged),
        )

    def _process_block(
        self,
        block_y: np.ndarray,
        block_x: np.ndarray,
        mask: np.ndarray,
        ref_img: np.ndarray,
        def_img: np.ndarray,
        abort_check: Callable[[], bool] | None = None,
        seed_x: np.ndarray | None = None,
        seed_y: np.ndarray | None = None,
    ):
        results = np.full((len(block_y), 6), np.nan, dtype=np.float64)
        for i, (cy, cx, m) in enumerate(zip(block_y, block_x, mask)):
            if abort_check and abort_check():
                break
            if m:
                seed = (
                    None
                    if (seed_x is None or seed_y is None)
                    else (seed_x[i], seed_y[i])
                )
                results[i] = self._process_point(cx, cy, ref_img, def_img, seed)
        return results

    def _dispatch(
        self,
        chunks,
        mask_flat: np.ndarray,
        ref_img: np.ndarray,
        def_img: np.ndarray,
        cy: np.ndarray,
        cx: np.ndarray,
        abort_check: Callable[[], bool] | None = None,
        seed_x: np.ndarray | None = None,
        seed_y: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run one threaded correlation pass, returning fresh flat arrays."""
        n = mask_flat.size
        x = np.full(n, np.nan, dtype=np.float64)
        y = np.full(n, np.nan, dtype=np.float64)
        s = np.full(n, np.nan, dtype=np.float64)
        c = np.full(n, np.nan, dtype=np.float64)
        if not chunks:
            return x, y, s, c
        with ThreadPoolExecutor(max_workers=self.config.n_workers) as pool:
            futures = [
                pool.submit(
                    self._process_block,
                    cy[ch],
                    cx[ch],
                    mask_flat[ch],
                    ref_img,
                    def_img,
                    abort_check,
                    None if seed_x is None else seed_x[ch],
                    None if seed_y is None else seed_y[ch],
                )
                for ch in chunks
            ]
            for ch, future in zip(chunks, futures):
                r = future.result()
                x[ch], y[ch], s[ch], c[ch] = r[:, 2], r[:, 3], r[:, 4], r[:, 5]
        return x, y, s, c

    # ------------------------------------------------------------------
    # Recovery passes
    # ------------------------------------------------------------------

    def _reject_rogues(self, x_out, y_out, shape) -> None:
        """NaN out nodes whose converged position disagrees with their neighbours."""
        converged = np.isfinite(x_out).reshape(shape)
        pred_x, pred_y = self.recovery.seed(
            x_out.reshape(shape),
            y_out.reshape(shape),
            self.tracker.x0,
            self.tracker.y0,
            self.tracker.mask,
            converged,
        )
        rogue = (
            converged.ravel()
            & np.isfinite(pred_x.ravel())
            & ~self.recovery.accept(x_out, y_out, pred_x.ravel(), pred_y.ravel(), 1)
        )
        x_out[rogue] = np.nan
        y_out[rogue] = np.nan

    def _retry(
        self,
        x_out,
        y_out,
        score_out,
        conv_out,
        recovered,
        targets,
        source_image,
        def_image,
        cy,
        cx,
        lost_frames,
        abort_check,
    ) -> None:
        """Seed `targets` from their neighbours, re-match, vet, and merge."""
        shape = self.tracker.x0.shape
        seed_x, seed_y = self.recovery.seed(
            x_out.reshape(shape),
            y_out.reshape(shape),
            self.tracker.x0,
            self.tracker.y0,
            self.tracker.mask,
            targets,
        )
        retry = (
            targets.ravel() & np.isfinite(seed_x.ravel()) & np.isfinite(seed_y.ravel())
        )
        if not retry.any():
            return  # no converged neighbours to predict from

        rx, ry, rs, rc = self._dispatch(
            self._chunks_for(retry),
            retry,
            source_image,
            def_image,
            cy,
            cx,
            abort_check,
            seed_x.ravel(),
            seed_y.ravel(),
        )
        ok = retry & self.recovery.accept(
            rx, ry, seed_x.ravel(), seed_y.ravel(), lost_frames
        )
        x_out[ok], y_out[ok], score_out[ok], conv_out[ok] = (
            rx[ok],
            ry[ok],
            rs[ok],
            rc[ok],
        )
        recovered.ravel()[ok] = True

    def run(
        self,
        progress_cb: Callable[[int, int], None] | None = None,
        abort_check: Callable[[], bool] | None = None,
    ) -> CorrelationResult:
        # Every run starts from the reference grid, so the engine can be run again.
        self._reset_state()
        ref_image = prev_def = _to_gray(self.setup.reference_frame)
        frames, frame_times = [], []

        # Frame 0: the reference
        frames.append(
            self._build_frame(
                0,
                "reference",
                self.tracker.x0,
                self.tracker.y0,
                np.zeros_like(self.tracker.x0),
                np.zeros_like(self.tracker.y0),
                np.ones_like(self.tracker.x0),
                np.zeros_like(self.tracker.x0, dtype=bool),
                converged=np.ones_like(self.tracker.x0, dtype=bool),
            )
        )

        flat_mask = self.tracker.mask.ravel()
        full_chunks = self._chunks_for(flat_mask)

        # Recovery re-matches against the last frame a node converged on, so
        # that many past frames are kept.
        keep = self.config.recovery.max_lost_frames + 1
        image_cache: dict[int, np.ndarray] = {0: ref_image}

        n_deformed = len(self.setup.deformed_list)

        for f_idx, f_name in enumerate(self.setup.deformed_list, start=1):
            if abort_check and abort_check():
                break
            t_start = perf_counter()

            def_image = _to_gray(self.frame_loader(f_name))
            base_image = (
                ref_image if self.config.correlation_mode == "spatial" else prev_def
            )
            cx_grid, cy_grid = self.tracker.get_centers(self.config.correlation_mode)

            flat_cx, flat_cy = cx_grid.ravel(), cy_grid.ravel()
            shape = self.tracker.x0.shape
            recovered = np.zeros(shape, dtype=bool)

            use_recovery = (
                self.recovery is not None
                and self.config.recovery.enabled
                and self.config.correlation_mode == "incremental"
                and self.config.tracking_mode != "eulerian"
            )

            if self.config.correlation_mode == "incremental":
                # A node lost on an earlier frame has no valid position in the previous
                # frame: its subset would belong to another material point. It stays
                # invalid unless recovery re-matches it below.
                live = flat_mask & (self.history.last_good_frame.ravel() == f_idx - 1)
                chunks = self._chunks_for(live)
            else:
                live = flat_mask
                chunks = full_chunks

            x_out, y_out, score_out, conv_out = self._dispatch(
                chunks, live, base_image, def_image, flat_cy, flat_cx, abort_check
            )

            if abort_check and abort_check():
                break

            if use_recovery:
                self._reject_rogues(x_out, y_out, shape)

            if use_recovery and self.recovery is not None:
                converged = np.isfinite(x_out).reshape(shape)

                handled = np.zeros(shape, dtype=bool)
                for group in self.recovery.plan(self.history, converged, f_idx):
                    source = image_cache.get(group.source_frame)
                    if source is None:
                        continue  # older than the retained window

                    targets = np.zeros(shape, dtype=bool)
                    targets.ravel()[group.flat_indices] = True
                    handled |= targets
                    self._retry(
                        x_out,
                        y_out,
                        score_out,
                        conv_out,
                        recovered,
                        targets,
                        source,
                        def_image,
                        self.history.last_good_y.ravel(),
                        self.history.last_good_x.ravel(),
                        group.lost_frames,
                        abort_check,
                    )

                if self.REACQUIRE_STALE:
                    stale = (
                        self.tracker.mask
                        & ~np.isfinite(x_out).reshape(shape)
                        & (self.history.last_good_frame >= 0)
                        & (
                            (f_idx - self.history.last_good_frame)
                            > self.config.recovery.max_lost_frames
                        )
                        & ~handled
                    )
                    if stale.any():
                        self._retry(
                            x_out,
                            y_out,
                            score_out,
                            conv_out,
                            recovered,
                            stale,
                            ref_image,
                            def_image,
                            self.tracker.y0.ravel(),
                            self.tracker.x0.ravel(),
                            1,
                            abort_check,
                        )

            x_mat, y_mat = x_out.reshape(shape), y_out.reshape(shape)

            if self.config.correlation_mode == "incremental":
                self.tracker.update(x_mat, y_mat)

            u_mat = x_mat - self.tracker.x0
            v_mat = y_mat - self.tracker.y0

            self.history.update(f_idx, x_mat, y_mat)
            if self.config.correlation_mode == "incremental":
                image_cache[f_idx] = def_image
                for stale_idx in [k for k in image_cache if 0 < k <= f_idx - keep]:
                    del image_cache[stale_idx]

            prev_def = def_image
            frame_times.append(perf_counter() - t_start)
            score_mat = score_out.reshape(shape)
            frames.append(
                self._build_frame(
                    f_idx,
                    f_name,
                    x_mat,
                    y_mat,
                    u_mat,
                    v_mat,
                    score_mat,
                    recovered,
                    converged=conv_out.reshape(shape) == 1.0,
                )
            )
            if progress_cb:
                progress_cb(f_idx, n_deformed)

        return CorrelationResult(
            self.tracker.x0,
            self.tracker.y0,
            frames,
            frame_times,
            bool(abort_check and abort_check()),
        )

    def _build_frame(
        self,
        idx: int,
        name: str,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        score: np.ndarray,
        recovered: np.ndarray | None = None,
        converged: np.ndarray | None = None,
    ) -> CorrelationFrame:
        scale = self.setup.calibration_mm_per_px
        valid = (
            self.tracker.mask
            & np.isfinite(x)
            & np.isfinite(y)
            & np.isfinite(u)
            & np.isfinite(v)
        )
        strain = self.tensor.compute(
            DisplacementField(u, v, valid),
            ReferenceGrid(
                self.tracker.x0,
                self.tracker.y0,
                self.tracker.mask,
                self.setup.step_size,
            ),
        )

        magnitude = np.sqrt(u**2 + v**2)
        return CorrelationFrame(
            frame_index=idx,
            frame_name=name,
            x_px=x,
            y_px=y,
            u_px=u,
            v_px=v,
            magnitude_px=magnitude,
            valid_mask=valid,
            x_mm=x * scale,
            y_mm=y * scale,
            u_mm=u * scale,
            v_mm=v * scale,
            magnitude_mm=magnitude * scale,
            strain=strain,
            score=score,
            recovered=recovered,
            converged=converged,
        )
