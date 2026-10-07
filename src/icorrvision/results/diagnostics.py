from collections import Counter
from dataclasses import dataclass

import numpy as np

from icorrvision.contracts import CorrelationFrame, CorrelationResult


@dataclass(frozen=True)
class FrameHealth:
    """One row of the per-frame diagnostics table."""

    frame_index: int
    frame_name: str
    valid_pct: float  # of the nodes inside the ROI, not of the grid bounding box
    n_valid: int
    recovered: int
    score_median: float
    score_p05: float  # the weak tail is what a threshold change would move


@dataclass(frozen=True)
class StrainOutlier:
    """The single node that sits furthest outside its frame's p1-p99 band."""

    frame_index: int
    frame_name: str
    row: int
    col: int
    value: float
    p1: float
    p99: float
    recovered: bool | None  # None when the result carries no recovery flags


_OUTLIER_SPAN_FACTOR = 1.0


def _finite_where(array: np.ndarray | None, mask: np.ndarray) -> np.ndarray:
    """Finite entries of `array` under `mask`, flattened. Empty if array is None."""
    if array is None:
        return np.empty(0, dtype=np.float64)
    values = np.asarray(array, dtype=np.float64)[mask]
    return values[np.isfinite(values)]


def _roi_node_count(result: CorrelationResult) -> int:
    """Nodes inside the ROI, read off the reference frame."""
    return int(result.frames[0].valid_mask.sum()) if result.frames else 0


def frame_health(result: CorrelationResult) -> list[FrameHealth]:
    n_roi = _roi_node_count(result)
    rows: list[FrameHealth] = []
    for frame in result.frames:
        valid = frame.valid_mask
        n_valid = int(valid.sum())
        scores = _finite_where(frame.score, valid)
        rows.append(
            FrameHealth(
                frame_index=frame.frame_index,
                frame_name=frame.frame_name,
                valid_pct=100.0 * n_valid / n_roi if n_roi else float("nan"),
                n_valid=n_valid,
                recovered=(
                    int(np.asarray(frame.recovered).sum())
                    if frame.recovered is not None
                    else 0
                ),
                score_median=float(np.median(scores)) if scores.size else float("nan"),
                score_p05=(
                    float(np.percentile(scores, 5)) if scores.size else float("nan")
                ),
            )
        )
    return rows


def score_distribution(
    result: CorrelationResult, frame_index: int, bins: int = 40
) -> tuple[np.ndarray, np.ndarray]:
    """Histogram of the per-node criterion value at the converged warp."""

    frame = result.frames[frame_index]
    scores = _finite_where(frame.score, frame.valid_mask)
    if scores.size == 0:
        return np.zeros(bins), np.linspace(-1.0, 1.0, bins + 1)
    return np.histogram(scores, bins=bins, range=(-1.0, 1.0))


def strain_percentiles(
    result: CorrelationResult, component: str = "e1"
) -> dict[str, np.ndarray]:
    """Per-frame p1, median, p99, minimum and maximum of a strain component."""
    keys = ("p1", "median", "p99", "min", "max")
    stats: dict[str, list[float]] = {k: [] for k in keys}
    indices: list[int] = []

    for frame in result.frames:
        if frame.strain is None:
            continue
        indices.append(frame.frame_index)
        values = _finite_where(getattr(frame.strain, component), frame.valid_mask)
        if values.size == 0:
            for k in keys:
                stats[k].append(np.nan)
            continue
        p1, p50, p99 = np.percentile(values, (1, 50, 99))
        stats["p1"].append(float(p1))
        stats["median"].append(float(p50))
        stats["p99"].append(float(p99))
        stats["min"].append(float(values.min()))
        stats["max"].append(float(values.max()))

    out = {k: np.asarray(v, dtype=np.float64) for k, v in stats.items()}
    out["frame_index"] = np.asarray(indices, dtype=np.int64)
    return out


def worst_strain_outlier(
    result: CorrelationResult, component: str = "e1"
) -> StrainOutlier | None:
    """Locate the node furthest outside its own frame's p1-p99 band."""
    best: tuple[float, CorrelationFrame, int, float, float] | None = None
    run_lo, run_hi = np.inf, -np.inf

    for frame in result.frames:
        if frame.strain is None:
            continue
        values = np.asarray(getattr(frame.strain, component), dtype=np.float64)
        good = frame.valid_mask & np.isfinite(values)
        if not good.any():
            continue
        p1, p99 = (float(p) for p in np.percentile(values[good], (1, 99)))
        run_lo, run_hi = min(run_lo, p1), max(run_hi, p99)

        # NaN out everything that isn't a valid node, then argmax/argmin.
        # The array keeps its shape, so the flat index maps back to (row, col).
        candidates = np.where(good, values, np.nan)
        hi_idx = int(np.nanargmax(candidates))
        lo_idx = int(np.nanargmin(candidates))
        hi_excess = candidates.flat[hi_idx] - p99
        lo_excess = p1 - candidates.flat[lo_idx]
        idx, excess = (
            (hi_idx, hi_excess) if hi_excess >= lo_excess else (lo_idx, lo_excess)
        )

        if best is None or excess > best[0]:
            best = (float(excess), frame, idx, p1, p99)

    span = run_hi - run_lo
    if best is None or span <= 0 or best[0] <= _OUTLIER_SPAN_FACTOR * span:
        return None

    _, frame, idx, p1, p99 = best
    row, col = np.unravel_index(idx, frame.valid_mask.shape)
    recovered = (
        None if frame.recovered is None else bool(np.asarray(frame.recovered).flat[idx])
    )
    return StrainOutlier(
        frame_index=frame.frame_index,
        frame_name=frame.frame_name,
        row=int(row),
        col=int(col),
        value=float(np.asarray(getattr(frame.strain, component)).flat[idx]),
        p1=p1,
        p99=p99,
        recovered=recovered,
    )


def dropout_gaps(result: CorrelationResult) -> Counter:
    """How many consecutive frames nodes go missing for, before returning."""
    if not result.frames:
        return Counter()

    stack = np.stack([f.valid_mask.ravel() for f in result.frames])
    run = np.zeros(stack.shape[1], dtype=np.int64)
    gaps: list[int] = []
    for ok in stack:
        closing = ok & (run > 0)
        gaps.extend(run[closing].tolist())
        run = np.where(ok, 0, run + 1)
    return Counter(gaps)


def run_status_line(result: CorrelationResult) -> str:
    """One line naming the frame count and whether the run finished."""
    status = "aborted" if result.aborted else "complete"
    return f"{len(result.frames)} frame(s), run {status}."


def summary_lines(result: CorrelationResult, component: str = "e1") -> list[str]:
    """Short summary of the run for the Analysis tab header."""
    rows = frame_health(result)
    if not rows:
        return ["No frames in this result."]

    lines = [run_status_line(result)]

    deformed = rows[1:]
    if not deformed:
        lines.append("Only the reference frame is present; nothing was correlated.")
        return lines

    lines.append(
        f"Valid nodes (of those inside the ROI): {deformed[-1].valid_pct:.0f}% at "
        f"the last frame, {min(r.valid_pct for r in deformed):.0f}% at the worst."
    )

    total_recovered = sum(r.recovered for r in rows)
    if total_recovered:
        lines.append(
            f"{total_recovered} node-frame(s) recovered from a previous frame: "
            "matched against the image, but seeded from their neighbours."
        )

    gaps = dropout_gaps(result)
    if gaps:
        common = ", ".join(
            f"{length}f x{count}" for length, count in sorted(gaps.items())[:5]
        )
        lines.append(
            f"Dropout gaps that closed again: {common}. "
            "recovery.max_lost_frames must cover these to retry them."
        )

    outlier = worst_strain_outlier(result, component)
    if outlier is not None:
        origin = {
            True: "a recovered node",
            False: "a directly matched node",
            None: "a node",
        }[outlier.recovered]
        lines.append(
            f"Strain outlier: {component} = {outlier.value:.3g} on frame "
            f"{outlier.frame_index}, grid node (row {outlier.row}, col {outlier.col}), "
            f"{origin}. That frame's p1-p99 band is {outlier.p1:.3g} to "
            f"{outlier.p99:.3g}. The colour scales clip it; check it before "
            "quoting a peak strain."
        )
    return lines
