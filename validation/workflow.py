"""Helpers shared by the Chapter 5 notebooks.

Config hashing, the run wrapper, loading an exported frame with reference-grid
geometry, run quality, line cuts and edge distance.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tomli_w
from scipy.ndimage import distance_transform_cdt

from .runner import Dataset, SweepPoint, run_one

# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def write_base_config(base: dict, name: str, results: Path) -> tuple[Path, Path]:
    """Write the base TOML; return (config_path, runs_root).

    The runs directory is keyed by a hash of the base config. The runner skips
    any configuration whose output already exists, and its config_id hashes
    only the per-run overrides, so without this key an edited base config
    would reuse results computed under the old settings.
    """
    text = tomli_w.dumps(base)
    root = Path(results) / name / f"base-{hashlib.sha1(text.encode()).hexdigest()[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.toml").write_text(text)
    return root / "config.toml", root / "runs"


def run_point(
    dataset: Dataset,
    runs_root: Path,
    point: SweepPoint,
    *,
    repo: Path,
    force: bool = False,
    quiet: bool = False,
):
    """One CLI run, reusing saved output when present. Prints a status line."""
    outcome = run_one(
        dataset, point, runs_root, python=sys.executable, repo_root=repo, force=force
    )
    if not quiet:
        status = "reused" if outcome.skipped else ("ok" if outcome.ok else "FAILED")
        print(f"  {point.config_id}  {status:>6}  {outcome.message}")
        if not outcome.ok:
            print("     see", outcome.output_dir / "stderr.txt")
    return outcome


# ---------------------------------------------------------------------------
# Reading exported results
# ---------------------------------------------------------------------------


def load_frame(output_dir: Path, frame: str | None = None) -> pd.DataFrame:
    """One exported frame, with reference-grid coordinates x0/y0 joined on.

    frame: a frame name as exported, or None for the last frame.

    Geometry comes from the reference frame, joined on (row, col). The exported
    x_px/y_px on a deformed frame are *matched* positions, and NaN wherever a
    node failed; the reference frame carries every node at its grid position.
    """
    df = pd.read_csv(Path(output_dir) / "data" / "correlation.csv")
    ref = df[df.frame_index == 0][["row_index", "col_index", "x_px", "y_px"]].rename(
        columns={"x_px": "x0", "y_px": "y0"}
    )
    sel = (
        df[df.frame_index == df.frame_index.max()]
        if frame is None
        else df[df.frame_name == frame]
    )
    out = sel.drop(columns=["x_px", "y_px"]).merge(ref, on=["row_index", "col_index"])
    out["valid"] = out["valid"].astype(str).str.strip().str.lower().eq("true")
    return out


def _as_bool(column: pd.Series) -> pd.Series:
    return column.astype(str).str.strip().str.lower().eq("true")


def run_quality(output_dir: Path) -> dict[str, float]:
    """Node counts for one completed run, pooled over its deformed frames.

    roi_nodes       nodes inside the ROI: the reference frame's valid nodes.
    valid_rate_roi  valid nodes / (roi_nodes x deformed frames). Unlike a rate over
                    every grid node, it is not lowered by grid points that fall in a
                    hole or outside the ROI, so it measures correlation success only.
    nonconverged_pct  valid nodes whose IC-GN iteration did not reach the tolerance,
                    as a percentage of valid nodes. The engine flags these but keeps
                    them if their score passes the threshold. NaN for results
                    exported before the engine wrote the `converged` column.
    """
    df = pd.read_csv(Path(output_dir) / "data" / "correlation.csv")
    roi_nodes = int(_as_bool(df.loc[df.frame_index == 0, "valid"]).sum())
    deformed = df[df.frame_index > 0]
    n_frames = deformed.frame_index.nunique()
    valid = _as_bool(deformed["valid"])
    n_valid = int(valid.sum())
    out = {
        "roi_nodes": roi_nodes,
        "valid_rate_roi": (
            n_valid / (roi_nodes * n_frames) if roi_nodes and n_frames else np.nan
        ),
        "nonconverged_pct": np.nan,
    }
    if "converged" in deformed and n_valid:
        nonconverged = int((valid & ~_as_bool(deformed["converged"])).sum())
        out["nonconverged_pct"] = 100.0 * nonconverged / n_valid
    return out


def row_cut(
    output_dir: Path, frame: str, column: str, row: int
) -> tuple[np.ndarray, np.ndarray]:
    """`column` along the grid row at reference y == row. Invalid nodes are NaN."""
    f = load_frame(output_dir, frame)
    f = f[np.isclose(f.y0, row)].sort_values("x0")
    return f.x0.to_numpy(), np.where(f.valid, f[column], np.nan)


def add_edge_distance(frame: pd.DataFrame) -> pd.DataFrame:
    """Chessboard distance, in grid nodes, to the nearest invalid node or grid edge.

    A square strain window of half-width h is fully supported iff edge_nodes > h.
    The grid is padded with invalid nodes, so the grid boundary counts as an edge.
    Chessboard, not Euclidean, because the windowed estimators use square
    neighbourhoods; a matched meshfree disc (r = h * step) sits inside the square.
    """
    rows, cols = frame.row_index.max() + 1, frame.col_index.max() + 1
    valid = np.zeros((rows, cols), bool)
    valid[frame.row_index, frame.col_index] = frame.valid
    d = distance_transform_cdt(
        np.pad(valid, 1, constant_values=False), metric="chessboard"
    )[1:-1, 1:-1]
    return frame.assign(edge_nodes=d[frame.row_index, frame.col_index])


def vertical_cut(
    frame: pd.DataFrame, x_cut: float, quantity: str = "e1"
) -> pd.DataFrame:
    """Every node in the grid column nearest x_cut, ordered by y.

    Invalid nodes are kept with NaN, so plotted lines break at holes instead of
    being drawn across them. Carries edge_nodes if present.
    """
    columns = frame.groupby("col_index").x0.first()
    col = (columns - x_cut).abs().idxmin()
    cut = frame[frame.col_index == col].sort_values("y0").copy()
    cut.loc[~cut.valid, quantity] = np.nan
    keep = ["y0", quantity] + (["edge_nodes"] if "edge_nodes" in cut else [])
    return cut[keep]
