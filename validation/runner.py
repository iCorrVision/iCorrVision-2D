"""Parameter sweeps over reference datasets, driven through the CLI.

WHY THE CLI RATHER THAN THE API
    Calling build_engine() directly would be faster to iterate on, but every
    result would then be reproducible only by re-running this harness. Going
    through `icorr run --config X --set k=v` means each row of the output
    table corresponds to a command a reader can type, which is the thesis's
    claim about reproducibility. It also exercises the CLI as an independent
    consumer of the engine, alongside the GUI and the Python API, which is the
    practical evidence for FR7.

RESUMABILITY
    A sweep of a few hundred configurations takes hours. Each run writes to
    its own directory and is skipped if that directory already holds a
    result, so an interrupted sweep resumes by being re-run. Nothing is
    cached in memory and no run depends on another.

OUTPUT SHAPE
    One tidy CSV, long format: one row per (configuration, frame, component).
    Swept parameters are columns, so filtering and grouping in pandas or a
    plotting script needs no parsing. Per-run aggregates are emitted as rows
    with frame_index = -1 rather than as a second file, so a plot that wants
    "the whole run" filters instead of joining.
"""

import hashlib
import subprocess
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import compute_metrics, virtual_strain_gauge
from .truth import GroundTruth, NoGroundTruthError

from components.archive.manifest import MANIFEST_NAME, read_manifest

# Columns the engine's CSV export always provides; see components/preview/export.py
_REQUIRED_COLUMNS = {
    "frame_index",
    "row_index",
    "col_index",
    "frame_name",
    "x_px",
    "y_px",
    "u_px",
    "v_px",
    "valid",
}


# ---------------------------------------------------------------------------
# A point in the sweep
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepPoint:
    """One configuration: a base TOML plus dotted-path overrides."""

    overrides: dict[str, str | int | float]

    @property
    def config_id(self) -> str:
        """Stable short identifier, safe as a directory name.

        A hash rather than a slug of the values: a readable slug of six
        overrides is too long, and a truncated one collides. The overrides
        themselves are columns in the output table, so the id never needs to
        be decoded by eye.
        """
        payload = "&".join(f"{k}={v}" for k, v in sorted(self.overrides.items()))
        return hashlib.sha1(payload.encode()).hexdigest()[:10]

    @property
    def cli_args(self) -> list[str]:
        out: list[str] = []
        for key, value in sorted(self.overrides.items()):
            # tomllib parses the right-hand side, so strings need quoting and
            # numbers must not be quoted.
            literal = f'"{value}"' if isinstance(value, str) else str(value)
            out += ["--set", f"{key}={literal}"]
        return out

    def describe(self) -> dict[str, str | int | float]:
        """Overrides as flat columns, with the dotted path as the column name."""
        return dict(self.overrides)


def expand_grid(**axes) -> list[SweepPoint]:
    """Cartesian product of named axes into SweepPoints.

        expand_grid(**{
            "correlation.image_interpolation.method": ["8tap", "bicubic_spline"],
            "correlation.refinement.shape_function": ["affine", "quadratic"],
        })

    gives four points. Axis names are the dotted TOML paths the CLI expects,
    so nothing has to translate between this and the config file.
    """
    keys = list(axes)
    return [
        SweepPoint(dict(zip(keys, values)))
        for values in product(*(axes[k] for k in keys))
    ]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Dataset:
    """A reference image set, its base configuration and its known solution."""

    name: str
    config: Path  # base TOML, e.g. dic-challenge/sample03.toml
    truth: GroundTruth

    #: Frames to exclude from metrics. The reference frame is correlated
    #: against itself and trivially reports zero error, which would bias every
    #: aggregate toward zero.
    skip_reference: bool = True


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


@dataclass
class RunOutcome:
    ok: bool
    output_dir: Path
    wall_seconds: float
    engine_seconds: float | None
    skipped: bool
    message: str = ""


def _result_csv(output_dir: Path) -> Path:
    return output_dir / "data" / "correlation.csv"


def run_one(
    dataset: Dataset,
    point: SweepPoint,
    out_root: Path,
    *,
    python: str = "python",
    module: str = "cli.main",
    repo_root: Path | None = None,
    timeout_s: float | None = None,
    force: bool = False,
) -> RunOutcome:
    """Execute one configuration, or skip it if a complete output already exists."""
    output_dir = out_root / dataset.name / point.config_id

    if _is_complete(output_dir) and not force:
        return RunOutcome(True, output_dir, 0.0, _engine_seconds(output_dir), True)

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python,
        "-m",
        module,
        "run",
        "--config",
        str(dataset.config),
        "--output-dir",
        str(output_dir),
        *point.cli_args,
    ]

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return RunOutcome(
            False, output_dir, time.perf_counter() - start, None, False, "timeout"
        )
    wall = time.perf_counter() - start

    # Record the invocation next to its output so a surprising row in the
    # table can be traced back to the exact command that produced it.
    (output_dir / "command.txt").write_text(" ".join(cmd) + "\n")
    if proc.returncode != 0:
        (output_dir / "stderr.txt").write_text(proc.stderr)
        return RunOutcome(
            False,
            output_dir,
            wall,
            None,
            False,
            f"exit {proc.returncode}: {proc.stderr.strip().splitlines()[-1:] or ['']}",
        )
    if not _is_complete(output_dir):
        return RunOutcome(False, output_dir, wall, None, False, "incomplete output")

    return RunOutcome(True, output_dir, wall, _engine_seconds(output_dir), False)


def _is_complete(output_dir: Path) -> bool:
    """True if the run finished writing: its CSV exists and its manifest parses.

    The CLI writes the manifest last, so an interrupted or failed export leaves
    no manifest and is run again instead of being reused.
    """
    manifest = output_dir / MANIFEST_NAME
    if not (_result_csv(output_dir).exists() and manifest.exists()):
        return False
    try:
        read_manifest(manifest.read_bytes())
    except (ValueError, UnicodeDecodeError):  # malformed TOML or unknown format
        return False
    return True


def _engine_seconds(output_dir: Path) -> float | None:
    """Engine-reported duration, which excludes interpreter start-up."""
    path = output_dir / MANIFEST_NAME
    if not path.exists():
        return None  # e.g. a run that failed before writing
    return float(read_manifest(path.read_bytes())["run"]["duration_sec"])


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_run(output_dir: Path, dataset: Dataset) -> tuple[list[dict], list[dict]]:
    """Per-frame and pooled metrics for one completed run.

    Returns (per_frame_rows, aggregate_rows). Both are lists of plain dicts so
    the caller can attach configuration columns without this function knowing
    what was swept.
    """
    frame = pd.read_csv(_result_csv(output_dir))
    missing = _REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{output_dir}: CSV missing columns {sorted(missing)}")

    per_frame: list[dict] = []
    pooled: dict[str, list[np.ndarray]] = {"u": [], "v": [], "u_t": [], "v_t": []}

    # GroundTruth.displacement_at expects REFERENCE positions. The x_px/y_px of a
    # deformed frame are matched positions (NaN where a node failed), so the
    # reference grid is taken from frame 0 and joined on (row, col). For the
    # uniform truths used so far this changes nothing; a spatially varying truth
    # requires it.
    reference = frame[frame.frame_index == 0].set_index(["row_index", "col_index"])[
        ["x_px", "y_px"]
    ]

    for frame_index, block in frame.groupby("frame_index", sort=True):
        if dataset.skip_reference and frame_index == 0:
            continue
        frame_name = str(block["frame_name"].iloc[0])

        at_reference = reference.loc[list(zip(block["row_index"], block["col_index"]))]
        x = at_reference["x_px"].to_numpy(dtype=float)
        y = at_reference["y_px"].to_numpy(dtype=float)
        u = block["u_px"].to_numpy(dtype=float)
        v = block["v_px"].to_numpy(dtype=float)
        valid = block["valid"].astype(str).str.strip().str.lower().eq("true").to_numpy()

        try:
            u_true, v_true = dataset.truth.displacement_at(
                x, y, frame_name, int(frame_index)
            )
        except NoGroundTruthError:
            return [], []  # comparison-only dataset: no error table to build

        for component, computed, true in (("u", u, u_true), ("v", v, v_true)):
            metrics = compute_metrics(computed, true, valid)
            per_frame.append(
                {
                    "frame_index": int(frame_index),
                    "frame_name": frame_name,
                    "component": component,
                    **metrics.as_dict(),
                }
            )

        pooled["u"].append(np.where(valid, u, np.nan))
        pooled["v"].append(np.where(valid, v, np.nan))
        pooled["u_t"].append(np.where(valid, u_true, np.nan))
        pooled["v_t"].append(np.where(valid, v_true, np.nan))

    aggregates: list[dict] = []
    if pooled["u"]:
        for component, key in (("u", "u"), ("v", "v")):
            metrics = compute_metrics(
                np.concatenate(pooled[key]), np.concatenate(pooled[f"{key}_t"])
            )
            aggregates.append(
                {
                    "frame_index": -1,  # sentinel: pooled over the whole run
                    "frame_name": "__all__",
                    "component": component,
                    **metrics.as_dict(),
                }
            )
    return per_frame, aggregates


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def _vsg_for(overrides: dict, defaults: dict) -> float | None:
    """VSG implied by a configuration, so runs are comparable on gauge length.

    Returns None for the meshfree estimator when no radius is known; a wrong
    gauge length would be worse than a missing one.
    """
    get = lambda k, d=None: overrides.get(k, defaults.get(k, d))
    subset = get("setup.subset_size")
    step = get("setup.step_size")
    if subset is None or step is None:
        return None
    method = get("correlation.strain.method", "lagrange_cloud")
    try:
        if method == "lagrange_cloud":
            radius = get("correlation.strain.radius_px")
            return (
                None
                if radius is None
                else virtual_strain_gauge(
                    int(subset), int(step), radius_px=float(radius)
                )
            )
        window = get("correlation.strain.strain_window")
        return (
            None
            if window is None
            else virtual_strain_gauge(int(subset), int(step), strain_window=int(window))
        )
    except (TypeError, ValueError):
        return None


def sweep(
    dataset: Dataset,
    points: list[SweepPoint],
    out_root: Path,
    *,
    defaults: dict | None = None,
    verbose: bool = True,
    **run_kwargs,
) -> pd.DataFrame:
    """Run every configuration and return the tidy metrics table.

    `defaults` holds parameter values that are fixed for this sweep but still
    recorded as columns, typically subset and step size from the base TOML:
    the VSG calculation needs them, and so does any reading of the table.
    """
    defaults = defaults or {}
    rows: list[dict] = []

    for n, point in enumerate(points, start=1):
        outcome = run_one(dataset, point, out_root, **run_kwargs)
        status = "skip" if outcome.skipped else ("ok" if outcome.ok else "FAIL")
        if verbose:
            print(
                f"[{n:>4}/{len(points)}] {dataset.name} {point.config_id} "
                f"{status:>4} {outcome.wall_seconds:6.1f}s {outcome.message}",
                flush=True,
            )
        if not outcome.ok:
            continue

        per_frame, aggregates = score_run(outcome.output_dir, dataset)
        common = {
            "dataset": dataset.name,
            "config_id": point.config_id,
            **defaults,
            **point.describe(),
            "vsg_px": _vsg_for(point.overrides, defaults),
            "engine_seconds": outcome.engine_seconds,
            "truth_provenance": dataset.truth.provenance,
        }
        rows += [{**common, **r} for r in (*per_frame, *aggregates)]

    return pd.DataFrame(rows)


def write_tidy(table: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path
