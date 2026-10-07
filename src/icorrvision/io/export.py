import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np

from components.contracts import CorrelationFrame, CorrelationResult

from .diagnostics import frame_health

STRAIN_COLUMNS = ("exx", "eyy", "exy", "e1", "e2", "principal_angle_deg")

CSV_COLUMNS = [
    "frame_index",
    "frame_name",
    "row_index",
    "col_index",
    "x_px",
    "y_px",
    "u_px",
    "v_px",
    "magnitude_px",
    "valid",
    "score",
    "recovered",
    "converged",
    "x_mm",
    "y_mm",
    "u_mm",
    "v_mm",
    "magnitude_mm",
    *STRAIN_COLUMNS,
]

SUMMARY_COLUMNS = [
    "frame_index",
    "frame_name",
    "valid_pct",
    "n_valid",
    "recovered",
    "score_median",
    "score_p05",
    "frame_time_sec",
]


def _column(array: np.ndarray | None, n: int) -> list:
    """One CSV column as a plain Python list; empty cells if the field is absent."""
    if array is None:
        return [""] * n
    return np.asarray(array).ravel().tolist()


def _frame_columns(frame: CorrelationFrame) -> list[list]:
    n = frame.x_px.size
    rows, cols = np.indices(frame.x_px.shape)
    columns = {
        "frame_index": [frame.frame_index] * n,
        "frame_name": [frame.frame_name] * n,
        "row_index": rows.ravel().tolist(),
        "col_index": cols.ravel().tolist(),
        "x_px": _column(frame.x_px, n),
        "y_px": _column(frame.y_px, n),
        "u_px": _column(frame.u_px, n),
        "v_px": _column(frame.v_px, n),
        "magnitude_px": _column(frame.magnitude_px, n),
        "valid": _column(frame.valid_mask, n),
        "score": _column(frame.score, n),
        "recovered": _column(frame.recovered, n),
        "converged": _column(frame.converged, n),
        "x_mm": _column(frame.x_mm, n),
        "y_mm": _column(frame.y_mm, n),
        "u_mm": _column(frame.u_mm, n),
        "v_mm": _column(frame.v_mm, n),
        "magnitude_mm": _column(frame.magnitude_mm, n),
    }
    for name in STRAIN_COLUMNS:
        field = None if frame.strain is None else getattr(frame.strain, name)
        columns[name] = _column(field, n)
    return [columns[name] for name in CSV_COLUMNS]


def export_full_csv(result: CorrelationResult, csv_path: str | Path) -> Path:
    """One row per grid node per frame."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for frame in result.frames:
            writer.writerows(zip(*_frame_columns(frame)))
    return path


def export_summary_csv(result: CorrelationResult, csv_path: str | Path) -> Path:
    """One row per frame: validity, recovery, match quality and timing."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # frame_times_sec holds one entry per *deformed* frame, in order.
    times = dict(
        zip((f.frame_index for f in result.frames[1:]), result.frame_times_sec)
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in frame_health(result):
            record = asdict(row)
            record["frame_time_sec"] = times.get(row.frame_index, "")
            writer.writerow(record)
    return path
