import csv
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable
import tempfile

import numpy as np
import cv2

from icorrvision.contracts import (
    CorrelationConfig,
    CorrelationFrame,
    CorrelationResult,
    StrainField,
    build_correlation_config,
)

from icorrvision.io.archive.manifest import MANIFEST_NAME, ROI_MASK_NAME, read_manifest

STRAIN_COMPONENTS = ("exx", "eyy", "exy", "e1", "e2", "principal_angle_deg")


@dataclass(frozen=True)
class LoadedArchive:
    result: CorrelationResult
    correlation_config: CorrelationConfig
    reference_frame_name: str
    reference_thumbnail: np.ndarray
    deformed_frames: list[str]
    frame_loader: Callable[[str], np.ndarray]  # deformed frames only
    roi_mask: np.ndarray
    subset_size: int
    step_size: int
    calibration_mm_per_px: float
    csv_path: Path


def _parse_float(s: str | None) -> float:
    """Parse a CSV cell; an empty or missing cell is NaN, not zero.

    DictWriter writes None as "", and DictReader returns None for the missing
    fields of a short row.
    """
    if s is None or s == "":
        return np.nan
    return float(s)  # handles "nan" natively


def _rows_to_result(rows: list[dict]) -> CorrelationResult:
    by_frame: dict[int, list[dict]] = {}
    names: dict[int, str] = {}
    for row in rows:
        idx = int(row["frame_index"])
        by_frame.setdefault(idx, []).append(row)
        names[idx] = row["frame_name"]

    frames: list[CorrelationFrame] = []
    grid_x = grid_y = None

    for idx in sorted(by_frame):
        frame_rows = by_frame[idx]
        n_rows = max(int(r["row_index"]) for r in frame_rows) + 1
        n_cols = max(int(r["col_index"]) for r in frame_rows) + 1
        arrays = {
            k: np.full((n_rows, n_cols), np.nan)
            for k in (
                "x_px",
                "y_px",
                "u_px",
                "v_px",
                "magnitude_px",
                "x_mm",
                "y_mm",
                "u_mm",
                "v_mm",
                "magnitude_mm",
                "score",
                "recovered",
                *STRAIN_COMPONENTS,
            )
        }
        valid = np.zeros((n_rows, n_cols), dtype=bool)

        has_strain = "exx" in frame_rows[0]
        has_score = "score" in frame_rows[0]
        has_recovered = "recovered" in frame_rows[0]
        recovered = np.zeros((n_rows, n_cols), dtype=bool)
        for r in frame_rows:
            ri, ci = int(r["row_index"]), int(r["col_index"])
            for key in arrays:
                if key in STRAIN_COMPONENTS and not has_strain:
                    continue
                if key in ("score", "recovered"):
                    continue  # handled separately: bool, and may be absent
                arrays[key][ri, ci] = _parse_float(r[key])
            if has_score:
                arrays["score"][ri, ci] = _parse_float(r["score"])
            if has_recovered:
                recovered[ri, ci] = r["recovered"] == "True"
            valid[ri, ci] = r["valid"] == "True"

        if idx == 0:
            grid_x, grid_y = arrays["x_px"].copy(), arrays["y_px"].copy()

        strain = (
            StrainField(
                exx=arrays["exx"],
                eyy=arrays["eyy"],
                exy=arrays["exy"],
                e1=arrays["e1"],
                e2=arrays["e2"],
                principal_angle_deg=arrays["principal_angle_deg"],
            )
            if has_strain
            else None
        )

        frames.append(
            CorrelationFrame(
                frame_index=idx,
                frame_name=names[idx],
                x_px=arrays["x_px"],
                y_px=arrays["y_px"],
                u_px=arrays["u_px"],
                v_px=arrays["v_px"],
                magnitude_px=arrays["magnitude_px"],
                valid_mask=valid,
                x_mm=arrays["x_mm"],
                y_mm=arrays["y_mm"],
                u_mm=arrays["u_mm"],
                v_mm=arrays["v_mm"],
                magnitude_mm=arrays["magnitude_mm"],
                strain=strain,
                score=arrays["score"] if has_score else None,
                recovered=recovered if has_recovered else None,
            )
        )

    return CorrelationResult(grid_x_px=grid_x, grid_y_px=grid_y, frames=frames)


def read_icorr_archive(archive_path: Path) -> LoadedArchive:
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"dic_loaded_{archive_path.stem}_"))

    with zipfile.ZipFile(archive_path, "r") as zf:
        manifest = read_manifest(zf.read(MANIFEST_NAME))
        deformed_frames = manifest["session"]["deformed_frames"]

        csv_bytes = zf.read("data/correlation.csv")
        rows = list(csv.DictReader(csv_bytes.decode("utf-8").splitlines()))
        result = replace(_rows_to_result(rows), aborted=manifest["run"]["aborted"])

        mask_bytes = zf.read(ROI_MASK_NAME)
        roi_mask = (
            cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE) > 0
        )
        original_shape = roi_mask.shape[
            :2
        ]  # (height, width): the resolution the correlation ran at

        ref_bytes = zf.read("thumbnails/reference.jpg")
        ref_thumb = cv2.imdecode(
            np.frombuffer(ref_bytes, np.uint8), cv2.IMREAD_GRAYSCALE
        )
        reference_thumbnail = cv2.resize(
            ref_thumb,
            (original_shape[1], original_shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )

        thumb_dir = tmp_dir / "thumbnails"
        thumb_dir.mkdir(parents=True)
        for i in range(len(deformed_frames)):
            name = f"frame_{i:04d}.jpg"
            (thumb_dir / name).write_bytes(zf.read(f"thumbnails/{name}"))

    tmp_csv_path = tmp_dir / "correlation.csv"
    tmp_csv_path.write_bytes(csv_bytes)

    def frame_loader(name: str) -> np.ndarray:
        idx = deformed_frames.index(name)
        thumb = cv2.imread(
            str(thumb_dir / f"frame_{idx:04d}.jpg"), cv2.IMREAD_GRAYSCALE
        )
        return cv2.resize(
            thumb, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_CUBIC
        )

    return LoadedArchive(
        result=result,
        correlation_config=build_correlation_config(manifest["correlation"]),
        reference_frame_name=manifest["session"]["reference_frame"],
        reference_thumbnail=reference_thumbnail,
        deformed_frames=deformed_frames,
        frame_loader=frame_loader,
        roi_mask=roi_mask,
        subset_size=manifest["setup"]["subset_size"],
        step_size=manifest["setup"]["step_size"],
        calibration_mm_per_px=manifest["setup"]["calibration_mm_per_px"],
        csv_path=tmp_csv_path,
    )
