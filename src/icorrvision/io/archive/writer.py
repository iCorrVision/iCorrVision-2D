import tempfile
import zipfile
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import tomli_w
from matplotlib.figure import Figure

from components.config.loader import BuiltRun
from components.contracts import CorrelationConfig, CorrelationResult, SetupOutput
from components.preview.export import export_full_csv
from components.preview.payload import (
    QUANTITY_LABELS,
    build_payload,
    color_scale,
)
from components.preview.render import render_heatmap
from components.archive.manifest import (
    MANIFEST_NAME,
    ROI_MASK_NAME,
    build_manifest,
)

THUMBNAIL_MAX_DIM = 800
THUMBNAIL_QUALITY = 85


def _generate_summary_plots(
    result: CorrelationResult, image: np.ndarray, out_dir: Path
) -> None:
    """Plot the last-frame map of every quantity, each on its own scale.

    Uses the payload and scale functions of the preview's Result tab, so an
    exported PNG matches the on-screen map.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    last = len(result.frames) - 1
    for quantity in QUANTITY_LABELS:
        figure = Figure(figsize=(6, 5), layout="tight")
        render_heatmap(
            figure,
            image,
            build_payload(result, quantity, last),
            color_scale(result, quantity, frame_index=last),
        )
        name = "magnitude" if quantity == "magnitude" else f"strain_{quantity}"
        figure.savefig(out_dir / f"{name}.png", dpi=150)


def _resize_max_dim(image: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale >= 1.0:
        return image
    return cv2.resize(
        image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
    )


def _write_thumbnail(image: np.ndarray, out_path: Path) -> None:
    thumb = _resize_max_dim(image, THUMBNAIL_MAX_DIM)
    cv2.imwrite(str(out_path), thumb, [cv2.IMWRITE_JPEG_QUALITY, THUMBNAIL_QUALITY])


def write_icorr_archive(
    result: CorrelationResult,
    setup_output: SetupOutput,
    correlation_config: CorrelationConfig,
    frame_loader: Callable[[str], np.ndarray],
    reference_frame_name: str,
    deformed_frames: list[str],
    archive_path: Path,
) -> None:
    """Write a results archive (.icorr).

    Independent of the CLI's BuiltRun and ResolvedSession, so both the CLI
    and the GUI call it, as they do build_engine().
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        export_full_csv(result, tmp_dir / "data" / "correlation.csv")

        last_img = (
            frame_loader(deformed_frames[-1])
            if deformed_frames
            else setup_output.reference_frame
        )
        _generate_summary_plots(result, last_img, tmp_dir / "plots")

        thumb_dir = tmp_dir / "thumbnails"
        thumb_dir.mkdir(parents=True, exist_ok=True)
        _write_thumbnail(setup_output.reference_frame, thumb_dir / "reference.jpg")
        for i, name in enumerate(deformed_frames):
            _write_thumbnail(frame_loader(name), thumb_dir / f"frame_{i:04d}.jpg")

        _write_roi_mask(setup_output.roi_mask, tmp_dir / ROI_MASK_NAME)

        manifest = build_manifest(
            result=result,
            setup_output=setup_output,
            correlation_config=correlation_config,
            reference_frame_name=reference_frame_name,
            deformed_frames=deformed_frames,
        )
        (tmp_dir / MANIFEST_NAME).write_bytes(tomli_w.dumps(manifest).encode())

        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in tmp_dir.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(tmp_dir))


def _write_roi_mask(mask: np.ndarray, path: Path) -> None:
    """Write the ROI as an 8-bit image: 255 inside, 0 outside."""
    cv2.imwrite(str(path), np.asarray(mask, dtype=bool).astype(np.uint8) * 255)


def _last_frame_image(result: CorrelationResult, built: BuiltRun) -> np.ndarray:
    last = result.frames[-1]
    if last.frame_index == 0:
        return built.setup_output.reference_frame
    return built.frame_loader(last.frame_name)


def write_directory_output(result, built, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    export_full_csv(result, out_dir / "data" / "correlation.csv")
    _generate_summary_plots(result, _last_frame_image(result, built), out_dir / "plots")
    _write_roi_mask(built.setup_output.roi_mask, out_dir / ROI_MASK_NAME)
    session = built.resolved_session
    # The manifest is written last: its presence marks a complete output, which is
    # what the validation runner checks before reusing a run.
    manifest = build_manifest(
        result=result,
        setup_output=built.setup_output,
        correlation_config=built.correlation_config,
        reference_frame_name=session.reference_frame_name,
        deformed_frames=session.deformed_frames,
    )
    (out_dir / MANIFEST_NAME).write_bytes(tomli_w.dumps(manifest).encode())
