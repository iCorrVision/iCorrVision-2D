"""The manifest written with every correlation output.

One place defines the format: the results archive (.icorr) and the CLI's
directory output both write it, and the archive reader and the validation
runner both read it. Any change to the format belongs here, with a new
FORMAT_VERSION.
"""

import tomllib
from dataclasses import asdict
from datetime import datetime, timezone

from icorrvision.contracts import CorrelationConfig, CorrelationResult, SetupOutput

FORMAT_VERSION = 1

# File names inside an output (archive or directory), shared by writer, reader and runner.
MANIFEST_NAME = "manifest.toml"
ROI_MASK_NAME = "roi_mask.tiff"
CSV_NAME = "data/correlation.csv"


def build_manifest(
    *,
    result: CorrelationResult,
    setup_output: SetupOutput,
    correlation_config: CorrelationConfig,
    reference_frame_name: str,
    deformed_frames: list[str],
) -> dict:
    """Describe one correlation run: its inputs, its settings and how it ended.

    All arguments are keyword-only, so they cannot be passed in the wrong order.
    """
    return {
        "format_version": FORMAT_VERSION,
        "session": {
            "mode": "explicit",
            "reference_frame": reference_frame_name,
            "deformed_frames": deformed_frames,
        },
        "setup": {
            "subset_size": setup_output.subset_size,
            "step_size": setup_output.step_size,
            "calibration_mm_per_px": setup_output.calibration_mm_per_px,
            "roi_mask": ROI_MASK_NAME,
        },
        "correlation": asdict(correlation_config),
        "run": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "frame_count": len(result.frames),
            "aborted": result.aborted,
            # Engine time only: the sum of the per-frame times, which excludes
            # interpreter start-up, setup and file writing.
            "duration_sec": round(sum(result.frame_times_sec), 2),
        },
    }


def read_manifest(data: bytes) -> dict:
    """Parse a manifest and refuse formats this version cannot read.

    Takes bytes rather than a path, so the caller decides where they come from
    (a member of a zip archive, or a file in an output directory).
    """
    manifest = tomllib.loads(data.decode())
    version = manifest.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"unsupported output format {version!r}: this version of iCorrVision "
            f"reads format {FORMAT_VERSION}"
        )
    return manifest
