import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import cv2
from icorrvision.contracts import (
    CorrelationConfig,
    SetupOutput,
    build_correlation_config,
    check_keys,
    from_table,
)
from icorrvision.io.folder import folder_loader, list_frames
from icorrvision.config.schema import RunConfig, SessionConfig, SetupConfig

# ------------------------------------------------------------------
# Parsing + overrides
# ------------------------------------------------------------------


def apply_overrides(config_dict: dict, overrides: list[str]) -> dict:
    """Apply CLI overrides ('key.path=value') to a nested TOML configuration."""
    for raw in overrides:
        key_path, sep, value_str = raw.partition("=")
        if not sep:
            raise ValueError(f"--set expects key.path=value, got {raw!r}")
        value = tomllib.loads(f"_={value_str}")["_"]
        *parents, leaf = key_path.split(".")
        node = config_dict
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    return config_dict


def load_run_config(toml_path: Path, set_overrides: list[str]) -> RunConfig:
    """Parse a TOML file, apply CLI overrides and return a validated RunConfig."""
    with toml_path.open("rb") as f:
        raw = tomllib.load(f)
    raw = apply_overrides(raw, set_overrides)
    check_keys(raw, {"session", "setup", "correlation"}, "top level")
    return RunConfig(
        session=from_table(SessionConfig, raw.get("session", {}), "session"),
        setup=from_table(SetupConfig, raw.get("setup", {}), "setup"),
        correlation=build_correlation_config(raw.get("correlation", {})),
    )


# ------------------------------------------------------------------
# Resolution: TOML-shaped intent -> runtime-ready objects
# ------------------------------------------------------------------



def _resolve_path(
    raw: str, toml_dir: Path, image_dir_override: Path | None, *, is_image_path: bool
) -> Path:
    """Resolve a configured path.

    An absolute path is kept. Otherwise image paths resolve against --image-dir
    when given, and every other path (roi_mask, explicit_frame_list) against
    toml_dir.
    """
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    if is_image_path and image_dir_override is not None:
        return (image_dir_override / p).resolve()
    return (toml_dir / p).resolve()


def _glob_tiffs(directory: Path, exclude: set[Path]) -> list[str]:
    names = [p.name for p in list_frames(directory) if p.resolve() not in exclude]
    if not names:
        raise ValueError(f"No .tiff images found in {directory}")
    return names


def _read_frame_list(path: Path) -> list[str]:
    """Read the image filenames from a text file, skipping blanks and comments."""
    lines = path.read_text().splitlines()
    return [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]


@dataclass(frozen=True)
class ResolvedSession:
    """Resolved session: image directory, reference and deformed frames, frame loader."""

    image_dir: Path
    reference_frame_name: str
    deformed_frames: list[str]
    loader: Callable[[str], np.ndarray]


def resolve_session(
    session: SessionConfig,
    toml_dir: Path,
    image_dir_override: Path | None,
    exclude_paths: set[Path],
) -> ResolvedSession:
    image_dir = _resolve_path(
        session.image_dir, toml_dir, image_dir_override, is_image_path=True
    )
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image_dir does not exist: {image_dir}")

    loader = folder_loader(image_dir)

    if session.mode == "all":
        names = _glob_tiffs(image_dir, exclude_paths)
        reference = session.reference_frame or names[0]
        deformed = [n for n in names if n != reference]
    elif session.mode == "stride":
        names = _glob_tiffs(image_dir, exclude_paths)
        reference = session.reference_frame or names[0]
        remaining = [n for n in names if n != reference]
        deformed = remaining[:: session.stride_n]
    elif session.mode == "explicit":
        frame_list_path = _resolve_path(
            session.explicit_frame_list, toml_dir, None, is_image_path=False
        )
        if not frame_list_path.exists():
            raise FileNotFoundError(f"explicit_frame_list not found: {frame_list_path}")
        reference = session.reference_frame
        deformed = _read_frame_list(frame_list_path)
        if reference in deformed:
            raise ValueError(
                f"reference_frame {reference!r} must not appear in explicit_frame_list"
            )

    else:
        raise ValueError(f"unknown session mode: {session.mode!r}")

    if not (image_dir / reference).exists():
        raise FileNotFoundError(f"reference_frame not found: {image_dir / reference}")
    missing = [n for n in deformed if not (image_dir / n).exists()]
    if missing:
        raise FileNotFoundError(f"deformed frames not found: {missing}")

    return ResolvedSession(image_dir, reference, deformed, loader)


def _resolve_roi_mask_path(setup: SetupConfig, toml_dir: Path) -> Path | None:
    if setup.roi_mask is None:
        return None
    return _resolve_path(setup.roi_mask, toml_dir, None, is_image_path=False)


def resolve_setup(
    setup: SetupConfig, resolved_session: ResolvedSession, toml_dir: Path
) -> SetupOutput:
    """Load the reference frame and ROI mask and build the SetupOutput."""
    reference_frame = resolved_session.loader(resolved_session.reference_frame_name)

    if setup.roi_mask is not None:
        mask_path = _resolve_path(setup.roi_mask, toml_dir, None, is_image_path=False)
        if not mask_path.exists():
            raise FileNotFoundError(f"roi_mask not found: {mask_path}")
        roi_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE).astype(bool)
    else:
        roi_mask = np.full(reference_frame.shape[:2], True, dtype=bool)

    return SetupOutput(
        reference_frame=reference_frame,
        roi_mask=roi_mask,
        subset_size=setup.subset_size,
        step_size=setup.step_size,
        deformed_list=resolved_session.deformed_frames,
        calibration_mm_per_px=setup.calibration_mm_per_px,
    )


@dataclass(frozen=True)
class BuiltRun:
    setup_output: SetupOutput
    correlation_config: CorrelationConfig
    frame_loader: Callable[[str], np.ndarray]
    resolved_session: (
        ResolvedSession  # kept for writing the archive after the run (needs image_dir, frame names)
    )


def build_run(
    toml_path: Path, set_overrides: list[str], image_dir_override: Path | None
) -> BuiltRun:
    """Parse the TOML, resolve session and setup, and return a BuiltRun for the CLI."""
    toml_path = toml_path.resolve()
    toml_dir = toml_path.parent

    run_config = load_run_config(toml_path, set_overrides)

    mask_path = _resolve_roi_mask_path(run_config.setup, toml_dir)
    exclude = {mask_path.resolve()} if mask_path is not None else set()

    resolved_session = resolve_session(
        run_config.session, toml_dir, image_dir_override, exclude
    )
    setup_output = resolve_setup(run_config.setup, resolved_session, toml_dir)

    return BuiltRun(
        setup_output=setup_output,
        correlation_config=run_config.correlation,
        frame_loader=resolved_session.loader,
        resolved_session=resolved_session,
    )
