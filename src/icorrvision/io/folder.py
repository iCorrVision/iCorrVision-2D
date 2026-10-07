import cv2
import csv
import logging
import re
import numpy as np
from collections.abc import Callable

from pathlib import Path

from icorrvision.contracts import ImageRecord


TIFF_SUFFIXES = (".tif", ".tiff")


def natural_key(name: str) -> list:
    """Sort key that orders the digit runs of a name by value: img_2 before img_10."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def list_frames(directory: Path | str) -> list[Path]:
    """TIFF frames in `directory`, in natural order, with suffixes in any case."""
    return sorted(
        (
            p
            for p in Path(directory).iterdir()
            if p.is_file() and p.suffix.lower() in TIFF_SUFFIXES
        ),
        key=lambda p: natural_key(p.name),
    )


def read_grayscale_frame(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def folder_loader(image_dir: Path | str) -> Callable[[str], np.ndarray]:
    """Return a frame loader for image_dir.

    The loader maps a frame name to a grayscale 2D array, as build_engine
    expects, and raises FileNotFoundError for a missing or unreadable frame.
    """
    image_dir = Path(image_dir)

    def loader(name: str) -> np.ndarray:
        return read_grayscale_frame(image_dir / name)

    return loader


# Leading columns of the manifest written by iCorrVision Grabber.
MANIFEST_COLUMNS = ("date", "left_frame_time", "left_frame_name")


def is_manifest(path: Path) -> bool:
    """True if the CSV starts with the Grabber manifest header."""
    try:
        with path.open(newline="", encoding="utf-8") as f:
            header = f.readline().strip().lower().split(";")
    except (OSError, UnicodeDecodeError):
        return False
    return tuple(header[: len(MANIFEST_COLUMNS)]) == MANIFEST_COLUMNS


class AmbiguousManifestError(Exception):
    """Raised when a folder contains more than one Grabber manifest."""


class FolderParser:
    def __init__(self, folder_path: str) -> None:
        self.folder_path: Path = Path(folder_path)
        self.csv_path: Path
        self.records: list[ImageRecord] = []

        self.find_manifest(self.folder_path)

    def find_manifest(self, folder_path: Path) -> None:
        """Locate and parse the CSV manifest in folder_path.

        A CSV counts as a manifest only if it carries the Grabber header; other
        CSV files (e.g. a load-cell record) are ignored. Without a manifest, the
        TIFF frames in the folder are listed instead. More than one manifest
        raises AmbiguousManifestError, which the presenter reports to the user.
        """
        csv_files = sorted(
            (p for p in folder_path.iterdir() if p.suffix.lower() == ".csv"),
            key=lambda p: natural_key(p.name),
        )
        candidates = [p for p in csv_files if is_manifest(p)]
        for other in sorted(set(csv_files) - set(candidates)):
            logging.info("Ignoring %s: not a Grabber manifest.", other.name)
        if not candidates:
            logging.info("No Grabber manifest in %s; listing its TIFF frames.", folder_path)
            frames = list_frames(folder_path)
            if not frames:
                logging.critical(f"No .tiff images found in {folder_path}")
                return
            else:
                self.parse_frames(frames)
                return

        if len(candidates) > 1:
            raise AmbiguousManifestError(
                f"Multiple Grabber manifests found in {folder_path}: "
                f"{[c.name for c in candidates]}"
            )
        self.csv_path = candidates[0]
        self.parse_manifest()

    def parse_manifest(self) -> None:
        """Parse a DICgrabber CSV manifest into ImageRecords."""
        with self.csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            fieldnames = reader.fieldnames or []
            expects_right = (
                "right_frame_time" in fieldnames and "right_frame_name" in fieldnames
            )

            records: list[ImageRecord] = []
            anomalous_rows = 0

            for row in reader:
                right_time = row.get("right_frame_time") or None
                right_name = row.get("right_frame_name") or None

                if expects_right:
                    if right_time is None or right_name is None:
                        anomalous_rows += 1
                        logging.critical(
                            "Row dated %s lacks right-channel data although %s "
                            "declares a stereo capture; check for a dropped frame "
                            "or a camera disconnect during acquisition.",
                            row["date"],
                            self.csv_path.name,
                        )
                    elif not (self.folder_path / row["right_frame_name"]).is_file():
                        logging.critical(f"frame missing: {row['right_frame_name']}")
                        right_name += " MISSING FILE"

                if not (self.folder_path / row["left_frame_name"]).is_file():
                    logging.critical(f"frame missing: {row['left_frame_name']}")
                    row["left_frame_name"] += " MISSING FILE"

                records.append(
                    ImageRecord(
                        date=row["date"],
                        left_frame_time=row["left_frame_time"],
                        left_frame_name=row["left_frame_name"],
                        right_frame_time=right_time,
                        right_frame_name=right_name,
                    )
                )

            if not expects_right:
                logging.info(
                    "%s has no right-channel columns: mono DIC session.",
                    self.csv_path.name,
                )
            elif anomalous_rows:
                logging.critical(
                    "%s: %d of %d rows lack right-channel data; the manifest "
                    "is likely corrupted. Verify the acquisition logs before "
                    "correlating.",
                    self.csv_path.name,
                    anomalous_rows,
                    len(records),
                )

        self.records = records

    def parse_frames(self, frames: list[Path]) -> None:
        logging.info("Listing found frames.")
        records: list[ImageRecord] = []

        for frame in frames:
            if "mask" in frame.name:
                pass
            else:
                records.append(
                    ImageRecord(
                        date=None,
                        left_frame_time=None,
                        left_frame_name=frame.name,
                    )
                )

        self.records = records

    type FrameLoader = Callable[[str], np.ndarray]

    def pull_image(self, image_name: str) -> np.ndarray | None:
        logging.debug(image_name)
        if not self.records:
            logging.critical("no images on record")
            return
        if "MISSING FILE" in image_name:
            logging.critical("image not found")
            return

        return read_grayscale_frame(self.folder_path / image_name)
