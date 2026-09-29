"""Known displacement fields for the DIC Challenge reference image sets.

One interface, one implementation per truth *convention* rather than per
dataset, because the nineteen Challenge sets between them use only four:

    1. the shift is written into the filename          (Samples 3, 4, 5)
    2. the shift is index x a documented constant      (Samples 1, 2, 6, 7, 9, Rot_s1)
    3. the field is tabulated in a supplied file       (Samples 14, 15, Stars)
    4. there is no ground truth at all                 (Samples 12, 13, 17)

Case 4 is represented explicitly. Samples 12 and 13 are experimental series
with no known solution; asking them for a truth field raises an error rather
than returning zeros, which would produce a plausible but meaningless error
table.

COORDINATE CONVENTION
    `displacement_at` receives node coordinates in the REFERENCE image, in
    pixels, with the origin at the top-left of the array and y increasing
    downward: the convention of the Challenge analysis codes (Reu et al.
    Fig. 10) and of the engine's ReferenceGrid.

EULERIAN VS LAGRANGIAN
    For a spatially uniform field (every rigid-translation set) the commanded
    displacement is the same whether evaluated at the reference or the
    deformed position, so no correction is needed and these classes are
    exact. For a spatially varying field it is NOT, and the Challenge
    provides MATLAB scripts (SampleNN_Euler2Lagrangian.m) that define the
    correction. `TabulatedTruth` therefore expects an already-corrected field
    and does not apply the correction itself.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class NoGroundTruthError(RuntimeError):
    """Raised when error metrics are requested for a dataset that has none."""


class GroundTruth(ABC):
    """A known displacement field, sampled at arbitrary reference positions."""

    #: Human-readable note on where the truth came from, for the results table.
    provenance: str = ""

    @abstractmethod
    def displacement_at(
        self, x: np.ndarray, y: np.ndarray, frame_name: str, frame_index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Commanded (u, v) at reference positions (x, y), in pixels.

        Returns arrays shaped like `x`. NaN marks positions where the truth
        is undefined, which the metrics then exclude.
        """

    def is_uniform(self) -> bool:
        """True if the field is constant in space.

        Uniform fields need no Eulerian-to-Lagrangian correction.
        """
        return False


# ---------------------------------------------------------------------------
# 1. Shift written into the filename
# ---------------------------------------------------------------------------

#: Matches "Sample3-001 X0.10 Y0.10 N2 C0 R0.tif" and the Sample 4/5 variants.
_XY_IN_NAME = re.compile(
    r"X(?P<x>-?\d+(?:\.\d+)?)\s+Y(?P<y>-?\d+(?:\.\d+)?)", re.IGNORECASE
)


@dataclass(frozen=True)
class FilenameShiftTruth(GroundTruth):
    """Rigid translation read from the image filename.

    Covers Samples 3, 4 and 5, whose names carry the imposed shift directly.
    Sample 5 additionally varies contrast (C) and rotation-free noise (R) per
    frame, which does not affect the commanded displacement.
    """

    provenance: str = "shift parsed from filename (X../Y..)"

    def displacement_at(self, x, y, frame_name, frame_index):
        match = _XY_IN_NAME.search(Path(frame_name).stem)
        if match is None:
            raise ValueError(
                f"no X../Y.. shift found in {frame_name!r}; "
                "use IndexedShiftTruth for sets that encode only an index"
            )
        u = float(match.group("x"))
        v = float(match.group("y"))
        return np.full_like(np.asarray(x, dtype=np.float64), u), np.full_like(
            np.asarray(y, dtype=np.float64), v
        )

    def is_uniform(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# 2. Shift is index x a documented constant
# ---------------------------------------------------------------------------

#: Trailing integer in a name: trxy_s2_07, Sample9-05, rot_s1_03, "Y07 X07".
_TRAILING_INDEX = re.compile(r"(\d+)(?!.*\d)")


@dataclass(frozen=True)
class IndexedShiftTruth(GroundTruth):
    """Rigid translation of `step` pixels per frame index.

    Covers Sample 1 and 2 (0.05 px/step), 6 and 7 (0.1 px/step). The step is
    documented in the set description rather than in the filename, so it must
    be supplied; there is no way to recover it from the images.

    The index is taken from the frame's own name rather than from its
    position in the sequence, so a run that skips frames (stride mode) still
    gets the right answer.
    """

    step_x: float
    step_y: float
    index_from_name: bool = True
    provenance: str = ""

    def __post_init__(self) -> None:
        if not self.provenance:
            object.__setattr__(
                self,
                "provenance",
                f"index x ({self.step_x}, {self.step_y}) px per step",
            )

    def _index(self, frame_name: str, frame_index: int) -> int:
        if not self.index_from_name:
            return frame_index
        match = _TRAILING_INDEX.search(Path(frame_name).stem)
        if match is None:
            raise ValueError(f"no trailing index in {frame_name!r}")
        return int(match.group(1))

    def displacement_at(self, x, y, frame_name, frame_index):
        n = self._index(frame_name, frame_index)
        shape = np.shape(x)
        return (
            np.full(shape, n * self.step_x, dtype=np.float64),
            np.full(shape, n * self.step_y, dtype=np.float64),
        )

    def is_uniform(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# 3. Tabulated / analytic field
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalyticFieldTruth(GroundTruth):
    """A closed-form field, evaluated wherever it is asked for.

    Used for Sample 14's swept sine and the Challenge 2.0 Star patterns. The
    callable receives (x, y, frame_name, frame_index) and returns (u, v).

    NOT automatically Lagrangian: a spatially varying commanded field is
    defined in Eulerian coordinates and must be corrected before comparison
    with DIC output; pass the corrected field, not the raw one.
    """

    field: callable
    provenance: str = "analytic field"
    uniform: bool = False

    def displacement_at(self, x, y, frame_name, frame_index):
        u, v = self.field(x, y, frame_name, frame_index)
        return np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)

    def is_uniform(self) -> bool:
        return self.uniform


# ---------------------------------------------------------------------------
# 4. No truth at all
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoTruth(GroundTruth):
    """An experimental set with no known solution.

    Samples 12 and 13 are real tensile tests: there is no commanded field, so
    error metrics are undefined. Such sets still serve to compare
    configurations with each other or with iCorrVision 1, but computing an
    error against them raises an error rather than returning zeros.
    """

    reason: str = "experimental series: no commanded displacement field exists"
    provenance: str = "none (experimental)"

    def displacement_at(self, x, y, frame_name, frame_index):
        raise NoGroundTruthError(self.reason)


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

#: Truth convention per Challenge 1.0 set, keyed by the directory name used
#: in the distributed archive. Steps are taken from the set descriptions
#: ("2D Challenge 1.0 set descriptions.xlsx"), not measured from the images.
CHALLENGE1_TRUTH: dict[str, GroundTruth] = {
    "Sample1": IndexedShiftTruth(0.05, 0.05),
    "Sample2": IndexedShiftTruth(0.05, 0.05),
    "Sample3": FilenameShiftTruth(),
    "Sample4": FilenameShiftTruth(),
    "Sample5": FilenameShiftTruth(),
    "Sample6": IndexedShiftTruth(0.1, 0.1),
    "Sample7": IndexedShiftTruth(0.1, 0.1),
    "Sample12": NoTruth("open-hole tension: experimental, no commanded field"),
    "Sample13": NoTruth("laser weld: experimental, no commanded field"),
}


def truth_for(dataset: str) -> GroundTruth:
    """Look up the truth provider for a Challenge 1.0 set directory name."""
    try:
        return CHALLENGE1_TRUTH[dataset]
    except KeyError:
        raise KeyError(
            f"no ground-truth convention registered for {dataset!r}. "
            f"Known: {sorted(CHALLENGE1_TRUTH)}. Rotation sets (Sample9, "
            "Rot_s1) need an angular truth, and Samples 14/15 and the Star "
            "sets need the tabulated fields plus a Lagrangian correction."
        ) from None
