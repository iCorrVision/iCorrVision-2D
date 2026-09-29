"""Public API of the iCorrVision 2.0 correlation engine.

Import from here and nowhere else. Every name below is a deliberate part of the
API; everything under ``components`` that is not re-exported here is internal
and may move without notice. This module contains no logic: it only re-exports
objects defined elsewhere, so ``icorrvision.build_engine`` *is* the factory's
``build_engine``, not a wrapper around it.

Nothing here imports Qt. The engine can be installed and used without PySide6;
the graphical application is a separate consumer of this same API.

The names fall into four groups:

Configuring
    Frozen dataclasses describing *what* to compute. A run is declared by
    building these objects; nothing is executed until ``build_engine(...).run()``.

Running
    ``build_engine`` turns a configuration into a ready engine and validates it
    before any correlation starts. ``folder_loader`` and ``build_run`` are the
    two ways of getting images in: a folder of TIFFs, or a CLI-style TOML file.

Reading results
    The result dataclasses, and the functions that write them to CSV or to an
    ``.icorr`` archive and read an archive back.

Extending
    The abstract base class for each algorithmic stage, the types those stages
    exchange, and the registries the factory selects from. The registries are
    plain dicts and are deliberately mutable: adding an entry, e.g.
    ``INTERPOLATOR_REGISTRY["bilinear"] = MyInterpolator``, makes a new method
    selectable by name without modifying any pipeline code.
"""

from importlib.metadata import PackageNotFoundError, version

# --- Configuring -------------------------------------------------------------
from components.contracts import (
    CorrelationConfig,
    InterpolationConfig,
    RecoveryConfig,
    RefinementConfig,
    SearchConfig,
    SetupOutput,
    StrainConfig,
    build_correlation_config,
)

# --- Running -----------------------------------------------------------------
from components.config.loader import BuiltRun, build_run
from components.session.folder import folder_loader
from components.correlation.orchestration.factory import EngineBuildError, build_engine

# --- Reading results ---------------------------------------------------------
from components.archive.reader import LoadedArchive, read_icorr_archive
from components.archive.writer import write_icorr_archive
from components.contracts import CorrelationFrame, CorrelationResult, StrainField
from components.preview.export import export_full_csv, export_summary_csv

# --- Extending ---------------------------------------------------------------
from components.correlation.orchestration.engine import DisplacementField, ReferenceGrid
from components.correlation.orchestration.factory import (
    CRITERION_REGISTRY,
    INTERPOLATOR_REGISTRY,
    REFINEMENT_REGISTRY,
    SEARCH_REGISTRY,
    SHAPE_FUNCTION,
    TENSOR_REGISTRY,
)
from components.correlation.pipeline.criteria import CorrelationCriterion
from components.correlation.pipeline.interpolation import SubpixelInterpolator
from components.correlation.pipeline.refinement import (
    DisplacementResult,
    SubpixelRefinement,
)
from components.correlation.pipeline.search import (
    InitialGuess,
    SearchRange,
    SearchStrategy,
)
from components.correlation.pipeline.shape_function import ShapeFunction
from components.correlation.pipeline.tensors import StrainTensor

try:
    # Read from the installed distribution's metadata so the version is written
    # once, in pyproject.toml. Running from a source checkout without installing
    # has no metadata to read, hence the fallback.
    __version__ = version("icorrvision")
except PackageNotFoundError:
    __version__ = "unknown (not installed)"

__all__ = [
    # configuring
    "CorrelationConfig",
    "SearchConfig",
    "RefinementConfig",
    "StrainConfig",
    "InterpolationConfig",
    "RecoveryConfig",
    "SetupOutput",
    "build_correlation_config",
    # running
    "build_engine",
    "EngineBuildError",
    "folder_loader",
    "build_run",
    "BuiltRun",
    # reading results
    "CorrelationResult",
    "CorrelationFrame",
    "StrainField",
    "export_full_csv",
    "export_summary_csv",
    "write_icorr_archive",
    "read_icorr_archive",
    "LoadedArchive",
    # extending
    "CorrelationCriterion",
    "SearchStrategy",
    "SearchRange",
    "InitialGuess",
    "ShapeFunction",
    "SubpixelInterpolator",
    "SubpixelRefinement",
    "DisplacementResult",
    "StrainTensor",
    "DisplacementField",
    "ReferenceGrid",
    "CRITERION_REGISTRY",
    "SEARCH_REGISTRY",
    "SHAPE_FUNCTION",
    "REFINEMENT_REGISTRY",
    "TENSOR_REGISTRY",
    "INTERPOLATOR_REGISTRY",
    # metadata
    "__version__",
]
