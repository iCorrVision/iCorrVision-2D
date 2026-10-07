from collections.abc import Callable

import numpy as np

from icorrvision.contracts import CorrelationConfig, SetupOutput
from icorrvision.engine.pipeline.shape_function import (
    AffineShapeFunction,
    QuadraticShapeFunction,
)
from icorrvision.engine.pipeline.interpolation import (
    EightTapInterpolator,
    BicubicSplineInterpolator,
    BiquinticSplineInterpolator,
)
from icorrvision.engine.pipeline.criteria import ZNCCCriterion, ZNSSDCriterion
from icorrvision.engine.pipeline.search import (
    BruteSearch,
    PyramidSearch,
)
from icorrvision.engine.pipeline.refinement import ICGNRefinement
from icorrvision.engine.pipeline.tensors import (
    PointCloudGreenLagrangeStrain,
    WindowGreenLagrangeStrain,
)
from icorrvision.engine.engine import CorrelationEngine
from icorrvision.engine.recovery import HistoryRecovery


CRITERION_REGISTRY = {
    "zncc": ZNCCCriterion,
    "znssd": ZNSSDCriterion,
}
SEARCH_REGISTRY = {
    "brute": BruteSearch,
    "pyramid": PyramidSearch,
}
SHAPE_FUNCTION = {
    "affine": AffineShapeFunction,
    "quadratic": QuadraticShapeFunction,
}

REFINEMENT_REGISTRY = {"icgn": ICGNRefinement}

TENSOR_REGISTRY = {
    "lagrange_cloud": PointCloudGreenLagrangeStrain,
    "lagrange_q4": WindowGreenLagrangeStrain,
    "lagrange_q9": WindowGreenLagrangeStrain,
}
INTERPOLATOR_REGISTRY = {
    "bicubic_spline": BicubicSplineInterpolator,
    "biquintic_spline": BiquinticSplineInterpolator,
    "8tap": EightTapInterpolator,
}

_REQUIRED_SETUP_FIELDS = (
    "reference_frame",
    "roi_mask",
    "subset_size",
    "step_size",
    "deformed_list",
)


class EngineBuildError(ValueError):
    """Raised when a CorrelationConfig cannot be built into a CorrelationEngine."""


def _lookup(registry: dict, key: str, field_name: str):
    try:
        return registry[key]
    except KeyError:
        raise EngineBuildError(
            f"unsupported {field_name}: {key!r}. Valid options: {sorted(registry)}"
        ) from None


def _check_setup(config: CorrelationConfig, setup: SetupOutput) -> None:
    """Reject an incomplete or inconsistent setup with a message naming the problem."""
    missing = [name for name in _REQUIRED_SETUP_FIELDS if getattr(setup, name) is None]
    if missing:
        raise EngineBuildError(
            f"SetupOutput is incomplete; missing: {', '.join(missing)}"
        )

    image_shape = setup.reference_frame.shape[:2]
    if setup.roi_mask.shape[:2] != image_shape:
        raise EngineBuildError(
            f"roi_mask shape {setup.roi_mask.shape[:2]} does not match "
            f"reference_frame shape {image_shape}"
        )
    if config.search_size <= setup.subset_size:
        raise EngineBuildError(
            f"search_size ({config.search_size}) must be larger than "
            f"subset_size ({setup.subset_size})"
        )


def _check_stage(stage_cls, stage_config, setup: SetupOutput) -> None:
    """Let a registered stage reject a setup it cannot work with."""
    try:
        stage_cls.check_setup(stage_config, setup)
    except ValueError as exc:
        raise EngineBuildError(str(exc)) from None


def build_engine(
    config: CorrelationConfig,
    setup: SetupOutput,
    frame_loader: Callable[[str], np.ndarray],
) -> CorrelationEngine:

    _check_setup(config, setup)

    interpolator_cls = _lookup(
        INTERPOLATOR_REGISTRY,
        config.image_interpolation.method,
        "interpolation method",
    )
    interpolator = interpolator_cls()

    criterion_cls = _lookup(
        CRITERION_REGISTRY,
        config.criterion_name,
        "criterion method",
    )
    criterion = criterion_cls()

    search_cls = _lookup(
        SEARCH_REGISTRY,
        config.search.method,
        "search method",
    )
    _check_stage(search_cls, config.search, setup)
    search = search_cls.from_config(criterion, config.match_threshold, config.search)

    shape_function_cls = _lookup(
        SHAPE_FUNCTION,
        config.refinement.shape_function,
        "shape function method",
    )
    shape_function = shape_function_cls()

    refinement_cls = _lookup(
        REFINEMENT_REGISTRY,
        config.refinement.method,
        "refinement method",
    )
    refinement = refinement_cls.from_config(
        criterion, interpolator, shape_function, config.refinement
    )

    tensor_cls = _lookup(
        TENSOR_REGISTRY,
        config.strain.method,
        "tensor",
    )
    _check_stage(tensor_cls, config.strain, setup)
    tensor = tensor_cls.from_config(config.strain)

    recovery = HistoryRecovery.from_config(config.recovery)

    return CorrelationEngine(
        search, refinement, tensor, setup, config, frame_loader, recovery
    )
