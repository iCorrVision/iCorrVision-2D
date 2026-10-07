# Changelog

All notable changes to iCorrVision 2D are recorded here. Versions follow
[semantic versioning](https://semver.org/).

## Unreleased

- **One installed package.** The code moved into `src/icorrvision/`, which is now the only package
  the wheel installs; 2.0.0 also installed the generic top-level packages `components`, `core`,
  `cli` and `UI`. Its subpackages are `engine` (with `engine.pipeline`), `config`, `io`,
  `results`, `cli` and `gui`. Only `gui` imports Qt, and no subpackage imports `gui` or `cli`
  except `gui` and `cli` themselves.
- The public interface (`import icorrvision`) and the `icorr` command are unchanged. Code that
  imported internal modules must use their new paths, e.g. `components.archive.manifest` is now
  `icorrvision.io.archive.manifest`.
- The graphical interface starts with `python -m icorrvision.gui.main` (was
  `python -m core.main`).
- `run.toml`, `template.toml` and the API tour moved to `examples/`, the validation notebooks to
  `validation/notebooks/` and the timing notebooks to `benchmarks/`.
- The Windows build workflow also runs on pull requests to `main`.

## 2.0.0 (tag `v2.0.0-msc`)

First release of iCorrVision 2D 2.0, the version evaluated in the MSc thesis *iCorrVision 2.0:
Development and Metrological Assessment of Modular Software for Two-Dimensional Digital Image
Correlation* (NOVA FCT, 2026). It is a reimplementation of the correlation part of
iCorrVision v1 and shares no code with it.

- **Engine** installable as the Python package `icorrvision`, without a graphical toolkit, and
  used through a single public interface. Each stage (criterion, coarse search, IC-GN
  refinement, shape function, interpolant, strain estimator) is selected by name from a
  registry, to which new methods can be added without changing the pipeline.
- **Command-line interface** `icorr`, which runs a correlation from a TOML configuration file
  and writes a results folder or a `.icorr` archive (`template.toml` lists every option).
- **Graphical application** (PySide6) covering the workflow from image selection to results:
  session, setup, correlation and a results preview.
- **Validation notebooks** that reproduce the results of the thesis on the DIC Challenge
  reference images, and an executable walkthrough of the public interface.
- **Windows executable** of the graphical application, built by the GitHub Actions workflow.

Tested with Python 3.12.13, NumPy 2.5.3, SciPy 1.18.1, OpenCV 5.0.0, Numba 0.67.0 and
PyOpenCL 2026.1.4. The known limitations of this release are collected in Table 4.4 of the
thesis.
