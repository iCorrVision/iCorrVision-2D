# Contributing to iCorrVision 2D

Thank you for your interest in iCorrVision 2D, a subset-based 2D digital image correlation (DIC) engine
that can be used as a Python package, from the command line (`icorr`) or through a PySide6 graphical
interface. This project began as an MSc thesis (João Pedro da Silva Duarte Récio, NOVA FCT, 2026,
supervised by José Manuel Cardoso Xavier) and is open source under GPL-3.0-or-later.
Contributions are welcome: bug reports, documentation, validation studies, and new features or methods.

## Ground rules

- **Version 2 only.** This repository is iCorrVision-2D version 2, a reimplementation that shares no code
  with version 1. Version 1 ([jcadf/iCorrVision_2D](https://github.com/jcadf/iCorrVision_2D)) has not changed
  since June 2022 and does not receive changes here.
- **The thesis tag is frozen.** The tag `v2.0.0-msc` is the version cited by the thesis. It must never be
  moved or deleted. All new work goes to `main` through pull requests.
- **Licence.** By contributing, you agree that your contribution is licensed under GPL-3.0-or-later, like
  the rest of the project (see `LICENSE`).
- Be respectful and constructive in issues and reviews. This project follows the
  [Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting a bug or proposing a feature

Open an issue first, especially for anything larger than a small fix. The issue forms ask for the
information listed below.

For a bug, include:

- the iCorrVision version (`python -c "import icorrvision; print(icorrvision.__version__)"`) or the commit,
  the Python version and the operating system;
- the configuration (TOML) or code that reproduces it;
- a small image set or synthetic example if possible (`examples/synthetic.py` generates synthetic images);
  do not attach confidential data;
- what you expected and what happened.

For a feature, describe the use case, the proposed behaviour and, for a new method, the reference it
comes from.

## Setting up a development environment

Python 3.12 or newer is required (`requires-python = ">=3.12"`).

With pip:

```sh
git clone https://github.com/iCorrVision/iCorrVision-2D.git
cd iCorrVision-2D
pip install -e ".[gui,notebooks,dev]"
```

The extras are defined in `pyproject.toml`, which is the only list of dependencies:

| Extra | Contents |
| --- | --- |
| `gui` | PySide6 (>=6.10) and Pillow, for the graphical interface |
| `notebooks` | Jupyter, ipykernel, nbconvert and pandas, for the notebooks |
| `acceleration` | numba and pyopencl, used only by `benchmarks/acceleration_routes.ipynb` |
| `dev` | black, isort, mypy and scipy-stubs |

With [devenv](https://devenv.sh/) (Nix): `devenv.nix` and `.envrc` (direnv) set up a Python 3.12 virtual
environment with the project installed in editable mode with every extra (`gui`, `notebooks`,
`acceleration`, `dev`), together with the system libraries Qt needs. It is written for Linux (it sets
`QT_QPA_PLATFORM` to `wayland;xcb` and lists the OpenCL driver path of NixOS). The thesis results were
produced on NixOS.
Platforms: the project has been cloned, installed and run on both Linux and Windows. devenv is only
needed on Linux (NixOS); on Windows, use a plain Python 3.12 virtual environment and the `pip install`
command above. macOS has not been tested.

To start the graphical interface from a development install: `python -m icorrvision.gui.main`.

## Code layout

The package lives in `src/icorrvision/`:

- `engine/` is the correlation engine: `engine.py`, `factory.py`, `recovery.py` and `pipeline/`, which holds
  one module per stage (`criteria.py`, `interpolation.py`, `refinement.py`, `search.py`, `shape_function.py`,
  `tensors.py`);
- `contracts.py` defines the shared data types, including the configuration dataclasses
  (`CorrelationConfig` and the dataclasses it contains) and their validation;
- `config/` reads the TOML run configuration (`loader.py`, `schema.py`);
- `io/` handles image input (`folder.py`) and the `.icorr` archive and CSV export (`archive/`, `export.py`);
- `results/` prepares result data for display (`diagnostics.py`, `payload.py`, `render.py`);
- `cli/` is the `icorr` command line (`icorr = "icorrvision.cli.main:main"`) and `gui/` is the PySide6 interface.

Outside the package: `examples/` (`run.toml`, `template.toml`, `synthetic.py`, `api_tour.ipynb`),
`validation/` (the harness code and `validation/notebooks/`), `benchmarks/` (timing notebooks) and `data/`
(inputs that are part of the repository, see `data/README.md`).

Rules for imports, kept since the move to the `src/` layout:

- Only `gui` imports Qt. `engine`, `config`, `io` and `results` must work without PySide6 installed.
- No subpackage imports `gui` or `cli`, except `gui` and `cli` themselves; `cli` may import anything except `gui`.
- Always use absolute imports, `icorrvision.<subpackage>.<module>`.
- The public API is what `src/icorrvision/__init__.py` re-exports. Everything else is internal and may move
  without notice, so examples, notebooks and user code should import from `icorrvision`.

## Adding a method (engine stage)

Each stage of the pipeline is selected by name from a registry, so a new method does not require changes to
the pipeline itself. The registries are plain dicts in `src/icorrvision/engine/factory.py`:

| Stage | Registry | Base class (in `engine/pipeline/`) |
| --- | --- | --- |
| Correlation criterion | `CRITERION_REGISTRY` | `CorrelationCriterion` (`criteria.py`) |
| Integer-pixel search | `SEARCH_REGISTRY` | `SearchStrategy` (`search.py`) |
| Shape function | `SHAPE_FUNCTION` | `ShapeFunction` (`shape_function.py`) |
| Sub-pixel refinement | `REFINEMENT_REGISTRY` | `SubpixelRefinement` (`refinement.py`) |
| Image interpolation | `INTERPOLATOR_REGISTRY` | `SubpixelInterpolator` (`interpolation.py`) |
| Strain estimator | `TENSOR_REGISTRY` | `StrainTensor` (`tensors.py`) |

1. Open (or comment on) an issue to agree on the design.
2. Create a branch from `main` and implement the stage as a subclass of the base class, in the matching
   module of `engine/pipeline/`. Stages that take options can define a `from_config` classmethod, and search
   and strain stages can define `check_setup`, which rejects a setup the method cannot work with by raising
   `ValueError` (the factory turns it into an `EngineBuildError`).
3. Register it in the matching dict of `engine/factory.py`; the key is the name used in the TOML file.
4. If it needs new options or new allowed values, add them to the configuration dataclasses in
   `contracts.py`, with validation in `__post_init__`. The `Literal` types there (for example
   `SearchMethodName`) list the permitted names but are not enforced at run time; the factory rejects an
   unregistered name with `EngineBuildError`.
5. Add the option and its allowed values to `examples/template.toml`, which lists every option.
6. If the new class is meant for users, export it from `src/icorrvision/__init__.py` (and `__all__`).
7. Show that it works (see the checks below) and describe the result in the pull request.
   The GUI does not read the registries: the choices it offers are listed by hand in label dictionaries in
   `src/icorrvision/gui/correlation/panel.py` (`_CRITERION_LABELS`, `_SEARCH_LABELS`,
   `_SHAPE_FUNCTION_LABELS`, `_INTERPOLATION_LABELS` and `_STRAIN_LABELS`). A method registered in the
   engine is therefore not shown in the graphical interface until it is added there, with the name used in
   the TOML file as the key and the text shown to the user as the value. The refinement method is
   deliberately not offered while IC-GN is the only one.

A method can also be registered at run time without editing the package, for example
`INTERPOLATOR_REGISTRY["bilinear"] = MyInterpolator`; this is shown in the docstring of
`src/icorrvision/__init__.py` and is useful for experiments.

## Checks before opening a pull request

There is no automated test suite in the repository yet, and the CI does not run linting or type checks.
The checks are therefore manual:

- **Run the example for the area you changed.** For the command line:
  set `image_dir` in `examples/run.toml` to a folder of TIFF frames, then
  `icorr run --config examples/run.toml --output-dir results/` (`results/` is ignored by git).
  `examples/api_tour.ipynb` runs on synthetic images and needs no data.
- **If you changed the engine**, run the relevant notebooks of `validation/notebooks/` from a fresh kernel and
  compare with the previous results:
  `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 <notebook>.ipynb`.
  The README lists what each notebook does, its inputs and its run time (from 1 to 45 minutes;
  `process_scaling` runs only on Linux). `star56_mei` compares the participants' MEI values of every
  execution with the frozen values in `data/star56/participant_mei_frozen.csv` and reports any value
  that changed by more than 1 %: explain such a change in the pull request.
- **Do not commit DIC Challenge data or large image sets.** The notebooks read the images through the
  environment variables `DIC_CHALLENGE` (Challenge 1.0), `DIC_CHALLENGE2` (Challenge 2.0) and `DIC_RESULTS`
  (run outputs; use an empty folder to recompute everything). See the README.
- **Style.** `black`, `isort` and `mypy` are installed by the `dev` extra, but the project has no
  configuration for them yet, and the existing code does not consistently follow their output. Follow the style of the
  surrounding code, and do not mix a reformatting of code you did not change into a feature or fix.
- **Tests.** If you add a test, say how to run it in the pull request.

## Pull requests and review

- Branch from `main` and open the pull request against `main`. Existing branches are named by kind of change,
  for example `docs/version-2-identity`, `fix/scrollable-panels` and `package-to-src`.
  Name new branches `<type>/<short-description>`, in lowercase with hyphens, where the type is one of `feat`
  (new feature or method), `fix` (bug fix), `docs` (documentation), `refactor` (restructuring without a
  change in behaviour) or `chore` (maintenance, CI, dependencies), for example `feat/zncc-criterion` or
  `docs/contributing-guide`.
- Keep pull requests focused: one feature or fix each, with a description of what changed and why.
- Write clear commit messages in the imperative mood, for example `Move the code into a src layout with one top-level package`.
- Update `CHANGELOG.md`: add an entry under `## Unreleased`, in the same prose style as the existing entries.
  Update the README when behaviour visible to users changes.
- The code owners in `.github/CODEOWNERS` (`@DanTheRobot` and `@josexavier3`) are asked to review pull requests.
- The GitHub Actions workflow "Build Windows exe" runs on every pull request to `main` (and on every push to
  `main`): it installs `.[gui]` with Python 3.12 and builds the graphical application with PyInstaller. It
  should pass before merging.
- Address review comments with new commits. Maintainers merge pull requests with a merge commit, so your
  individual commits are kept in the history; keep them tidy. Do not force-push to a branch under review
  unless a reviewer asks you to.

## Releases and versions

`pyproject.toml` holds the version (currently `2.0.0`), and `CHANGELOG.md` says versions follow
[semantic versioning](https://semver.org/). The tag `v2.0.0-msc` marks the thesis version.
Contributors do not change the version number or create tags; maintainers do.
The version appears in two places, which must always agree: `version` in `pyproject.toml` and `version`
and `date-released` in `CITATION.cff`. To make a release, a maintainer:

1. opens a pull request from a branch such as `chore/release-X.Y.Z` that renames the `## Unreleased`
   heading of `CHANGELOG.md` to `## X.Y.Z (YYYY-MM-DD)` (and adds a new, empty `## Unreleased` above it),
   and updates the version in `pyproject.toml` and `CITATION.cff`;
2. merges it after review;
3. creates an annotated tag `vX.Y.Z` on the merge commit of `main`
   (`git tag -a vX.Y.Z -m "iCorrVision 2D X.Y.Z"` and `git push origin vX.Y.Z`) and publishes a GitHub
   release from that tag, using the changelog entry as its notes.

The number follows semantic versioning: a bug fix raises the last number, a new backwards-compatible
feature the middle one, and a change that breaks the public interface (`import icorrvision` or the `icorr`
command) the first. The tag `v2.0.0-msc` is never moved or reused.

## Citing and contact

If you use iCorrVision in your work, please cite it as described in `CITATION.cff` (GitHub's
"Cite this repository" button) and give the version, so that it is not confused with version 1.

Questions are welcome in [GitHub issues](https://github.com/iCorrVision/iCorrVision-2D/issues).

To report a security problem privately, follow [SECURITY.md](SECURITY.md); do not open a public issue.
Conduct concerns go to the contact in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
