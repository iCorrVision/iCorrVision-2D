# iCorrVision 2D

A subset-based 2D digital image correlation (DIC) engine, usable as a Python package
(`icorrvision`), from the command line (`icorr`), or through a PySide6 graphical interface.

This is the software of the MSc thesis *iCorrVision 2.0: Development and Metrological Assessment of Modular Software for Two-Dimensional Digital Image Correlation* (João Pedro da Silva Duarte Récio, NOVA FCT, 2026), supervised by José Manuel Cardoso Xavier. The tag `v2.0.0-msc` is the version the thesis cites.

## Install

Python 3.12 or newer.

```sh
pip install .            # engine and command line only
pip install ".[gui]"     # with the graphical interface
```

Optional extras: `notebooks` (Jupyter and pandas, for the notebooks below) and
`acceleration` (numba and pyopencl, for `notebooks/acceleration_routes.ipynb` only).

A self-contained Windows executable of the graphical interface is built with PyInstaller by the
repository's GitHub Actions workflow on every push to `main`; it is available as the
`icorr-correlation-windows` artifact of the workflow run.

## Use

- **Graphical interface:** `python -m core.main`
- **Command line:** set `image_dir` in `run.toml` to a folder of TIFF frames, then
  `icorr run --config run.toml --output-dir results/`. `template.toml` lists every
  configuration option and its allowed values.
- **Python API:** `example/api_tour.ipynb` walks through the public API on synthetic images.

## Notebooks

`notebooks/` holds the validation and timing studies reported in the thesis. The DIC Challenge
images are not included; download them from the DIC Challenge and point the notebooks at them
through environment variables:

| Variable | Contents |
| --- | --- |
| `DIC_CHALLENGE` | 2D DIC Challenge 1.0 samples (Sample 3, Sample 12) |
| `DIC_CHALLENGE2` | 2D DIC Challenge 2.0 samples (Star 1, Star 5/6) |
| `DIC_RESULTS` | Folder for run outputs |

Runs are cached by configuration, so use an empty `DIC_RESULTS` folder to recompute everything.
`data/` holds the inputs that are part of this repository: the Sample 12 ROI mask and the frozen
participant results used as a regression check in the Star 5/6 notebook (see `data/README.md`).

Each notebook is executed from a fresh kernel with

```sh
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 <notebook>.ipynb
```

| Notebook | Purpose | Data | Run time | Thesis |
| --- | --- | --- | --- | --- |
| `example/api_tour` | Executable walkthrough of the public interface, with PySide6 made unimportable | synthetic | 2 min | Section 5.3 |
| `notebooks/sample03_interpolation_bias` | Interpolation bias of the three interpolants | Sample 3 | 5 min | Section 5.4 |
| `notebooks/sample12_vsg_study` | Virtual strain gauge and agreement between strain estimators | Sample 12 | 5 min | Section 5.5 |
| `notebooks/star1_shape_function_theory` | Spatial resolution against the Savitzky–Golay prediction | Star 1 | 1 min | Section 5.6 |
| `notebooks/star56_mei` | Metrological efficiency and placement among the Challenge participants | Stars 5, 6; participant results | 45 min | Section 5.7 |
| `notebooks/parallel_scaling` | Determinism, speed-up and memory with worker threads | synthetic | 11 min | Section 5.3.3 |
| `notebooks/process_scaling` | Frames distributed over processes (Linux only) | synthetic | 15 min | Section 5.3.3 |
| `notebooks/acceleration_routes` | Profile of the engine and benchmark of compiled and GPU forms of the refinement | synthetic | 1 min | Section 5.3.4 |

- `star56_mei` compares the participant MEI values of every execution with the frozen values in
  `data/star56/`, reports any value that changed by more than 1 %, and stops if the frozen file is
  missing.
- `process_scaling` forks the notebook's own process and therefore runs only on Linux.
- `acceleration_routes` runs its Numba and OpenCL variants only if those packages import and, for
  OpenCL, a GPU is found; otherwise it skips them and records why.
- `parallel_scaling` and `process_scaling` measure time and should run on an otherwise idle machine.

## Tested with

The results of the thesis were obtained with Python 3.12.13, NumPy 2.5.3, SciPy 1.18.1,
OpenCV 5.0.0, Numba 0.67.0 and PyOpenCL 2026.1.4 on NixOS (Linux). `pyproject.toml` gives lower
bounds only, so a new installation may use newer versions.

## Citation

See `CITATION.cff`, or use GitHub's "Cite this repository" button.

## Licence

GPL-3.0-or-later; see `LICENSE`.
