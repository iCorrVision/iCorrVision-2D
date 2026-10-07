# iCorrVision 2D (version 2)

A subset-based 2D digital image correlation (DIC) engine, usable as a Python package
(`icorrvision`), from the command line (`icorr`), or through a PySide6 graphical interface.

> [!NOTE]
> This is **version 2** of iCorrVision 2D, released in 2026. It is a reimplementation that
> shares no code with **iCorrVision-2D version 1** (de Deus Filho, da Silva Nunes and Xavier,
> *SoftwareX*, 2022), which remains available unchanged. See
> [Relation to iCorrVision version 1](#relation-to-icorrvision-version-1).

This is the software of the MSc thesis *iCorrVision 2.0: Development and Metrological Assessment of Modular Software for Two-Dimensional Digital Image Correlation* (João Pedro da Silva Duarte Récio, NOVA FCT, 2026), supervised by José Manuel Cardoso Xavier. The tag `v2.0.0-msc` is the version the thesis cites.

## Install

Python 3.12 or newer.

```sh
pip install .            # engine and command line only
pip install ".[gui]"     # with the graphical interface
```

Optional extras: `notebooks` (Jupyter and pandas, for the notebooks below) and
`acceleration` (numba and pyopencl, for `benchmarks/acceleration_routes.ipynb` only).

A self-contained Windows executable of the graphical interface is built with PyInstaller by the
repository's GitHub Actions workflow on every push and pull request to `main`; it is available as the
`icorr-correlation-windows` artifact of the workflow run.

## Use

- **Graphical interface:** `python -m icorrvision.gui.main`
- **Command line:** set `image_dir` in `examples/run.toml` to a folder of TIFF frames, then
  `icorr run --config examples/run.toml --output-dir results/`. `examples/template.toml` lists
  every configuration option and its allowed values.
- **Python API:** `examples/api_tour.ipynb` walks through the public API on synthetic images.

## Notebooks

`validation/notebooks/` holds the validation studies and `benchmarks/` the timing studies reported
in the thesis. The notebooks import the installed package, so install it first, in editable mode
while developing (`pip install -e ".[notebooks]"`; the devenv shell does this itself). They also
import `validation/` and `examples/synthetic.py` from the repository, which they locate from
their own folder. The DIC Challenge images are not included; download them from the DIC
Challenge and point the notebooks at them through environment variables:

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
| `examples/api_tour` | Executable walkthrough of the public interface, with PySide6 made unimportable | synthetic | 2 min | Section 5.3 |
| `validation/notebooks/sample03_interpolation_bias` | Interpolation bias of the three interpolants | Sample 3 | 5 min | Section 5.4 |
| `validation/notebooks/sample12_vsg_study` | Virtual strain gauge and agreement between strain estimators | Sample 12 | 5 min | Section 5.5 |
| `validation/notebooks/star1_shape_function_theory` | Spatial resolution against the Savitzky–Golay prediction | Star 1 | 1 min | Section 5.6 |
| `validation/notebooks/star56_mei` | Metrological efficiency and placement among the Challenge participants | Stars 5, 6; participant results | 45 min | Section 5.7 |
| `benchmarks/parallel_scaling` | Determinism, speed-up and memory with worker threads | synthetic | 11 min | Section 5.3.3 |
| `benchmarks/process_scaling` | Frames distributed over processes (Linux only) | synthetic | 15 min | Section 5.3.3 |
| `benchmarks/acceleration_routes` | Profile of the engine and benchmark of compiled and GPU forms of the refinement | synthetic | 1 min | Section 5.3.4 |

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

## Relation to iCorrVision version 1

iCorrVision was first released in 2022 as two Tkinter applications, iCorrVision-2D and
iCorrVision-3D, developed in the doctoral work of J. C. A. de Deus Filho and published as
companion articles in *SoftwareX*. Version 2 is a new implementation. It shares no code with
version 1, separates the correlation engine from the interface, and can be used as a Python
package, from the command line or through a PySide6 interface. In version 2, image acquisition
is a separate application, iCorrVision Grabber. Version 1 has not changed since June 2022 and
remains available.

| Version | Application | Toolkit | Repository |
| --- | --- | --- | --- |
| 1 (2022) | iCorrVision-2D | Tkinter | [jcadf/iCorrVision_2D](https://github.com/jcadf/iCorrVision_2D); journal copy [ElsevierSoftwareX/SOFTX-D-22-00075](https://github.com/ElsevierSoftwareX/SOFTX-D-22-00075) |
| 1 (2022) | iCorrVision-3D | Tkinter | [jcadf/iCorrVision_3D](https://github.com/jcadf/iCorrVision_3D); journal copy [ElsevierSoftwareX/SOFTX-D-22-00076](https://github.com/ElsevierSoftwareX/SOFTX-D-22-00076) |
| 2 (2026) | iCorrVision 2D (this repository) | PySide6 | [iCorrVision/iCorrVision-2D](https://github.com/iCorrVision/iCorrVision-2D) |
| 2 (2026) | iCorrVision Grabber | PySide6 | [iCorrVision/iCorrVision-Grabber](https://github.com/iCorrVision/iCorrVision-Grabber) |
| 2, planned | iCorrVision 3D | | |

## Citation

See `CITATION.cff`, or use GitHub's "Cite this repository" button. Please give the version
(2.0.0), so that the citation is not mistaken for iCorrVision-2D version 1.

Version 1 is described in:

- J. C. A. de Deus Filho, L. C. da Silva Nunes, J. M. C. Xavier, "iCorrVision-2D: An integrated
  python-based open-source Digital Image Correlation software for in-plane measurements (Part 1)",
  *SoftwareX* 19 (2022) 101131. <https://doi.org/10.1016/j.softx.2022.101131>
- J. C. A. de Deus Filho, L. C. da Silva Nunes, J. M. C. Xavier, "iCorrVision-3D: An integrated
  python-based open-source Digital Image Correlation Software for in-plane and out-of-plane
  measurements (Part 2)", *SoftwareX* 19 (2022) 101132.
  <https://doi.org/10.1016/j.softx.2022.101132>

## Licence

GPL-3.0-or-later; see `LICENSE`.
