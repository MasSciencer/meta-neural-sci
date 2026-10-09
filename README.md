# meta-neural-sci

Research notebooks for metasurface lens design using Meep electromagnetic simulations,
adjoint optimization, and PyTorch parameter updates. The workflow computes Airy and
plane-wave normalization, optimizes lens segments for multiple incidence angles and
wavelengths, evaluates PSF/MTF and focusing efficiency, and exports GDSII geometry.
Lengths are expressed in microns.

## Project layout

- `main/`: main notebooks, material CSV files, reference arrays, and saved parameters.
- `lib/`: geometry, sources, plotting, optimization helpers, and a local Meep adjoint module.
- `configs/conf_test.py`: configuration used by the main notebooks and library modules.
- `MEEPExamples/`: separate Meep examples.
- `utilites/`: additional analysis notebooks and scripts.

## Environment

Use Linux or Ubuntu on WSL with Conda available in the Linux shell. The existing
project environment uses Python 3.8.20, Meep 1.29.0, and PyTorch 2.4.1. These commands
specify the main dependency versions from that environment; this is not a full lockfile.

```bash
conda create -n meta-neural-sci --override-channels -c conda-forge \
  python=3.8.20 "pymeep=1.29.0=nompi_*" numpy=1.24.4 scipy=1.9.1 \
  autograd=1.6.2 gdspy=1.6.13 nlopt=2.7.1 pip
conda activate meta-neural-sci
python -m pip install torch==2.4.1 pandas==2.0.3 matplotlib==3.7.3 \
  trimesh==4.8.1 shapely==2.0.7 ipywidgets==8.1.5 \
  jupyterlab==4.2.5 ipykernel==6.29.5 nbformat==5.10.4
python -m ipykernel install --user --name meta-neural-sci \
  --display-name "Python (meta-neural-sci)"
```

Meep is installed as `pymeep` through Conda. The single-process (`nompi`) build is
sufficient for this workflow. The main notebooks use CPU tensors; a GPU is optional.
JAX is optional and needed only for `lib.adjoint.MeepJaxWrapper`, which the main
notebooks do not use.

## Run the main notebook

From the repository root in a Linux/WSL shell:

```bash
conda activate meta-neural-sci
cd main
export PYTHONPATH="$(pwd)/..${PYTHONPATH:+:$PYTHONPATH}"
python -c "import meep, torch; from configs import conf_test; from lib import core_finctions, numpy_functions, my_gui; print('Imports OK')"
python -m jupyterlab
```

Open the local URL printed by Jupyter, select **Python (meta-neural-sci)**, and open
`CC_adjoint_1d_base.ipynb`. Keep the kernel working directory set to `main/`: the
configuration reads `aSi_1-20.csv` relative to that directory, and the notebooks load
and save arrays and parameters there. `PYTHONPATH` makes `lib` and `configs` importable.

Run the cells in order:

1. **Project setup**: imports and objective/gradient functions.
2. **Normalization and comparison calculations**: initialize parameters and sources,
   then load the Airy and plane-wave reference arrays.
3. **Optimizer**: run the multi-angle local optimization loop. Start with
   `num_steps = 1` and `opt_zone_end = 1` for a shorter initial trial.
4. **TEST**: evaluate the current optimized parameters and plot PSF/MTF and efficiency.
5. **GDS or STL**: optionally export geometry after checking the parameter filenames.

The current `conf_test.py` uses a 6000 um aperture, 2000 um focal length, two incidence
angles, and three wavelengths from 3.6 to 4.4 um. The base notebook needs these files
in `main/`:

- `focusAiry_D6000.0_f2000.0_diaph2000_res5_nfreq3.npy`
- `baseAiry_D6000.0_f2000.0_diaph2000_res5_nfreq3.npy`
- `basePlane_D6000.0_f2000.0_diaph2000_res5_nfreq3.npy`
- `aSi_1-20.csv` (material columns: `wl`, `n`, `k`).

For a different configuration or missing reference arrays, set `filename_focus_IAiry`,
`filename_base_IAiry`, and `file_name_IBase` to `None` in the base-parameters cell to
recompute and save normalization data. Restart the kernel after changing
`configs/conf_test.py`, since several library functions capture configuration values
at import time.

`CC_adjoint_1d_origin_3f.ipynb` and `CC_adjoint_1d_7f.ipynb` are experimental variants.
They share the same configuration import: check the frequency count and reference
filenames before running them; the notebook filename does not select a configuration.

## Saved results

Optimization writes `global_a_params_<orders>.pt/.npy`,
`global_h_params_<orders>.pt/.npy`, and `loc_history_angle_<index>.pkl` in `main/`.
Plotting cells save PDF figures, and export cells save GDS files. Re-running these
cells can overwrite files with the same names, so copy results you want to keep.

The export section reloads saved `.pt` files. Update those filenames to match your
run (the default configuration has 536 orders), or skip the reload cell to export
the parameters currently in memory. Ensure radius arrays and parameter tensors
have matching lengths.

Imports and the initial setup/normalization cells were checked in the existing Conda
environment. Full optimization and a fresh dependency installation were not run as
part of the documentation check; optimization can take substantial CPU time and RAM.
