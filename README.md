# Machine Learning for Adaptive Optics — OAO School 2026

Hands-on tutorials on machine learning applied to adaptive optics (AO), developed for the **OAO School 2026** (Asiago). The tutorials are Jupyter notebooks built with [PyTorch](https://pytorch.org/) and trained on simulated data of the **EKARUS** single-conjugate AO system with a pyramid wavefront sensor (PWFS) at the 1.82 m Copernico telescope.

Each notebook follows the same workflow, so that the second tutorial reuses what was learned in the first:

1. physics of the problem and inspection of the data,
2. train / validation / test split,
3. normalisation computed on the training set only,
4. PyTorch `Dataset` and `DataLoader`,
5. network architecture,
6. training and validation, with best-epoch checkpointing,
7. evaluation on the test set against a simple baseline,
8. residual analysis,
9. saving the model and using it for inference,
10. exercises.

## Tutorials

| # | Notebook | Task | Network | Data |
|---|---|---|---|---|
| 1 | [`dataset_and_jupyter_1st_tutorial/ml_tutorial_npz.ipynb`](dataset_and_jupyter_1st_tutorial/ml_tutorial_npz.ipynb) | Atmospheric parameters (seeing, $L_0$, wind speeds) from AO performance metrics (SR, EE, FWHM) | Dense network (MLP) | `sweep_dataset.npz`, 1675 SPECULA simulations (included) |
| 2 | [`pyramid_wfs_to_zernike_cnn_tutorial_dictionary.ipynb`](pyramid_wfs_to_zernike_cnn_tutorial_dictionary.ipynb) | Zernike coefficients of the wavefront from pyramid WFS images | Convolutional network (CNN) | `dataset_2.pt`, PWFS images and Zernike coefficients |

### Tutorial 1 — Retrieving atmospheric parameters from AO telemetry with a neural network

The AO system itself "sees" the turbulence it corrects. This tutorial trains a neural network as an **inverse model**: from three numbers that every AO system measures on the long-exposure PSF — Strehl ratio, encircled energy and FWHM — it predicts six atmospheric parameters: the seeing, the outer scale $L_0$ and the wind speeds of four turbulent layers.

The training set is a parameter sweep of 1675 closed-loop end-to-end simulations of EKARUS made with [SPECULA](https://github.com/ArcetriAdaptiveOptics/SPECULA), with random seeing, $L_0$, wind speeds, zenith angle and guide-star magnitude. The tutorial covers data exploration (correlations between inputs and targets, nuisance parameters such as the zenith angle), log and z-score pre-processing, data leakage, a dense network for multi-output regression, learning curves and over-fitting, evaluation with MAE and $R^2$ against a predict-the-mean baseline, and inference with a saved checkpoint. A pretrained model is provided, so the notebook can be run without training.

Main lesson: the network retrieves the seeing well, but not $L_0$ or the wind speeds, because the three inputs do not contain that information. *Explore your data before training, and always compare with a baseline.*

### Tutorial 2 — Retrieving Zernike wavefront modes from pyramid WFS images with a CNN

A pyramid wavefront sensor produces four images of the telescope pupil. This tutorial trains a **convolutional neural network** to reconstruct the wavefront directly from these images, as the first (up to) 50 Zernike coefficients.

It extends the workflow of Tutorial 1 from scalar inputs to images: inspection of the four PWFS channels and of the distribution of the Zernike coefficients, channel-wise normalisation, why a CNN suits image inputs better than a dense network, a compact CNN architecture (convolution, pooling, batch normalisation, dropout), and per-mode diagnostics: predicted versus true coefficients, $R^2$ as a function of the mode number, and residuals. The exercises explore network depth, a dense-network baseline, noise robustness, generalisation outside the training range and group-aware data splitting.

Main lesson: a single global loss hides the details. Per-mode diagnostics show which spatial frequencies the PWFS + CNN can actually retrieve.

## Getting started

You need Python (≥ 3.10) and Jupyter:

```bash
git clone https://github.com/alphaCassi/ML_tutorial_OAO2026.git
cd ML_tutorial_OAO2026
pip install jupyterlab
jupyter lab
```

Open the notebooks in the order of the table above. The first notebook checks the required libraries (numpy, pandas, matplotlib, scikit-learn, PyTorch) and installs any missing ones.

Both tutorials run on a CPU; a GPU only makes training faster. For GPU training, install the PyTorch build that matches your CUDA version (see [pytorch.org](https://pytorch.org/get-started/locally/)).

Optionally, you can work in a separate environment (e.g. `venv`, `conda`, `uv` or a Docker container) to keep the tutorial libraries apart from your other projects.

## Data

**Tutorial 1** uses `dataset_and_jupyter_1st_tutorial/sweep_dataset.npz` (about 11 MB, included). It contains, for each of the 1675 simulations, the long-exposure SR, EE and FWHM (inputs), the six atmospheric parameters (targets), the zenith angle and guide-star magnitude, the RMS of the 400 DM modal commands, and the frame-by-frame SR, EE and FWHM time series (500 steps). It was built from the raw SPECULA outputs with:

```bash
python dataset_and_jupyter_1st_tutorial/build_sweep_npz.py \
    --root dataset_specula_ekarus/output/parameter_sweep \
    --out  dataset_and_jupyter_1st_tutorial/sweep_dataset.npz \
    --ml-workflow ml_workflow
```

The SPECULA configuration of the simulated system is in `dataset_specula_ekarus/ekarus_main.yml`, and the script that runs the parameter sweep is `dataset_specula_ekarus/create_dataset.py`.

**Tutorial 2** expects a file `dataset_2.pt` next to the notebook, containing a dictionary with

```python
dataset["wfs"]    # (N, 4, H, W)  pyramid WFS images
dataset["comm"]   # (N, N_modes)  Zernike coefficients
```

The loader also accepts `.npy` and `.pkl` files with the same dictionary (set `DATASET_PATH` in the setup cell). The dataset is generated with `tutorials/create_comm_wfs_dataset.py`.

## Repository structure

```
ML_tutorial_OAO2026/
├── dataset_and_jupyter_1st_tutorial/
│   ├── ml_tutorial_npz.ipynb                     ← Tutorial 1
│   ├── sweep_dataset.npz                         ← Tutorial 1 dataset
│   └── build_sweep_npz.py                        ← packs SPECULA outputs into the .npz
├── pyramid_wfs_to_zernike_cnn_tutorial_dictionary.ipynb   ← Tutorial 2
├── pretrained_models/                            ← pretrained network for Tutorial 1
├── dataset_specula_ekarus/                       ← SPECULA configuration, calibration and sweep script
├── ml_workflow/                                  ← script version of Tutorial 1 (Hydra configs, train/test)
├── tutorials/                                    ← development versions and dataset generation for Tutorial 2
└── LICENSE
```

`ml_workflow/` contains the same atmospheric-retrieval problem as Tutorial 1, organised as a training pipeline configured with [Hydra](https://hydra.cc/) (`python main.py` from inside `ml_workflow/`). It is used in the last exercise of Tutorial 1.

## License

This project is released under the GNU General Public License v3.0. See [LICENSE](LICENSE).
