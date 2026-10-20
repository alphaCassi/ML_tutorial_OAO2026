#!/usr/bin/env python3
"""
Pack the parameter_sweep/ folders into a single compressed .npz file.

Run it from the project root, for example:

    python build_sweep_npz.py \
        --root dataset_specula_ekarus/output_2/parameter_sweep \
        --out  dataset_specula_ekarus/output_2/sweep_dataset.npz \
        --ml-workflow ml_workflow

The inputs (SR, EE, FWHM) and the atmospheric targets are read with the same
functions the notebook uses (dataset.dataset_utils), so the values are identical.
Only the arrays the tutorial needs are stored: psfint.fits and params.yml are
not copied, and comm.fits is reduced to the RMS of each mode.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits

N_STEPS = 500     # frames per simulation (loop time series)
N_MODES = 400     # DM modes


def scalar(loader, root):
    """The *_int.fits files contain a single number."""
    value = np.asarray(loader(root), dtype=np.float32).squeeze()
    if value.size != 1:
        raise ValueError(f'expected one value in {root}, got shape {value.shape}')
    return float(value)


def load_series(path):
    """Frame-by-frame time series, 500 values."""
    arr = np.asarray(fits.getdata(path), dtype=np.float32).squeeze()
    if arr.shape != (N_STEPS,):
        raise ValueError(f'{path.name}: expected shape ({N_STEPS},), got {arr.shape}')
    return arr


def comm_rms(path):
    """comm.fits (500 steps x 400 modes) -> RMS over time for each mode: shape (400,)."""
    comm = np.asarray(fits.getdata(path), dtype=np.float64).squeeze()
    if comm.shape == (N_MODES, N_STEPS):      # saved transposed
        comm = comm.T
    if comm.shape != (N_STEPS, N_MODES):
        raise ValueError(f'comm.fits: expected ({N_STEPS}, {N_MODES}), got {comm.shape}')
    return np.sqrt(np.mean(comm ** 2, axis=0)).astype(np.float32)


def load_nuisance(root):
    """Zenith angle and NGS magnitude: stored for the exercises, not given to the network."""
    with open(root / 'params.yml') as f:
        d = yaml.safe_load(f)
    return float(d['main']['zenithAngleInDeg']), float(d['source_ngs']['magnitude'])


def main():
    parser = argparse.ArgumentParser(description='Pack parameter_sweep into one .npz file')
    parser.add_argument('--root', type=Path, required=True,
                        help='folder with parametri_simulazioni.csv and sim_XXXXX/ subfolders')
    parser.add_argument('--out', type=Path, required=True, help='output .npz file')
    parser.add_argument('--ml-workflow', type=Path, default=Path('ml_workflow'),
                        help='folder of the ml_workflow package (contains dataset/dataset_utils.py)')
    args = parser.parse_args()

    # Same loaders as the notebook
    sys.path.insert(0, str(args.ml_workflow.resolve()))
    from dataset.dataset_utils import load_atm_params, load_psfee_int, load_psffwhm_int, load_sr_int

    sample_paths = [p for p in sorted(args.root.glob('sim_*/*')) if p.is_dir()]
    print(f'Found {len(sample_paths)} simulation folders in {args.root}')

    ids, X, y, zenith, ngs_mag, comm, sr_ts, psfee_ts, psffwhm_ts = ([] for _ in range(9))
    skipped = []

    for i, p in enumerate(sample_paths):
        name = p.parent.name                      # sim_XXXXX
        try:
            x = [scalar(load_sr_int, p), scalar(load_psfee_int, p), scalar(load_psffwhm_int, p)]
            t = np.asarray(load_atm_params(p), dtype=np.float32)
            if t.shape != (6,):
                raise ValueError(f'expected 6 atmospheric targets, got {t.shape}')
            z, m = load_nuisance(p)
            c = comm_rms(p / 'comm.fits')
            s1 = load_series(p / 'sr.fits')
            s2 = load_series(p / 'psfee.fits')
            s3 = load_series(p / 'psffwhm.fits')
        except Exception as e:
            skipped.append((name, repr(e)))
            print(f'  SKIPPED {name}: {e}')
            continue

        ids.append(name)
        X.append(x); y.append(t); zenith.append(z); ngs_mag.append(m)
        comm.append(c); sr_ts.append(s1); psfee_ts.append(s2); psffwhm_ts.append(s3)

        if (i + 1) % 100 == 0:
            print(f'  processed {i + 1}/{len(sample_paths)}')

    if len(set(ids)) != len(ids):
        raise RuntimeError('duplicate simulation names: more than one timestamp folder per sim_XXXXX')
    if not ids:
        raise RuntimeError('no valid simulation found')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        ids=np.array(ids),                              # (N,)      simulation names
        X_long=np.array(X, dtype=np.float32),           # (N, 3)    SR, EE (4 λ/D), FWHM [λ/D]
        y=np.array(y, dtype=np.float32),                # (N, 6)    seeing, L0, v1..v4
        zenith=np.array(zenith, dtype=np.float32),      # (N,)      nuisance
        ngs_mag=np.array(ngs_mag, dtype=np.float32),    # (N,)      nuisance
        comm_rms=np.array(comm, dtype=np.float32),      # (N, 400)  RMS of DM commands per mode
        sr_ts=np.array(sr_ts, dtype=np.float32),        # (N, 500)  frame-by-frame SR
        psfee_ts=np.array(psfee_ts, dtype=np.float32),  # (N, 500)  frame-by-frame EE
        psffwhm_ts=np.array(psffwhm_ts, dtype=np.float32),  # (N, 500) frame-by-frame FWHM
        input_names=np.array(['SR', 'EE (4 λ/D)', 'FWHM [λ/D]']),
        target_names=np.array(['Seeing', 'L0', 'v1', 'v2', 'v3', 'v4']),
        target_units=np.array(['arcsec', 'm', 'm/s', 'm/s', 'm/s', 'm/s']),
    )

    size_mb = args.out.stat().st_size / 1e6
    print(f'\nSaved {args.out} ({size_mb:.2f} MB)')
    print(f'Valid simulations: {len(ids)}   skipped: {len(skipped)}')


if __name__ == '__main__':
    main()
