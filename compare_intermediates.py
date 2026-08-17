"""compare_intermediates.py -- diff every pipeline stage: Python vs FID-A/MATLAB.

Loads the Python dump (run_rosette_pipeline.py, save_dir=rosette_py_stages) and the
MATLAB dump (dump_intermediates.m -> rosette_matlab_stages), pairs the stages,
aligns dim order by name, and prints corr + max|diff|/scale per stage.

Run AFTER:
  python run_rosette_pipeline.py            # (with skip_water_removal=True; see __main__)
  matlab -batch "run('dump_intermediates.m')"
"""
import os, json, warnings; warnings.filterwarnings('ignore')
import numpy as np, h5py

PYDIR  = r'C:\Users\divya\Downloads\mrsi_pipeline\rosette_py_stages'
MATDIR = r'C:\Users\divya\Downloads\mrsi_pipeline\rosette_matlab_stages'

# (python .npz key, matlab .mat key)
PAIRS = [
    ('s02_prep_met', 's01_load'),      # FID-A io_CSIload_twix_pair does prep internally
    ('s03_recon_met', 's03_recon'),
    ('s04_combine_met', 's04_combine'),
    ('s05_ccav_met', 's05_ccav'),
    ('s06_spec_met', 's06_spec'),
    ('s07_ssp_met', 's07_ssp'),
    ('s09_b0_met', 's09_b0'),
    ('s10_masked_met', 's10_masked'),
    ('s11_smooth_met', 's11_smooth'),
]


def load_py(key):
    z = np.load(os.path.join(PYDIR, key + '.npz'), allow_pickle=True)
    meta = json.loads(str(z['meta']))
    return z['data'], {k: int(v) for k, v in meta['dims'].items() if v}


def load_mat(key):
    with h5py.File(os.path.join(MATDIR, key + '.mat'), 'r') as f:
        d = f['data'][()]
        a = (d['real'] + 1j * d['imag']) if (d.dtype.names and 'real' in d.dtype.names) else d
        data = np.transpose(a, tuple(range(a.ndim)[::-1]))
        dims = {k: int(np.ravel(f['dims'][k][()])[0]) for k in f['dims']
                if int(np.ravel(f['dims'][k][()])[0]) > 0}
    return data, dims


def align(py, pyd, mad):
    """permute py array to MATLAB dim-name order."""
    names = [k for k, v in sorted(mad.items(), key=lambda kv: kv[1])]
    order = [pyd[n] - 1 for n in names if n in pyd]
    return np.transpose(py, order) if len(order) == py.ndim else py


def main():
    print(f'{"stage":16s} {"shapes (py / mat)":30s} {"mag corr":>9s} {"max|diff|/scale":>15s}')
    print('-' * 74)
    for pk, mk in PAIRS:
        try:
            py, pyd = load_py(pk); ma, mad = load_mat(mk)
        except (FileNotFoundError, OSError):
            print(f'{mk:16s} (missing)'); continue
        pa = align(py, pyd, mad)
        if pa.shape != ma.shape:
            print(f'{mk:16s} SHAPE {pa.shape} vs {ma.shape}'); continue
        c = np.corrcoef(np.abs(pa).ravel(), np.abs(ma).ravel())[0, 1]
        md = np.abs(pa - ma).max() / (np.abs(ma).max() or 1)
        print(f'{mk:16s} {str(pa.shape):30s} {c:9.4f} {md:15.2e}')


if __name__ == '__main__':
    main()
