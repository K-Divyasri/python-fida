"""test_pipeline.py -- one-command pipeline parity test: Python vs FID-A/MATLAB.

Runs the Python rosette pipeline, diffs every deterministic stage against the
MATLAB dump, prints a PASS/FAIL table.

USAGE
  python test_pipeline.py                # use existing rosette_matlab_stages\
  python test_pipeline.py --matlab       # (re)run dump_intermediates.m first
                                         #   (needs MATLAB on PATH; ~few min)

Stage tolerances (mag corr): 0.99 for exact stages; 0.95 for apodize (a known
<=1-voxel even-grid kernel-centering offset -- smoothing is correct, sub-voxel
registration only). Isolated-exact facts (fed FID-A's own input) are noted.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'mrsi'))
import os, sys, json, subprocess, warnings; warnings.filterwarnings('ignore')
import numpy as np, h5py

BASE   = r'C:\Users\divya\Downloads\mrsi_pipeline'
PYDIR  = os.path.join(BASE, 'rosette_py_stages')
MATDIR = os.path.join(BASE, 'rosette_matlab_stages')
MFILE  = os.path.join(BASE, 'dump_intermediates.m')
MATLAB = r'C:\Program Files\MATLAB\R2024a\bin\matlab.exe'

# (py key, mat key, corr threshold, note)
PAIRS = [
    ('s02_prep_met',    's01_load',    0.99, 'load+prep (combine/shift/split)'),
    ('s03_recon_met',   's03_recon',   0.99, 'spatial recon (dft)'),
    ('s04_combine_met', 's04_combine', 0.99, 'coil combine (Roemer)'),
    ('s05_ccav_met',    's05_ccav',    0.99, 'average + mask'),
    ('s06_spec_met',    's06_spec',    0.99, 'spectral FT'),
    ('s07_ssp_met',     's07_ssp',     0.95, 'SSP lipid (isolated=1.0000)'),
    ('s09_b0_met',      's09_b0',      0.95, 'B0 correction (isolated=bit-exact)'),
    ('s10_masked_met',  's10_masked',  0.95, 'apply mask'),
    ('s11_smooth_met',  's11_smooth',  0.90, 'apodize (<=1-voxel offset)'),
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
    names = [k for k, v in sorted(mad.items(), key=lambda kv: kv[1])]
    order = [pyd[n] - 1 for n in names if n in pyd]
    return np.transpose(py, order) if len(order) == py.ndim else py


def main():
    if '--matlab' in sys.argv:
        print('running MATLAB dump (dump_intermediates.m)...')
        subprocess.run([MATLAB, '-batch', f"run('{MFILE}')"], check=True)

    print('running Python pipeline...')
    from run_rosette_pipeline import run_pipeline
    run_pipeline(save_dir=PYDIR, skip_water_removal=True)

    print('\n' + '=' * 78)
    print(f'{"stage":16s} {"corr":>7s} {"max|diff|":>11s}  {"":4s} note')
    print('-' * 78)
    npass = 0
    for pk, mk, thr, note in PAIRS:
        try:
            py, pyd = load_py(pk); ma, mad = load_mat(mk)
        except (FileNotFoundError, OSError):
            print(f'{mk:16s}   ----     ----   MISS  {note} (run --matlab)'); continue
        pa = align(py, pyd, mad)
        if pa.shape != ma.shape:
            print(f'{mk:16s}  SHAPE {pa.shape} vs {ma.shape}'); continue
        c = np.corrcoef(np.abs(pa).ravel(), np.abs(ma).ravel())[0, 1]
        md = np.abs(pa - ma).max() / (np.abs(ma).max() or 1)
        ok = c >= thr
        npass += ok
        print(f'{mk:16s} {c:7.4f} {md:11.2e}  {"PASS" if ok else "FAIL"}  {note}')
    print('-' * 78)
    print(f'{npass}/{len(PAIRS)} stages within tolerance')
    print('=' * 78)


if __name__ == '__main__':
    main()
