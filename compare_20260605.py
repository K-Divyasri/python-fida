"""compare_20260605.py -- per-stage FID-A vs Python parity on the 20260605 met
dataset (real metabolite, 2 averages), 1% tolerance.

Compares the DETERMINISTIC chain (prep -> recon -> combine -> avg -> spectral ->
SSP -> B0). HSVD water removal + ACME phasing are Python-only additions and are
NOT part of this comparison (different algorithms than FID-A).

Needs:
  gt_20260605_stages\        (dump_20260605_stages.m)
  rosette_py_stages_20260605_cmp\   (run_pipeline skip_water_removal, do_phase=False)
"""
import os, json, warnings; warnings.filterwarnings('ignore')
import numpy as np, h5py

PY = r'C:\Users\divya\Downloads\mrsi_pipeline\rosette_py_stages_20260605_cmp'
GT = r'C:\Users\divya\Downloads\mrsi_pipeline\gt_20260605_stages'
TOL = 0.01   # 1%

PAIRS = [
    ('s02_prep_met', 's01_prep',    'load+prep'),
    ('s03_recon_met', 's03_recon',  'recon (dft+nn DCF)'),
    ('s04_combine_met', 's04_combine', 'coil combine'),
    ('s05_ccav_met', 's05_ccav',    'average'),
    ('s06_spec_met', 's06_spec',    'spectral FT'),
    ('s07_ssp_met', 's07_ssp',      'SSP lipid (SVD)'),
    ('s09_b0_met', 's09_b0',        'B0 correction'),
]


def load_py(key):
    z = np.load(os.path.join(PY, key + '.npz'), allow_pickle=True)
    meta = json.loads(str(z['meta']))
    return z['data'], {k: int(v) for k, v in meta['dims'].items() if v}


def load_gt(key):
    with h5py.File(os.path.join(GT, key + '.mat'), 'r') as f:
        d = f['data'][()]
        a = (d['real'] + 1j * d['imag']) if (d.dtype.names and 'real' in d.dtype.names) else d
        data = np.transpose(a, tuple(range(a.ndim)[::-1]))
        dims = {k: int(np.ravel(f['dims'][k][()])[0]) for k in f['dims']
                if int(np.ravel(f['dims'][k][()])[0]) > 0}
    return data, dims


def align(py, pyd, gd):
    names = [k for k, v in sorted(gd.items(), key=lambda kv: kv[1])]
    order = [pyd[n] - 1 for n in names if n in pyd]
    return np.transpose(py, order) if len(order) == py.ndim else py


def main():
    print(f'{"stage":13s} {"desc":22s} {"corr":>8s} {"max|d|/sc":>10s} {"p99|d|/sc":>10s}  {"1%?":4s}')
    print('-' * 78)
    npass = 0
    for pk, gk, desc in PAIRS:
        try:
            py, pyd = load_py(pk); gt, gd = load_gt(gk)
        except (FileNotFoundError, OSError):
            print(f'{gk:13s} {desc:22s}  (missing)'); continue
        pa = align(py, pyd, gd)
        if pa.shape != gt.shape:
            print(f'{gk:13s} {desc:22s}  SHAPE {pa.shape} vs {gt.shape}'); continue
        sc = np.abs(gt).max() or 1.0
        dif = np.abs(pa - gt) / sc
        corr = np.corrcoef(np.abs(pa).ravel(), np.abs(gt).ravel())[0, 1]
        mx, p99 = dif.max(), np.percentile(dif, 99)
        ok = p99 < TOL
        npass += ok
        print(f'{gk:13s} {desc:22s} {corr:8.4f} {mx:10.2e} {p99:10.2e}  {"PASS" if ok else "FAIL"}')
    print('-' * 78)
    print(f'{npass}/{len(PAIRS)} stages within 1% (p99 |diff|/scale)')


if __name__ == '__main__':
    main()
