"""compare_matlab_vs_python.py -- step-by-step FID-A (MATLAB) vs Python.

Production config (nufft + pipe_menon). Compares the saved Python stages
(rosette_py_stages_nufft576, current fixed-apodize code) against the FID-A MATLAB
dump (fida_stages_s02_nufft) for each directly-comparable [t/f, y, x] stage,
masked. Metric per voxel: best complex scale |a| (a=<fa,py>/<py,py>), shape diff
after that scale, and magnitude correlation.
"""
import os, json, numpy as np, h5py

PY = r'C:\Users\divya\Downloads\mrsi_pipeline\rosette_py_stages_nufft576'
GT = r'C:\Users\divya\Downloads\mrsi_pipeline\fida_stages_s02_nufft'


def load_py(key):
    z = np.load(os.path.join(PY, key + '.npz'), allow_pickle=True)
    m = json.loads(str(z['meta'])); dims = {k: int(v) for k, v in m['dims'].items() if v}
    d = z['data']; fax = (dims.get('f') or dims.get('t')) - 1
    return np.transpose(d, (fax, dims['y'] - 1, dims['x'] - 1))          # [f/t, y, x]


def load_gt(key):
    with h5py.File(os.path.join(GT, key + '.mat'), 'r') as f:
        d = f['data'][()]
        a = (d['real'] + 1j * d['imag']) if (d.dtype.names and 'real' in d.dtype.names) else d
    return np.transpose(a, tuple(range(a.ndim))[::-1])                    # [f/t, y, x]


def mask_gt():
    with h5py.File(os.path.join(GT, 'mask.mat'), 'r') as f:
        return np.transpose(f['mask'][()], (1, 0)).astype(bool)


def stat(py, gt, mask):
    if py.shape != gt.shape:
        if py.shape == gt.transpose(0, 2, 1).shape:
            gt = gt.transpose(0, 2, 1)
        else:
            return None
    ys, xs = np.where(mask); sc, sh = [], []
    for y, x in zip(ys, xs):
        a = np.vdot(py[:, y, x], gt[:, y, x]) / np.vdot(py[:, y, x], py[:, y, x])
        if not np.isfinite(a):
            continue
        sc.append(abs(a)); sh.append(np.linalg.norm(py[:, y, x] * a - gt[:, y, x]) / (np.linalg.norm(gt[:, y, x]) + 1e-30))
    corr = np.corrcoef(np.abs(py[:, mask]).ravel(), np.abs(gt[:, mask]).ravel())[0, 1]
    return np.median(sc), np.median(sh) * 100, corr


PAIRS = [
    ('5  ccav (time [t,y,x])',  's05_ccav_met',   's05_ccav_met'),
    ('6  spectral FT',          's06_spec_met',   's06_spec_met'),
    ('9  B0 (+L2 water rm*)',   's09_b0_met',     's09_b0_met'),
    ('11 apodize (final)',      's11_smooth_met', 's11_smooth_met'),
    ('-- water channel (deterministic) --', None, None),
    ('6  spectral FT (H2O)',    's06_spec_ref',   's06_spec_ref'),
    ('9  B0 (H2O)',             's09_b0_ref',     's09_b0_ref'),
    ('11 apodize (H2O)',        's11_smooth_ref', 's11_smooth_ref'),
]


def main():
    mask = mask_gt()
    print('=== FID-A (MATLAB) vs Python | nufft + pipe_menon | masked (%d vox) ===' % int(mask.sum()))
    print(f'{"stage":30s} {"scale py/fa":>11s} {"shape diff":>11s} {"|corr|":>8s}')
    print('-' * 64)
    for label, pk, gk in PAIRS:
        if pk is None:
            print(label); continue
        try:
            r = stat(load_py(pk), load_gt(gk), mask)
        except Exception as e:
            print(f'{label:30s}  (missing: {e})'); continue
        if r is None:
            print(f'{label:30s}  shape mismatch'); continue
        sc, sh, corr = r
        print(f'{label:30s} {sc:11.3f} {sh:10.2f}% {corr:8.4f}')
    print('\n* metabolite B0/apodize include the STOCHASTIC L2 water-removal basis')
    print('  (random in both FID-A and Python) -> a few % is expected there.')
    print('  Water channel has NO water removal -> fully deterministic.')


if __name__ == '__main__':
    main()
