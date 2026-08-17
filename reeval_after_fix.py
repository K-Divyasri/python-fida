"""reeval_after_fix.py -- post apodize-fix end-to-end vs FID-A, no recon rerun.

s06/s09 (recon..B0) are unchanged by the apodize fix; only s11 changes. Rebuild
met_final = fixed_apodize(applymask(s09_met)), wat_final = fixed_apodize(s09_wat)
from the saved trace2 s09 arrays, and compare to FID-A's 576-pt RAW (final).
"""
import os, sys, glob, re
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mrsi'))
import numpy as np
from op_CSIpostproc import op_CSIApodize, op_CSIapplymask

ARR = r'C:\Users\divya\Downloads\mrsi_pipeline\trace2_arrays.npz'
FALCM = r'F:\fida\divya\20260605_phantom_test\subject02\outputs\lcm_out'
FOV = dict(x=240.0, y=240.0, z=15.0); VOX = dict(x=6.0, y=6.0, z=15.0)
DIMS = {'t': 0, 'f': 1, 'y': 2, 'x': 3, 'coils': 0, 'averages': 0,
        'kx': 0, 'ky': 0, 'z': 0, 'kpts': 0, 'kshot': 0, 'extras': 0}


def parse_raw(p):
    c1, c2, st = [], [], False
    for ln in open(p, errors='ignore'):
        if '$END' in ln: st = True; continue
        if not st: continue
        q = ln.split()
        if len(q) == 2:
            try: c1.append(float(q[0])); c2.append(float(q[1]))
            except ValueError: pass
    return np.array(c1) - 1j * np.array(c2)


def load_fida():
    met, wat = {}, {}
    for f in glob.glob(os.path.join(FALCM, '*_ftSpec_smooth_lcm')):
        m = re.match(r'(\d+)x(\d+)_ftSpec_smooth_lcm$', os.path.basename(f))
        if not m: continue
        x, y = int(m.group(1)), int(m.group(2))
        met[(x, y)] = parse_raw(f)
        if os.path.exists(f + '_w'): wat[(x, y)] = parse_raw(f + '_w')
    return met, wat


def s2f(spec): return np.fft.fft(np.fft.fftshift(spec))


def struct(data, mask=None):
    s = dict(data=data.copy(), fov=FOV, voxelSize=VOX, dims=DIMS,
             flags={'spatialft': 1}, sz=data.shape)
    if mask is not None:
        s['mask'] = {'brainmasks': mask}
    return s


def cmp(pyfyx, faref, ny, nx, label):
    scs, shs, raws = [], [], []
    for (x, y), fref in faref.items():
        if not (0 <= y < ny and 0 <= x < nx): continue
        fpy = s2f(pyfyx[:, y, x])
        if fpy.shape != fref.shape or np.linalg.norm(fref) < 1e-9: continue
        a = np.vdot(fpy, fref) / np.vdot(fpy, fpy)
        scs.append(np.abs(a)); shs.append(np.linalg.norm(fpy * a - fref) / np.linalg.norm(fref))
        raws.append(np.linalg.norm(fpy - fref) / np.linalg.norm(fref))
    print(f'{label:26s} n={len(scs)}  scale {np.median(scs):.3f}  shape {np.median(shs)*100:6.2f}%  raw {np.median(raws)*100:6.2f}%')


def load_fresh_s11():
    """FID-A current-code fresh dump s11 (the run_rosette_40x40 truth)."""
    import h5py
    GT = r'C:\Users\divya\Downloads\mrsi_pipeline\fida_stages_s02_nufft'
    def mfyx(k):
        with h5py.File(os.path.join(GT, k + '.mat'), 'r') as f:
            d = f['data'][()]
            a = (d['real'] + 1j * d['imag']) if (d.dtype.names and 'real' in d.dtype.names) else d
        return np.transpose(a, tuple(range(a.ndim))[::-1])
    s11w = mfyx('s11_smooth_ref'); s11m = mfyx('s11_smooth_met')
    metd = {}; watd = {}
    ny, nx = s11w.shape[1:]
    for y in range(ny):
        for x in range(nx):
            metd[(x, y)] = s2f(s11m[:, y, x]); watd[(x, y)] = s2f(s11w[:, y, x])
    return metd, watd


def main():
    z = np.load(ARR)
    s09_met = z['s09_met']; s09_wat = z['s09_wat']; mask = z['mask'].astype(bool)
    nf, ny, nx = s09_met.shape
    fmet, fwat = load_fida()
    gmet, gwat = load_fresh_s11()

    wat_final = op_CSIApodize(struct(s09_wat), 'gaussian', 20)['data']
    met_masked = op_CSIapplymask(struct(s09_met, mask))['data']
    met_final = op_CSIApodize(struct(met_masked), 'gaussian', 20)['data']

    print('=== post apodize-fix, nufft/pipe_menon, NO zerofill ===')
    print('--- vs FID-A FRESH DUMP s11 (current run_rosette_40x40 = truth) ---')
    cmp(wat_final, gwat, ny, nx, 'WATER final vs fresh s11')
    cmp(met_final, gmet, ny, nx, 'MET   final vs fresh s11')
    print('--- vs FID-A production RAW (older code, stale reference) ---')
    cmp(wat_final, fwat, ny, nx, 'WATER final vs prod RAW')
    cmp(met_final, fmet, ny, nx, 'MET   final vs prod RAW')


if __name__ == '__main__':
    main()
