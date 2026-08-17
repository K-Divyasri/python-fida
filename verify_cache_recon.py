"""verify_cache_recon.py -- with the cached FID-A DCF, does Python nufft/pipe_menon
now match FID-A? Water channel (deterministic) through spectral FT vs FID-A dump.
"""
import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), 'mrsi'))
import numpy as np, h5py
from io_CSIload_twix import io_CSIload_twix_pair
from op_CSIRosettePrep import prep_noncartesian
from op_CSIRecon import op_CSIRecon
from op_CSICombineCoils1 import op_CSICombineCoils1
from op_CSIAverage import op_CSIAverage
from op_CSISegment_simple import op_CSISegment_simple
from op_CSIFourierTransform import op_CSIFourierTransform

MET = r'F:\fida\divya\20260605_phantom_test\subject02\met\meas_MID00138_FID48082_Rosette_40x40_isoctr.dat'
REF = r'F:\fida\divya\20260605_phantom_test\subject02\mrs_ref\meas_MID00139_FID48083_Rosette_40x40_isoctr_w.dat'
KFILE = r'C:\Users\divya\Downloads\fida codes\fid_a\processingTools\MRSI\kFiles\Rosette_traj_40x40.txt'
GT = r'C:\Users\divya\Downloads\mrsi_pipeline\fida_stages_s02_nufft'


def gt(key):
    with h5py.File(os.path.join(GT, key + '.mat'), 'r') as f:
        d = f['data'][()]
        a = (d['real'] + 1j * d['imag']) if (d.dtype.names and 'real' in d.dtype.names) else d
    return np.transpose(a, tuple(range(a.ndim))[::-1])


def to_fyx(s):
    dims = s['dims']; fax = (dims.get('f') or dims.get('t')) - 1
    return np.transpose(s['data'], (fax, dims['y'] - 1, dims['x'] - 1))


def cmp(py, g, mask, lbl):
    if py.shape != g.shape and py.shape == g.transpose(0, 2, 1).shape:
        g = g.transpose(0, 2, 1)
    ys, xs = np.where(mask); sc, sh = [], []
    for y, x in zip(ys, xs):
        a = np.vdot(py[:, y, x], g[:, y, x]) / np.vdot(py[:, y, x], py[:, y, x])
        if np.isfinite(a):
            sc.append(abs(a)); sh.append(np.linalg.norm(py[:, y, x] * a - g[:, y, x]) / (np.linalg.norm(g[:, y, x]) + 1e-30))
    print(f'{lbl:26s} scale {np.median(sc):.4f}  shape {np.median(sh)*100:6.3f}%')


def main():
    met, ref = io_CSIload_twix_pair(MET, REF, KFILE, 'rosette')
    wp = prep_noncartesian(ref, KFILE, 'rosette')
    ftw = op_CSIRecon(wp, KFILE, 'pipe_menon', 'nufft')       # <- cached DCF
    ccw = op_CSICombineCoils1(ftw)[0]
    ccav_w = op_CSIAverage(ccw)
    ccav_w = op_CSISegment_simple(ccav_w)
    mask = ccav_w['mask']['brainmasks']
    ftSpec_w = op_CSIFourierTransform(ccav_w, spatial=False, spectral=True)
    print('\n=== Python nufft/pipe_menon (CACHED DCF) vs FID-A, water, masked ===')
    cmp(to_fyx(ccav_w), gt('s05_ccav_ref'), mask, 'ccav (time)')
    cmp(to_fyx(ftSpec_w), gt('s06_spec_ref'), mask, 'spectral FT')


if __name__ == '__main__':
    main()
