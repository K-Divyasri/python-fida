"""compare_dftnn.py -- exact-match proof: Python dft+nn vs FID-A dft+nn (no
nufft anywhere, no water removal, no zerofill, apodize fixed). Per-stage masked.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mrsi'))
import numpy as np, h5py
from op_CSIpostproc import op_CSIApodize, op_CSIapplymask
from viz._common import to_fyx

MET = r'F:\fida\divya\20260605_phantom_test\subject02\met\meas_MID00138_FID48082_Rosette_40x40_isoctr.dat'
REF = r'F:\fida\divya\20260605_phantom_test\subject02\mrs_ref\meas_MID00139_FID48083_Rosette_40x40_isoctr_w.dat'
KFILE = r'C:\Users\divya\Downloads\fida codes\fid_a\processingTools\MRSI\kFiles\Rosette_traj_40x40.txt'
GT = r'C:\Users\divya\Downloads\mrsi_pipeline\fida_stages_s02_dftnn'


def mfyx(k):
    with h5py.File(os.path.join(GT, k + '.mat'), 'r') as f:
        d = f['data'][()]
        a = (d['real'] + 1j * d['imag']) if (d.dtype.names and 'real' in d.dtype.names) else d
    return np.transpose(a, tuple(range(a.ndim))[::-1])


def cmp(py, fa, mask, lbl):
    ys, xs = np.where(mask); sc, sh, rw = [], [], []
    for y, x in zip(ys, xs):
        a = np.vdot(py[:, y, x], fa[:, y, x]) / np.vdot(py[:, y, x], py[:, y, x])
        if not np.isfinite(a): continue
        sc.append(abs(a)); sh.append(np.linalg.norm(py[:, y, x] * a - fa[:, y, x]) / (np.linalg.norm(fa[:, y, x]) + 1e-30))
        rw.append(np.linalg.norm(py[:, y, x] - fa[:, y, x]) / (np.linalg.norm(fa[:, y, x]) + 1e-30))
    print(f'{lbl:26s} scale {np.median(sc):.4f}  shape {np.median(sh)*100:6.3f}%  raw {np.median(rw)*100:6.3f}%')


def main():
    from run_rosette_pipeline import run_pipeline
    st = run_pipeline(metFile=MET, refFile=REF, kfile=KFILE, seq_type='rosette',
                      method='dft', dcf='nn', save_dir=None, water_removal='none',
                      do_phase=False, do_lcmodel=False, recenter=False,
                      phantom=True, zerofill=None)
    mask = st['s05_ccav_met']['mask']['brainmasks'].astype(bool)
    s06m = to_fyx(st['s06_spec_met']['data'], st['s06_spec_met']['dims'])
    s06w = to_fyx(st['s06_spec_ref']['data'], st['s06_spec_ref']['dims'])
    s09m = to_fyx(st['s09_b0_met']['data'], st['s09_b0_met']['dims'])
    s09w = to_fyx(st['s09_b0_ref']['data'], st['s09_b0_ref']['dims'])
    FOV = st['s06_spec_met']['fov']; VOX = st['s06_spec_met']['voxelSize']
    DIMS = st['s09_b0_met']['dims']

    def struct(data, m=None):
        s = dict(data=data.copy(), fov=FOV, voxelSize=VOX, dims=DIMS,
                 flags={'spatialft': 1}, sz=data.shape)
        if m is not None: s['mask'] = {'brainmasks': m}
        return s
    watf = op_CSIApodize(struct(s09w), 'gaussian', 20)['data']
    metf = op_CSIApodize(struct(op_CSIapplymask(struct(s09m, mask))['data']), 'gaussian', 20)['data']
    watf = to_fyx(watf, DIMS); metf = to_fyx(metf, DIMS)

    print('=== Python dft+nn  vs  FID-A dft+nn (deterministic, apodize fixed) ===')
    cmp(s06w, mfyx('s06_spec_ref'), mask, 'WATER s06 spectral')
    cmp(s09w, mfyx('s09_b0_ref'), mask, 'WATER s09 B0')
    cmp(watf, mfyx('s11_smooth_ref'), mask, 'WATER s11 apodize')
    cmp(s06m, mfyx('s06_spec_met'), mask, 'MET   s06 spectral')
    cmp(s09m, mfyx('s09_b0_met'), mask, 'MET   s09 B0')
    cmp(metf, mfyx('s11_smooth_met'), mask, 'MET   s11 apodize')


if __name__ == '__main__':
    main()
