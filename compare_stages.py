"""compare_stages.py -- stage-by-stage Python(config A) vs FID-A dump.

WATER channel is fully deterministic (no water removal) so it localises the
scale + shape deviation to a single stage:
  s06 spectral FT  ->  s09 B0  ->  s11 apodize(smooth)
MET s06 is also deterministic (pre water-removal); met s09/s11 involve the
stochastic lipid basis so are expected to differ and are shown for context.

Per stage: best complex scale a=<fref,fpy>/<fpy,fpy>, |a| (py/fida amplitude),
shape diff after removing a, and raw diff. Median over signal voxels.
"""
import os, numpy as np, h5py

ARR = r'C:\Users\divya\Downloads\mrsi_pipeline\trace2_arrays.npz'
GT = r'C:\Users\divya\Downloads\mrsi_pipeline\fida_stages_s02_nufft'


def load_mat_fyx(key):
    """FID-A dumped stage -> data as [f,y,x] (MATLAB [f,y,x] -> h5py reversed)."""
    with h5py.File(os.path.join(GT, key + '.mat'), 'r') as f:
        d = f['data'][()]
        a = (d['real'] + 1j * d['imag']) if (d.dtype.names and 'real' in d.dtype.names) else d
    return np.transpose(a, tuple(range(a.ndim))[::-1])   # -> [f,y,x]


def stage_metrics(pyarr, faarr, mask):
    nf, ny, nx = pyarr.shape
    scales, shapes, raws = [], [], []
    ys, xs = np.where(mask)
    for y, x in zip(ys, xs):
        fpy = pyarr[:, y, x]; fref = faarr[:, y, x]
        if np.linalg.norm(fref) < 1e-9:
            continue
        raws.append(np.linalg.norm(fpy - fref) / np.linalg.norm(fref))
        a = np.vdot(fpy, fref) / np.vdot(fpy, fpy)
        scales.append(np.abs(a))
        shapes.append(np.linalg.norm(fpy * a - fref) / np.linalg.norm(fref))
    return (np.median(scales), np.median(shapes), np.median(raws), len(raws))


def main():
    z = np.load(ARR)
    mask = z['mask'].astype(bool)
    print(f'signal voxels (mask): {int(mask.sum())}')
    print(f'{"stage":22s} {"scale py/fida":>13s} {"shape(after scale)":>19s} {"raw diff":>10s}')
    print('-' * 68)
    pairs = [
        ('WATER s06 spectral', z['s06_wat'], 's06_spec_ref'),
        ('WATER s09 B0',       z['s09_wat'], 's09_b0_ref'),
        ('WATER s11 apodize',  z['wat_final'], 's11_smooth_ref'),
        ('MET   s06 spectral', z['s06_met'], 's06_spec_met'),
        ('MET   s09 B0(+rmw)', z['s09_met'], 's09_b0_met'),
        ('MET   s11 smooth',   z['met_final'], 's11_smooth_met'),
    ]
    for label, pyarr, gkey in pairs:
        try:
            fa = load_mat_fyx(gkey)
        except Exception as e:
            print(f'{label:22s}  load fail: {e}'); continue
        if fa.shape != pyarr.shape:
            # tolerate y/x swap
            if fa.shape == pyarr.transpose(0, 2, 1).shape:
                fa = fa.transpose(0, 2, 1)
            else:
                print(f'{label:22s}  SHAPE {pyarr.shape} vs {fa.shape}'); continue
        sc, sh, rw, n = stage_metrics(pyarr, fa, mask)
        print(f'{label:22s} {sc:13.3f} {sh*100:17.2f}% {rw*100:8.2f}%')


if __name__ == '__main__':
    main()
