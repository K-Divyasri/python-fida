"""test_pynufft_dcf.py -- does a pynufft (Fessler min-max KB) pipe_menon match
FID-A's DCF weights better than my finufft one?"""
import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), 'mrsi'))
import numpy as np, h5py
from op_CSIRecon import read_kfile, dcf_pipe_menon, _radial_smooth

KFILE = r'C:\Users\divya\Downloads\fida codes\fid_a\processingTools\MRSI\kFiles\Rosette_traj_40x40.txt'
FOV, VOX = 240.0, 6.0
Nx = round(FOV / VOX); dx = FOV / Nx


def load_fida_w():
    with h5py.File(r'C:\Users\divya\Downloads\mrsi_pipeline\dcf_fida.mat', 'r') as f:
        return (np.ravel(f['weights'][()]), np.ravel(f['kx'][()]), np.ravel(f['ky'][()]))


def pynufft_dcf(kx, ky, iters=25):
    from pynufft import NUFFT
    Nk = kx.size
    om = np.column_stack([2 * np.pi * ky * dx, 2 * np.pi * kx * dx])   # [ky, kx] like FID-A
    om = np.mod(om + np.pi, 2 * np.pi) - np.pi                          # wrap to [-pi, pi)
    A = NUFFT(); A.plan(om, (Nx, Nx), (2 * Nx, 2 * Nx), (6, 6))
    w = np.ones(Nk, dtype=complex)
    for _ in range(iters):
        Gw = A.forward(A.adjoint(w))                                    # E (E^H w) = Gram
        den = np.maximum(np.abs(Gw), 1e-4 * np.abs(Gw).max())
        w = w / den; w = w / w.mean()
    w = np.abs(w)
    lo, hi = np.percentile(w, 2), np.percentile(w, 98)
    w = np.clip(w, lo, hi)
    w = _radial_smooth(kx, ky, w, max(round(Nk / 80), 5))
    return w * (Nk / w.sum())


def cmp(name, w, faw):
    a = np.vdot(w, faw) / np.vdot(w, w)
    shape = np.linalg.norm(w * a - faw) / np.linalg.norm(faw)
    corr = np.corrcoef(w, faw)[0, 1]
    print(f'{name:22s} scale {abs(a):.4f}  shape {shape*100:6.2f}%  corr {corr:.4f}')


def main():
    faw, fkx, fky = load_fida_w()
    tr = read_kfile(KFILE); kx, ky = tr['kx'], tr['ky']
    print(f'Nk={kx.size}  (FID-A weights {faw.size})')
    print('align check: kx max diff', np.abs(kx - fkx).max() if kx.size == fkx.size else 'N/A')
    myw = dcf_pipe_menon(kx, ky, FOV, VOX, iters=25); myw = myw * (kx.size / myw.sum())
    pyw = pynufft_dcf(kx, ky)
    print('\n=== DCF weights vs FID-A ===')
    cmp('my finufft DCF', myw, faw)
    cmp('pynufft DCF', pyw, faw)


if __name__ == '__main__':
    main()
