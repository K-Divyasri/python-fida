"""test_recon_ft.py -- with FID-A's EXACT DCF weights, which FT reproduces FID-A's
recon? Reconstruct the water (s02_prep_ref) and compare to FID-A s03_recon_ref.
Tests my dft (exact), my finufft, and pynufft (Fessler KB) FTs.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.getcwd(), 'mrsi'))
import numpy as np, h5py
from op_CSIRecon import read_kfile, _merge_kpts_time, _image_grid, _coords

KFILE = r'C:\Users\divya\Downloads\fida codes\fid_a\processingTools\MRSI\kFiles\Rosette_traj_40x40.txt'
PY = r'C:\Users\divya\Downloads\mrsi_pipeline\rosette_py_stages_nufft576'
GT = r'C:\Users\divya\Downloads\mrsi_pipeline\fida_stages_s02_nufft'
FOV, VOX = 240.0, 6.0
Nx = round(FOV / VOX); dx = FOV / Nx
NIT = 6                                   # compare first NIT spectral points


def load_py(key):
    z = np.load(os.path.join(PY, key + '.npz'), allow_pickle=True)
    m = json.loads(str(z['meta'])); dims = {k: int(v) for k, v in m['dims'].items() if v}
    return dict(data=z['data'], dims=dims, fov={'x': FOV, 'y': FOV},
                voxelSize={'x': VOX, 'y': VOX}, sz=z['data'].shape,
                dwelltime=1.0, flags={})


def load_gt(key):
    with h5py.File(os.path.join(GT, key + '.mat'), 'r') as f:
        d = f['data'][()]
        a = (d['real'] + 1j * d['imag']) if (d.dtype.names and 'real' in d.dtype.names) else d
    return np.transpose(a, tuple(range(a.ndim))[::-1])       # reverse -> FID-A [t,coils,y,x]


def load_faw():
    with h5py.File(r'C:\Users\divya\Downloads\mrsi_pipeline\dcf_fida.mat', 'r') as f:
        return np.ravel(f['weights'][()])


def make_ft_ops():
    tr = read_kfile(KFILE); kx, ky = tr['kx'], tr['ky']; Nk = tr['Nk']
    xg, yg = _coords(FOV, VOX), _coords(FOV, VOX)
    om_x = 2 * np.pi * kx * dx; om_y = 2 * np.pi * ky * dx
    x_shift = np.median(np.arange(Nx) - xg / dx); y_shift = np.median(np.arange(Nx) - yg / dx)

    # my finufft adjoint (matches op_CSIRecon._spatial_nufft)
    import finufft
    nshift_ph = np.exp(1j * (om_x * (Nx / 2 - x_shift) + om_y * (Nx / 2 - y_shift)))
    def fin(y):                       # y:[Nk] -> [Ny,Nx]
        return finufft.nufft2d1(om_y, om_x, (y * nshift_ph).astype(complex), (Nx, Nx)) / Nk

    # exact dense adjoint (dft)
    xx, yy = np.meshgrid(xg, yg); xp = xx.ravel(order='F'); yp = yy.ravel(order='F')
    EH = np.exp(2j * np.pi * (np.outer(xp, kx) + np.outer(yp, ky))) / Nk
    def dft(y):
        return (EH @ y).reshape(Nx, Nx, order='F')

    # pynufft (Fessler min-max KB), n_shift via half-pixel phase
    from pynufft import NUFFT
    om = np.column_stack([om_y, om_x]); om = np.mod(om + np.pi, 2 * np.pi) - np.pi
    A = NUFFT(); A.plan(om, (Nx, Nx), (2 * Nx, 2 * Nx), (6, 6))
    hp = np.exp(1j * (om_y * (Nx / 2 - y_shift) + om_x * (Nx / 2 - x_shift)))
    def pyn(y):
        return A.adjoint((y * hp).astype(complex)) / Nk
    return dict(finufft=fin, dft=dft, pynufft=pyn), Nk


def main():
    faw = load_faw()
    ops, Nk = make_ft_ops()
    prep = load_py('s02_prep_ref')
    X, nkpt, nshot, extras = _merge_kpts_time(prep)     # [Ntot, nshot, Nextra]
    Nextra = X.shape[2]
    gt = load_gt('s03_recon_ref')                        # [t, coils, y, x]
    # FID-A layout [t,coils,y,x]; my recon gives [y,x] per (it, extra=coil)
    print('prep', prep['data'].shape, ' gt recon', gt.shape, ' Nextra', Nextra)

    W = faw.reshape(nkpt, nshot, order='F')
    res = {k: [] for k in ops}
    for it in range(NIT):
        i0, i1 = it * nkpt, (it + 1) * nkpt
        for e in range(Nextra):
            sl = X[i0:i1, :, e].reshape(Nk, order='F') * W.reshape(Nk, order='F')
            g = gt[it, e]                                # [y,x] FID-A
            for name, op in ops.items():
                img = op(sl)
                a = np.vdot(img, g) / np.vdot(img, img)
                res[name].append((abs(a), np.linalg.norm(img * a - g) / (np.linalg.norm(g) + 1e-30)))
    print('\n=== recon vs FID-A (FID-A weights, first %d spectral pts x %d coils) ===' % (NIT, Nextra))
    for name in ops:
        arr = np.array(res[name])
        print(f'{name:10s} scale {np.median(arr[:,0]):.4f}  shape {np.median(arr[:,1])*100:6.2f}%')


if __name__ == '__main__':
    main()
