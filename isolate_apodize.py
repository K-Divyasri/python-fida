"""isolate_apodize.py -- is my op_CSIApodize the bug, or just amplifying recon err?

Feed FID-A's OWN s09 (B0) output through my apodize and compare to FID-A's s11
(smooth). If it reproduces FID-A s11 <1%, my apodize is correct. Also test a
reference numpy re-implementation of FID-A's exact conv2('same') to find the
right crop offset.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mrsi'))
import numpy as np, h5py
from scipy.signal import convolve2d

GT = r'C:\Users\divya\Downloads\mrsi_pipeline\fida_stages_s02_nufft'
FOV = dict(x=240.0, y=240.0, z=15.0); VOX = dict(x=6.0, y=6.0, z=15.0)


def load_mat_fyx(key):
    with h5py.File(os.path.join(GT, key + '.mat'), 'r') as f:
        d = f['data'][()]
        a = (d['real'] + 1j * d['imag']) if (d.dtype.names and 'real' in d.dtype.names) else d
    return np.transpose(a, tuple(range(a.ndim))[::-1])


def _coords(fov_mm, vox_mm):
    b = fov_mm / 2.0
    c = np.arange(-b + vox_mm / 2.0, b, vox_mm)
    return c


def build_weightsFT(fwhm=20.0):
    xco = _coords(FOV['x'], VOX['x']); yco = _coords(FOV['y'], VOX['y'])
    xco = xco - xco.mean(); yco = yco - yco.mean()          # FID-A centres coords
    hfw = fwhm / 2.0
    sigma2 = hfw ** 2 / 2 * np.log(0.5)                      # negative
    def g(x):
        return -np.exp(x ** 2 / (2 * sigma2))               # FID-A gaussian(x,sigma)
    Xw = np.fft.fftshift(np.fft.fft(np.fft.fftshift(g(xco))))
    Yw = np.fft.fftshift(np.fft.fft(np.fft.fftshift(g(yco))))
    wm = np.outer(Xw, Yw)                                    # [Nx,Ny]
    if wm.shape[0] % 2 == 1: wm = np.roll(wm, 1, axis=0)
    if wm.shape[1] % 2 == 1: wm = np.roll(wm, 1, axis=1)
    wFT = np.fft.fftshift(np.fft.fft(np.fft.fftshift(wm, 0), axis=0), 0)
    wFT = np.fft.fftshift(np.fft.fft(np.fft.fftshift(wFT, 1), axis=1), 1)
    return wFT / wFT.size


def fida_apodize_ref(data_fyx, crop='floor'):
    """Exact FID-A conv2('same') per spectral slice. data [f,y,x]. weightsFT is
    [Nx,Ny] -> for conv over (y,x) we need it as [y,x]; FID-A reshapes to {y,x}
    then conv2. weightMatrix = xWeights'*yWeights is [Nx,Ny]; data reshaped {y,x}
    so conv2(data[y,x], weightsFT). weightsFT indexed [?]. Test transpose too."""
    wFT = build_weightsFT()
    nf, ny, nx = data_fyx.shape
    for W, tag in [(wFT, 'wFT[Nx,Ny]'), (wFT.T, 'wFT.T[Ny,Nx]')]:
        mb, nb = W.shape
        if crop == 'floor':
            r0, c0 = mb // 2, nb // 2
        else:
            r0, c0 = (mb - 1) // 2, (nb - 1) // 2
        out = np.empty_like(data_fyx)
        for k in range(nf):
            full = convolve2d(data_fyx[k], W, mode='full')
            out[k] = full[r0:r0 + ny, c0:c0 + nx]
        yield tag, crop, out


def metrics(py, fa, mask):
    ys, xs = np.where(mask)
    sc, sh = [], []
    for y, x in zip(ys, xs):
        a = np.vdot(py[:, y, x], fa[:, y, x]) / np.vdot(py[:, y, x], py[:, y, x])
        sc.append(np.abs(a))
        sh.append(np.linalg.norm(py[:, y, x] * a - fa[:, y, x]) / (np.linalg.norm(fa[:, y, x]) + 1e-30))
    return np.median(sc), np.median(sh)


def main():
    b0 = load_mat_fyx('s09_b0_ref')          # apodize INPUT (water, FID-A)
    s11 = load_mat_fyx('s11_smooth_ref')     # apodize OUTPUT (water, FID-A)
    with h5py.File(os.path.join(GT, 'mask.mat'), 'r') as f:
        mask = np.transpose(f['mask'][()], (1, 0)).astype(bool)
    print('mask vox', int(mask.sum()), 'b0', b0.shape)

    # 1) my production op_CSIApodize
    from op_CSIpostproc import op_CSIApodize
    s = dict(data=b0.copy(), fov=FOV, voxelSize=VOX,
             dims={'t': 0, 'f': 1, 'y': 2, 'x': 3, 'coils': 0, 'averages': 0,
                   'kx': 0, 'ky': 0, 'z': 0, 'kpts': 0, 'kshot': 0, 'extras': 0},
             flags={'spatialft': 1}, sz=b0.shape)
    mine = op_CSIApodize(s, 'gaussian', 20)['data']
    print(f'{"mine op_CSIApodize":28s} scale {metrics(mine, s11, mask)[0]:.4f}  shape {metrics(mine, s11, mask)[1]*100:6.2f}%')

    # 2) reference variants (kernel orientation x crop offset)
    for crop in ('floor', 'floor-1'):
        for tag, cr, out in fida_apodize_ref(b0, crop):
            m = metrics(out, s11, mask)
            print(f'  ref {tag:14s} crop={cr:8s} scale {m[0]:.4f}  shape {m[1]*100:6.2f}%')


if __name__ == '__main__':
    main()
