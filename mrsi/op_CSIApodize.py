"""op_CSIApodize.py -- spatial smoothing (FID-A op_CSIApodize).

For spatial-FT'd data: build a k-space weight (gaussian/hamming/cosine), FT it to
an image-space kernel, and conv2('same') each spectral slice. FWHM in mm (gaussian).

IMPORTANT: MATLAB `weightMatrix = xWeights' * yWeights` uses the CONJUGATE
transpose. The gaussian sits half a sample off-center (coords +/-3, +/-9, ...) so
xWeights is complex; using np.outer(np.conj(Xw), Yw) reproduces FID-A exactly.
Dropping the conj shifts the whole apodized image by one pixel in x and y.
"""
import copy
import numpy as np
from mrsi_common import _ax, _faxis, _coords, _gaussian_negexp


def op_CSIApodize(struct, functionType='gaussian', fullWidthHalfMax=20.0):
    from scipy.signal import convolve2d
    s = copy.deepcopy(struct)
    af = _faxis(s); ay = _ax(s, 'y'); ax = _ax(s, 'x')
    fov = s['fov']; vox = s['voxelSize']
    xco = _coords(fov['x'], vox['x']); yco = _coords(fov['y'], vox['y'])
    Nx, Ny = xco.size, yco.size
    ft = functionType.lower()
    if ft == 'gaussian' and fullWidthHalfMax:
        xco = xco - xco.mean(); yco = yco - yco.mean()        # FID-A centres coords (gaussian-FWHM branch only)
        hfw = fullWidthHalfMax / 2.0
        sig2 = hfw ** 2 / 2 * np.log(0.5)                      # real, negative
        gx = _gaussian_negexp(xco, sig2); gy = _gaussian_negexp(yco, sig2)
        Xw = np.fft.fftshift(np.fft.fft(np.fft.fftshift(gx)))
        Yw = np.fft.fftshift(np.fft.fft(np.fft.fftshift(gy)))
    else:
        # k-space grids for hamming/cosine (spatial-FT'd -> uses x/y coords as k proxy)
        kx, ky = xco, yco
        kMaxX = np.abs(kx).max() + (kx[1] - kx[0]) / 2
        kMaxY = np.abs(ky).max() + (ky[1] - ky[0]) / 2
        if ft == 'cosine':
            Xw = np.cos(np.pi * kx / (2 * kMaxX)); Yw = np.cos(np.pi * ky / (2 * kMaxY))
        else:  # hamming
            Xw = 0.54 + 0.46 * np.cos(np.pi * kx / kMaxX)
            Yw = 0.54 + 0.46 * np.cos(np.pi * ky / kMaxY)
    weightMatrix = np.outer(np.conj(Xw), Yw)                   # MATLAB xWeights'*yWeights (conj transpose on x)
    if not s['flags'].get('spatialft'):
        # k-space multiply (data still [.. ky, kx]); rare in our flow
        raise NotImplementedError('op_CSIApodize: pre-spatial-FT path not used here')
    if weightMatrix.shape[0] % 2 == 1:
        weightMatrix = np.roll(weightMatrix, 1, axis=0)
    if weightMatrix.shape[1] % 2 == 1:
        weightMatrix = np.roll(weightMatrix, 1, axis=1)
    wFT = np.fft.fftshift(np.fft.fft(np.fft.fftshift(weightMatrix, axes=0), axis=0), axes=0)
    wFT = np.fft.fftshift(np.fft.fft(np.fft.fftshift(wFT, axes=1), axis=1), axes=1)
    wFT = wFT / wFT.size
    d = np.transpose(s['data'], (ay, ax, af))                  # [Ny, Nx, spec]
    ma, na = d.shape[:2]; mb, nb = wFT.shape
    r0, c0 = mb // 2, nb // 2                                   # MATLAB conv2 'same' offset
    out = np.empty_like(d)
    for k in range(d.shape[2]):
        full = convolve2d(d[:, :, k], wFT, mode='full')
        out[:, :, k] = full[r0:r0 + ma, c0:c0 + na]
    s['data'] = np.transpose(out, np.argsort((ay, ax, af)))
    s['flags']['apodized'] = 1
    return s
