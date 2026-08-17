"""op_CSIFourierTransform.py -- FID-A-style spatial/spectral FT for CSI.

Mirrors FID-A op_CSIFourierTransform / op_CSIRecon (cartesian FFT branch):
takes an MRSIStruct from io_CSIload_twix, transforms, and updates dims/flags
exactly as FID-A does so the struct stays comparable field-for-field.

  spatial=True  : ifft2 over (kx,ky) -> image; dims kx->x, ky->y; flags.spatialft=1
  spectral=True : ifft over t -> spectrum; dims t->f; flags.spectralft=1; fills .ppm

Conventions (validated against FID-A on this data):
  spatial : fftshift(ifft2(ifftshift(k))) * sqrt(Nx*Ny)   (centred, per-voxel scale)
  spectral: fftshift(ifft(fid))                            (matched FID-A exactly, corr 1.0)
"""
import copy
import numpy as np


def op_CSIFourierTransform(struct, spatial=True, spectral=False, ppm_ref=4.65):
    s = copy.deepcopy(struct)
    data = s['data']
    dims = s['dims']

    if spatial:
        if s['flags'].get('spatialft'):
            raise RuntimeError('op_CSIFourierTransform: already spatial-FT')
        axx = dims['kx'] - 1
        axy = dims['ky'] - 1
        Nx = data.shape[axx]; Ny = data.shape[axy]
        data = np.fft.ifftshift(data, axes=(axy, axx))
        data = np.fft.ifft2(data, axes=(axy, axx))
        data = np.fft.fftshift(data, axes=(axy, axx)) * np.sqrt(Nx * Ny)
        # relabel dims kx->x, ky->y (same axis indices, FID-A convention)
        dims['x'] = dims['kx']; dims['y'] = dims['ky']
        dims['kx'] = 0; dims['ky'] = 0
        s['flags']['spatialft'] = 1

    if spectral:
        if s['flags'].get('spectralft'):
            raise RuntimeError('op_CSIFourierTransform: already spectral-FT')
        axt = dims['t'] - 1
        n = data.shape[axt]
        data = np.fft.fftshift(np.fft.ifft(data, axis=axt), axes=axt)
        dims['f'] = dims['t']; dims['t'] = 0
        s['flags']['spectralft'] = 1
        f = np.fft.fftshift(np.fft.fftfreq(n, s['dwelltime']))     # Hz
        s['ppm'] = -f / (s['txfrq'] / 1e6) + ppm_ref
        s['spectralFreq'] = f

    s['data'] = data
    s['sz'] = tuple(data.shape)
    s['dims'] = dims
    return s


if __name__ == '__main__':
    import sys
    from io_CSIload_twix import io_CSIload_twix, print_struct
    met = sys.argv[1] if len(sys.argv) > 1 else r'tmp\met.dat'
    m = io_CSIload_twix(met)
    ft = op_CSIFourierTransform(m, spatial=True)
    print_struct(ft, 'after spatial FT')
    print('  flags.spatialft =', ft['flags']['spatialft'], ' dims x,y =', ft['dims']['x'], ft['dims']['y'])
