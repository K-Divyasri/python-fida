"""op_CSIspecZeroFill.py -- spectral zero-fill (FID-A op_CSIspecZeroFill).

Invert the spectral FT, zero-pad the FID to Ntarget, re-transform. Pure spectral
interpolation (finer ppm grid for LCModel). Applied AFTER B0, on native length.
"""
import copy
import numpy as np
from mrsi_common import _faxis


def op_CSIspecZeroFill(struct, Ntarget=4096, ppm0=4.65):
    s = copy.deepcopy(struct)
    af = _faxis(s)
    N0 = s['data'].shape[af]
    if Ntarget <= N0:
        return s
    fid = np.fft.fft(np.fft.ifftshift(s['data'], axes=af), axis=af)
    shp = list(s['data'].shape); shp[af] = Ntarget
    fidPad = np.zeros(shp, dtype=complex)
    idx = [slice(None)] * s['data'].ndim; idx[af] = slice(0, N0)
    fidPad[tuple(idx)] = fid
    s['data'] = np.fft.fftshift(np.fft.ifft(fidPad, axis=af), axes=af)
    s['sz'] = tuple(s['data'].shape)
    sw = float(s['spectralWidth']); step = sw / Ntarget
    f = np.linspace(-sw / 2 + step / 2, sw / 2 - step / 2, Ntarget)
    s['ppm'] = -f / (float(s['txfrq']) / 1e6) + ppm0
    s['spectralFreq'] = f
    dt = float(s.get('dwelltime', 1.0) or 1.0)
    s['spectralTime'] = np.arange(Ntarget) * dt
    s['adcTime'] = np.arange(Ntarget) * dt
    s.setdefault('flags', {})['zerofilled'] = 1
    return s
