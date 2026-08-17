"""op_CSIB0Correction_v2.py -- B0 correction (FID-A op_CSIB0Correction_v2).

Per masked voxel: phase_offset = unwrap(angle(fft(ifftshift(water_freq)))),
subtract it from BOTH met and water FIDs, transform back to freq.
Returns (met_c, wat_c, freqMap, R2Map). freqMap = linear-fit slope/(2pi) of the
water FID phase at the best-R^2 max_tpt (diagnostic). Deterministic -> reproduces
FID-A bit-for-bit. Requires coil-combined + spatial+spectral FT.
"""
import copy
import numpy as np
from mrsi_common import _faxis, _spatial_shape, TWO_PI


def op_CSIB0Correction_v2(met, wat, mask=None, spectralTime=None):
    if not met['flags'].get('addedrcvrs'):
        raise RuntimeError('op_CSIB0Correction_v2: combine coils first')
    if not (met['flags'].get('spatialft') and met['flags'].get('spectralft')):
        raise RuntimeError('op_CSIB0Correction_v2: need spatial + spectral FT')
    m = copy.deepcopy(met); w = copy.deepcopy(wat)
    af = _faxis(m)
    Nf = m['data'].shape[af]
    sh = _spatial_shape(m, af)                         # (Ny, Nx)
    st = spectralTime if spectralTime is not None else np.asarray(m['spectralTime'])
    if mask is None:
        mask = m.get('mask', {}).get('brainmasks')
    mask = np.ones(sh, bool) if mask is None else np.asarray(mask, bool)

    dm = np.moveaxis(m['data'], af, 0).reshape(Nf, -1, order='F')
    dw = np.moveaxis(w['data'], af, 0).reshape(Nf, -1, order='F')
    idx = np.where(mask.ravel(order='F'))[0]

    # ----- freqMap / R2Map (diagnostic): slope of unwrapped water FID phase -----
    fidw_all = np.fft.fft(np.fft.ifftshift(dw[:, idx], axes=0), axis=0)   # [Nf, Nmask]
    ph_all = np.unwrap(np.angle(fidw_all), axis=0)      # unwrap(full)[0:mt]==unwrap(prefix)
    mt_range = np.arange(10, Nf + 1)                    # MATLAB 10:sz(1)
    Nvox = dm.shape[1]
    freqMapF = np.zeros((Nvox, mt_range.size))
    R2MapF = np.zeros((Nvox, mt_range.size))
    for j, mt in enumerate(mt_range):
        x = st[:mt]; Y = ph_all[:mt, :]
        xm = x.mean(); Sxx = ((x - xm) ** 2).sum()
        slope = ((x - xm)[:, None] * (Y - Y.mean(0))).sum(0) / Sxx
        inter = Y.mean(0) - slope * xm
        yhat = inter[None, :] + slope[None, :] * x[:, None]
        SSres = ((Y - yhat) ** 2).sum(0)
        SStot = ((Y - Y.mean(0)) ** 2).sum(0)
        R2 = np.where(SStot > 0, 1 - SSres / SStot, 0.0)
        freqMapF[idx, j] = slope / TWO_PI
        R2MapF[idx, j] = R2
    meanR2 = R2MapF[idx, :].sum(0) / idx.size
    r2idx = int(np.argmax(meanR2))
    freqMap = freqMapF[:, r2idx].reshape(sh, order='F')
    R2Map = R2MapF[:, r2idx].reshape(sh, order='F')

    # ----- correction: subtract full water FID phase from met + water -----
    fidw = np.fft.fft(np.fft.ifftshift(dw[:, idx], axes=0), axis=0)
    poff = np.unwrap(np.angle(fidw), axis=0)
    fidm = np.fft.fft(np.fft.ifftshift(dm[:, idx], axes=0), axis=0)
    fidm = fidm * np.exp(-1j * poff)
    dm[:, idx] = np.fft.fftshift(np.fft.ifft(fidm, axis=0), axes=0)
    fidw = fidw * np.exp(-1j * poff)
    dw[:, idx] = np.fft.fftshift(np.fft.ifft(fidw, axis=0), axes=0)

    m['data'] = np.moveaxis(dm.reshape((Nf,) + sh, order='F'), 0, af)
    w['data'] = np.moveaxis(dw.reshape((Nf,) + sh, order='F'), 0, af)
    m['freqMap'] = freqMap; m['R2Map'] = R2Map
    return m, w, freqMap, R2Map
