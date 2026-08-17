"""hsvd_water_removal.py -- per-voxel HSVD residual-water removal.

Models each FID with `rank` damped exponentials, subtracts components within
+/- water_hw_hz of the water frequency. Needs f-domain, coil-combined data +
struct dwelltime/txfrq. (Alternative to the stochastic L2 water removal.)
"""
import copy
import numpy as np
from mrsi_common import _faxis


def hsvd_water_removal(struct, rank=30, water_hw_hz=35.0, mask=None, ppm0=4.65):
    import suspect
    from suspect.processing.water_suppression import hsvd, construct_fid
    s = copy.deepcopy(struct)
    af = _faxis(s); ya, xa = s['dims']['y'] - 1, s['dims']['x'] - 1
    d = np.transpose(s['data'], (af, ya, xa)).copy()          # [f,y,x]
    nf, ny, nx = d.shape
    dt = float(s['dwelltime']); f0 = float(s['txfrq']) / 1e6
    tax = np.arange(nf) * dt
    m = np.ones((ny, nx), bool) if mask is None else np.asarray(mask, bool)
    for y, x in zip(*np.where(m)):
        spec = d[:, y, x]
        fid = np.fft.fft(np.fft.ifftshift(spec))
        try:
            comps = hsvd(suspect.MRSData(fid, dt, f0, ppm0=ppm0), rank)
            water = [c for c in comps if abs(c['frequency']) < water_hw_hz]
            if water:
                wfid = np.asarray(construct_fid(water, tax))
                d[:, y, x] = np.fft.fftshift(np.fft.ifft(fid - wfid))
        except Exception:
            continue
    s['data'] = np.transpose(d, np.argsort((af, ya, xa)))
    s['flags']['waterremoved'] = 1
    return s
