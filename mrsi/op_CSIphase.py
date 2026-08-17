"""op_CSIphase.py -- per-voxel automatic phasing (ACME entropy minimisation, suspect).

Applies zero- and (optionally) first-order phase so metabolite peaks are
absorptive. NOTE: FID-A production has no ACME step; use only for a phased view.
"""
import copy
import numpy as np
from mrsi_common import _faxis


def op_CSIphase(struct, range_ppm=(0.5, 4.0), mask=None, apply_p1=True, ppm0=4.65):
    from suspect.processing import phase as PH
    import suspect
    s = copy.deepcopy(struct)
    af = _faxis(s); ya, xa = s['dims']['y'] - 1, s['dims']['x'] - 1
    d = np.transpose(s['data'], (af, ya, xa)).copy()
    nf, ny, nx = d.shape
    dt = float(s['dwelltime']); f0 = float(s['txfrq']) / 1e6
    k = np.arange(nf)
    m = np.ones((ny, nx), bool) if mask is None else np.asarray(mask, bool)
    for y, x in zip(*np.where(m)):
        spec = d[:, y, x]
        fid = np.fft.fft(np.fft.ifftshift(spec))
        try:
            p0, p1 = PH.acme(suspect.MRSData(fid, dt, f0, ppm0=ppm0), range_ppm=range_ppm)
            d[:, y, x] = spec * np.exp(1j * (p0 + (p1 * k if apply_p1 else 0.0)))
        except Exception:
            continue
    s['data'] = np.transpose(d, np.argsort((af, ya, xa)))
    s['flags']['phased'] = 1
    return s
