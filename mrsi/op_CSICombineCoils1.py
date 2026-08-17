"""op_CSICombineCoils1.py -- Roemer receive-coil combination (FID-A op_CSICombineCoils1).

Two-call workflow (exactly like the MATLAB):
    ccw, phase, weights = op_CSICombineCoils1(ft_w)             # maps from water ref
    cc                  = op_CSICombineCoils1(ft, 1, phase, weights)  # apply to met

Input must be spatial-FT'd and still have a 'coils' dim; combine is in the time
domain (first dim 't', samplePoint=1 = first/highest-SNR FID point).
"""
import copy
import numpy as np
from mrsi_common import _ax


def op_CSICombineCoils1(struct, samplePoint=1, phaseMap=None, weightMap=None):
    """Roemer receive-coil combination.

    phaseMap/weightMap None  -> derive from THIS data (water ref), return them.
    phaseMap/weightMap given -> apply those (met), reuse the water-ref maps.

    Returns (combined_struct, phaseMap, weightMap). Maps are CANONICAL [coils, y, x]
    so a water-ref map applies to a met struct even when the met has an extra
    'averages' dim (they broadcast into any [t, coils, (avg), y, x] layout).
    """
    if struct['flags'].get('addedrcvrs'):
        raise RuntimeError('op_CSICombineCoils1: data already coil-combined')
    if not struct['flags'].get('spatialft'):
        raise RuntimeError('op_CSICombineCoils1: needs spatial FT first')
    s = copy.deepcopy(struct)
    at, ac, ay, ax = _ax(s, 't'), _ax(s, 'coils'), _ax(s, 'y'), _ax(s, 'x')
    aav = _ax(s, 'averages')
    sp = samplePoint - 1                                  # 1-based -> 0-based

    # transpose to canonical [t, coils, (avg), y, x]
    order = [at, ac] + ([aav] if aav is not None else []) + [ay, ax]
    if sorted(order) != list(range(s['data'].ndim)):
        raise ValueError(f'unexpected dims for combine: {s["dims"]}')
    D = np.transpose(s['data'], order)
    has_avg = aav is not None

    def bcast(m):                                          # [coils,y,x] -> broadcast vs D
        shp = [1] * D.ndim
        shp[1] = m.shape[0]; shp[-2] = m.shape[1]; shp[-1] = m.shape[2]
        return m.reshape(shp)

    # ---------------- (1) phase map [coils,y,x] ----------------
    if phaseMap is None:
        ref = D[sp]                                        # [coils,(avg),y,x]
        if has_avg:
            ref = ref.mean(axis=1)                         # -> [coils,y,x]
        phaseMap = np.angle(ref)
    D = D * np.exp(-1j * bcast(phaseMap))

    # ---------------- (2) weight map (Roemer magnitude, per-voxel unit norm) ----
    if weightMap is None:
        mag = np.abs(D[sp])                                # phase-corrected ref
        if has_avg:
            mag = mag.mean(axis=1)                         # -> [coils,y,x]
        weightMap = mag / np.sqrt(np.nansum(mag ** 2, axis=0, keepdims=True))
    D = D * bcast(weightMap)

    # ---------------- collapse coils (axis 1) ----------------
    D = D.sum(axis=1)                                      # [t, (avg), y, x]
    nd = {k: 0 for k in s['dims']}
    nd['t'] = 1; nxt = 2
    if has_avg:
        nd['averages'] = 2; nxt = 3
    nd['y'] = nxt; nd['x'] = nxt + 1
    s['data'] = D; s['sz'] = tuple(D.shape); s['dims'] = nd
    s['flags']['addedrcvrs'] = 1
    return s, phaseMap, weightMap
