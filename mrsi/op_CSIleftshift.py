"""op_CSIleftshift.py -- drop leading FID points (FID-A op_CSIleftshift).

The FID's first point isn't at t=0 (ADC/echo delay), which puts a first-order
phase across the spectrum. Removing the leading `ls` points before the spectral
FT flattens that phase. Call on the TIME-domain struct, BEFORE op_CSIFourierTransform.
ls defaults to struct['pointsToLeftshift'] (0 for this rosette data -> no-op).
"""
import copy
import numpy as np
from mrsi_common import _ax


def op_CSIleftshift(struct, ls=None):
    if ls is None:
        ls = int(struct.get('pointsToLeftshift', 0) or 0)
    ls = int(round(ls))
    if ls <= 0:
        return copy.deepcopy(struct)
    if struct.get('flags', {}).get('leftshifted'):
        return copy.deepcopy(struct)
    s = copy.deepcopy(struct)
    td = _ax(s, 't')
    if td is None:
        raise RuntimeError('op_CSIleftshift needs a time dimension (dims.t)')
    Nt0 = s['data'].shape[td]
    if ls >= Nt0:
        raise ValueError(f'ls ({ls}) must be < number of time points ({Nt0})')
    idx = [slice(None)] * s['data'].ndim
    idx[td] = slice(ls, Nt0)
    s['data'] = s['data'][tuple(idx)]
    s['sz'] = tuple(s['data'].shape)
    Nt = s['data'].shape[td]
    dt = s.get('spectralDwellTime')
    if dt:
        s['spectralTime'] = np.arange(Nt) * float(dt)
    adt = s.get('adcDwellTime')
    if adt:
        s['adcTime'] = np.arange(Nt) * float(adt)
    s.setdefault('flags', {})['leftshifted'] = 1
    print(f'op_CSIleftshift: dropped {ls} leading FID point(s); t dim {Nt0} -> {Nt}.')
    return s
