"""op_CSIFlip180.py -- 180 deg spatial flip (rot180 = flip x & y; overlay orientation).

Reindexes the spatial grid only so MRSI overlays on the T1; spectra unchanged.
"""
import copy
import numpy as np
from mrsi_common import _ax


def op_CSIFlip180(struct):
    s = copy.deepcopy(struct)
    ay = _ax(s, 'y'); ax = _ax(s, 'x')
    if ay is not None:
        s['data'] = np.flip(s['data'], axis=ay)
    if ax is not None:
        s['data'] = np.flip(s['data'], axis=ax)
    if 'mask' in s and s['mask'].get('brainmasks') is not None:
        s['mask']['brainmasks'] = np.flip(np.flip(s['mask']['brainmasks'], 0), 1)
    s['flags']['flipped180'] = 1
    return s
