"""op_CSIapplymask.py -- zero out voxels outside mask.brainmasks (FID-A op_CSIapplymask)."""
import copy
import numpy as np
from mrsi_common import _ax, _faxis


def op_CSIapplymask(struct):
    """Zero out voxels outside mask.brainmasks across the spectral dim."""
    if 'mask' not in struct or struct['mask'].get('brainmasks') is None:
        raise RuntimeError('op_CSIapplymask: no mask (segment first)')
    s = copy.deepcopy(struct)
    af = _faxis(s); ay = _ax(s, 'y'); ax = _ax(s, 'x')
    mask = np.asarray(s['mask']['brainmasks'], bool)          # [y, x]
    full = [1] * s['data'].ndim
    full[ay] = mask.shape[0]; full[ax] = mask.shape[1]
    s['data'] = s['data'] * mask.reshape(full)
    return s
