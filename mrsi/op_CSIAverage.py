"""op_CSIAverage.py -- mean over the 'averages' dim (FID-A op_CSIAverage)."""
import copy
from mrsi_common import _ax, _drop_dim


def op_CSIAverage(struct):
    """Mean over the 'averages' dim; drop it. No-op if absent/size<=1/already done."""
    aav = _ax(struct, 'averages')
    if aav is None or struct['flags'].get('averaged') or struct['data'].shape[aav] <= 1:
        return copy.deepcopy(struct)
    s = copy.deepcopy(struct)
    s['data'] = s['data'].mean(axis=aav)
    s['dims'] = _drop_dim(s['dims'], 'averages')
    s['sz'] = tuple(s['data'].shape)
    s['flags']['averaged'] = 1
    return s
