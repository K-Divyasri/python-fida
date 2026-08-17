"""op_CSISegment_simple.py -- mean-threshold water mask (FID-A op_CSISegment_simple).

intensity = |data[0]| (first point along leading dim, 2D y-x) ->
bwareaopen(intensity>thr, min_size) -> imfill holes. Stores mask.brainmasks.
"""
import copy
import numpy as np
from mrsi_common import _ax


def op_CSISegment_simple(struct, threshold=None, min_size=3):
    from scipy import ndimage
    s = copy.deepcopy(struct)
    d = s['data']
    # first point along the leading (t or f) dim -> 2D [y, x]
    lead = _ax(s, 't')
    if lead is None:
        lead = _ax(s, 'f')
    intensity = np.abs(np.take(d, 0, axis=lead if lead is not None else 0))
    intensity = np.squeeze(intensity)
    if threshold is None:
        threshold = float(intensity.mean())
    bw = intensity > threshold
    # bwareaopen: drop connected components smaller than min_size (8-connectivity)
    lbl, n = ndimage.label(bw, structure=np.ones((3, 3)))
    if n:
        sizes = np.bincount(lbl.ravel())
        keep = np.isin(lbl, np.nonzero(sizes >= min_size)[0][1:] if sizes.size else [])
        bw = keep
    mask = ndimage.binary_fill_holes(bw)
    s.setdefault('mask', {})
    s['mask']['brainmasks'] = mask.astype(bool)
    s['mask']['intensity'] = intensity
    s['mask']['threshold'] = threshold
    print(f'op_CSISegment_simple: {int(mask.sum())}/{mask.size} voxels '
          f'({100 * mask.mean():.1f}%)  threshold={threshold:.3g}')
    return s
