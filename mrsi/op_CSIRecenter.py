"""op_CSIRecenter.py -- data-driven recenter of an off-centre phantom.

computeFOVShift / op_CSIShift only correct the VOI offset via the through-plane
component (pTra) for axial slices, so an off-centre *in-plane* VOI (large dCor/
dSag on an oblique slice) leaves the phantom shifted in the reconstructed image
(e.g. 28thJuly: VOI dCor=-43.8mm -> phantom ~37mm off-centre; the mask/maps look
"not captured fully"). This recentres the spatial grid by integer voxels so the
WATER-reference phantom centroid sits at the FOV centre. Applied to met + ref +
mask identically, so segmentation and maps stay aligned.

Integer np.roll (no interpolation). Guarded: skips the roll if it would wrap the
phantom across an edge.
"""
import copy
import numpy as np


def _ax(s, n):
    v = s['dims'].get(n, 0); return (v - 1) if v else None


def _water_map(ref_struct):
    """[y,x] water map: band-integrate |spectrum| (spectral data) or first FID point."""
    from viz._common import water_map
    return water_map(ref_struct)


def water_centroid(ref_struct):
    """(cy, cx, shape) centroid of the water-ref phantom."""
    w = _water_map(ref_struct)
    yy, xx = np.indices(w.shape)
    tot = w.sum() or 1.0
    return (w * yy).sum() / tot, (w * xx).sum() / tot, w.shape


def compute_roll(ref_struct):
    """Integer (dy, dx) roll that centres the water phantom; (0,0) if it would wrap."""
    cy, cx, (ny, nx) = water_centroid(ref_struct)
    dy, dx = int(round(ny / 2 - cy)), int(round(nx / 2 - cx))
    w = _water_map(ref_struct)
    ys = np.where(w.sum(1) > w.sum(1).max() * 0.05)[0]
    xs = np.where(w.sum(0) > w.sum(0).max() * 0.05)[0]
    if ys.size and xs.size:
        if not (0 <= ys.min() + dy and ys.max() + dy < ny and
                0 <= xs.min() + dx and xs.max() + dx < nx):
            return 0, 0        # would clip/wrap -> don't roll
    return dy, dx


def op_CSIRecenter(struct, dy, dx):
    """Roll the spatial (y,x) dims of a struct by (dy,dx)."""
    s = copy.deepcopy(struct)
    ya, xa = _ax(s, 'y'), _ax(s, 'x')
    if ya is not None:
        s['data'] = np.roll(s['data'], dy, axis=ya)
    if xa is not None:
        s['data'] = np.roll(s['data'], dx, axis=xa)
    if 'mask' in s and s['mask'].get('brainmasks') is not None:
        s['mask']['brainmasks'] = np.roll(np.roll(s['mask']['brainmasks'], dy, 0), dx, 1)
    s['flags']['recentered'] = 1
    s['recenter_roll'] = (dy, dx)
    return s


def recenter_pair(met, ref):
    """Recenter met + ref together using the ref water centroid. -> (met2, ref2, (dy,dx))."""
    dy, dx = compute_roll(ref)
    return op_CSIRecenter(met, dy, dx), op_CSIRecenter(ref, dy, dx), (dy, dx)
