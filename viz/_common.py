"""viz/_common.py -- shared helpers for the MRSI visualisers.

Load pipeline structs OR saved stage .npz, reorder to canonical [f, y, x],
handle the MRS ppm axis, integrate ppm bands into 2D maps, and build masks.
"""
import os, json
import numpy as np


# ------------------------------------------------------------------ loading
def load_stage(path):
    """Load a saved stage .npz (from run_rosette_pipeline save_dir) ->
    dict(data, ppm, dims, fov, voxelSize). data kept in its stored layout."""
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z['meta']))
    ppm = z['ppm']
    return dict(data=np.asarray(z['data']), ppm=(ppm if ppm.size else None),
                dims={k: int(v) for k, v in meta['dims'].items() if v},
                fov=meta.get('fov', {}), voxelSize=meta.get('voxelSize', {}))


def as_struct(x):
    """Accept a pipeline struct (dict with data/dims/ppm) or a stage dict; return it."""
    return x


def to_fyx(data, dims):
    """Reorder an array to [f(or t), y, x] using the 1-based dims dict."""
    fax = dims.get('f') or dims.get('t')
    yax = dims.get('y'); xax = dims.get('x')
    if not (fax and yax and xax):
        raise ValueError(f'need f/t,y,x dims; got {dims}')
    return np.transpose(data, (fax - 1, yax - 1, xax - 1))


# ------------------------------------------------------------------ ppm axis
def ppm_window(ppm, lo, hi):
    """Boolean mask + sorted index for ppm in [lo,hi] (ppm may be descending)."""
    m = (ppm >= lo) & (ppm <= hi)
    return m


def mrs_xaxis(ax, ppm, xlim=None):
    """Set an MRS-convention ppm x-axis: high ppm on the LEFT."""
    lo, hi = (min(ppm), max(ppm)) if xlim is None else xlim
    ax.set_xlim(hi, lo)                 # reversed
    ax.set_xlabel('chemical shift (ppm)')


# ------------------------------------------------------------------ maps
def integrate_band(data_fyx, ppm, lo, hi, mode='mag'):
    """Integrate |spectrum| (or real) over a ppm band -> [y, x] map.
    mode: 'mag' (sum |.|), 'real' (sum real), 'max' (peak |.|)."""
    sel = ppm_window(ppm, lo, hi)
    band = data_fyx[sel, :, :]
    if mode == 'real':
        return np.real(band).sum(axis=0)
    if mode == 'max':
        return np.abs(band).max(axis=0)
    return np.abs(band).sum(axis=0)


def first_point_map(data_fyx):
    """|data[0]| -> [y,x] (time-domain first-point intensity; water map proxy)."""
    return np.abs(data_fyx[0])


def water_map(struct):
    """2D water-intensity map from a struct/stage. Uses the water band if spectral,
    else the first time point."""
    d = to_fyx(struct['data'], struct['dims'])
    ppm = struct.get('ppm')
    if ppm is not None and 'f' in struct['dims']:
        return integrate_band(d, ppm, 4.0, 5.4, mode='max')
    return first_point_map(d)


def make_mask(intensity, threshold=None, min_size=3):
    """Mean-threshold mask (op_CSISegment_simple): bwareaopen + fill holes."""
    from scipy import ndimage
    if threshold is None:
        threshold = float(intensity.mean())
    bw = intensity > threshold
    lbl, n = ndimage.label(bw, structure=np.ones((3, 3)))
    if n:
        sizes = np.bincount(lbl.ravel())
        keep = np.isin(lbl, np.nonzero(sizes >= min_size)[0][1:] if sizes.size else [])
        bw = keep
    return ndimage.binary_fill_holes(bw)


def savefig(fig, out_path, dpi=110):
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    return out_path
