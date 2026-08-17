"""viz/spectral_grid.py -- op_CSIPlot equivalent: grid of voxel spectra.

Classic MRSI grid: each voxel's spectrum drawn as a small line at its (x,y)
image position, on one axes (offset-stack), optionally restricted to the mask
bounding box. High ppm to the left (MRS convention).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ._common import to_fyx, ppm_window, make_mask, water_map, savefig


def spectral_grid(struct, mode='real', ppm_range=(0.2, 5.0), mask=None,
                  bbox=True, out_path=None, title='spectral grid (op_CSIPlot)',
                  color='k', bg_map=True):
    """Grid of voxel spectra over the FOV.

    mode      real | abs | imag
    ppm_range ppm window shown per cell
    mask      [y,x] bool (default: auto from water map). Only masked voxels drawn.
    bbox      crop to the mask bounding box
    bg_map    show the water map faintly behind the grid
    """
    d = to_fyx(struct['data'], struct['dims'])
    nf, ny, nx = d.shape
    ppm = np.asarray(struct.get('ppm', np.arange(nf)))
    sel = ppm_window(ppm, *ppm_range)
    # order ppm ascending-in-index for a left-to-right plot, then flip x display
    order = np.argsort(ppm[sel])
    pp = ppm[sel][order]
    proj = {'real': np.real, 'abs': np.abs, 'imag': np.imag, 'phase': np.angle}[mode]

    wmap = water_map(struct)
    if mask is None:
        mask = make_mask(wmap)
    ys, xs = np.where(mask)
    if bbox and ys.size:
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    else:
        y0, y1, x0, x1 = 0, ny, 0, nx

    # global amplitude for consistent per-cell scaling
    band = proj(d[sel][order][:, mask]) if mask.any() else proj(d[sel][order])
    amp = np.percentile(np.abs(band), 99) or 1.0

    fig, ax = plt.subplots(figsize=(min(0.5 * (x1 - x0) + 2, 18),
                                    min(0.5 * (y1 - y0) + 2, 18)))
    if bg_map:
        ax.imshow(wmap[y0:y1, x0:x1], cmap='gray', origin='lower',
                  extent=[x0 - 0.5, x1 - 0.5, y0 - 0.5, y1 - 0.5], alpha=0.35)
    # x within a cell: map ppm (reversed) to [-0.45, +0.45]
    xn = (pp - pp.min()) / (pp.max() - pp.min() + 1e-12)      # 0..1 (low->high ppm)
    xcell = 0.45 - 0.9 * xn                                   # high ppm left
    for yy, xx in zip(ys, xs):
        s = proj(d[sel, yy, xx][order])
        ax.plot(xx + xcell, yy + 0.42 * s / amp, color=color, lw=0.4)
    ax.set_xlim(x0 - 0.6, x1 - 0.4); ax.set_ylim(y0 - 0.6, y1 - 0.4)
    ax.set_aspect('equal'); ax.set_title(f'{title}\n{mode}, {ppm_range[0]}-{ppm_range[1]} ppm (high ppm left in each cell)')
    ax.set_xlabel('x voxel'); ax.set_ylabel('y voxel')
    if out_path:
        savefig(fig, out_path); plt.close(fig); return out_path
    return fig
