"""viz/op_CSIPlot.py -- FID-A op_CSIPlot, in Python.

The classic MRSI grid plot: every voxel's spectrum drawn as a little line at its
(x, y) grid position, so the spatial layout and the per-voxel lineshape are shown
at once. Faithful to processingTools/MRSI/op_CSIPlot.m:

  - plane_type : 'real' | 'imag' | 'abs' | 'phase'
  - ppmBounds  : (lo, hi) ppm window drawn in each cell (high ppm to the LEFT)
  - xIndecies / yIndecies : (lo, hi) 1-based inclusive voxel ranges to draw
  - yMul       : vertical amplitude multiplier
  - lineWidth  : spectrum line width
  - voxel index labels, (1,1) at bottom-left (idx1 -> right, idx2 -> up)

Works on a pipeline struct or a saved-stage dict (dims [f/t, y, x]).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ._common import to_fyx, water_map, make_mask, savefig

_PROJ = {'real': np.real, 'imag': np.imag, 'abs': np.abs, 'phase': np.angle}


def op_CSIPlot(struct, plane_type='real', ppmBounds=(0.2, 5.0),
               xIndecies=None, yIndecies=None, yMul=1.0, lineWidth=0.6,
               mask=None, bbox=False, bg_map=False, color='k', grid=True,
               labels=True, label_step=1, ax=None, out_path=None,
               title='op_CSIPlot'):
    """Grid of voxel spectra over the FOV.

    struct     pipeline struct / stage dict with data, dims, ppm
    plane_type real | imag | abs | phase
    ppmBounds  (lo, hi) ppm window per cell
    xIndecies  (lo, hi) 1-based inclusive x voxel range (default: full or mask bbox)
    yIndecies  (lo, hi) 1-based inclusive y voxel range
    mask       [y,x] bool; if given, only masked voxels are drawn
    bbox       crop the drawn range to the mask bounding box
    yMul       vertical amplitude scale (per-cell spectra ~ +/- 0.45*yMul)
    labels     annotate each cell with (idxRight, idxUp), (1,1) bottom-left
    Returns the figure (or out_path if saved).
    """
    d = to_fyx(struct['data'], struct['dims'])
    nf, ny, nx = d.shape
    ppm = np.asarray(struct.get('ppm'))
    if ppm is None or ppm.size != nf:
        ppm = np.arange(nf)                      # time-domain fallback
    proj = _PROJ[plane_type]

    # ppm window, ordered low->high ppm index for a consistent per-cell x-map
    lo, hi = min(ppmBounds), max(ppmBounds)
    sel = np.where((ppm >= lo) & (ppm <= hi))[0]
    if sel.size == 0:
        raise ValueError(f'no ppm points in {ppmBounds} (ppm {ppm.min():.2f}..{ppm.max():.2f})')
    order = sel[np.argsort(ppm[sel])]
    pp = ppm[order]

    # ---- voxel ranges (1-based inclusive, like FID-A) ----
    wmap = water_map(struct) if bg_map or mask is None else None
    if mask is None and (bbox or bg_map):
        mask = make_mask(wmap)
    if xIndecies is not None:
        x0, x1 = int(xIndecies[0]) - 1, int(xIndecies[1])
    else:
        x0, x1 = 0, nx
    if yIndecies is not None:
        y0, y1 = int(yIndecies[0]) - 1, int(yIndecies[1])
    else:
        y0, y1 = 0, ny
    if bbox and mask is not None and mask.any():
        ys, xs = np.where(mask)
        x0, x1 = max(x0, xs.min()), min(x1, xs.max() + 1)
        y0, y1 = max(y0, ys.min()), min(y1, ys.max() + 1)

    def drawn(yy, xx):
        if not (y0 <= yy < y1 and x0 <= xx < x1):
            return False
        return True if mask is None else bool(mask[yy, xx])

    # ---- global amplitude for consistent per-cell scaling (99th pct over drawn) ----
    drawn_cells = [(yy, xx) for yy in range(y0, y1) for xx in range(x0, x1) if drawn(yy, xx)]
    if not drawn_cells:
        raise ValueError('no voxels to draw (check mask / ranges)')
    band = np.array([proj(d[order, yy, xx]) for yy, xx in drawn_cells])
    amp = np.percentile(np.abs(band), 99) or 1.0

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(min(0.55 * (x1 - x0) + 2, 20),
                                        min(0.55 * (y1 - y0) + 2, 20)))
    else:
        fig = ax.figure

    if bg_map and wmap is not None:
        ax.imshow(wmap[y0:y1, x0:x1], cmap='gray', origin='lower',
                  extent=[x0 - 0.5, x1 - 0.5, y0 - 0.5, y1 - 0.5], alpha=0.30)

    # per-cell x: ppm reversed into [-0.45, +0.45] so HIGH ppm is on the LEFT
    xn = (pp - pp.min()) / (pp.max() - pp.min() + 1e-12)     # 0(low)..1(high)
    xcell = 0.45 - 0.9 * xn

    if grid:
        for gx in range(x0, x1 + 1):
            ax.axvline(gx - 0.5, color='0.85', lw=0.4, zorder=0)
        for gy in range(y0, y1 + 1):
            ax.axhline(gy - 0.5, color='0.85', lw=0.4, zorder=0)

    for yy, xx in drawn_cells:
        s = proj(d[order, yy, xx])
        ax.plot(xx + xcell, yy + 0.45 * yMul * s / amp, color=color,
                lw=lineWidth, zorder=2)
        if labels and ((xx - x0) % label_step == 0) and ((yy - y0) % label_step == 0):
            ax.text(xx - 0.47, yy + 0.47, f'({xx - x0 + 1},{yy - y0 + 1})',
                    fontsize=6, color='0.5', ha='left', va='top', zorder=1)

    ax.set_xlim(x0 - 0.6, x1 - 0.4)
    ax.set_ylim(y0 - 0.6, y1 - 0.4)                          # origin lower -> (1,1) bottom-left
    ax.set_aspect('equal')
    ax.set_xlabel('x voxel'); ax.set_ylabel('y voxel')
    ax.set_title(f'{title}  |  {plane_type}, {lo:g}-{hi:g} ppm (high ppm left in each cell)')
    if own and out_path:
        savefig(fig, out_path); plt.close(fig); return out_path
    return fig
