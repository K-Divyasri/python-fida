"""viz/plotspec.py -- FID-A op_plotspec / op_plotfid / imagesc, in Python.

op_plotspec(ppm, specs)   real spectrum vs ppm, MRS axis (high ppm left)  [op_plotspec.m]
op_plotfid(t, fids)       FID real/imag vs time                           [op_plotfid.m]
imagesc(M)                MATLAB imagesc: row 1 at top, axis image, colorbar
plot_voxel_spec/fid       pull a voxel out of a pipeline struct and plot it

All accept raw arrays OR a (struct, x, y) via the *_voxel helpers.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _save_or_return(fig, out_path):
    if out_path:
        import os
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        fig.savefig(out_path, dpi=110, bbox_inches='tight'); plt.close(fig)
        return out_path
    return fig


# ------------------------------------------------------------------ op_plotspec
def op_plotspec(ppm, specs, ppmmin=0.2, ppmmax=5.2, xlab='Frequency (ppm)',
                ylab='', title='', part='real', ax=None, out_path=None, lw=2.0,
                color='b'):
    """FID-A op_plotspec: plot the (real) spectrum with a reversed ppm axis.
    part: 'real' (default, absorptive) | 'imag' | 'abs'."""
    ppm = np.asarray(ppm); specs = np.asarray(specs).squeeze()
    y = {'real': np.real, 'imag': np.imag, 'abs': np.abs}[part](specs)
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(ppm, y, color=color, lw=lw)
    ax.set_xlim(ppmmax, ppmmin)                       # XDir reverse (high ppm left)
    ax.set_xlabel(xlab)
    if ylab:
        ax.set_ylabel(ylab)
    else:
        ax.set_yticks([])
    if title:
        ax.set_title(title)
    ax.spines[['top', 'right']].set_visible(False)
    return _save_or_return(ax.figure, out_path) if own else ax


# ------------------------------------------------------------------ op_plotfid
def op_plotfid(t, fids, tmax=None, title='', ax=None, out_path=None, lw=1.0):
    """FID-A op_plotfid: plot the FID (real + imag) vs time."""
    t = np.asarray(t); fids = np.asarray(fids).squeeze()
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(t, np.real(fids), 'b', lw=lw, label='real')
    ax.plot(t, np.imag(fids), 'r', lw=lw, alpha=0.7, label='imag')
    if tmax:
        ax.set_xlim(0, tmax)
    ax.set_xlabel('time (s)'); ax.set_ylabel('amplitude'); ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    return _save_or_return(ax.figure, out_path) if own else ax


# ------------------------------------------------------------------ imagesc
def imagesc(M, title='', cmap='viridis', clim=None, origin='upper', colorbar=True,
            mask=None, ax=None, out_path=None, xlabel='', ylabel=''):
    """MATLAB imagesc: display a 2D matrix with a colorbar. origin='upper' = row 1
    at top (MATLAB axis ij); 'lower' = y increasing up (image convention)."""
    M = np.asarray(M, float)
    if mask is not None:
        M = np.where(np.asarray(mask, bool), M, np.nan)
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(M, cmap=cmap, origin=origin, aspect='equal',
                   vmin=None if clim is None else clim[0],
                   vmax=None if clim is None else clim[1])
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save_or_return(ax.figure, out_path) if own else ax


# ------------------------------------------------- voxel helpers (from a struct)
def _voxel(struct, x, y):
    """Return (ppm, spec, t, fid) for voxel (x,y) of an [f,y,x] struct."""
    from ._common import to_fyx
    d = to_fyx(struct['data'], struct['dims'])
    spec = d[:, y, x]
    ppm = np.asarray(struct.get('ppm', np.arange(d.shape[0])))
    dt = float(struct.get('dwelltime', 1.0) or 1.0)
    t = np.arange(d.shape[0]) * dt
    fid = np.fft.fft(np.fft.fftshift(spec))           # op_CSItoMRS convention
    return ppm, spec, t, fid


def plot_voxel_spec(struct, x, y, **kw):
    """op_plotspec for voxel (x,y). NB: x first, y second (op_CSItoMRS order)."""
    ppm, spec, _, _ = _voxel(struct, x, y)
    kw.setdefault('title', f'voxel (x={x}, y={y})')
    return op_plotspec(ppm, spec, **kw)


def plot_voxel_fid(struct, x, y, **kw):
    ppm, spec, t, fid = _voxel(struct, x, y)
    kw.setdefault('title', f'FID voxel (x={x}, y={y})')
    return op_plotfid(t, fid, **kw)
