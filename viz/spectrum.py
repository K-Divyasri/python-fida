"""viz/spectrum.py -- single-voxel spectrum plot (MRS convention).

plot_spectrum(struct, y, x)  ->  spectrum at voxel (y,x) vs ppm, high ppm left.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ._common import to_fyx, mrs_xaxis, savefig


def voxel_spectrum(struct, y, x):
    """Return (ppm, complex spectrum) at voxel (y,x)."""
    d = to_fyx(struct['data'], struct['dims'])
    ppm = struct.get('ppm')
    if ppm is None:
        ppm = np.arange(d.shape[0])
    return np.asarray(ppm), d[:, y, x]


def plot_spectrum(struct, y, x, mode='real', xlim=(0.2, 5.0), ax=None,
                  out_path=None, title=None):
    """Plot one voxel spectrum. mode: real | abs | imag. xlim in ppm."""
    ppm, spec = voxel_spectrum(struct, y, x)
    yv = {'real': np.real, 'abs': np.abs, 'imag': np.imag, 'phase': np.angle}[mode](spec)
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(ppm, yv, lw=0.9, color='tab:blue')
    mrs_xaxis(ax, ppm, xlim)
    ax.set_ylabel(f'{mode} intensity')
    ax.set_title(title or f'voxel (y={y}, x={x})')
    ax.grid(alpha=0.3)
    if own and out_path:
        savefig(fig, out_path); plt.close(fig); return out_path
    return ax
