"""viz/voxel_viewer.py -- interactive: click a voxel -> its spectrum.

launch_viewer(struct)  opens a two-panel figure (map | spectrum); click a voxel on
the map to plot its spectrum (op_CSIPlotVoxelSpec-style). Needs an interactive
matplotlib backend (TkAgg/QtAgg) -- run from a normal Python session, not headless.

render_voxel(struct, y, x, out_path)  static fallback: map with a marker + that
voxel's spectrum, saved to PNG (works headless).
"""
import numpy as np

from ._common import to_fyx, water_map, make_mask, mrs_xaxis, savefig


def _bg_map(struct, bg, ppm_range):
    if bg == 'water':
        return water_map(struct)
    from .maps import op_CSIintegrate
    return np.abs(op_CSIintegrate(struct, ppm_range[0], ppm_range[1], 'mag'))


def render_voxel(struct, y, x, ppm_range=(0.2, 5.2), mode='real', bg='water',
                 out_path=None, bg_struct=None):
    """Static: background map (marker at y,x) + the voxel spectrum. bg_struct lets
    the background come from the water ref (met water is suppressed/removed)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = to_fyx(struct['data'], struct['dims'])
    ppm = np.asarray(struct.get('ppm', np.arange(d.shape[0])))
    proj = {'real': np.real, 'abs': np.abs, 'imag': np.imag, 'phase': np.angle}[mode]
    bgm = _bg_map(bg_struct if bg_struct is not None else struct, bg, ppm_range)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 4))
    a0.imshow(bgm, cmap='gray', origin='lower'); a0.plot(x, y, 'r+', ms=14, mew=2)
    a0.set_title(f'{bg} map'); a0.axis('off')
    a1.plot(ppm, proj(d[:, y, x]), lw=0.9)
    mrs_xaxis(a1, ppm, ppm_range); a1.set_title(f'voxel (y={y}, x={x})')
    a1.set_ylabel(f'{mode}'); a1.grid(alpha=0.3)
    if out_path:
        savefig(fig, out_path); plt.close(fig); return out_path
    return fig


def launch_viewer(struct, ppm_range=(0.2, 5.2), mode='real', bg='water'):
    """Interactive click-to-spectrum viewer. Returns the figure; keep a reference
    so callbacks stay alive. Requires a GUI backend."""
    import matplotlib.pyplot as plt
    d = to_fyx(struct['data'], struct['dims'])
    ppm = np.asarray(struct.get('ppm', np.arange(d.shape[0])))
    proj = {'real': np.real, 'abs': np.abs, 'imag': np.imag, 'phase': np.angle}[mode]
    bgm = _bg_map(struct, bg, ppm_range)
    ny, nx = bgm.shape

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.5))
    a0.imshow(bgm, cmap='gray', origin='lower'); a0.set_title(f'{bg} map — click a voxel')
    marker, = a0.plot([], [], 'r+', ms=14, mew=2)
    line, = a1.plot(ppm, proj(d[:, ny // 2, nx // 2]), lw=0.9)
    mrs_xaxis(a1, ppm, ppm_range); a1.set_ylabel(mode); a1.grid(alpha=0.3)

    def on_click(ev):
        if ev.inaxes is not a0 or ev.xdata is None:
            return
        x = int(round(ev.xdata)); y = int(round(ev.ydata))
        if not (0 <= x < nx and 0 <= y < ny):
            return
        line.set_ydata(proj(d[:, y, x])); a1.relim(); a1.autoscale_view()
        a1.set_title(f'voxel (y={y}, x={x})'); marker.set_data([x], [y])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig._mrsi_cb = on_click                          # keep ref alive
    return fig
