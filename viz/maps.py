"""viz/maps.py -- metabolite / water / CRLB maps (imagesc equivalents).

- water_map_fig       water-intensity map
- metabolite_map      peak-integration map over a ppm band (no fit needed)
- metabolite_panel    grid of integration maps for several metabolites
- lcmodel_maps        conc + CRLB (+ optional SNR/FWHM) panels from a fit dict
                       (op_CSILCModelMaps equivalent)

imagesc(origin='lower'); apply flip180=True to view in overlay orientation.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ._common import to_fyx, integrate_band, water_map, make_mask, savefig

# default phantom / brain metabolite integration bands (ppm)
DEFAULT_BANDS = {
    'Lac': (1.25, 1.40), 'NAA': (1.95, 2.10), 'Act': (1.85, 1.95),
    'Cr': (2.95, 3.10), 'Cho': (3.15, 3.25), 'water': (4.4, 4.9),
}


def _flip180(m):
    return np.flip(np.flip(m, 0), 1)


def show_map(map2d, title='', cmap='viridis', mask=None, flip180=False,
             ax=None, out_path=None, clim=None):
    """imagesc a single 2D map."""
    m = np.asarray(map2d, float)
    if mask is not None:
        m = np.where(mask, m, np.nan)
    if flip180:
        m = _flip180(m)
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(m, cmap=cmap, origin='lower',
                   vmin=None if clim is None else clim[0],
                   vmax=None if clim is None else clim[1])
    ax.set_title(title); ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if own and out_path:
        savefig(fig, out_path); plt.close(fig); return out_path
    return ax


def water_map_fig(struct, mask=None, flip180=False, out_path=None):
    w = water_map(struct)
    if mask is None:
        mask = make_mask(w)
    return show_map(w, 'water map', cmap='gray', mask=mask, flip180=flip180, out_path=out_path)


def op_CSIintegrate(struct, ppmmin, ppmmax, mode='re'):
    """Faithful FID-A op_CSIintegrate: numeric [y,x] map, real-part sum over a
    STRICT (ppm>min & ppm<max) band. mode: 're' | 'im' | 'mag'."""
    d = to_fyx(struct['data'], struct['dims'])
    ppm = np.asarray(struct['ppm'])
    sel = (ppm > ppmmin) & (ppm < ppmmax)
    band = d[sel, :, :]
    if mode == 'im':
        return np.imag(band).sum(axis=0)
    if mode == 'mag':
        return np.abs(band).sum(axis=0)
    return np.real(band).sum(axis=0)


def metabolite_map(struct, band, name='', mode='mag', mask=None, flip180=False,
                   out_path=None):
    """Peak-integration map: integrate |spectrum| over `band` (lo,hi) ppm."""
    d = to_fyx(struct['data'], struct['dims'])
    ppm = np.asarray(struct['ppm'])
    m = integrate_band(d, ppm, band[0], band[1], mode=mode)
    if mask is None:
        mask = make_mask(water_map(struct))
    return show_map(m, name or f'{band[0]}-{band[1]} ppm', mask=mask,
                    flip180=flip180, out_path=out_path)


def metabolite_panel(struct, bands=None, mode='mag', mask=None, flip180=False,
                     out_path=None, title='peak-integration maps'):
    """Grid of integration maps for several metabolites."""
    bands = bands or DEFAULT_BANDS
    d = to_fyx(struct['data'], struct['dims'])
    ppm = np.asarray(struct['ppm'])
    if mask is None:
        mask = make_mask(water_map(struct))
    n = len(bands); ncol = min(n, 3); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.4 * nrow), squeeze=False)
    for i, (name, band) in enumerate(bands.items()):
        m = integrate_band(d, ppm, band[0], band[1], mode=mode)
        show_map(m, f'{name} ({band[0]}-{band[1]})', mask=mask, flip180=flip180,
                 ax=axes[i // ncol][i % ncol])
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    fig.suptitle(title)
    if out_path:
        savefig(fig, out_path); plt.close(fig); return out_path
    return fig


def lcmodel_maps(conc, crlb=None, snr=None, fwhm=None, mask=None, flip180=False,
                 crlb_cap=999, out_path=None, title='LCModel maps'):
    """Panels from fit results. conc/crlb/... = {metabolite: 2D array}.
    CRLB shown as %SD (voxels with CRLB>crlb_cap blanked). op_CSILCModelMaps equiv."""
    mets = list(conc.keys())
    rows = [('conc', conc)]
    if crlb: rows.append(('CRLB %SD', crlb))
    if snr:  rows.append(('S/N', snr))
    if fwhm: rows.append(('FWHM', fwhm))
    nrow = len(rows); ncol = len(mets)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.2 * nrow), squeeze=False)
    for r, (label, dct) in enumerate(rows):
        for c, met in enumerate(mets):
            m = np.asarray(dct.get(met, np.full_like(next(iter(conc.values())), np.nan)), float)
            if label.startswith('CRLB'):
                m = np.where(m > crlb_cap, np.nan, m)
            cmap = 'viridis' if r == 0 else ('inferno' if label.startswith('CRLB') else 'magma')
            show_map(m, f'{met} {label}', cmap=cmap, mask=mask, flip180=flip180,
                     ax=axes[r][c])
    fig.suptitle(title)
    if out_path:
        savefig(fig, out_path); plt.close(fig); return out_path
    return fig
