"""mrsi_common.py -- shared helpers for the individual FID-A MRSI function files.

One-function-per-file (FID-A style) means the small dim/axis helpers live here so
every op_* file imports them from one place. Reuses _coords / read_kfile from
op_CSIRecon (recon has no dependency on this module -> no import cycle).
"""
import copy  # noqa: F401  (re-exported for convenience)
import numpy as np
from op_CSIRecon import _coords, read_kfile  # noqa: F401  (shared, single source)

TWO_PI = 2 * np.pi


def _ax(s, name):
    """0-based axis index of a dim, or None if absent (dims are 1-based)."""
    v = s['dims'].get(name, 0)
    return (v - 1) if v else None


def _faxis(s):
    """0-based spectral (f) / time (t) axis; defaults to axis 0."""
    return (s['dims'].get('f') or s['dims'].get('t') or 1) - 1


def _fyx(s):
    """View of data as [f, y, x] plus the permutation order used."""
    dims = s['dims']
    fax = (dims.get('f') or dims.get('t')) - 1
    ya, xa = dims['y'] - 1, dims['x'] - 1
    order = (fax, ya, xa)
    return np.transpose(s['data'], order), order


def _drop_dim(dims, name):
    """Remove a dim: zero it, decrement every dim with a higher axis index."""
    ax = dims.get(name, 0)
    out = {}
    for k, v in dims.items():
        if k == name:
            out[k] = 0
        elif v and ax and v > ax:
            out[k] = v - 1
        else:
            out[k] = v
    return out


def _spatial_shape(s, af):
    """Shape of the data with the spectral axis `af` removed (e.g. (Ny, Nx))."""
    return tuple(n for i, n in enumerate(s['data'].shape) if i != af)


def _gaussian_negexp(x, sig2):
    """FID-A gaussian(x, sigma) = -exp(x^2 / (2 sigma^2)); sig2 = sigma^2 (negative)."""
    return -np.exp(x ** 2 / (2 * sig2))
