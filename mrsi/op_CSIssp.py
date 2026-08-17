"""op_CSIssp.py -- SVD subspace projection lipid suppression (FID-A op_CSIssp).

Removes the top-m spatial components that dominate the [minppm, maxppm] band:
P = I - Um Um^H applied to the [y*x, spec] data matrix. (Skip for phantoms -- the
0.8-1.88 ppm band would remove lactate.)
"""
import copy
import numpy as np
from mrsi_common import _ax, _faxis


def op_CSIssp(struct, minppm, maxppm, m=6, useEcon=False):
    if not struct['flags'].get('spectralft'):
        raise RuntimeError('op_CSIssp: needs spectral FT')
    s = copy.deepcopy(struct)
    af = _faxis(s); ay = _ax(s, 'y'); ax = _ax(s, 'x')
    d = np.transpose(s['data'], (ay, ax, af))                  # -> [ny, nx, nspec]
    ny, nx, nspec = d.shape
    doricol = d.reshape(ny * nx, nspec)                        # [y*x, spec]
    ppm = np.asarray(s['ppm'])
    endidx = int(np.argmin(np.abs(ppm - minppm)))
    startidx = int(np.argmin(np.abs(ppm - maxppm)))
    lo, hi = min(startidx, endidx), max(startidx, endidx)
    dorilip = doricol[:, lo:hi + 1]                            # lipid-band slice
    U, _, _ = np.linalg.svd(dorilip, full_matrices=not useEcon)
    Um = U[:, :m]
    P = np.eye(Um.shape[0]) - Um @ Um.conj().T                 # [y*x, y*x] spatial projection
    dsup = (P @ doricol).reshape(ny, nx, nspec)
    s['data'] = np.transpose(dsup, np.argsort((ay, ax, af)))   # back to original order
    return s
