"""op_CSIRemoveLipids.py -- L2 lipid/water removal (FID-A op_CSIRemoveLipids).

data <- inv(I + beta*B*B^H) @ data along the spectral (f) axis. Pass lipidBasis
to reuse a fixed basis (e.g. FID-A's) for a bit-exact match; the SOLVE is exact,
only the basis is stochastic in FID-A.
"""
import copy
import numpy as np
from mrsi_common import _faxis, _spatial_shape
from make_lipid_basis import make_lipid_basis


def op_CSIRemoveLipids(struct, lipidComponents=1000, lineWidthRange=(1, 80),
                       lipidPPMRange=(0.3, 1.9), beta=1e-4, lipidBasis=None, rng=None):
    s = copy.deepcopy(struct)
    af = _faxis(s)
    Nf = s['data'].shape[af]
    sh = _spatial_shape(s, af)
    if lipidBasis is None:
        lipidBasis = make_lipid_basis(Nf, s['spectralWidth'], lipidComponents,
                                      lineWidthRange, lipidPPMRange,
                                      s.get('txfrq', 123.25e6) / 1e6, rng=rng)
    L2 = np.linalg.inv(np.eye(Nf) + beta * (lipidBasis @ lipidBasis.conj().T))
    d = np.moveaxis(s['data'], af, 0).reshape(Nf, -1, order='F')
    d = L2 @ d
    s['data'] = np.moveaxis(d.reshape((Nf,) + sh, order='F'), 0, af)
    s['lipidBasis'] = lipidBasis
    return s
