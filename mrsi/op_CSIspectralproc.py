"""op_CSIspectralproc.py -- compatibility shim.

Spectral-domain ops now live in individual files (FID-A one-function-per-file).
Re-exported so existing imports keep working.
"""
from op_CSIB0Correction_v2 import op_CSIB0Correction_v2
from op_CSIRemoveLipids import op_CSIRemoveLipids
from op_CSIspecZeroFill import op_CSIspecZeroFill
from op_CSIphase import op_CSIphase
from hsvd_water_removal import hsvd_water_removal
from make_lipid_basis import make_lipid_basis
from mrsi_common import _faxis, _fyx, _spatial_shape, TWO_PI  # noqa: F401  (legacy)

__all__ = ['op_CSIB0Correction_v2', 'op_CSIRemoveLipids', 'op_CSIspecZeroFill',
           'op_CSIphase', 'hsvd_water_removal', 'make_lipid_basis']
