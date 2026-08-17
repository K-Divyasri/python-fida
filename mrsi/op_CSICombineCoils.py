"""op_CSICombineCoils.py -- compatibility shim.

The coil-combine / average / segment functions now live in individual files
(FID-A one-function-per-file style). This re-exports them so existing imports
`from op_CSICombineCoils import op_CSICombineCoils1, op_CSIAverage, op_CSISegment_simple`
keep working.
"""
from op_CSICombineCoils1 import op_CSICombineCoils1
from op_CSIAverage import op_CSIAverage
from op_CSISegment_simple import op_CSISegment_simple
from mrsi_common import _ax, _drop_dim  # noqa: F401  (legacy helper imports)

__all__ = ['op_CSICombineCoils1', 'op_CSIAverage', 'op_CSISegment_simple']
