"""compare_rosette_ref.py -- step-by-step FID-A vs Python on the water ref.

Stages:
  s0 raw read   : pymapVBVD squeezed  vs  MATLAB mapVBVD squeezed
  s1 prep       : Python prep_noncartesian  vs  FID-A io_CSIload_twix_pair
  s2 recon (dft): Python op_CSIRecon  vs  FID-A op_CSIRecon

Reports, per stage: shape match, best magnitude corr, max|diff|/scale.
Also prints FID-A's dims for each stage (settles the Par interleave question).
Run AFTER export_rosette_ref_stages.m has written rosette_ref_stages\*.mat.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'mrsi'))
import os, warnings; warnings.filterwarnings('ignore')
import numpy as np, h5py

FIDA = r'C:\Users\divya\Downloads\mrsi_pipeline\rosette_ref_stages'
REF = r'F:\fida\divya\28thJULYpHANTOM40X40\subject01\mrs_ref\meas_MID01095_FID59456_XA60_RosetteSpinEcho_2_avg_8mste_4sTR_w.dat'
KF = r'C:\Users\divya\Downloads\fida codes\fid_a\processingTools\MRSI\kFiles\Rosette_traj_40x40.txt'


def h5arr(g):
    d = g[()]
    if d.dtype.names and 'real' in d.dtype.names:
        d = d['real'] + 1j * d['imag']
    return np.transpose(d, tuple(range(d.ndim)[::-1]))   # F->C = MATLAB sz order


def h5dims(f, path='dims'):
    out = {}
    grp = f[path]
    for k in grp:
        try: out[k] = int(np.ravel(grp[k][()])[0])
        except Exception: pass
    return out


def stats(a, b):
    """best magnitude corr over spatial flips + relative max diff (complex-aware)."""
    a = np.asarray(a); b = np.asarray(b)
    if a.shape != b.shape:
        return f'SHAPE MISMATCH {a.shape} vs {b.shape}', None
    am, bm = np.abs(a).ravel(), np.abs(b).ravel()
    corr = np.corrcoef(am, bm)[0, 1]
    scale = np.abs(b).max() or 1.0
    maxdiff = np.abs(a - b).max() / scale if a.dtype.kind == 'c' == b.dtype.kind else np.abs(am - bm).max() / scale
    return f'corr {corr:.4f}   max|diff|/scale {maxdiff:.2e}', corr


def align(mine, my_dims, fa_dims):
    """permute my array to FID-A's dim-name order."""
    names = [k for k, v in sorted(fa_dims.items(), key=lambda kv: kv[1]) if v]
    order = []
    for nm in names:
        if my_dims.get(nm):
            order.append(my_dims[nm] - 1)
    if len(order) == mine.ndim:
        return np.transpose(mine, order)
    return mine


print('=== STAGE 0: raw read (pymapVBVD vs MATLAB mapVBVD) ===')
from mapvbvd import mapVBVD
tw = mapVBVD(REF, quiet=True); im = tw[-1] if isinstance(tw, list) else tw
im.image.flagRemoveOS = False; im.image.squeeze = True
py_raw = np.asarray(im.image[''])
print('  pymapVBVD sqzDims:', im.image.sqzDims, py_raw.shape)
with h5py.File(os.path.join(FIDA, 's0_raw.mat'), 'r') as f:
    fa_raw = h5arr(f['raw'])
    print('  MATLAB sqzDims  :', [b.decode() if isinstance(b, bytes) else b for b in
          (f['sqzDims'][()] if 'sqzDims' in f else [])] or '(see below)', fa_raw.shape)
print('  ', stats(py_raw, fa_raw)[0])

print('\n=== STAGE 1: prep (Python vs FID-A load pair) ===')
from io_CSIload_twix import io_CSIload_twix_noncart
from op_CSIRosettePrep import prep_noncartesian
with h5py.File(os.path.join(FIDA, 's1_prep.mat'), 'r') as f:
    fa1 = h5arr(f['data']); fa1_dims = h5dims(f)
print('  FID-A prep dims:', fa1_dims, fa1.shape)
sref = io_CSIload_twix_noncart(REF, KF, seq_type='rosette')
print('  auto VOI shift (mm):', round(sref['xShift_mm'], 4), round(sref['yShift_mm'], 4))
sp = prep_noncartesian(sref, KF, seq_type='rosette')   # voi_shift=None -> auto computeFOVShift
print('  Python prep dims:', {k: v for k, v in sp['dims'].items() if v}, sp['sz'])
my1 = align(sp['data'], sp['dims'], fa1_dims)
print('  after align:', my1.shape)
print('  ', stats(my1, fa1)[0])

print('\n=== STAGE 2: recon dft (Python vs FID-A) ===')
from op_CSIRecon import op_CSIRecon
with h5py.File(os.path.join(FIDA, 's2_recon_dft.mat'), 'r') as f:
    fa2 = h5arr(f['data']); fa2_dims = h5dims(f)
print('  FID-A recon dims:', fa2_dims, fa2.shape)
rec = op_CSIRecon(sp, KF, dcfMethod='nn', ftMethod='dft')
print('  Python recon dims:', {k: v for k, v in rec['dims'].items() if v}, rec['sz'])
my2 = align(rec['data'], rec['dims'], fa2_dims)
print('  after align:', my2.shape)
print('  ', stats(my2, fa2)[0])
