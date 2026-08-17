"""TWIX -> FID-A struct via spec2nii (NIfTI-MRS).  Run: python read_spec2nii.py
NB: spec2nii recons standard CSI, but for this CUSTOM csi_fid sequence it emits an
EMPTY geometry template (no data) -- so this path only works for sequences spec2nii
knows. Kept for completeness / standard data.
"""
import os, sys, subprocess, glob
import numpy as np, numpy.typing  # noqa
import nibabel as nib
sys.path.insert(0, r'C:\Users\divya\Downloads\mrsi_pipeline\readers')
from to_fida import write_fida

DAT = r'F:\fida\divya\20260605_phantom_test\subject04\met\meas_MID00151_FID48095_csi_fid_24x24_isoctr.dat'
OUTDIR = r'C:\Users\divya\Downloads\mrsi_pipeline\readers\out'
os.makedirs(OUTDIR, exist_ok=True)
S2N = r'C:\Users\divya\miniconda3\envs\fida\Scripts\spec2nii.exe'

subprocess.run([S2N, 'twix', '-e', 'image', '-f', 'csi_s2n', '-o', OUTDIR, DAT], check=False)
nii = sorted(glob.glob(os.path.join(OUTDIR, 'csi_s2n*.nii.gz')))
if not nii:
    raise SystemExit('spec2nii produced no NIfTI-MRS')
h = nib.load(nii[0]); d = np.asarray(h.dataobj)     # [x,y,z,t,(coils)]
print('spec2nii NIfTI-MRS', nii[0], 'shape', d.shape)
if d.size == 0 or not np.any(d):
    raise SystemExit('empty template (custom sequence not reconstructed by spec2nii)')
# NIfTI-MRS: [x,y,z,t,coils?] -> FID-A [t,coils,x,y]
dt = float(h.header['pixdim'][4]); import json
ext = json.loads(h.header.extensions[0].get_content())
txfrq = ext['SpectrometerFrequency'][0] * 1e6; te = ext.get('EchoTime', 0.0023) * 1000
tr = ext.get('RepetitionTime', np.nan); tr = tr * 1000 if tr == tr else np.nan
x, y, z, t = d.shape[:4]
coils = d.shape[4] if d.ndim > 4 else 1
d = d.reshape(x, y, z, t, coils)[:, :, 0]           # [x,y,t,coils]
fids = np.transpose(d, (2, 3, 0, 1))                # [t,coils,x,y]
write_fida(fids, dt, txfrq, te, tr, os.path.join(OUTDIR, 'fida_spec2nii.mat'))
