"""TWIX -> FID-A struct via suspect.  Run: python read_suspect.py
suspect.io.load_twix returns an MRSData; reshape its k-space to [t,coils,x,y].
"""
import os, sys
import numpy as np
sys.path.insert(0, r'C:\Users\divya\Downloads\mrsi_pipeline\readers')
from to_fida import write_fida
import suspect.io

DAT = r'F:\fida\divya\20260605_phantom_test\subject04\met\meas_MID00151_FID48095_csi_fid_24x24_isoctr.dat'
OUT = r'C:\Users\divya\Downloads\mrsi_pipeline\readers\out\fida_suspect.mat'
os.makedirs(os.path.dirname(OUT), exist_ok=True)

data = suspect.io.load_twix(DAT)         # MRSData; .np array carries the dims
arr = np.asarray(data)
print('suspect shape', arr.shape, 'dt', data.dt, 'f0', data.f0)
# suspect returns time on last axis; other axes are spatial/coil. Move time first,
# then order remaining by size -> coils, x, y.
tax = arr.ndim - 1
rest = [a for a in range(arr.ndim) if a != tax]
rest_sorted = sorted(rest, key=lambda a: -arr.shape[a])    # coils(16), x(24), y(24)
fids = np.transpose(arr, (tax, *rest_sorted))              # [t,coils,x,y]
dt = data.dt; txfrq = data.f0 * 1e6
te = getattr(data, 'te', 2.3); tr = getattr(data, 'tr', np.nan)
write_fida(fids, dt, txfrq, te, tr, OUT)
