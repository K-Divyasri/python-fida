"""TWIX -> FID-A struct via pymapVBVD (mapvbvd).  Run: python read_pymapvbvd.py
Reorders k-space to [t, coils, x(=kx/seg), y(=ky/lin)] and writes a FID-A .mat.
"""
import os, sys
import numpy as np
sys.path.insert(0, r'C:\Users\divya\Downloads\mrsi_pipeline\mrsi_pipeline')
from read_twix import read_twix          # pymapVBVD wrapper
from to_fida import write_fida

DAT = r'F:\fida\divya\20260605_phantom_test\subject04\met\meas_MID00151_FID48095_csi_fid_24x24_isoctr.dat'
OUT = r'C:\Users\divya\Downloads\mrsi_pipeline\readers\out\fida_pymapvbvd.mat'
os.makedirs(os.path.dirname(OUT), exist_ok=True)

r = read_twix(DAT, remove_os=False)
k, d, t = r['kdata'], r['dims'], r['timing']       # k: [Col,Cha,Lin,Seg]
fids = np.transpose(k, (d['col'], d['cha'], d['seg'], d['lin']))   # [t,coils,x,y]
txfrq = t['txfrq_hz'] if not np.isnan(t.get('txfrq_hz', np.nan)) else 123.25e6
write_fida(fids, t['dwell_s'], txfrq, t['te_ms'], t.get('tr_ms', np.nan), OUT)
