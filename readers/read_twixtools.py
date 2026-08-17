"""TWIX -> FID-A struct via twixtools.  Run: python read_twixtools.py
twixtools.map_twix gives a labelled k-space array; squeeze to [t,coils,x,y].
"""
import os, sys
import numpy as np
sys.path.insert(0, r'C:\Users\divya\Downloads\mrsi_pipeline\readers')
from to_fida import write_fida
import twixtools

DAT = r'F:\fida\divya\20260605_phantom_test\subject04\met\meas_MID00151_FID48095_csi_fid_24x24_isoctr.dat'
OUT = r'C:\Users\divya\Downloads\mrsi_pipeline\readers\out\fida_twixtools.mat'
os.makedirs(os.path.dirname(OUT), exist_ok=True)

mapped = twixtools.map_twix(DAT)
im = mapped[-1]['image']            # last measurement, image data
im.flagRemoveOS = False
arr = im[:].squeeze()               # labelled dims -> squeezed ndarray
# twixtools sqzDims tells the order; map Col/Cha/Lin/Seg -> [t,coils,x,y]
dims = list(im.non_singleton_dims) if hasattr(im, 'non_singleton_dims') else im.dims
print('twixtools dims', dims, 'shape', arr.shape)
# expected squeezed order (Siemens): [Lin, Seg, Cha, Col] or similar -> find axes
name = {n: i for i, n in enumerate([d for d in im.dims if im.sqzSize[im.dims.index(d)] > 1])} \
    if hasattr(im, 'sqzSize') else {}
# robust: identify by size (Col=2048/1024 largest, Cha=16, Lin=Seg=24)
order = np.argsort(arr.shape)[::-1]           # [Col, Cha, {Lin,Seg}, {Lin,Seg}]
col = order[0]; cha = order[1]; sp = [order[2], order[3]]
fids = np.transpose(arr, (col, cha, sp[0], sp[1]))     # [t,coils,x,y]
# timing from header
hdr = mapped[-1]['hdr']['MeasYaps']
dt = hdr[('sRXSPEC', 'alDwellTime', '0')] * 1e-9
te = hdr[('alTE', '0')] / 1000.0
try: tr = hdr[('alTR', '0')] / 1000.0
except Exception: tr = np.nan
txfrq = hdr[('sTXSPEC', 'asNucleusInfo', '0', 'lFrequency')]
write_fida(fids, dt, txfrq, te, tr, OUT)
