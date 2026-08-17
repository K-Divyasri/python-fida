"""to_fida.py -- write a FID-A-compatible MRSI struct to .mat.

Input: fids as [t, coils, x, y] (time-domain, k-space in x/y = kx/ky, uncombined).
Output: a .mat holding a struct `fida` with the fields FID-A's op_* functions
expect (dims are MATLAB 1-indexed; 0 = absent).
Load in MATLAB:  load('out.mat')  ->  fida.fids, fida.dims, ...
"""
import numpy as np
from scipy.io import savemat


def write_fida(fids, dt, txfrq_hz, te_ms, tr_ms, path, seq='csi_fid'):
    fids = np.asarray(fids)                       # [t, coils, x, y]
    Nt = fids.shape[0]
    specs = np.fft.fftshift(np.fft.fft(fids, axis=0), axes=0)
    sw = 1.0 / dt
    ppm = np.fft.fftshift(np.fft.fftfreq(Nt, dt)) / -(txfrq_hz / 1e6) + 4.65
    t = np.arange(Nt) * dt
    fida = dict(
        fids=fids, specs=specs,
        sz=np.array(fids.shape, float),
        dims=dict(t=1, coils=2, x=3, y=4, z=0, averages=0, subSpecs=0, extras=0),
        spectralwidth=float(sw), dwelltime=float(dt),
        txfrq=float(txfrq_hz), Bo=float(txfrq_hz / 42.577478e6),
        te=float(te_ms), tr=float(tr_ms), n=int(Nt),
        ppm=ppm, t=t, seq=seq, date='', sim='', pointsToLeftshift=0,
        flags=dict(writtentostruct=1, gotparams=1, leftshifted=0, filtered=0,
                   zeropadded=0, freqcorrected=0, phasecorrected=0, averaged=0,
                   addedrcvrs=0, subtracted=0, writtentotext=0, downsampled=0,
                   isFourSteps=0),
    )
    savemat(path, {'fida': fida}, do_compression=True)
    print('wrote FID-A struct', path, 'fids', fids.shape,
          f'SW={sw:.0f} txfrq={txfrq_hz/1e6:.4f}MHz TE={te_ms} TR={tr_ms}')
    return path
