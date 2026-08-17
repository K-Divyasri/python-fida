"""io_CSIload_twix.py -- FID-A-style CSI TWIX loader (Python).

Mirrors FID-A's io_CSIload_twix / MRSIStruct: reads a Siemens TWIX .dat with
pymapVBVD and returns a struct (dict) with the SAME fields/layout FID-A uses, so
you can compare field-by-field against the MATLAB output.

MRSIStruct fields (matching FID-A):
    data          complex ndarray, layout [t, coils, kx, ky]   (FID-A sz order)
    sz            tuple(shape)
    dims          {name: 1-based axis, 0 if absent}  -- FID-A dim convention
    spectralWidth Hz            (= 1/dwelltime)
    dwelltime     s
    spectralTime  ndarray, s    (0 .. (Nt-1)*dwelltime)
    adcTime       alias of spectralTime
    txfrq         Hz            (transmit frequency)
    te, tr        ms
    Bo            T             (= txfrq / 42.577e6)
    ppm           None until spectral FT (placeholder; FID-A fills after FT)
    fov           {x,y,z} mm ; voxelSize {x,y,z} mm ; imageOrigin [3] mm ; normal [3]
    flags         {writtentostruct, gotparams, addedrcvrs, spatialft,
                   spectralft, averaged, isCartesian, ...}
    seq, te_source, filename

Usage:
    from io_CSIload_twix import io_CSIload_twix, io_CSIload_twix_pair, save_struct
    m = io_CSIload_twix(metFile)                 # single
    m, w = io_CSIload_twix_pair(metFile, refFile)
    save_struct(m, 'py_load_met.mat')            # -> compare in MATLAB/Python
"""
import os
import numpy as np
from mapvbvd import mapVBVD

GAMMA_1H = 42.5774e6   # Hz/T


def _yaps(my, *keys, default=None):
    """Fetch from the flat MeasYaps dict (tuple-of-string keys)."""
    key = tuple(str(k) for k in keys)
    try:
        v = my[key]
        return v if v is not None else default
    except (KeyError, TypeError):
        return default


def io_CSIload_twix(filename, remove_os=False, txfrq_override=None):
    """Load one CSI TWIX .dat -> FID-A-style MRSIStruct (dict).

    txfrq_override: set the transmit freq (Hz) explicitly. pymapVBVD reports the
    raw transmit freq (~123.2528 MHz); MATLAB mapVBVD/FID-A report ~123.1988 MHz
    (an internal-adjusted value NOT present in the raw header). For basis-matched
    fitting, pass the basis centre freq here (e.g. 123198765) to match FID-A."""
    tw = mapVBVD(filename, quiet=True)
    im = tw[-1] if isinstance(tw, list) else tw
    im.image.flagRemoveOS = bool(remove_os)
    im.image.squeeze = True

    data = im.image['']                      # squeezed k-space
    sqz = list(im.image.sqzDims)             # e.g. ['Col','Cha','Lin','Seg']

    # --- map Siemens loop names -> FID-A dim names ---
    name_map = {'Col': 't', 'Cha': 'coils', 'Lin': 'kx', 'Seg': 'ky',
                'Ave': 'averages', 'Set': 'subspec', 'Par': 'kz', 'Rep': 'extras'}
    fida_order = ['t', 'coils', 'averages', 'timeinterleave', 'kx', 'ky', 'kz',
                  'x', 'y', 'z', 'kpts', 'kshot', 'subspec', 'extras', 'f']
    # build dims: 1-based axis per FID-A name (0 = absent)
    axis_of = {}
    for ax, nm in enumerate(sqz):
        axis_of[name_map.get(nm, nm.lower())] = ax + 1     # 1-based (MATLAB)
    dims = {nm: axis_of.get(nm, 0) for nm in fida_order}

    my = im.hdr.MeasYaps
    dwell = float(_yaps(my, 'sRXSPEC', 'alDwellTime', '0', default=250000)) * 1e-9   # ns->s
    if remove_os:
        dwell *= 2.0                          # OS removal halves points, doubles dwell
    te = float(_yaps(my, 'alTE', '0', default=2300)) / 1000.0                        # us->ms
    tr = float(_yaps(my, 'alTR', '0', default=720000)) / 1000.0                       # us->ms
    # txfrq: FID-A uses twix.hdr.Meas.lFrequency (NOT MeasYaps sTXSPEC.lFrequency,
    # which is the transmit freq ~54 kHz / 0.44 ppm higher -> a referencing shift).
    txfrq = np.nan
    meas = getattr(im.hdr, 'Meas', None)
    for acc in ('__getitem__', 'get', 'attr'):
        try:
            if acc == '__getitem__':
                txfrq = float(meas['lFrequency'])
            elif acc == 'get':
                txfrq = float(meas.get('lFrequency'))
            else:
                txfrq = float(getattr(meas, 'lFrequency'))
            if np.isfinite(txfrq):
                break
        except Exception:
            continue
    if not np.isfinite(txfrq):     # fallback: MeasYaps transmit freq
        txfrq = float(_yaps(my, 'sTXSPEC', 'asNucleusInfo', '0', 'lFrequency', default=np.nan))
    if txfrq_override is not None:
        txfrq = float(txfrq_override)   # match FID-A / basis frequency
    Nt = data.shape[0]

    # --- geometry (sSliceArray.asSlice{0}) ---
    g = lambda *k: _yaps(my, 'sSliceArray', 'asSlice', '0', *k, default=0.0)
    fov = {'x': float(g('dReadoutFOV')), 'y': float(g('dPhaseFOV')), 'z': float(g('dThickness'))}
    pos = [float(g('sPosition', 'dSag')), float(g('sPosition', 'dCor')), float(g('sPosition', 'dTra'))]
    nrm = [float(g('sNormal', 'dSag')), float(g('sNormal', 'dCor')), float(g('sNormal', 'dTra'))]
    nx = dims['kx'] and data.shape[dims['kx'] - 1] or 1
    ny = dims['ky'] and data.shape[dims['ky'] - 1] or 1
    vox = {'x': fov['x'] / nx if nx else fov['x'], 'y': fov['y'] / ny if ny else fov['y'], 'z': fov['z']}

    struct = {
        'data': np.asarray(data),
        'sz': tuple(data.shape),
        'dims': dims,
        'spectralWidth': 1.0 / dwell,
        'dwelltime': dwell,
        'spectralTime': np.arange(Nt) * dwell,
        'adcTime': np.arange(Nt) * dwell,
        'txfrq': txfrq,
        'te': te, 'tr': tr,
        'Bo': (txfrq / GAMMA_1H) if np.isfinite(txfrq) else np.nan,
        'ppm': None,                          # filled after spectral FT (FID-A convention)
        'fov': fov, 'voxelSize': vox, 'imageOrigin': pos, 'normal': nrm,
        'seq': str(_yaps(my, 'tSequenceString', default='')).strip("'"),
        'filename': os.path.abspath(filename),
        'flags': {
            'writtentostruct': 1, 'gotparams': 1, 'leftshifted': 0, 'filtered': 0,
            'zeropadded': 0, 'freqcorrected': 0, 'phasecorrected': 0,
            'addedrcvrs': 0, 'subtracted': 0, 'averaged': 0,
            'spatialft': 0, 'spectralft': 0, 'isCartesian': 1,
        },
    }
    return struct


def _compute_fov_shift(my):
    """Port of FID-A computeFOVShift: VOI off-centre (mm) -> op_CSIShift deltas.
    Reads sSpecPara.sVoI.sNormal / .sPosition."""
    v = lambda *k: float(_yaps(my, 'sSpecPara', 'sVoI', *k, default=0.0))
    nSag, nCor, nTra = v('sNormal', 'dSag'), v('sNormal', 'dCor'), v('sNormal', 'dTra')
    normal = [nSag, nCor, nTra]
    idx = int(np.argmax(np.abs(normal)))                 # 0 Sag,1 Cor,2 Tra
    theta = np.deg2rad(np.degrees(np.arccos(min(abs(normal[idx]), 1.0))))
    pSag, pCor, pTra = v('sPosition', 'dSag'), v('sPosition', 'dCor'), v('sPosition', 'dTra')
    if idx == 2:      # Axial
        return pTra * np.sin(theta), pTra * np.sin(theta)
    if idx == 0:      # Sagittal
        return pSag * np.cos(theta), pSag * np.sin(theta)
    return pCor * np.sin(theta), pCor * np.cos(theta)     # Coronal


def io_CSIload_twix_noncart(filename, kfile, seq_type='rosette', remove_os=False,
                            txfrq_override=None, dim_map=None):
    """Load a NON-cartesian (rosette / concentric-ring) CSI twix -> FID-A struct
    laid out [t, coils, (averages), kshot, (extras)], ready for op_CSIRosettePrep
    (combine_time -> split_readout_kpts) then op_CSIRecon.

    Requires the raw .dat to be STANDARD twix (VD/VE) that pymapVBVD can read.
    dwelltime + kPtsPerCycle come from the kFile (FID-A chooseDwellTime: kFile
    time column overrides the header); txfrq from header lFrequency.

    dim_map overrides the Siemens-label -> FID-A-name mapping. Defaults, from
    FID-A io_CSIload_twix_pair buildDims + this XA rosette's counters (Lin=petals):
        rosette:    Col->t Cha->coils Ave->averages Lin->kshot Set->kshot
                    Par->extras Eco->extras
        concentric: Col->t Cha->coils Lin->kshot Seg->kshot Par->extras"""
    import pandas as pd
    tw = mapVBVD(filename, quiet=True)
    im = tw[-1] if isinstance(tw, list) else tw
    im.image.flagRemoveOS = bool(remove_os)
    im.image.squeeze = True
    data = im.image['']
    sqz = list(im.image.sqzDims)

    if dim_map is None:
        if seq_type == 'concentric':
            dim_map = {'Col': 't', 'Cha': 'coils', 'Lin': 'kshot', 'Seg': 'kshot',
                       'Ave': 'averages', 'Par': 'extras', 'Eco': 'extras'}
        else:  # rosette
            dim_map = {'Col': 't', 'Cha': 'coils', 'Lin': 'kshot', 'Set': 'kshot',
                       'Ave': 'averages', 'Par': 'extras', 'Eco': 'extras'}
    fida_order = ['t', 'coils', 'averages', 'timeinterleave', 'kx', 'ky', 'kz',
                  'x', 'y', 'z', 'kpts', 'kshot', 'subspec', 'extras', 'f']
    axis_of = {}
    for ax, nm in enumerate(sqz):
        axis_of[dim_map.get(nm, nm.lower())] = ax + 1
    dims = {nm: axis_of.get(nm, 0) for nm in fida_order}

    # --- kFile: dwell + kPtsPerCycle (authoritative for non-cartesian) ---
    kdf = pd.read_csv(kfile)
    kcols = {c.lower(): c for c in kdf.columns}
    nshots = int(kdf[kcols['tr']].max()) if 'tr' in kcols else 1
    kPtsPerCycle = len(kdf) // nshots
    if 'time' in kcols:
        t = kdf[kcols['time']].to_numpy(float)
        dwell = float(t[2] - t[1]) if len(t) > 2 else float(t[1] - t[0])
    else:
        my0 = im.hdr.MeasYaps
        dwell = float(_yaps(my0, 'sRXSPEC', 'alDwellTime', '0', default=250000)) * 1e-9

    my = im.hdr.MeasYaps
    te = float(_yaps(my, 'alTE', '0', default=0)) / 1000.0
    tr = float(_yaps(my, 'alTR', '0', default=0)) / 1000.0
    txfrq = np.nan
    meas = getattr(im.hdr, 'Meas', None)
    for acc in ('__getitem__', 'get', 'attr'):
        try:
            txfrq = float(meas['lFrequency'] if acc == '__getitem__' else
                          meas.get('lFrequency') if acc == 'get' else getattr(meas, 'lFrequency'))
            if np.isfinite(txfrq):
                break
        except Exception:
            continue
    if not np.isfinite(txfrq):
        txfrq = float(_yaps(my, 'sTXSPEC', 'asNucleusInfo', '0', 'lFrequency', default=np.nan))
    if txfrq_override is not None:
        txfrq = float(txfrq_override)

    g = lambda *k: _yaps(my, 'sSliceArray', 'asSlice', '0', *k, default=0.0)
    fov = {'x': float(g('dReadoutFOV')), 'y': float(g('dPhaseFOV')), 'z': float(g('dThickness'))}
    # non-cartesian recon grid from kFile Kmax: Nx = round(2*FOV*Kmax); vox = FOV/Nx
    kmax = float(np.max(np.hypot(kdf[kcols.get('kx', kdf.columns[1])].to_numpy(float),
                                 kdf[kcols.get('ky', kdf.columns[2])].to_numpy(float))))
    Nx = max(int(round(2 * fov['x'] * kmax)), 1) if (fov['x'] and kmax) else 1
    Ny = max(int(round(2 * fov['y'] * kmax)), 1) if (fov['y'] and kmax) else 1
    voxel = {'x': fov['x'] / Nx, 'y': fov['y'] / Ny, 'z': fov['z']}
    pos = [float(g('sPosition', 'dSag')), float(g('sPosition', 'dCor')), float(g('sPosition', 'dTra'))]
    nrm = [float(g('sNormal', 'dSag')), float(g('sNormal', 'dCor')), float(g('sNormal', 'dTra'))]
    Nt = data.shape[dims['t'] - 1]
    struct = {
        'data': np.asarray(data), 'sz': tuple(data.shape), 'dims': dims,
        'spectralWidth': 1.0 / dwell, 'dwelltime': dwell,
        'adcDwellTime': dwell,                       # pre-split ADC dwell
        'spectralTime': np.arange(Nt) * dwell, 'adcTime': np.arange(Nt) * dwell,
        'txfrq': txfrq, 'te': te, 'tr': tr,
        'Bo': (txfrq / GAMMA_1H) if np.isfinite(txfrq) else np.nan, 'ppm': None,
        'fov': fov, 'voxelSize': voxel, 'reconMatrix': (Ny, Nx),
        'imageOrigin': pos, 'normal': nrm, 'kPtsPerCycle': kPtsPerCycle, 'nShots': nshots,
        'xShift_mm': _compute_fov_shift(my)[0], 'yShift_mm': _compute_fov_shift(my)[1],
        'seq': str(_yaps(my, 'tSequenceString', default='')).strip("'"),
        'filename': os.path.abspath(filename), 'seq_type': seq_type,
        'flags': {'writtentostruct': 1, 'gotparams': 1, 'addedrcvrs': 0,
                  'spatialft': 0, 'spectralft': 0, 'isCartesian': 0},
    }
    return struct


def io_CSIload_twix_pair(metFile, refFile, kfile=None, seq_type='rosette', remove_os=False,
                         txfrq_override=None):
    """Load metabolite + water-reference pair (FID-A io_CSIload_twix_pair).
    kfile=None -> cartesian; kfile given -> non-cartesian (rosette/concentric)."""
    if kfile in (None, ''):
        return io_CSIload_twix(metFile, remove_os), io_CSIload_twix(refFile, remove_os)
    return (io_CSIload_twix_noncart(metFile, kfile, seq_type, remove_os, txfrq_override),
            io_CSIload_twix_noncart(refFile, kfile, seq_type, remove_os, txfrq_override))


def print_struct(s, name='MRSIStruct'):
    print(f'=== {name} ===')
    print(f'  data {s["data"].shape} {s["data"].dtype}   sz {s["sz"]}')
    print('  dims:', {k: v for k, v in s['dims'].items() if v})
    print(f'  spectralWidth {s["spectralWidth"]:.1f} Hz   dwelltime {s["dwelltime"]:.3e} s')
    print(f'  txfrq {s["txfrq"]:.1f} Hz   Bo {s["Bo"]:.3f} T   te {s["te"]} ms   tr {s["tr"]} ms')
    print(f'  fov {s["fov"]}   voxelSize {s["voxelSize"]}')
    print(f'  imageOrigin {s["imageOrigin"]}   normal {s["normal"]}   seq {s["seq"]}')
    print('  flags:', {k: v for k, v in s['flags'].items() if v})


def save_struct(s, path):
    """Save to .mat (v7) for side-by-side comparison with FID-A's MRSIStruct."""
    from scipy.io import savemat
    out = {k: v for k, v in s.items() if k not in ('dims', 'flags', 'fov', 'voxelSize', 'ppm')}
    out['dims'] = s['dims']; out['flags'] = s['flags']
    out['fov'] = s['fov']; out['voxelSize'] = s['voxelSize']
    if s['ppm'] is not None:
        out['ppm'] = s['ppm']
    savemat(path, {'py': out}, do_compression=True)
    print('saved', path)


if __name__ == '__main__':
    import sys
    met = sys.argv[1] if len(sys.argv) > 1 else None
    s = io_CSIload_twix(met)
    print_struct(s, 'PYTHON load (met)')
