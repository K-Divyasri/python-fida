"""viz/lcmodel_read.py -- parse LCModel .table output into maps (op_CSILCModelMaps).

Reads all f<yy>x<xx>.table files in a folder; extracts per-voxel concentration,
CRLB (%SD), FWHM (ppm) and S/N from the $$CONC and $$MISC blocks; lays them into
[ny, nx] maps keyed by metabolite. Mirrors FID-A op_CSILCModelMaps (and fixes its
zeros(size_x,size_y) allocation bug -> allocates [ny,nx]).

Voxel index parsed from filename: 'f01x02...' -> x=1, y=2  (FID-A convention).
"""
import os, re, glob
import numpy as np

# FID-A default matched metabolites (field name = '+' removed)
DEFAULT_METS = ['Cr+PCr', 'GPC+PCh', 'NAA+NAAG', 'Glu+Gln', 'Ins']


def _parse_table(path, mets):
    """Return dict: {met: (conc, crlb%)} + ('_LW', fwhm_ppm), ('_SN', snr)."""
    txt = open(path, errors='ignore').read()
    out = {}
    if '$$CONC' not in txt and 'Conc.' not in txt:
        return out                                    # LCModel FATAL -> skip
    for line in txt.splitlines():
        for met in mets:
            # a metabolite row: "<conc>  <%SD>%   <conc/ref>   <name>"
            if line.strip().endswith(met) or f' {met}' in line:
                toks = line.split()
                try:
                    conc = float(toks[0]); crlb = float(toks[1].replace('%', ''))
                    out[met] = (conc, crlb)
                except (ValueError, IndexError):
                    pass
    m = re.search(r'FWHM\s*=\s*([\d.]+)\s*ppm', txt)
    if m: out['_LW'] = float(m.group(1))
    m = re.search(r'S/N\s*=\s*([\d.]+)', txt)
    if m: out['_SN'] = float(m.group(1))
    return out


def parse_lcmodel_tables(folder, size_x, size_y, mets=None):
    """-> (conc, crlb, LW, SNR) dicts of [ny,nx] arrays."""
    mets = mets or DEFAULT_METS
    fields = {m.replace('+', ''): m for m in mets}
    conc = {k: np.full((size_y, size_x), np.nan) for k in fields}
    crlb = {k: np.full((size_y, size_x), np.nan) for k in fields}
    LW = np.full((size_y, size_x), np.nan)
    SNR = np.full((size_y, size_x), np.nan)
    for path in glob.glob(os.path.join(folder, '*.table')):
        name = os.path.basename(path)
        m = re.search(r'f?(\d+)x(\d+)', name)
        if not m:
            continue
        x = int(m.group(1)) - 1; y = int(m.group(2)) - 1
        if not (0 <= x < size_x and 0 <= y < size_y):
            continue
        vals = _parse_table(path, mets)
        for field, met in fields.items():
            if met in vals:
                conc[field][y, x], crlb[field][y, x] = vals[met]
        if '_LW' in vals: LW[y, x] = vals['_LW']
        if '_SN' in vals: SNR[y, x] = vals['_SN']
    return conc, crlb, LW, SNR
