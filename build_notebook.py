"""build_notebook.py -- generate run_rosette_pipeline.ipynb.

FID-A-style notebook: adds the mrsi/ folder to sys.path (== addpath(genpath)),
imports each op_ function individually, and calls them one-by-one exactly like
run_MRSI_Rosette_40x40.m. Output stored in the dataset's outputs/ folder.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("""# Rosette MRSI pipeline (pure-Python FID-A port)

Mirrors `run_MRSI_Rosette_40x40.m` step-by-step. The `mrsi/` folder is added to
`sys.path` (the `addpath(genpath(...))` equivalent) so every `op_*.py` function is
importable by name, then called one at a time. Outputs land in the dataset's
`outputs/` folder, the same place FID-A writes.

**Config knobs** (below): recon `DCF`/`FT`, `SMOOTH_FWHM`, `PHANTOM` (skip SSP),
`ZEROFILL` (None = NUNFIL 576 = matches the FID-A subject02 RAW; 4096 to interpolate).""")

code("""# --- setup: put the entire mrsi/ folder on the path, import each function ---
import os, sys
import numpy as np

ROOT     = r'C:\\Users\\divya\\Downloads\\mrsi_pipeline'
MRSI_DIR = os.path.join(ROOT, 'mrsi')
sys.path.insert(0, MRSI_DIR)   # addpath(genpath(mrsi))  -> every op_*.py importable
sys.path.insert(0, ROOT)       # for fit_lcmodel_rosette + viz

from io_CSIload_twix     import io_CSIload_twix_pair
from op_CSIRosettePrep   import prep_noncartesian
from op_CSIRecon         import op_CSIRecon
from op_CSICombineCoils1 import op_CSICombineCoils1
from op_CSIAverage       import op_CSIAverage
from op_CSISegment_simple import op_CSISegment_simple
from op_CSIleftshift     import op_CSIleftshift
from op_CSIFourierTransform import op_CSIFourierTransform
from op_CSIssp           import op_CSIssp
from op_CSIRemoveLipids  import op_CSIRemoveLipids
from op_CSIB0Correction_v2 import op_CSIB0Correction_v2
from op_CSIspecZeroFill  import op_CSIspecZeroFill
from op_CSIapplymask     import op_CSIapplymask
from op_CSIApodize       import op_CSIApodize
from op_CSIFlip180       import op_CSIFlip180
from fit_lcmodel_rosette import fit_maps
from viz._common         import to_fyx
print('imported', len([f for f in dir() if f.startswith('op_')]), 'op_ functions')""")

code("""# --- USER INPUTS (match run_MRSI_Rosette_40x40.m) ---
DATASET = r'F:\\fida\\divya\\20260605_phantom_test\\subject02'
MET   = os.path.join(DATASET, 'met',     'meas_MID00138_FID48082_Rosette_40x40_isoctr.dat')
REF   = os.path.join(DATASET, 'mrs_ref', 'meas_MID00139_FID48083_Rosette_40x40_isoctr_w.dat')
KFILE = r'C:\\Users\\divya\\Downloads\\fida codes\\fid_a\\processingTools\\MRSI\\kFiles\\Rosette_traj_40x40.txt'

# output stored in the SAME dataset location as FID-A (outputs/); _py suffix keeps
# FID-A's own lcm_out intact -- drop the suffix to write into FID-A's exact folder.
OUT = os.path.join(DATASET, 'outputs', 'lcm_out_py')
os.makedirs(OUT, exist_ok=True)

# reconstruction + processing choices
DCF, FT      = 'pipe_menon', 'nufft'   # FID-A production config
SMOOTH_FWHM  = 20                       # spatial Gaussian FWHM (mm)
PHANTOM      = True                     # phantom -> SKIP SSP (0.8-1.88 band removes Lac)
ZEROFILL     = None                     # None = NUNFIL 576 (matches subject02 RAW); 4096 to interpolate
DO_LCMODEL   = True                     # set False to stop after apodize (fast)
print('dataset:', DATASET, '\\noutput ->', OUT)""")

md("## 1. Load TWIX pair  (`io_CSIload_twix_pair`)")
code("""tc, tc_w = io_CSIload_twix_pair(MET, REF, KFILE, 'rosette')
print('met', tc['data'].shape, tc['dims'])""")

md("""## 1b. Rosette prep  (`prep_noncartesian`)

Reshape the raw readout into `[t, coils, avg, kpts, kshot]` (adds the kpts/kshot
trajectory dims recon needs). FID-A's loader returns this already-reshaped; the
Python port keeps it as an explicit step.""")
code("""tc   = prep_noncartesian(tc,   KFILE, 'rosette')
tc_w = prep_noncartesian(tc_w, KFILE, 'rosette')
print('prepped met', tc['data'].shape, tc['dims'])""")

md("## 2. Spatial reconstruction  (`op_CSIRecon`)  — DCF + FT")
code("""ft   = op_CSIRecon(tc,   KFILE, DCF, FT)
ft_w = op_CSIRecon(tc_w, KFILE, DCF, FT)
print('recon met', ft['data'].shape)""")

md("## 3. Coil combination  (`op_CSICombineCoils1`) — Roemer, maps from water ref")
code("""cc_w, phase, weights = op_CSICombineCoils1(ft_w)
cc                    = op_CSICombineCoils1(ft, 1, phase, weights)[0]
print('combined met', cc['data'].shape)""")

md("## 4. Average + water mask  (`op_CSIAverage`, `op_CSISegment_simple`)")
code("""ccav   = op_CSIAverage(cc)
ccav_w = op_CSIAverage(cc_w)
ccav_w = op_CSISegment_simple(ccav_w)
ccav['mask'] = ccav_w['mask']
mask = ccav_w['mask']['brainmasks']""")

md("## 4b. Left-shift  (`op_CSIleftshift`) — remove FID first-point phase (ls=0 here -> no-op)")
code("""ccav   = op_CSIleftshift(ccav)
ccav_w = op_CSIleftshift(ccav_w)""")

md("## 5. Spectral FT  (`op_CSIFourierTransform`, spatial already done in recon)")
code("""ftSpec   = op_CSIFourierTransform(ccav,   spatial=False, spectral=True)
ftSpec_w = op_CSIFourierTransform(ccav_w, spatial=False, spectral=True)
print('ppm', ftSpec['ppm'][0], '..', ftSpec['ppm'][-1])""")

md("## 6. Lipid/water removal + B0  (`op_CSIssp` skipped for phantom, `op_CSIRemoveLipids`, `op_CSIB0Correction_v2`)")
code("""if PHANTOM:
    rmlip = ftSpec                          # skip SSP: 0.8-1.88 ppm band removes lactate
else:
    rmlip = op_CSIssp(ftSpec, 0.8, 1.88)
# water removal in the [4.5 5.0] ppm band (FID-A run_MRSI_Rosette_40x40 settings)
ftSpec_rmw = op_CSIRemoveLipids(rmlip, lipidPPMRange=(4.5, 5.0), lineWidthRange=(1, 10))
ftSpec_B0, ftSpec_B0_w, freqMap, R2Map = op_CSIB0Correction_v2(ftSpec_rmw, ftSpec_w)""")

md("## 6b. (optional) spectral zero-fill  (`op_CSIspecZeroFill`)")
code("""if ZEROFILL:
    ftSpec_B0   = op_CSIspecZeroFill(ftSpec_B0,   ZEROFILL)
    ftSpec_B0_w = op_CSIspecZeroFill(ftSpec_B0_w, ZEROFILL)
    print('zero-filled ->', ftSpec_B0['data'].shape[0])
else:
    print('no zero-fill; NUNFIL =', ftSpec_B0['data'].shape[0])""")

md("## 7. Apply mask + spatial smoothing  (`op_CSIapplymask`, `op_CSIApodize`)")
code("""ftSpec_B0['mask'] = ccav['mask']
ftSpec_masked = op_CSIapplymask(ftSpec_B0)
if SMOOTH_FWHM > 0:
    ftSpec_smooth   = op_CSIApodize(ftSpec_masked, 'gaussian', SMOOTH_FWHM)
    ftSpec_smooth_w = op_CSIApodize(ftSpec_B0_w,   'gaussian', SMOOTH_FWHM)
else:
    ftSpec_smooth, ftSpec_smooth_w = ftSpec_masked, ftSpec_B0_w
print('smoothed met', ftSpec_smooth['data'].shape)""")

md("""## 8. LCModel + metabolite maps  (`fit_maps`)

Per-voxel LCModel over the mask: writes RAW / control / table for every voxel into
`OUT` (FID-A layout) and returns conc / CRLB / LW / SNR maps + `maps.npz` + `lcm_maps.png`.""")
code("""if DO_LCMODEL:
    met_fyx = to_fyx(ftSpec_smooth['data'],   ftSpec_smooth['dims'])
    wat_fyx = to_fyx(ftSpec_smooth_w['data'], ftSpec_smooth_w['dims'])
    res = fit_maps(met_fyx, wat_fyx, np.asarray(ftSpec_smooth['ppm']), mask, OUT)
    print('maps + tables ->', OUT)
else:
    print('DO_LCMODEL = False (skipped)')""")

md("## 9. View maps")
code("""import matplotlib.pyplot as plt
from PIL import Image
img = os.path.join(OUT, 'lcm_maps.png')
if os.path.exists(img):
    plt.figure(figsize=(15, 10)); plt.imshow(Image.open(img)); plt.axis('off'); plt.show()
else:
    print('run step 8 first')""")

nb['cells'] = cells
nb['metadata']['kernelspec'] = {'name': 'fida', 'display_name': 'fida', 'language': 'python'}
out = r'C:\Users\divya\Downloads\mrsi_pipeline\run_rosette_pipeline.ipynb'
with open(out, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print('wrote', out, 'with', len(cells), 'cells')
