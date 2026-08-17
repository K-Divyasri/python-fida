"""run_rosette.py -- full non-cartesian (rosette / concentric) MRSI pipeline.

    load pair -> prep (combine_time + split_readout_kpts) -> op_CSIRecon
    -> spectral FT -> coil combine -> save maps/spectra

Pure-Python FID-A-style toolbox (no FID-A, no MATLAB). Works on STANDARD twix
(VD/VE) rosette data. NOTE: the met .dat in 28thJULYpHANTOM40X40\subject01 is a
corrupted 'INDX' container -- re-copy / re-export it as standard twix first
(the water ref there is already standard twix and loads fine).

Run:  python run_rosette.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'mrsi'))
import os
import numpy as np

from io_CSIload_twix import io_CSIload_twix_pair
from op_CSIRosettePrep import prep_noncartesian
from op_CSIRecon import op_CSIRecon
from op_CSIFourierTransform import op_CSIFourierTransform

# ============================ CONFIG ============================
MET = r'F:\fida\divya\28thJULYpHANTOM40X40\subject01\met\meas_MID01094_FID59455_XA60_RosetteSpinEcho_2_avg_8mste_4sTR.dat'
REF = r'F:\fida\divya\28thJULYpHANTOM40X40\subject01\mrs_ref\meas_MID01095_FID59456_XA60_RosetteSpinEcho_2_avg_8mste_4sTR_w.dat'
KFILE = r'C:\Users\divya\Downloads\fida codes\fid_a\processingTools\MRSI\kFiles\Rosette_traj_40x40.txt'
OUTDIR = r'C:\Users\divya\Downloads\mrsi_pipeline\out_rosette'
SEQ = 'rosette'
DCF = 'voronoi'          # nn | voronoi | pipe_menon  (skipped for tikhonov)
FT = 'dft'               # nufft (fast) | dft (exact) | tikhonov (exact, no DCF)
COMBINE_INTERLEAVES = True   # combine Par/'extras' into the time/spectral axis
# ===============================================================


def process(struct, tag):
    print(f'[{tag}] loaded {struct["sz"]} dims',
          {k: v for k, v in struct['dims'].items() if v})
    sp = prep_noncartesian(struct, KFILE, seq_type=SEQ) if COMBINE_INTERLEAVES else struct
    print(f'[{tag}] after prep {sp["sz"]} dims',
          {k: v for k, v in sp['dims'].items() if v})
    rec = op_CSIRecon(sp, KFILE, dcfMethod=DCF, ftMethod=FT)   # [t, coils, y, x]
    spec = op_CSIFourierTransform(rec, spatial=False, spectral=True)
    d = spec['data']
    ax_c = spec['dims'].get('coils', 0)
    combined = np.sqrt((np.abs(d) ** 2).sum(axis=ax_c - 1)) if ax_c else np.abs(d)
    print(f'[{tag}] recon {d.shape} -> combined {combined.shape}  ppm axis n={spec["ppm"].size}')
    return spec, combined


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    met, ref = io_CSIload_twix_pair(MET, REF, kfile=KFILE, seq_type=SEQ)
    for s, tag in ((ref, 'ref'), (met, 'met')):
        spec, comb = process(s, tag)
        np.savez(os.path.join(OUTDIR, f'{tag}_maps.npz'),
                 combined=comb, ppm=spec['ppm'], fov=spec['fov']['x'],
                 vox=spec['voxelSize']['x'])
        print(f'[{tag}] saved -> {tag}_maps.npz')
    print('done ->', OUTDIR)


if __name__ == '__main__':
    main()
