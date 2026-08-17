"""run_l2_repro.py -- reproduce the lcm_l2 run, deterministically.

One command: full 20260605 pipeline (L2 water removal, seeded -> reproducible;
recenter on) -> per-voxel LCModel fit (TE15/SW1587 basis, licensed) -> maps.

Deterministic: recon/combine/HSVD/ACME/LCModel are deterministic; the only random
step (L2 water-removal basis) is seeded with lipid_seed=0. Same inputs -> same maps.

Preserves the original lcm_l2/; writes to *_repro dirs. Compares to lcm_l2 at the end.

Run:  python run_l2_repro.py
"""
import os, sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mrsi'))
import numpy as np

MET = r'F:\fida\divya\20260605_phantom_test\subject02\met\meas_MID00138_FID48082_Rosette_40x40_isoctr.dat'
REF = r'F:\fida\divya\20260605_phantom_test\subject02\mrs_ref\meas_MID00139_FID48083_Rosette_40x40_isoctr_w.dat'
KF = r'C:\Users\divya\Downloads\fida codes\fid_a\processingTools\MRSI\kFiles\Rosette_traj_40x40.txt'
STAGES = r'C:\Users\divya\Downloads\mrsi_pipeline\rosette_py_stages_l2_repro'
LCM_OUT = r'C:\Users\divya\Downloads\mrsi_pipeline\lcm_l2_repro'
METS = ['Cr', 'Cho', 'Lac', 'Act']


def main():
    from run_rosette_pipeline import run_pipeline
    print('=== reproducing lcm_l2 (20260605, L2 water removal seed=0, recenter) ===', flush=True)
    st = run_pipeline(metFile=MET, refFile=REF, kfile=KF, seq_type='rosette',
                      method='dft', dcf='nn',
                      water_removal='l2', lipid_seed=0,      # seeded -> reproducible
                      do_phase=False,                        # LCModel fits no-ACME data
                      recenter=True,
                      save_dir=STAGES,
                      do_lcmodel=True, lcm_out=LCM_OUT)
    # compare to the preserved lcm_l2
    print('\n=== vs original lcm_l2 ===')
    try:
        a = np.load(LCM_OUT + r'\maps.npz'); b = np.load(r'C:\Users\divya\Downloads\mrsi_pipeline\lcm_l2\maps.npz')
        for m in METS:
            ca, cb = a[f'crlb_{m}'], b[f'crlb_{m}']
            print(f'  {m}: repro median {np.nanmedian(ca):.0f}%/n{int(np.nansum(ca<=20))}   '
                  f'lcm_l2 median {np.nanmedian(cb):.0f}%/n{int(np.nansum(cb<=20))}')
    except Exception as e:
        print('  compare skipped:', e)
    print('done -> stages', STAGES, ' maps', LCM_OUT)


if __name__ == '__main__':
    main()
