"""run_maps_nufft576.py -- regenerate metabolite maps at FID-A production config:
nufft + pipe_menon recon, NUNFIL 576 (no zerofill), no ACME, no recenter,
phantom (no SSP), L2 water removal, apodize (fixed). Per-voxel LCModel.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mrsi'))

MET = r'F:\fida\divya\20260605_phantom_test\subject02\met\meas_MID00138_FID48082_Rosette_40x40_isoctr.dat'
REF = r'F:\fida\divya\20260605_phantom_test\subject02\mrs_ref\meas_MID00139_FID48083_Rosette_40x40_isoctr_w.dat'
KFILE = r'C:\Users\divya\Downloads\fida codes\fid_a\processingTools\MRSI\kFiles\Rosette_traj_40x40.txt'
SAVE = r'C:\Users\divya\Downloads\mrsi_pipeline\rosette_py_stages_nufft576'
LCMOUT = r'C:\Users\divya\Downloads\mrsi_pipeline\lcm_nufft576'

if __name__ == '__main__':
    from run_rosette_pipeline import run_pipeline
    run_pipeline(metFile=MET, refFile=REF, kfile=KFILE, seq_type='rosette',
                 method='nufft', dcf='pipe_menon', save_dir=SAVE,
                 water_removal='l2', do_phase=False, do_lcmodel=True,
                 lcm_out=LCMOUT, recenter=False, phantom=True, zerofill=None)
    print('maps ->', LCMOUT)
