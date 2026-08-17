"""trace2_save.py -- run Python pipeline at FID-A config (pipe_menon/nufft,
no zerofill, no ACME, no recenter), SAVE met+water final spectra AND the key
intermediate stages, so all deviation analysis is offline/cheap.

Saves to trace2_arrays.npz:  met_final[f,y,x], wat_final[f,y,x], ppm,
 s06_met, s06_wat (spectral FT), s09_met, s09_wat (B0), and mask.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mrsi'))
import numpy as np

MET = r'F:\fida\divya\20260605_phantom_test\subject02\met\meas_MID00138_FID48082_Rosette_40x40_isoctr.dat'
REF = r'F:\fida\divya\20260605_phantom_test\subject02\mrs_ref\meas_MID00139_FID48083_Rosette_40x40_isoctr_w.dat'
KFILE = r'C:\Users\divya\Downloads\fida codes\fid_a\processingTools\MRSI\kFiles\Rosette_traj_40x40.txt'
OUT = r'C:\Users\divya\Downloads\mrsi_pipeline\trace2_arrays.npz'


def main():
    from run_rosette_pipeline import run_pipeline
    from op_CSIpostproc import op_CSIapplymask, op_CSIApodize
    from viz._common import to_fyx
    st = run_pipeline(metFile=MET, refFile=REF, kfile=KFILE, seq_type='rosette',
                      method='nufft', dcf='pipe_menon', save_dir=None,
                      water_removal='l2', do_phase=False, do_lcmodel=False,
                      recenter=False, phantom=True, zerofill=None)
    mask = st['s05_ccav_met']['mask']['brainmasks']
    b0 = st['s09_b0_met']; b0['mask'] = st['s05_ccav_met']['mask']
    met_final = op_CSIApodize(op_CSIapplymask(b0), 'gaussian', 20)
    wat_final = op_CSIApodize(st['s09_b0_ref'], 'gaussian', 20)

    def fyx(s):
        return to_fyx(s['data'], s['dims'])
    np.savez(OUT,
             met_final=fyx(met_final), wat_final=fyx(wat_final),
             s06_met=fyx(st['s06_spec_met']), s06_wat=fyx(st['s06_spec_ref']),
             s09_met=fyx(st['s09_b0_met']), s09_wat=fyx(st['s09_b0_ref']),
             ppm=np.asarray(met_final['ppm']), mask=np.asarray(mask))
    print('saved', OUT, 'met_final', fyx(met_final).shape)


if __name__ == '__main__':
    main()
