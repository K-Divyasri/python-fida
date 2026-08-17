"""Run Python recon at dft (exact inverse) + pipe_menon DCF, save s06, compare
to FID-A's nufft fresh-dump s06 -> is exact-DFT closer to FID-A's Fessler NUFFT
than my finufft is?"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mrsi'))
import numpy as np, h5py
MET = r'F:\fida\divya\20260605_phantom_test\subject02\met\meas_MID00138_FID48082_Rosette_40x40_isoctr.dat'
REF = r'F:\fida\divya\20260605_phantom_test\subject02\mrs_ref\meas_MID00139_FID48083_Rosette_40x40_isoctr_w.dat'
KFILE = r'C:\Users\divya\Downloads\fida codes\fid_a\processingTools\MRSI\kFiles\Rosette_traj_40x40.txt'
GT = r'C:\Users\divya\Downloads\mrsi_pipeline\fida_stages_s02_nufft'
ARR = r'C:\Users\divya\Downloads\mrsi_pipeline\trace2_arrays.npz'


def main():
    from run_rosette_pipeline import run_pipeline
    from viz._common import to_fyx
    st = run_pipeline(metFile=MET, refFile=REF, kfile=KFILE, seq_type='rosette',
                      method='dft', dcf='pipe_menon', save_dir=None,
                      water_removal='none', do_phase=False, do_lcmodel=False,
                      recenter=False, phantom=True, zerofill=None)
    dft_s06_wat = to_fyx(st['s06_spec_ref']['data'], st['s06_spec_ref']['dims'])
    dft_s06_met = to_fyx(st['s06_spec_met']['data'], st['s06_spec_met']['dims'])

    def mfyx(k):
        with h5py.File(os.path.join(GT, k + '.mat'), 'r') as f:
            d = f['data'][()]
            a = (d['real'] + 1j * d['imag']) if (d.dtype.names and 'real' in d.dtype.names) else d
        return np.transpose(a, tuple(range(a.ndim))[::-1])
    fa_wat = mfyx('s06_spec_ref'); fa_met = mfyx('s06_spec_met')
    with h5py.File(os.path.join(GT, 'mask.mat'), 'r') as f:
        mask = np.transpose(f['mask'][()], (1, 0)).astype(bool)
    fin = np.load(ARR)  # my finufft s06

    def cmp(py, fa, lbl):
        ys, xs = np.where(mask); sc, sh = [], []
        for y, x in zip(ys, xs):
            a = np.vdot(py[:, y, x], fa[:, y, x]) / np.vdot(py[:, y, x], py[:, y, x])
            if not np.isfinite(a): continue
            sc.append(abs(a)); sh.append(np.linalg.norm(py[:, y, x] * a - fa[:, y, x]) / (np.linalg.norm(fa[:, y, x]) + 1e-30))
        print(f'{lbl:38s} scale {np.median(sc):.3f}  shape {np.median(sh)*100:6.2f}%')

    print('=== s06 recon vs FID-A nufft(Fessler) fresh dump, masked ===')
    cmp(fin['s06_wat'], fa_wat, 'WATER  my finufft  vs FID-A nufft')
    cmp(dft_s06_wat, fa_wat, 'WATER  my dft      vs FID-A nufft')
    cmp(fin['s06_met'], fa_met, 'MET    my finufft  vs FID-A nufft')
    cmp(dft_s06_met, fa_met, 'MET    my dft      vs FID-A nufft')


if __name__ == '__main__':
    main()
