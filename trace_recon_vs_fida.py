"""trace_recon_vs_fida.py -- find WHERE Python deviates from FID-A subject02.

Authoritative reference = FID-A's actual per-voxel LCModel RAW files in
  F:\...\subject02\outputs\lcm_out\*_ftSpec_smooth_lcm     (576-pt FINAL spectra)
These are the exact spectra FID-A fed LCModel (nufft/pipe_menon recon, no
zerofill, NUNFIL=576, no ACME). Compare my Python final met spectrum voxel-wise
at two recon configs to localise the deviation:
  A: nufft + pipe_menon   (== FID-A)
  B: dft   + nn           (my old default)

met_fit = op_CSIApodize(op_CSIapplymask(b0_576), 'gaussian', 20)   [pre-ACME]
FID convention (io_writelcm): col1=Re(fid), col2=-Im(fid), fid=fft(fftshift(spec)).
"""
import os, sys, glob, re
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mrsi'))
import numpy as np

FALCM = r'F:\fida\divya\20260605_phantom_test\subject02\outputs\lcm_out'
MET = r'F:\fida\divya\20260605_phantom_test\subject02\met\meas_MID00138_FID48082_Rosette_40x40_isoctr.dat'
REF = r'F:\fida\divya\20260605_phantom_test\subject02\mrs_ref\meas_MID00139_FID48083_Rosette_40x40_isoctr_w.dat'
KFILE = r'C:\Users\divya\Downloads\fida codes\fid_a\processingTools\MRSI\kFiles\Rosette_traj_40x40.txt'


def parse_raw(path):
    """Read an LCModel RAW -> complex fid (undo io_writelcm conjugate)."""
    c1, c2 = [], []
    started = False
    for ln in open(path, errors='ignore'):
        if '$END' in ln:
            started = True; continue
        if not started:
            continue
        p = ln.split()
        if len(p) == 2:
            try:
                c1.append(float(p[0])); c2.append(float(p[1]))
            except ValueError:
                pass
    c1 = np.array(c1); c2 = np.array(c2)
    return c1 - 1j * c2                      # fid = real + i*imag, imag = -col2


def load_fida_fids():
    """All FID-A met RAW -> dict[(x,y)] = fid[576] + also water."""
    met, wat = {}, {}
    for f in glob.glob(os.path.join(FALCM, '*_ftSpec_smooth_lcm')):
        b = os.path.basename(f)
        m = re.match(r'(\d+)x(\d+)_ftSpec_smooth_lcm$', b)
        if not m:
            continue
        x, y = int(m.group(1)), int(m.group(2))
        met[(x, y)] = parse_raw(f)
        wf = f + '_w'
        if os.path.exists(wf):
            wat[(x, y)] = parse_raw(wf)
    return met, wat


def py_final_spectra(dcf, method):
    """Run Python pipeline (zerofill off, no ACME, no recenter) -> met_fit [f,y,x]."""
    from run_rosette_pipeline import run_pipeline
    from op_CSICombineCoils import op_CSIAverage  # noqa
    from op_CSIpostproc import op_CSIapplymask, op_CSIApodize
    from viz._common import to_fyx
    st = run_pipeline(metFile=MET, refFile=REF, kfile=KFILE, seq_type='rosette',
                      method=method, dcf=dcf, save_dir=None, water_removal='l2',
                      do_phase=False, do_lcmodel=False, recenter=False,
                      phantom=True, zerofill=None)
    b0 = st['s09_b0_met']                      # 576, post-B0, pre-ACME
    b0['mask'] = st['s05_ccav_met']['mask']
    met_fit = op_CSIApodize(op_CSIapplymask(b0), 'gaussian', 20)
    wat_fit = op_CSIApodize(st['s09_b0_ref'], 'gaussian', 20)
    return (to_fyx(met_fit['data'], met_fit['dims']),
            to_fyx(wat_fit['data'], wat_fit['dims']),
            np.asarray(met_fit['ppm']))


def spec_to_fid(spec):
    return np.fft.fft(np.fft.fftshift(spec))


def compare(tag, dcf, method, fida_met):
    print(f'\n=== config {tag}: dcf={dcf} method={method} ===')
    met_fyx, wat_fyx, ppm = py_final_spectra(dcf, method)
    nf, ny, nx = met_fyx.shape
    rels, corrs, ratios = [], [], []
    for (x, y), fref in fida_met.items():
        if not (0 <= y < ny and 0 <= x < nx):
            continue
        fpy = spec_to_fid(met_fyx[:, y, x])
        if fpy.shape != fref.shape:
            continue
        a, b = np.abs(fpy), np.abs(fref)
        if b.max() < 1e-12:
            continue
        rels.append(np.linalg.norm(fpy - fref) / np.linalg.norm(fref))
        corrs.append(np.corrcoef(a, b)[0, 1])
        ratios.append(a.max() / b.max())
    rels = np.array(rels); corrs = np.array(corrs); ratios = np.array(ratios)
    print(f'  voxels compared: {len(rels)}')
    print(f'  |py-fida|/|fida|  median {np.median(rels)*100:6.2f}%   p90 {np.percentile(rels,90)*100:6.2f}%   %<2%: {np.mean(rels<0.02)*100:.0f}%')
    print(f'  mag corr          median {np.median(corrs):.4f}')
    print(f'  amp ratio         median {np.median(ratios):.3f}')
    return rels, corrs, ratios


def main():
    print('Loading FID-A RAW (final 576-pt spectra)...')
    fida_met, fida_wat = load_fida_fids()
    print(f'  {len(fida_met)} FID-A met RAW voxels, N={len(next(iter(fida_met.values())))}')
    compare('A', 'pipe_menon', 'nufft', fida_met)
    compare('B', 'nn', 'dft', fida_met)


if __name__ == '__main__':
    main()
