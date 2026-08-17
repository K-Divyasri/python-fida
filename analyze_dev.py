"""analyze_dev.py -- offline deviation analysis (no recon rerun).

Uses trace2_arrays.npz (my Python config-A final + intermediate stages) and
FID-A's per-voxel RAW (576-pt final spectra) from subject02 lcm_out.

Answers:
  1. Water channel (fully deterministic, NO water removal): my wat_final vs
     FID-A water RAW -> per-voxel amplitude scale, 0-order phase, and the
     REAL-spectrum diff after phase+scale alignment. Localises scale/phase.
  2. Met channel: raw complex diff vs phase-aligned diff vs phase+scale-aligned
     -> decompose the 30% into {phase, scale, true shape}.
"""
import os, glob, re
import numpy as np

ARR = r'C:\Users\divya\Downloads\mrsi_pipeline\trace2_arrays.npz'
FALCM = r'F:\fida\divya\20260605_phantom_test\subject02\outputs\lcm_out'


def parse_raw(path):
    c1, c2, started = [], [], False
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
    return np.array(c1) - 1j * np.array(c2)


def load_fida():
    met, wat = {}, {}
    for f in glob.glob(os.path.join(FALCM, '*_ftSpec_smooth_lcm')):
        m = re.match(r'(\d+)x(\d+)_ftSpec_smooth_lcm$', os.path.basename(f))
        if not m:
            continue
        x, y = int(m.group(1)), int(m.group(2))
        met[(x, y)] = parse_raw(f)
        if os.path.exists(f + '_w'):
            wat[(x, y)] = parse_raw(f + '_w')
    return met, wat


def spec_to_fid(spec):
    return np.fft.fft(np.fft.fftshift(spec))


def align_metrics(fpy, fref):
    """Return (raw, phase_aligned, phasescale_aligned) relative L2 diffs + scale + dphase(deg)."""
    raw = np.linalg.norm(fpy - fref) / np.linalg.norm(fref)
    # best 0-order phase: phi = angle(<fref, fpy>)
    ip = np.vdot(fref, fpy)
    phi = np.angle(ip)
    fpy_p = fpy * np.exp(-1j * phi)
    pa = np.linalg.norm(fpy_p - fref) / np.linalg.norm(fref)
    # best complex scale a = <fref,fpy>/<fpy,fpy>
    a = np.vdot(fpy, fref) / np.vdot(fpy, fpy)
    fpy_s = fpy * a
    psa = np.linalg.norm(fpy_s - fref) / np.linalg.norm(fref)
    return raw, pa, psa, np.abs(a), np.degrees(phi)


def main():
    z = np.load(ARR)
    met_f = z['met_final']; wat_f = z['wat_final']; ppm = z['ppm']
    nf, ny, nx = met_f.shape
    fida_met, fida_wat = load_fida()
    print(f'grid {ny}x{nx}, ppm {ppm[0]:.2f}..{ppm[-1]:.2f}, FID-A vox {len(fida_met)}')

    for label, pyarr, faref in [('WATER (deterministic)', wat_f, fida_wat),
                                ('METAB', met_f, fida_met)]:
        R = {'raw': [], 'phase': [], 'phasescale': [], 'scale': [], 'dphi': []}
        for (x, y), fref in faref.items():
            if not (0 <= y < ny and 0 <= x < nx):
                continue
            fpy = spec_to_fid(pyarr[:, y, x])
            if fpy.shape != fref.shape or np.linalg.norm(fref) < 1e-9:
                continue
            raw, pa, psa, sc, dphi = align_metrics(fpy, fref)
            R['raw'].append(raw); R['phase'].append(pa); R['phasescale'].append(psa)
            R['scale'].append(sc); R['dphi'].append(dphi)
        for k in R:
            R[k] = np.array(R[k])
        print(f'\n--- {label}  (n={len(R["raw"])}) ---')
        print(f'  raw complex diff        median {np.median(R["raw"])*100:6.2f}%')
        print(f'  after 0-order phase     median {np.median(R["phase"])*100:6.2f}%')
        print(f'  after phase+scale       median {np.median(R["phasescale"])*100:6.2f}%')
        print(f'  amplitude scale (py/fida) median {np.median(R["scale"]):.3f}  (mean {np.mean(R["scale"]):.3f})')
        print(f'  0-order phase offset    median {np.median(R["dphi"]):6.1f} deg  IQR [{np.percentile(R["dphi"],25):.0f},{np.percentile(R["dphi"],75):.0f}]')


if __name__ == '__main__':
    main()
