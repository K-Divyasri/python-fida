"""compare_full_cached.py -- full step-by-step FID-A vs Python with the cached DCF.

Runs the Python pipeline (nufft/pipe_menon, cached DCF, no zerofill, no ACME, no
recenter, L2 water removal, phantom) and compares EVERY stage to the FID-A MATLAB
dump (fida_stages_s02_nufft). Per stage (met + water): correlation, median
voxel-wise %diff, and shape diff (after best complex scale).
"""
import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), 'mrsi'))
import numpy as np, h5py

MET = r'F:\fida\divya\20260605_phantom_test\subject02\met\meas_MID00138_FID48082_Rosette_40x40_isoctr.dat'
REF = r'F:\fida\divya\20260605_phantom_test\subject02\mrs_ref\meas_MID00139_FID48083_Rosette_40x40_isoctr_w.dat'
KFILE = r'C:\Users\divya\Downloads\fida codes\fid_a\processingTools\MRSI\kFiles\Rosette_traj_40x40.txt'
GT = r'C:\Users\divya\Downloads\mrsi_pipeline\fida_stages_s02_nufft'


def to_feat_yx(data, dims):
    """Reorder any struct array to [features..., y, x] flattened -> [F, ny, nx]."""
    ya, xa = dims['y'] - 1, dims['x'] - 1
    rest = [a for a in range(data.ndim) if a not in (ya, xa)]
    A = np.transpose(data, rest + [ya, xa])
    ny, nx = A.shape[-2:]
    return A.reshape(-1, ny, nx)


def gt_load(key):
    with h5py.File(os.path.join(GT, key + '.mat'), 'r') as f:
        d = f['data'][()]
        a = (d['real'] + 1j * d['imag']) if (d.dtype.names and 'real' in d.dtype.names) else d
        a = np.transpose(a, tuple(range(a.ndim))[::-1])          # -> FID-A dim order
        dims = {k: int(np.ravel(f['dims'][k][()])[0]) for k in f['dims'] if int(np.ravel(f['dims'][k][()])[0]) > 0}
    return to_feat_yx(a, dims)


def mask_load():
    with h5py.File(os.path.join(GT, 'mask.mat'), 'r') as f:
        return np.transpose(f['mask'][()], (1, 0)).astype(bool)


def metrics(py, gt, mask):
    if py.shape != gt.shape:
        # tolerate a y<->x transpose of the feature-collapsed array
        if py.shape == gt.transpose(0, 2, 1).shape:
            gt = gt.transpose(0, 2, 1)
        else:
            return None
    corr = np.corrcoef(np.abs(py[:, mask]).ravel(), np.abs(gt[:, mask]).ravel())[0, 1]
    ys, xs = np.where(mask); sc, sh, pd = [], [], []
    for y, x in zip(ys, xs):
        pv, gv = py[:, y, x], gt[:, y, x]
        if np.linalg.norm(gv) < 1e-30:
            continue
        pd.append(np.linalg.norm(pv - gv) / np.linalg.norm(gv))           # raw %diff
        a = np.vdot(pv, gv) / np.vdot(pv, pv)
        sc.append(abs(a)); sh.append(np.linalg.norm(pv * a - gv) / np.linalg.norm(gv))
    return corr, np.median(pd) * 100, np.median(sh) * 100, np.median(sc)


def main():
    from run_rosette_pipeline import run_pipeline
    print('running Python pipeline (nufft/pipe_menon, CACHED DCF)...')
    st = run_pipeline(metFile=MET, refFile=REF, kfile=KFILE, seq_type='rosette',
                      method='nufft', dcf='pipe_menon', save_dir=None,
                      water_removal='l2', do_phase=False, do_lcmodel=False,
                      recenter=False, phantom=True, zerofill=None)
    mask = st['s05_ccav_met']['mask']['brainmasks'].astype(bool)

    # add the final apodized met/water (post applymask+apodize)
    from op_CSIpostproc import op_CSIapplymask, op_CSIApodize
    b0 = st['s09_b0_met']; b0['mask'] = st['s05_ccav_met']['mask']
    st['s11_smooth_met'] = op_CSIApodize(op_CSIapplymask(b0), 'gaussian', 20)
    st['s11_smooth_ref'] = op_CSIApodize(st['s09_b0_ref'], 'gaussian', 20)

    PAIRS = [
        ('recon',            's03_recon_met', 's03_recon_met'),
        ('combine',          's04_combine_met', 's04_combine_met'),
        ('average (ccav)',   's05_ccav_met',  's05_ccav_met'),
        ('spectral FT',      's06_spec_met',  's06_spec_met'),
        ('B0 (+L2 water*)',  's09_b0_met',    's09_b0_met'),
        ('apodize (final)',  's11_smooth_met', 's11_smooth_met'),
        ('--- WATER (deterministic) ---', None, None),
        ('recon (H2O)',      's03_recon_ref', 's03_recon_ref'),
        ('spectral FT (H2O)','s06_spec_ref',  's06_spec_ref'),
        ('B0 (H2O)',         's09_b0_ref',    's09_b0_ref'),
        ('apodize (H2O)',    's11_smooth_ref', 's11_smooth_ref'),
    ]
    print(f'\n=== FID-A (MATLAB) vs Python | nufft + pipe_menon (cached DCF) | {int(mask.sum())} vox ===')
    print(f'{"stage":26s} {"corr":>8s} {"%diff":>9s} {"shape":>9s} {"scale":>7s}')
    print('-' * 62)
    for label, pk, gk in PAIRS:
        if pk is None:
            print(label); continue
        try:
            py = to_feat_yx(st[pk]['data'], st[pk]['dims']); gt = gt_load(gk)
        except Exception as e:
            print(f'{label:26s}  (skip: {e})'); continue
        r = metrics(py, gt, mask)
        if r is None:
            print(f'{label:26s}  shape {py.shape} vs {gt.shape}'); continue
        corr, pd, sh, sc = r
        print(f'{label:26s} {corr:8.4f} {pd:8.2f}% {sh:8.3f}% {sc:7.3f}')
    print('\n* metabolite B0/apodize include the stochastic L2 water-removal basis.')


if __name__ == '__main__':
    main()
