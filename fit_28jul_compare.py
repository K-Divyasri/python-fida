"""fit_28jul_compare.py -- fit 28thJuly my-pipeline data with FID-A's EXACT LCModel
params, then compare to FID-A's saved maps (outputs\lcm_out\map.mat / crlb.mat).

FID-A 28thJuly params (from its control): basis RosettePhantom_LacAceCrCho_TE15_SW1587,
ECHOT=8.5, DELTAT=6.3e-4, HZPPPM=123.19877, NUNFIL=4096 (FID zero-filled to 4096),
phantom control (NUSE1=3 Cr/Cho/Lac, DKNTMN=1.0). conditionTail: met tail->0,
water tail->small decay (avoids DOECC 0/0).
"""
import os, re, json, subprocess, glob, warnings; warnings.filterwarnings('ignore')
import numpy as np, h5py
from fit_one_voxel_lcm import win2wsl, spec_to_fid, parse_table, run_lcm

D = r'C:\Users\divya\Downloads\mrsi_pipeline\rosette_py_stages_28jul'
OUT = r'C:\Users\divya\Downloads\mrsi_pipeline\lcm_28jul_te8p5'
FIDA = r'F:\fida\divya\28thJULYpHANTOM40X40\subject01\outputs\lcm_out'
BASIS = r'C:\Users\divya\Downloads\mrsi_pipeline\basis_te8p5\RosettePhantom_LacAceCrCho_TE8p5.basis'  # TE-matched
OWNER = 'Divyasri Krishnkaumar, Sunnybrook Research Institute'; KEY = os.environ.get('LCMODEL_KEY', '')
DT, F0, TE, NUNFIL, PPMST, PPMEND = 6.300e-4, 123.19877, 8.5, 4096, 4.25, 0.2
METS = ['Cr', 'Cho', 'Lac', 'Act']


def load_fyx(dir_, key):
    z = np.load(os.path.join(dir_, key + '.npz'), allow_pickle=True)
    m = json.loads(str(z['meta'])); dm = {k: int(v) for k, v in m['dims'].items() if v}
    d = z['data']; fax = (dm.get('f') or dm.get('t')) - 1
    return np.transpose(d, (fax, dm['y'] - 1, dm['x'] - 1)), np.asarray(z['ppm'])


def zerofill(fid, N, is_water):
    """FID-A conditionTail: pad to N; met tail->0, water tail->tiny decay (>0)."""
    n = fid.size
    if n >= N:
        return fid[:N]
    out = np.zeros(N, complex); out[:n] = fid
    if is_water:
        env = abs(fid[-1]) or np.abs(fid).max(); tau = max(n / 3, 1)
        out[n:] = env * 1e-3 * np.exp(-(np.arange(1, N - n + 1)) / tau)
    return out


def write_raw(fid, path):
    fid = np.asarray(fid).ravel()
    with open(path, 'w', newline='\n') as f:
        f.write(' $SEQPAR\n echot= %.2f\n seq= \'PRESS\'\n hzpppm= %.6f' % (TE, F0))
        f.write('\n NumberOfPoints= %d\n dwellTime= %.6f\n $END\n $NMID' % (fid.size, DT))
        f.write("\n id='ANONYMOUS ', fmtdat='(2E15.6)'\n volume=8.0\n tramp=1.0\n $END\n")
        for v in fid:
            f.write('  % .6e  % .6e\n' % (v.real, -v.imag))     # conjugate (io_writelcm)


def write_control(stem, tag):
    ctrl = stem + '.control'
    L = ['$LCMODL', f" OWNER='{OWNER}'", f' KEY={KEY}', f" TITLE='{tag}'",
         f" FILRAW='{win2wsl(stem + '.RAW')}'", f" FILBAS='{win2wsl(BASIS)}'",
         f" FILH2O='{win2wsl(stem + '_w.RAW')}'", f" FILTAB='{win2wsl(stem + '.table')}'",
         f" FILCOO='{win2wsl(stem + '.coord')}'", f" FILPS='{win2wsl(stem + '.ps')}'",
         f' HZPPPM={F0:.5f}', f' DELTAT={DT:.8f}', f' NUNFIL={NUNFIL}', f' ECHOT={TE:g}',
         ' LTABLE=7', ' LCOORD=9', ' DOREFS(1)=T', f' PPMST={PPMST:g}', f' PPMEND={PPMEND:g}',
         ' RFWHM=0.15', ' WDLINE(1)=0.025', ' DKNTMN=1.0', ' NEACH=999', ' NSIMUL=0',
         ' NOMIT=0', ' NNORAT=0', ' NRATIO=0', ' NUSE1=3', " CHUSE1(1)='Cr'",
         " CHUSE1(2)='Cho'", " CHUSE1(3)='Lac'", ' DOECC=T', ' DOWS=T', ' ATTH2O=1.0',
         ' WCONC=55556', '$END']
    open(ctrl, 'w', newline='\n').write('\n'.join(L) + '\n')
    return ctrl


def load_fida_maps(ny, nx):
    def h5a(f, p):
        d = f[p][()]; a = (d['real'] + 1j * d['imag']) if (d.dtype.names and 'real' in d.dtype.names) else d
        return np.transpose(a, tuple(range(a.ndim)[::-1])) if a.ndim > 1 else a
    conc, crlb = {}, {}
    with h5py.File(os.path.join(FIDA, 'map.mat'), 'r') as f, h5py.File(os.path.join(FIDA, 'crlb.mat'), 'r') as fc:
        for m in METS:
            conc[m] = h5a(f, f'map/{m}'); crlb[m] = h5a(fc, f'crlb/{m}')
    return conc, crlb


def main():
    os.makedirs(OUT, exist_ok=True)
    from viz._common import make_mask, water_map, load_stage
    met, ppm = load_fyx(D, 's11_smooth_met'); wat, _ = load_fyx(D, 's11_smooth_ref')
    mask = make_mask(water_map(load_stage(os.path.join(D, 's11_smooth_ref.npz'))))
    ny, nx = mask.shape; ijk = np.argwhere(mask)
    conc = {m: np.full((ny, nx), np.nan) for m in METS}
    crlb = {m: np.full((ny, nx), np.nan) for m in METS}
    print(f'fitting {len(ijk)} voxels (zero-fill {met.shape[0]}->{NUNFIL}, TE {TE}, rosette basis)...')
    nok = 0
    for n, (y, x) in enumerate(ijk, 1):
        tag = f'{x}x{y}'; stem = os.path.join(OUT, tag)
        write_raw(zerofill(spec_to_fid(met[:, y, x]), NUNFIL, False), stem + '.RAW')
        write_raw(zerofill(spec_to_fid(wat[:, y, x]), NUNFIL, True), stem + '_w.RAW')
        run_lcm(write_control(stem, tag))
        r = parse_table(stem + '.table')
        if r and r['mets']:
            nok += 1
            for m, (c, sd) in r['mets'].items():
                if m in conc: conc[m][y, x] = c; crlb[m][y, x] = sd
        if n % 50 == 0 or n == len(ijk):
            print(f'  [{n}/{len(ijk)}] {nok} ok', flush=True)
    np.savez(os.path.join(OUT, 'maps.npz'), **{f'conc_{m}': conc[m] for m in METS},
             **{f'crlb_{m}': crlb[m] for m in METS}, mask=mask)

    # ---- compare to FID-A ----
    fc, fq = load_fida_maps(ny, nx)
    print(f'\n{"met":5s} {"mine med":>9s} {"FID-A med":>10s} {"ratio":>7s} {"corr":>7s} {"n both":>7s}')
    print('-' * 52)
    for m in METS:
        both = (crlb[m] <= 20) & (fq[m] <= 20) & mask & np.isfinite(fc[m]) & np.isfinite(conc[m])
        a, b = conc[m][both], fc[m][both]
        if a.size < 3:
            print(f'{m:5s} (n={a.size})'); continue
        print(f'{m:5s} {np.median(a):9.1f} {np.median(b):10.1f} {np.median(a / b):7.3f} {np.corrcoef(a, b)[0, 1]:7.3f} {a.size:7d}')
    # deviation figure
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from viz.plotspec import imagesc
    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    for i, m in enumerate(METS):
        both = (crlb[m] <= 20) & (fq[m] <= 20) & mask
        imagesc(np.where(both, conc[m], np.nan), title=f'mine {m}', ax=ax[0][i], cmap='viridis')
        imagesc(np.where(both, fc[m], np.nan), title=f'FID-A {m}', ax=ax[1][i], cmap='viridis')
    plt.tight_layout(); plt.savefig(os.path.join(OUT, 'compare_fida.png'), dpi=100)
    print('saved', os.path.join(OUT, 'compare_fida.png'))


if __name__ == '__main__':
    main()
