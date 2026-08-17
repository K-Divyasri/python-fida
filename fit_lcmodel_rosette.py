"""fit_lcmodel_rosette.py -- batch LCModel fit over all masked voxels.

Python port of run_lcm_rosette_portable.m: per masked voxel write LCModel RAW
(conjugate FID convention, io_writelcm) + control (licensed KEY + phantom params),
run LCModel (WSL) in one batch loop, parse .table -> conc/CRLB/LW/SNR maps, and
render concentration + CRLB panels.

Run:  python fit_lcmodel_rosette.py
"""
import os, glob, re, json, subprocess
import numpy as np

D = r'C:\Users\divya\Downloads\mrsi_pipeline\rosette_py_stages_20260605_cmp'  # FID-A-faithful (no ACME)
OUT = r'C:\Users\divya\Downloads\mrsi_pipeline\lcm_20260605_ff'
BASIS = r'C:\Users\divya\Downloads\fida codes\fid_a\basis_rosette_phantom\RosettePhantom_LacAceCrCho_TE15_SW1587.basis'  # FID-A's 20260605 basis
LCM = '/home/divya/.lcmodel/bin/lcmodel'
OWNER = 'Divyasri Krishnkaumar, Sunnybrook Research Institute'
KEY = os.environ.get('LCMODEL_KEY', '')
DT, F0, TE = 6.300e-4, 123.19877, 15.0
PPMST, PPMEND = 4.25, 0.2
METS = ['Cr', 'Cho', 'Lac', 'Act']


def win2wsl(p):
    p = os.path.abspath(p); return '/mnt/' + p[0].lower() + p[2:].replace('\\', '/')


def load_fyx(key):
    z = np.load(os.path.join(D, key + '.npz'), allow_pickle=True)
    m = json.loads(str(z['meta'])); dims = {k: int(v) for k, v in m['dims'].items() if v}
    d = z['data']; fax = (dims.get('f') or dims.get('t')) - 1
    return np.transpose(d, (fax, dims['y'] - 1, dims['x'] - 1)), np.asarray(z['ppm'])


def spec_to_fid(spec):
    return np.fft.fft(np.fft.fftshift(spec))


def write_raw(fid, path):
    fid = np.asarray(fid).ravel()
    with open(path, 'w', newline='\n') as f:
        f.write(' $SEQPAR\n echot= %.2f\n seq= \'PRESS\'\n hzpppm= %.6f' % (TE, F0))
        f.write('\n NumberOfPoints= %d\n dwellTime= %.6f\n $END\n $NMID' % (fid.size, DT))
        f.write("\n id='ANONYMOUS ', fmtdat='(2E15.6)'\n volume=8.0\n tramp=1.0\n $END\n")
        for v in fid:
            f.write('  % .6e  % .6e\n' % (v.real, -v.imag))     # conjugate (io_writelcm)


def write_control(stem, out, tag, nunfil=576):
    ctrl = stem + '.control'
    L = ['$LCMODL', f" OWNER='{OWNER}'", f' KEY={KEY}', f" TITLE='{tag}'",
         f" FILRAW='{win2wsl(stem + '.RAW')}'", f" FILBAS='{win2wsl(BASIS)}'",
         f" FILH2O='{win2wsl(stem + '_w.RAW')}'",
         f" FILTAB='{win2wsl(out + '/' + tag + '.table')}'",
         f" FILCOO='{win2wsl(out + '/' + tag + '.coord')}'",
         f" FILCSV='{win2wsl(out + '/' + tag + '.csv')}'",
         f" FILPS='{win2wsl(out + '/' + tag + '.ps')}'",
         f' HZPPPM={F0:.5f}', f' DELTAT={DT:.8f}', f' NUNFIL={nunfil}', f' ECHOT={TE:g}',
         ' LTABLE=7', ' LCOORD=9', ' LCSV=11', ' DOREFS(1)=T',
         f' PPMST={PPMST:g}', f' PPMEND={PPMEND:g}', ' RFWHM=0.15', ' WDLINE(1)=0.025',
         ' DKNTMN=1.0', ' NEACH=999', ' NSIMUL=0', ' NOMIT=0', ' NNORAT=0', ' NRATIO=0',
         ' NUSE1=3', " CHUSE1(1)='Cr'", " CHUSE1(2)='Cho'", " CHUSE1(3)='Lac'",
         ' DOECC=F', ' DOWS=T', ' ATTH2O=1.0', ' WCONC=55556', '$END']  # FID-A: DOECC=F
    open(ctrl, 'w', newline='\n').write('\n'.join(L) + '\n')
    return ctrl


def parse_table(tbl):
    if not os.path.exists(tbl):
        return None
    txt = open(tbl, errors='ignore').read()
    if 'FATAL' in txt.split('$$CONC')[0]:
        return None
    out = {}
    for line in txt.splitlines():
        m = re.match(r'\s*([\d.E+-]+)\s+(\d+)%\s+[\d.E+-]+\s+(\w+)', line)
        if m:
            out[m.group(3)] = (float(m.group(1)), float(m.group(2)))
    fw = re.search(r'FWHM\s*=\s*([\d.]+)', txt); sn = re.search(r'S/N\s*=\s*(\d+)', txt)
    return dict(mets=out, fwhm=float(fw.group(1)) if fw else np.nan,
                snr=float(sn.group(1)) if sn else np.nan)


def fit_maps(met, wat, ppm, mask, out_dir):
    """Per-voxel LCModel fit (one control at a time) -> conc/CRLB/LW/SNR maps.
    met/wat: [f, y, x] arrays (apodized met + apodized water, NO ACME = FID-A-
    faithful). Writes per-voxel RAW/control/table, maps.npz, lcm_maps.png. Returns
    dict(conc, crlb, LW, SNR, mask)."""
    os.makedirs(out_dir, exist_ok=True)
    ny, nx = mask.shape
    ijk = np.argwhere(mask)
    conc = {m: np.full((ny, nx), np.nan) for m in METS}
    crlb = {m: np.full((ny, nx), np.nan) for m in METS}
    LW = np.full((ny, nx), np.nan); SNR = np.full((ny, nx), np.nan)
    nunfil = met.shape[0]                          # zero-filled length (e.g. 4096)
    print(f'LCModel: fitting {len(ijk)} voxels ONE-BY-ONE (NUNFIL={nunfil})...')
    nok = 0
    for n, (y, x) in enumerate(ijk, 1):
        tag = f'{x}x{y}'; stem = os.path.join(out_dir, tag)
        write_raw(spec_to_fid(met[:, y, x]), stem + '.RAW')
        write_raw(spec_to_fid(wat[:, y, x]), stem + '_w.RAW')
        ctrl = write_control(stem, out_dir, tag, nunfil=nunfil)
        subprocess.run(['wsl', 'bash', '-lc', f'"{LCM}" < "{win2wsl(ctrl)}"'],
                       capture_output=True, text=True)       # one voxel at a time
        r = parse_table(stem + '.table')
        if r and r['mets']:
            nok += 1
            for m, (c, sd) in r['mets'].items():
                if m in conc:
                    conc[m][y, x] = c; crlb[m][y, x] = sd
            LW[y, x] = r['fwhm']; SNR[y, x] = r['snr']
        if n % 25 == 0 or n == len(ijk):
            print(f'  [{n:4d}/{len(ijk)}]  {nok} fit OK', flush=True)
    print(f'{nok}/{len(ijk)} voxels fit OK')
    np.savez(os.path.join(out_dir, 'maps.npz'),
             **{f'conc_{m}': conc[m] for m in METS},
             **{f'crlb_{m}': crlb[m] for m in METS}, LW=LW, SNR=SNR, mask=mask)
    for m in METS:
        c = crlb[m]; print(f'  {m}: median CRLB {np.nanmedian(c):.0f}%  CRLB<=20 in {int(np.nansum(c <= 20))} vox')
    render_lcm_maps(conc, crlb, LW, SNR, mask, os.path.join(out_dir, 'lcm_maps.png'))
    return dict(conc=conc, crlb=crlb, LW=LW, SNR=SNR, mask=mask)


def render_lcm_maps(conc, crlb, LW, SNR, mask, out_path, crlb_cap=20):
    """conc (CRLB-masked) + CRLB + S/N/FWHM/ratio panel from the fit maps."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from viz.plotspec import imagesc
    fig, ax = plt.subplots(3, 4, figsize=(15, 10.5))
    for i, met in enumerate(METS):
        c = conc[met].copy(); q = crlb[met]
        c[(q > crlb_cap) | ~mask] = np.nan
        imagesc(c, title=f'{met} conc (CRLB<={crlb_cap}%)', ax=ax[0][i], cmap='viridis')
        imagesc(np.where(mask, q, np.nan), title=f'{met} CRLB %', ax=ax[1][i], cmap='inferno', clim=(0, 30))
    imagesc(np.where(mask, SNR, np.nan), title='S/N', ax=ax[2][0], cmap='cividis')
    imagesc(np.where(mask, LW, np.nan), title='FWHM (ppm)', ax=ax[2][1], cmap='cividis', clim=(0, 0.08))
    rat = conc['Cho'] / conc['Cr']; rat[(crlb['Cr'] > crlb_cap) | (crlb['Cho'] > crlb_cap) | ~mask] = np.nan
    imagesc(rat, title='Cho/Cr ratio', ax=ax[2][2], cmap='magma', clim=(0, 1.2))
    ok = (~np.isnan(conc['Cr'])) & mask
    ax[2][3].imshow(ok, cmap='gray', origin='upper'); ax[2][3].set_title('fit OK mask'); ax[2][3].axis('off')
    fig.suptitle('LCModel maps (per-voxel .table outputs)', fontsize=13)
    plt.tight_layout(); plt.savefig(out_path, dpi=100); plt.close(fig)
    print('rendered', out_path)


def main(met_stage='s11_smooth_met', wat_stage='s11_smooth_ref'):
    from viz._common import make_mask, water_map, load_stage
    met, ppm = load_fyx(met_stage); wat, _ = load_fyx(wat_stage)
    mask = make_mask(water_map(load_stage(os.path.join(D, 's11_smooth_ref.npz'))))
    fit_maps(met, wat, ppm, mask, OUT)


if __name__ == '__main__':
    main()
