"""fit_one_voxel_lcm.py -- single-voxel LCModel fit to check basis + fit quality.

Fits one strong-metabolite voxel of the processed rosette data with LCModel
(via WSL), using the TE/SW-matched phantom basis. Prints conc + CRLB + FWHM + S/N
so we can see whether the basis was the problem. Optionally compares a second
(mismatched) basis.

Data: SW 1587 Hz -> DELTAT 6.3e-4, NUNFIL 576, ECHOT 15 ms, HZPPPM 123.199.
"""
import os, subprocess, re
import numpy as np
import suspect
from suspect.io.lcmodel import save_raw

D = r'C:\Users\divya\Downloads\mrsi_pipeline\rosette_py_stages_20260605'
OUT = r'C:\Users\divya\Downloads\mrsi_pipeline\lcm_onevox'
BASIS_GOOD = r'C:\Users\divya\Downloads\fida codes\fid_a\basis_r14_phantom\CRPhantom_LacAceCrCho_TE15_SW1587.basis'
BASIS_BAD = r'C:\Users\divya\Downloads\fida codes\fid_a\basis_phantom\Phantom_LacAceCrCho_TE2p3_SW4000_2048.basis'
LCM = '/home/divya/.lcmodel/bin/lcmodel'
OWNER = 'Divyasri Krishnkaumar, Sunnybrook Research Institute'
KEY = os.environ.get('LCMODEL_KEY', '')                       # LCModel license key (from run_lcm_rosette_portable)
DT = 6.300e-4; F0 = 123.19877; TE = 15.0    # HZPPPM matches basis header (avoids MYBASI)
PPMST, PPMEND = 4.25, 0.2


def win2wsl(p):
    p = os.path.abspath(p); return '/mnt/' + p[0].lower() + p[2:].replace('\\', '/')


def load_fyx(key):
    import json
    z = np.load(os.path.join(D, key + '.npz'), allow_pickle=True)
    meta = json.loads(str(z['meta'])); dims = {k: int(v) for k, v in meta['dims'].items() if v}
    d = z['data']; fax = (dims.get('f') or dims.get('t')) - 1
    return np.transpose(d, (fax, dims['y'] - 1, dims['x'] - 1)), np.asarray(z['ppm'])


def spec_to_fid(spec):
    # op_CSItoMRS convention: fids = fft(fftshift(specs))
    return np.fft.fft(np.fft.fftshift(spec))


def write_lcm_raw(fid, outfile, te, txfrq_hz, dwelltime):
    """Port of FID-A io_writelcm: LCModel RAW with real + (-imag) [conjugate],
    fmtdat='(2E15.6)'. This sign convention is what LCModel expects."""
    fid = np.asarray(fid).ravel()
    n = fid.size
    with open(outfile, 'w', newline='\n') as f:
        f.write(' $SEQPAR')
        f.write(f'\n echot= {te:.2f}')
        f.write("\n seq= 'PRESS'")
        f.write(f'\n hzpppm= {txfrq_hz / 1e6:.6f}')
        f.write(f'\n NumberOfPoints= {n}')
        f.write(f'\n dwellTime= {dwelltime:.6f}')
        f.write('\n $END')
        f.write('\n $NMID')
        f.write("\n id='ANONYMOUS ', fmtdat='(2E15.6)'")
        f.write('\n volume=8.0')
        f.write('\n tramp=1.0')
        f.write('\n $END\n')
        for v in fid:
            f.write('  % .6e  % .6e\n' % (v.real, -v.imag))   # -imag = conjugate


def write_control(stem, basis, with_water):
    """Mirror run_lcm_rosette_portable's control exactly (licensed + phantom)."""
    ctrl = stem + '.control'
    L = ['$LCMODL', f" OWNER='{OWNER}'", f' KEY={KEY}', " TITLE='rosette voxel'",
         f" FILRAW='{win2wsl(stem + '.RAW')}'", f" FILBAS='{win2wsl(basis)}'",
         f" FILPS='{win2wsl(stem + '.ps')}'", f" FILTAB='{win2wsl(stem + '.table')}'",
         f" FILCOO='{win2wsl(stem + '.coord')}'", f" FILCSV='{win2wsl(stem + '.csv')}'",
         f" FILPRI='{win2wsl(stem + '.print')}'",
         f' HZPPPM={F0:.5f}', f' DELTAT={DT:.8f}', f' NUNFIL={576}', f' ECHOT={TE:g}',
         ' LPRINT=6', ' LTABLE=7', ' LCOORD=9', ' LCSV=11', ' LPS=8', ' DOREFS(1)=T',
         f' PPMST={PPMST:g}', f' PPMEND={PPMEND:g}', ' RFWHM=0.15', ' WDLINE(1)=0.025',
         ' DKNTMN=1.0', ' NEACH=999', ' NSIMUL=0', ' NOMIT=0', ' NNORAT=0', ' NRATIO=0',
         ' NUSE1=3', " CHUSE1(1)='Cr'", " CHUSE1(2)='Cho'", " CHUSE1(3)='Lac'"]
    if with_water:
        L += [' DOECC=T', ' DOWS=T', f" FILH2O='{win2wsl(stem + '_h2o.RAW')}'",
              ' ATTH2O=1.0', ' WCONC=55556']
    L += ['$END']
    open(ctrl, 'w', newline='\n').write('\n'.join(L) + '\n')
    return ctrl


def run_lcm(ctrl):
    inner = f'{LCM} < "{win2wsl(ctrl)}"'
    return subprocess.run(['wsl', 'bash', '-lc', inner], capture_output=True, text=True)


def parse_table(tbl):
    if not os.path.exists(tbl):
        return None
    txt = open(tbl, errors='ignore').read()
    out = {}
    for line in txt.splitlines():
        m = re.match(r'\s*([\d.E+-]+)\s+([\d.]+)%\s+([\d.E+-]+)\s+(\w[\w+]*)', line)
        if m:
            out[m.group(4)] = (float(m.group(1)), float(m.group(2)))   # conc, %SD
    fw = re.search(r'FWHM\s*=\s*([\d.]+)', txt); sn = re.search(r'S/N\s*=\s*([\d.]+)', txt)
    return dict(mets=out, fwhm=float(fw.group(1)) if fw else None,
               snr=float(sn.group(1)) if sn else None)


def fit(basis, label):
    os.makedirs(OUT, exist_ok=True)
    met, ppm = load_fyx('s09b_phased_met')
    wat, _ = load_fyx('s06_spec_ref')
    from viz.maps import op_CSIintegrate  # noqa
    # strong-Cr voxel
    cr = np.abs(np.real(met[(ppm > 2.95) & (ppm < 3.10)]).sum(0))
    y, x = np.unravel_index(np.argmax(cr), cr.shape)
    stem = os.path.join(OUT, f'vox_{label}')
    mfid = spec_to_fid(met[:, y, x]); wfid = spec_to_fid(wat[:, y, x])
    write_lcm_raw(mfid, stem + '.RAW', TE, F0 * 1e6, DT)
    write_lcm_raw(wfid, stem + '_h2o.RAW', TE, F0 * 1e6, DT)
    ctrl = write_control(stem, basis, with_water=True)
    p = run_lcm(ctrl)
    res = parse_table(stem + '.table')
    print(f'\n=== {label}: basis {os.path.basename(basis)}  voxel ({y},{x}) ===')
    if res is None:
        print('  NO .table produced. stderr tail:', (p.stdout + p.stderr)[-400:])
        return
    print(f'  FWHM {res["fwhm"]} ppm   S/N {res["snr"]}')
    for m, (c, sd) in sorted(res['mets'].items()):
        print(f'    {m:10s} conc {c:8.3f}   CRLB {sd:5.0f}%')


if __name__ == '__main__':
    fit(BASIS_GOOD, 'TE15_SW1587_correct')
    fit(BASIS_BAD, 'TE2p3_SW4000_wrong')
