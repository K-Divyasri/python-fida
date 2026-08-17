"""compare_l2_stepwise.py -- FID-A vs Python, step by step, voxel-wise (<2%).

Stages: deterministic chain (prep->recon->combine->avg->spectral->SSP) —
  my rosette_py_stages_l2_ref  vs  FID-A gt_20260605_stages.
  Per stage: mag corr + per-voxel relative diff (median, %voxels<2%).
Maps: LCModel conc (Cr/Cho/Lac) — my lcm_l2_ref  vs  FID-A fida_maps_20260605.
  Per metabolite: corr + per-voxel |diff|/FID-A (median, %voxels<2%), in mask & CRLB<=20.
"""
import os, json, re, glob, warnings; warnings.filterwarnings('ignore')
import numpy as np, h5py

PY = r'C:\Users\divya\Downloads\mrsi_pipeline\rosette_py_stages_l2_ref'
GT = r'C:\Users\divya\Downloads\mrsi_pipeline\gt_20260605_stages'
MYLCM = r'C:\Users\divya\Downloads\mrsi_pipeline\lcm_l2_ref'
# USER'S real FID-A LCModel reference (no-SSP phantom run) -- the correct reference.
FALCM = r'F:\fida\divya\20260605_phantom_test\subject02\outputs\lcm_out'

STAGE_PAIRS = [('s02_prep_met', 's01_prep', 'prep'), ('s03_recon_met', 's03_recon', 'recon'),
               ('s04_combine_met', 's04_combine', 'combine'), ('s05_ccav_met', 's05_ccav', 'average'),
               ('s06_spec_met', 's06_spec', 'spectral FT')]
# NB: SSP intentionally skipped for phantom (removes Lac) -> not compared.


def load_py(key):
    z = np.load(os.path.join(PY, key + '.npz'), allow_pickle=True)
    m = json.loads(str(z['meta']))
    return z['data'], {k: int(v) for k, v in m['dims'].items() if v}


def load_gt(key):
    with h5py.File(os.path.join(GT, key + '.mat'), 'r') as f:
        d = f['data'][()]
        a = (d['real'] + 1j * d['imag']) if (d.dtype.names and 'real' in d.dtype.names) else d
        data = np.transpose(a, tuple(range(a.ndim)[::-1]))
        dims = {k: int(np.ravel(f['dims'][k][()])[0]) for k in f['dims'] if int(np.ravel(f['dims'][k][()])[0]) > 0}
    return data, dims


def align(py, pd, gd):
    names = [k for k, v in sorted(gd.items(), key=lambda kv: kv[1])]
    order = [pd[n] - 1 for n in names if n in pd]
    return (np.transpose(py, order) if len(order) == py.ndim else py,
            [n for n in names if n in pd])


def stage_stats(pk, gk):
    py, pd = load_py(pk); gt, gd = load_gt(gk)
    pya, names = align(py, pd, gd)
    if pya.shape != gt.shape:
        return ('shape', pya.shape, gt.shape)
    corr = np.corrcoef(np.abs(pya).ravel(), np.abs(gt).ravel())[0, 1]
    scale = np.abs(gt).max() or 1.0
    p99 = float(np.percentile(np.abs(pya - gt) / scale, 99))
    vox = None
    if 'y' in names and 'x' in names:
        yi, xi = names.index('y'), names.index('x')
        pm = np.moveaxis(pya, [yi, xi], [-2, -1]); gm = np.moveaxis(gt, [yi, xi], [-2, -1])
        ny, nx = pm.shape[-2:]
        pm = pm.reshape(-1, ny, nx); gm = gm.reshape(-1, ny, nx)
        num = np.linalg.norm(pm - gm, axis=0); den = np.linalg.norm(gm, axis=0)
        sig = den > np.percentile(den, 50)
        rd = num[sig] / den[sig]
        vox = (float(np.median(rd)), float(np.mean(rd < 0.02) * 100), int(sig.sum()))
    return corr, p99, vox


def parse_conc(tbl):
    if not os.path.exists(tbl):
        return {}
    out = {}
    for line in open(tbl, errors='ignore'):
        m = re.match(r'\s*([\d.E+-]+)\s+(\d+)%\s+[\d.E+-]+\s+(\w+)', line)
        if m:
            out[m.group(3)] = (float(m.group(1)), float(m.group(2)))
    return out


def fida_conc_maps(shape, mets):
    conc = {m: np.full(shape, np.nan) for m in mets}
    crlb = {m: np.full(shape, np.nan) for m in mets}
    for tbl in glob.glob(os.path.join(FALCM, '*_ftSpec_smooth_lcm.table')) or \
               glob.glob(os.path.join(FALCM, '*', '*.table')):
        mm = re.match(r'(\d+)x(\d+)', os.path.basename(tbl))
        if not mm:
            continue
        x, y = int(mm.group(1)), int(mm.group(2))
        c = parse_conc(tbl)
        for met in mets:
            if met in c and 0 <= y < shape[0] and 0 <= x < shape[1]:
                conc[met][y, x], crlb[met][y, x] = c[met]
    return conc, crlb


def main():
    print('=== STAGES: FID-A vs Python ===')
    print(f'{"stage":13s} {"corr":>8s} {"p99|d|/scale":>13s} {"vox %diff(med)":>15s} {"%vox<2%":>9s}')
    print('-' * 62)
    for pk, gk, name in STAGE_PAIRS:
        r = stage_stats(pk, gk)
        if r and r[0] == 'shape':
            print(f'{name:13s}  SHAPE {r[1]} vs {r[2]}'); continue
        corr, p99, vox = r
        v = f'{vox[0]*100:14.3f}% {vox[1]:8.1f}%' if vox else f'{"(k-space)":>24s}'
        print(f'{name:13s} {corr:8.4f} {p99*100:12.2f}% {v}')

    print('\n=== LCModel CONC MAPS: FID-A vs Python (voxel-wise, in mask & CRLB<=20) ===')
    mets = ['Cr', 'Cho', 'Lac']
    my = np.load(os.path.join(MYLCM, 'maps.npz')); mask = my['mask'].astype(bool)
    fac, facr = fida_conc_maps(mask.shape, mets)
    print(f'{"metab":6s} {"corr":>8s} {"median vox %diff":>16s} {"% vox <2%":>10s} {"n":>5s}')
    print('-' * 52)
    for met in mets:
        a = my[f'conc_{met}']; b = fac[met]
        good = mask & (my[f'crlb_{met}'] <= 20) & (facr[met] <= 20) & np.isfinite(a) & np.isfinite(b) & (b != 0)
        if good.sum() == 0:
            print(f'{met:6s}  no overlap'); continue
        corr = np.corrcoef(a[good], b[good])[0, 1]
        rd = np.abs(a[good] - b[good]) / np.abs(b[good])
        print(f'{met:6s} {corr:8.4f} {np.median(rd) * 100:15.2f}% {np.mean(rd < 0.02) * 100:9.1f}% {int(good.sum()):5d}')


if __name__ == '__main__':
    main()
