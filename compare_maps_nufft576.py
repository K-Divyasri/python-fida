"""compare_maps_nufft576.py -- my new maps (nufft/pipe_menon 576, apodize fixed)
vs FID-A production tables. NOTE: production tables are from OLDER FID-A code
(pre apodize-fix, 1-px shifted) -> a stale reference; true validation needs a
current-code FID-A LCModel run. Reports corr/ratio/voxel-diff + side-by-side.
"""
import os, re, glob, warnings; warnings.filterwarnings('ignore')
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

MY = r'C:\Users\divya\Downloads\mrsi_pipeline\lcm_nufft576\maps.npz'
FALCM = r'F:\fida\divya\20260605_phantom_test\subject02\outputs\lcm_out'
METS = ['Cr', 'Cho', 'Act', 'Lac']


def parse_conc(tbl):
    out = {}
    for line in open(tbl, errors='ignore'):
        m = re.match(r'\s*([\d.E+-]+)\s+(\d+)%\s+[\d.E+-]+\s+(\w+)', line)
        if m:
            out[m.group(3)] = (float(m.group(1)), float(m.group(2)))
    return out


def fida_maps(shape):
    conc = {m: np.full(shape, np.nan) for m in METS}
    crlb = {m: np.full(shape, np.nan) for m in METS}
    for tbl in glob.glob(os.path.join(FALCM, '*_ftSpec_smooth_lcm.table')):
        mm = re.match(r'(\d+)x(\d+)', os.path.basename(tbl))
        if not mm: continue
        x, y = int(mm.group(1)), int(mm.group(2))
        c = parse_conc(tbl)
        for met in METS:
            if met in c and 0 <= y < shape[0] and 0 <= x < shape[1]:
                conc[met][y, x], crlb[met][y, x] = c[met]
    return conc, crlb


def main():
    z = np.load(MY); mask = z['mask'].astype(bool)
    fac, facr = fida_maps(mask.shape)
    print('=== my nufft576 (apodize fixed)  vs  FID-A production tables (STALE) ===')
    print(f'{"met":5s} {"corr":>7s} {"ratio my/fa":>12s} {"med vox %diff":>14s} {"n":>5s}')
    for met in METS:
        a = z[f'conc_{met}']; b = fac[met]
        good = mask & (z[f'crlb_{met}'] <= 20) & (facr[met] <= 20) & np.isfinite(a) & np.isfinite(b) & (b != 0)
        if good.sum() < 5:
            print(f'{met:5s}  (n={int(good.sum())}, skip)'); continue
        corr = np.corrcoef(a[good], b[good])[0, 1]
        rat = np.median(a[good] / b[good])
        rd = np.median(np.abs(a[good] - b[good]) / np.abs(b[good]))
        print(f'{met:5s} {corr:7.3f} {rat:12.3f} {rd*100:13.1f}% {int(good.sum()):5d}')

    # side-by-side
    fig, ax = plt.subplots(3, 4, figsize=(15, 10))
    for i, met in enumerate(['Cr', 'Cho', 'Act']):
        a = z[f'conc_{met}'].copy(); b = fac[met].copy()
        a[(z[f'crlb_{met}'] > 20) | ~mask] = np.nan
        b[(facr[met] > 20) | ~mask] = np.nan
        vmax = np.nanpercentile(np.concatenate([a[np.isfinite(a)], b[np.isfinite(b)]]), 98)
        for j, (M, t) in enumerate([(a, f'{met} PYTHON'), (b, f'{met} FID-A(stale)'), (a - b, f'{met} diff')]):
            im = ax[i][j].imshow(M, origin='upper', cmap='viridis' if j < 2 else 'RdBu_r',
                                 vmin=0 if j < 2 else -vmax * .4, vmax=vmax if j < 2 else vmax * .4)
            ax[i][j].set_title(t); plt.colorbar(im, ax=ax[i][j], fraction=.046)
        ax[i][3].axis('off')
    fig.suptitle('nufft/pipe_menon 576, apodize FIXED  |  Python vs FID-A(stale prod)')
    plt.tight_layout()
    out = r'C:\Users\divya\Downloads\mrsi_pipeline\lcm_nufft576\py_vs_fida_nufft576.png'
    plt.savefig(out, dpi=100); print('saved', out)


if __name__ == '__main__':
    main()
