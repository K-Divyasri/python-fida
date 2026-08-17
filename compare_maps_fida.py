"""compare_maps_fida.py -- my LCModel metabolite maps vs FID-A's, voxel-by-voxel.

Parses FID-A per-voxel .table (from run_lcm_rosette_portable -> fida_maps_20260605\)
into conc/CRLB maps, loads my maps (lcm_full\maps.npz), and compares where BOTH
voxels fit reliably: per-metabolite median ratio, correlation, %diff, + a Cr
deviation map. Tells us whether the deviation is a global scale, a spatial trend,
or random voxel noise.
"""
import os, re, glob, warnings; warnings.filterwarnings('ignore')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from fit_lcmodel_rosette import parse_table
from viz.plotspec import imagesc

FDIR = r'C:\Users\divya\Downloads\mrsi_pipeline\fida_maps_20260605'
MYMAPS = r'C:\Users\divya\Downloads\mrsi_pipeline\lcm_full\maps.npz'
METS = ['Cr', 'Cho', 'Lac', 'Act']


def load_fida_maps(ny, nx):
    conc = {m: np.full((ny, nx), np.nan) for m in METS}
    crlb = {m: np.full((ny, nx), np.nan) for m in METS}
    n = 0
    for tbl in glob.glob(os.path.join(FDIR, '*.table')):
        mm = re.search(r'(\d+)x(\d+)', os.path.basename(tbl))
        if not mm:
            continue
        x, y = int(mm.group(1)), int(mm.group(2))
        if not (0 <= x < nx and 0 <= y < ny):
            continue
        r = parse_table(tbl)
        if not r or not r['mets']:
            continue
        n += 1
        for met, (c, sd) in r['mets'].items():
            if met in conc:
                conc[met][y, x] = c; crlb[met][y, x] = sd
    print(f'FID-A tables parsed: {n}')
    return conc, crlb


def main():
    my = np.load(MYMAPS); mask = my['mask'].astype(bool); ny, nx = mask.shape
    fc, fq = load_fida_maps(ny, nx)
    print(f'\n{"met":5s} {"n both":>7s} {"median ratio(mine/fida)":>24s} {"corr":>7s} {"mean|%diff|":>11s}')
    print('-' * 62)
    for met in METS:
        mc = my[f'conc_{met}']; mq = my[f'crlb_{met}']
        both = (mq <= 20) & (fq[met] <= 20) & mask & np.isfinite(fc[met]) & np.isfinite(mc)
        a = mc[both]; b = fc[met][both]
        if a.size < 3:
            print(f'{met:5s}  (too few common voxels: {a.size})'); continue
        ratio = np.median(a / b); corr = np.corrcoef(a, b)[0, 1]
        pdiff = np.mean(np.abs(a - b) / b) * 100
        print(f'{met:5s} {a.size:7d} {ratio:24.3f} {corr:7.3f} {pdiff:10.1f}%')
    # Cr deviation map + scatter
    met = 'Cr'; mc = my[f'conc_{met}']; both = (my[f'crlb_{met}'] <= 20) & (fq[met] <= 20) & mask
    fig, ax = plt.subplots(1, 4, figsize=(17, 4))
    imagesc(np.where(both, mc, np.nan), title='mine Cr', ax=ax[0], cmap='viridis')
    imagesc(np.where(both, fc[met], np.nan), title='FID-A Cr', ax=ax[1], cmap='viridis')
    imagesc(np.where(both, mc / fc[met], np.nan), title='ratio mine/FID-A', ax=ax[2], cmap='RdBu_r', clim=(0.7, 1.3))
    ax[3].scatter(fc[met][both], mc[both], s=6, alpha=0.5); mx = np.nanmax(fc[met][both])
    ax[3].plot([0, mx], [0, mx], 'k--', lw=1); ax[3].set_xlabel('FID-A Cr'); ax[3].set_ylabel('mine Cr'); ax[3].set_title('scatter')
    plt.tight_layout(); out = r'C:\Users\divya\Downloads\mrsi_pipeline\lcm_full\cr_vs_fida.png'
    plt.savefig(out, dpi=100); print('saved', out)


if __name__ == '__main__':
    main()
