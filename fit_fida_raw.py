"""fit_fida_raw.py -- pass ALL of FID-A's RAW files through my fitter, compare to
FID-A's own .table across every voxel. Validates the fitter end-to-end and
isolates whether any map difference is the fit or the pipeline data.

FID-A RAW live in <subj>\outputs\lcm_out\<x>x<y>_ftSpec_smooth_lcm (+ _w), with
FID-A's fits in <x>x<y>_ftSpec_smooth_lcm.table. Copies them local (F: is flaky),
re-fits each with my write_control (RosettePhantom basis, DOECC=F), and reports
per-metabolite corr + median ratio + max per-voxel diff.
"""
import os, sys, glob, re, time, shutil, subprocess, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mrsi'))
import numpy as np
import fit_lcmodel_rosette as F

SRC = r'F:\fida\divya\20260605_phantom_test\subject02\outputs\lcm_out'
LOC = r'C:\Users\divya\Downloads\mrsi_pipeline\tmp\fida_raw_all'
OUT = r'C:\Users\divya\Downloads\mrsi_pipeline\lcm_fidaRAW_myfit'
METS = ['Cr', 'Cho', 'Lac', 'Act']


def _retry(fn):
    for _ in range(15):
        try:
            r = fn()
            if r:
                return r
        except Exception:
            pass
        time.sleep(3)
    return None


def copy_all():
    os.makedirs(LOC, exist_ok=True)
    mets = _retry(lambda: glob.glob(os.path.join(SRC, '*_ftSpec_smooth_lcm')))
    if not mets:
        print('F: down, no RAW copied'); return []
    tags = []
    for m in mets:
        tag = os.path.basename(m).replace('_ftSpec_smooth_lcm', '')
        for suf, dst in [('_ftSpec_smooth_lcm', tag + '.RAW'),
                         ('_ftSpec_smooth_lcm_w', tag + '_w.RAW'),
                         ('_ftSpec_smooth_lcm.table', tag + '.fida.table')]:
            src = os.path.join(SRC, tag + suf)
            if os.path.exists(src):
                try: shutil.copy2(src, os.path.join(LOC, dst))
                except Exception: pass
        if os.path.exists(os.path.join(LOC, tag + '.RAW')) and os.path.exists(os.path.join(LOC, tag + '_w.RAW')):
            tags.append(tag)
    print(f'copied {len(tags)} voxel RAW pairs')
    return tags


def tabconc(p):
    if not os.path.exists(p):
        return None
    txt = open(p, errors='ignore').read()
    if 'FATAL' in txt.split('$$CONC')[0]:
        return None
    out = {}
    for l in txt.splitlines():
        m = re.match(r'\s*([\d.E+-]+)\s+(\d+)%\s+[\d.E+-]+\s+(\w+)', l)
        if m:
            out[m.group(3)] = (float(m.group(1)), float(m.group(2)))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    tags = copy_all()
    if not tags:
        return
    xs = [int(t.split('x')[0]) for t in tags]; ys = [int(t.split('x')[1]) for t in tags]
    ny, nx = max(ys) + 1, max(xs) + 1
    my = {m: np.full((ny, nx), np.nan) for m in METS}
    fa = {m: np.full((ny, nx), np.nan) for m in METS}
    myq = {m: np.full((ny, nx), np.nan) for m in METS}; faq = {m: np.full((ny, nx), np.nan) for m in METS}
    print(f'fitting {len(tags)} FID-A RAW voxels with my control...', flush=True)
    for n, tag in enumerate(tags, 1):
        x, y = int(tag.split('x')[0]), int(tag.split('x')[1])
        stem = os.path.join(LOC, tag)
        ctrl = F.write_control(stem, LOC, tag)
        subprocess.run(['wsl', 'bash', '-lc', f'"{F.LCM}" < "{F.win2wsl(ctrl)}"'], capture_output=True, text=True)
        r = tabconc(stem + '.table'); fr = tabconc(os.path.join(LOC, tag + '.fida.table'))
        for m in METS:
            if r and m in r: my[m][y, x], myq[m][y, x] = r[m]
            if fr and m in fr: fa[m][y, x], faq[m][y, x] = fr[m]
        if n % 50 == 0 or n == len(tags):
            print(f'  [{n}/{len(tags)}]', flush=True)
    np.savez(os.path.join(OUT, 'compare.npz'),
             **{f'my_{m}': my[m] for m in METS}, **{f'fa_{m}': fa[m] for m in METS},
             **{f'myq_{m}': myq[m] for m in METS}, **{f'faq_{m}': faq[m] for m in METS})
    print('\n=== my-fitter-on-FIDA-RAW  vs  FID-A .table (all voxels) ===')
    print(f'{"met":5s} {"n both":>7s} {"corr":>7s} {"med ratio":>10s} {"max|d|/med":>11s}')
    for m in METS:
        a, b = my[m], fa[m]
        v = np.isfinite(a) & np.isfinite(b) & (b > 0)
        if v.sum() < 10:
            print(f'{m:5s} (too few)'); continue
        corr = np.corrcoef(a[v], b[v])[0, 1]
        ratio = np.nanmedian(a[v] / b[v]); md = np.abs(a[v] - b[v]).max() / (np.nanmedian(b[v]) or 1)
        print(f'{m:5s} {int(v.sum()):7d} {corr:7.4f} {ratio:10.3f} {md:11.2e}')
    print('saved', os.path.join(OUT, 'compare.npz'))


if __name__ == '__main__':
    main()
