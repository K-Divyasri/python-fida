"""run_viz.py -- produce all MRSI visualisers from a pipeline stage.

Generates (into viz_out/): water map, spectral grid (op_CSIPlot1), single-voxel
spectrum, peak-integration metabolite panel, an MRSI-on-T1 overlay (static +
NIfTI), a 4D MRSI NIfTI, and a click-voxel static render. If an LCModel output
folder is given, also builds concentration + CRLB map panels and per-metabolite
NIfTIs.

Run:  python run_viz.py
      python run_viz.py <stage.npz> <out_dir> [t1.nii.gz] [lcm_dir] [empty_template.nii]
"""
import os, sys
import numpy as np

from viz._common import load_stage, water_map, make_mask
from viz.spectrum import plot_spectrum
from viz.spectral_grid import spectral_grid
from viz.maps import water_map_fig, metabolite_panel, lcmodel_maps, op_CSIintegrate
from viz.mrsi_on_t1 import overlay_static, overlay_nifti
from viz.nifti_out import write_mrsi_4d, write_map_nifti, write_metabolite_niftis
from viz.voxel_viewer import render_voxel
from viz.lcmodel_read import parse_lcmodel_tables

# defaults
STAGE = r'C:\Users\divya\Downloads\mrsi_pipeline\rosette_py_stages\s11_smooth_ref.npz'
OUT   = r'C:\Users\divya\Downloads\mrsi_pipeline\viz_out'


def main(stage=STAGE, out=OUT, t1=None, lcm=None, empty=None):
    os.makedirs(out, exist_ok=True)
    s = load_stage(stage)
    tag = os.path.splitext(os.path.basename(stage))[0]
    # MASK from the WATER REF (met water is suppressed/HSVD-removed -> can't mask from it)
    ref_path = stage.replace('_met.npz', '_ref.npz')
    sref = load_stage(ref_path) if (os.path.exists(ref_path) and ref_path != stage) else s
    w = water_map(sref); mask = make_mask(w)
    # spectrum/voxel display at a STRONG-metabolite voxel (not the low-signal centroid)
    cr = np.where(mask, np.abs(op_CSIintegrate(s, 2.95, 3.10, 'mag')), 0) if s.get('ppm') is not None else np.zeros_like(w)
    if cr.max() > 0:
        cy, cx = map(int, np.unravel_index(np.argmax(cr), cr.shape))
    else:
        ys, xs = np.where(mask); cy, cx = int(ys.mean()), int(xs.mean())
    print(f'stage {tag}  data{s["data"].shape}  mask {int(mask.sum())} vox (from ref)  peak-voxel ({cy},{cx})')

    figs = {}
    figs['water_map']   = water_map_fig(sref, mask=mask, out_path=f'{out}\\{tag}_water.png')
    figs['spectrum']    = plot_spectrum(s, cy, cx, mode='real', xlim=(0.2, 5.2),
                                        out_path=f'{out}\\{tag}_spectrum.png')
    figs['grid']        = spectral_grid(s, mode='real', ppm_range=(0.2, 5.2), mask=mask,
                                        out_path=f'{out}\\{tag}_grid.png')
    figs['met_panel']   = metabolite_panel(s, mask=mask, out_path=f'{out}\\{tag}_metpanel.png')
    figs['overlay']     = overlay_static(s, t1_path=t1, ppm_range=(1.8, 3.5), mask=mask,
                                         ref_struct=sref, out_path=f'{out}\\{tag}_overlay.png')
    figs['voxel']       = render_voxel(s, cy, cx, bg_struct=sref, out_path=f'{out}\\{tag}_voxel.png')
    # NIfTI exports (fsleyes/nii_viewer)
    figs['mrsi_4d.nii'] = write_mrsi_4d(s, f'{out}\\{tag}_mrsi4d.nii.gz')
    figs['overlay.nii'] = overlay_nifti(s, f'{out}\\{tag}_overlay.nii.gz',
                                        empty_template=empty, ppm_range=(1.8, 3.5))
    figs['water.nii']   = write_map_nifti(w, s, f'{out}\\{tag}_water.nii.gz')

    if lcm and os.path.isdir(lcm):
        d = load_stage(stage)  # for size
        from viz._common import to_fyx
        _, ny, nx = to_fyx(d['data'], d['dims']).shape
        conc, crlb, LW, SNR = parse_lcmodel_tables(lcm, nx, ny)
        figs['lcm_maps'] = lcmodel_maps(conc, crlb, mask=mask,
                                        out_path=f'{out}\\{tag}_lcm_maps.png')
        figs['lcm.nii'] = write_metabolite_niftis(conc, s, f'{out}\\lcm_nii', crlb=crlb)
        print('LCModel maps + NIfTIs written')

    print('\nwrote:')
    for k, v in figs.items():
        print(f'  {k:14s} {v}')
    return figs


if __name__ == '__main__':
    a = sys.argv[1:]
    main(*(a + [None] * (5 - len(a)))[:5]) if a else main()
