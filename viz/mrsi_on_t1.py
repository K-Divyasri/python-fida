"""viz/mrsi_on_t1.py -- overlay MRSI on a T1 (mrsi_on_t1_map / nii_viewer_mrsi).

Two paths:
  overlay_nifti  : write an MRSI overlay NIfTI (map or 4D spectra) whose affine is
                   copied VERBATIM from a spec2nii empty template (as FID-A does),
                   then view in fsleyes/nii_viewer alongside the T1.
  overlay_static : matplotlib overlay of an MRSI map on a T1 slice (quick look).

Uses rot180 (flip both spatial axes) -- the overlay orientation confirmed correct
by test_overlay_flip (FID-A's mrsi_on_t1_map hardcodes 'lr'; rot180 is the fix).
"""
import os
import numpy as np

from ._common import to_fyx, water_map, make_mask
from .maps import op_CSIintegrate


def _rot180(a):
    return np.flip(np.flip(a, 0), 1)


def integrate_signal_map(struct, ppm_range=(1.8, 3.5), mode='mag'):
    """nii_viewer_mrsi-style map: sum |spectrum| over a ppm band -> [y,x] (normalised)."""
    m = op_CSIintegrate(struct, ppm_range[0], ppm_range[1], mode='mag' if mode == 'mag' else 're')
    m = np.abs(m)
    p99 = np.percentile(m[m > 0], 99) if np.any(m > 0) else 1.0
    return np.minimum(m / (p99 or 1.0), 1.0)


def overlay_nifti(struct, out_path, empty_template=None, ppm_range=None,
                  as_4d=False, flip='rot180', mode='real'):
    """Write an MRSI overlay NIfTI.
    empty_template : path to a spec2nii empty NIfTI -> its affine/header are copied
                     verbatim (FID-A behaviour). If None, a simple geometry affine.
    as_4d          : write full spectra [x,y,1,f]; else a 2D integration map [x,y,1].
    ppm_range      : band for the 2D map (default whole spectrum peak)."""
    import nibabel as nib
    d = to_fyx(struct['data'], struct['dims'])
    flipper = {'none': lambda a: a, 'lr': lambda a: a[::-1],
               'ud': lambda a: a[:, ::-1], 'rot180': _rot180}[flip]
    if as_4d:
        proj = {'real': np.real, 'abs': np.abs, 'imag': np.imag}[mode]
        v = proj(d).astype(np.float32)               # [f,y,x]
        v = np.stack([flipper(v[k]) for k in range(v.shape[0])], 0)
        vol = np.transpose(v, (2, 1, 0))[:, :, None, :]  # [x,y,1,f]
    else:
        rng = ppm_range or (float(np.min(struct['ppm'])), float(np.max(struct['ppm'])))
        m = flipper(integrate_signal_map(struct, rng))
        vol = np.transpose(m, (1, 0))[:, :, None]        # [x,y,1]

    if empty_template and os.path.exists(empty_template):
        tmpl = nib.load(empty_template)
        affine, header = tmpl.affine, tmpl.header.copy()
    else:
        from .nifti_out import mrsi_affine
        affine, header = mrsi_affine(struct), None
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nib.save(nib.Nifti1Image(vol, affine, header), out_path)
    return out_path


def overlay_static(struct, t1_path=None, ppm_range=(1.8, 3.5), slice_idx=None,
                   flip='rot180', alpha=0.5, cmap='hot', out_path=None,
                   mask=True, ref_struct=None):
    """matplotlib overlay of an MRSI integration map on a T1 slice (quick look).
    If t1_path is None, uses the (ref) water map as pseudo-anatomy. mask may be a
    [y,x] array (e.g. from the water ref); True = derive from ref_struct/struct."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    flipper = {'none': lambda a: a, 'lr': lambda a: a[::-1],
               'ud': lambda a: a[:, ::-1], 'rot180': _rot180}[flip]
    src = ref_struct if ref_struct is not None else struct        # water anatomy from ref
    mp = flipper(integrate_signal_map(struct, ppm_range))
    if mask is not None and mask is not False:
        mm = mask if not isinstance(mask, bool) else make_mask(water_map(src))
        mp = np.where(flipper(np.asarray(mm)), mp, np.nan)

    fig, ax = plt.subplots(figsize=(5, 5))
    if t1_path and os.path.exists(t1_path):
        import nibabel as nib
        t1 = nib.load(t1_path).get_fdata()
        if slice_idx is None:
            slice_idx = t1.shape[2] // 2
        bg = t1[:, :, slice_idx].T
        ax.imshow(bg, cmap='gray', origin='lower')
        # resize overlay to bg (nearest) -- coarse; real registration via fsleyes
        from scipy.ndimage import zoom
        zy, zx = bg.shape[0] / mp.shape[0], bg.shape[1] / mp.shape[1]
        mp_rs = zoom(np.nan_to_num(mp), (zy, zx), order=0)
        mp_rs = np.where(mp_rs > 0, mp_rs, np.nan)
        ax.imshow(mp_rs, cmap=cmap, origin='lower', alpha=alpha)
        ax.set_title(f'MRSI ({ppm_range[0]}-{ppm_range[1]} ppm) on T1 [slice {slice_idx}]')
    else:
        bg = flipper(water_map(src))
        ax.imshow(bg, cmap='gray', origin='lower')
        ax.imshow(mp, cmap=cmap, origin='lower', alpha=alpha)
        ax.set_title(f'MRSI ({ppm_range[0]}-{ppm_range[1]} ppm) on water map (no T1)')
    ax.axis('off')
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=110, bbox_inches='tight'); plt.close(fig)
        return out_path
    return fig
