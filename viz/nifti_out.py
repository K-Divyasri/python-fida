"""viz/nifti_out.py -- write MRSI maps/spectra to NIfTI for fsleyes / nii_viewer.

- mrsi_affine        build a voxel->world affine from struct geometry (mm)
- write_map_nifti    a 2D/3D map -> NIfTI
- write_mrsi_4d      spectra [f,y,x] -> 4D NIfTI (x,y,1,f) like mrsi_on_t1_map
- write_metabolite_niftis   {met: 2D map} -> one NIfTI each
                            (create_separate_metabolite_niftis_v2 equivalent)

The 180 flip (flip both spatial axes) puts the map in overlay orientation vs the
T1, matching FID-A's mrsi_on_t1_map default.
"""
import os
import numpy as np

from ._common import to_fyx, water_map, make_mask


def _flip180(a):
    return np.flip(np.flip(a, 0), 1)


def mrsi_affine(struct):
    """Simple axial voxel->world affine (mm) from fov/voxelSize + imageOrigin.
    Good enough to place the MRSI slab; exact T1 registration should use the
    scanner VOI affine when available."""
    vx = struct.get('voxelSize', {}) or {}
    dx = float(vx.get('x', 1)); dy = float(vx.get('y', 1)); dz = float(vx.get('z', 1))
    org = struct.get('imageOrigin', [0, 0, 0]) or [0, 0, 0]
    A = np.diag([dx, dy, dz, 1.0]).astype(float)
    A[:3, 3] = np.asarray(org[:3], float)
    return A


def _save(arr, affine, path):
    import nibabel as nib
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(arr), affine), path)
    return path


def write_map_nifti(map2d, struct, path, flip180=True):
    """2D [y,x] map -> 3D NIfTI [x,y,1]."""
    m = np.asarray(map2d, float)
    if flip180:
        m = _flip180(m)
    vol = np.transpose(m, (1, 0))[:, :, None]        # [x, y, 1]
    return _save(vol, mrsi_affine(struct), path)


def write_mrsi_4d(struct, path, flip180=True, mode='real'):
    """Spectra [f,y,x] -> 4D NIfTI [x, y, 1, f] (nii_viewer time dimension = ppm)."""
    d = to_fyx(struct['data'], struct['dims'])
    proj = {'real': np.real, 'abs': np.abs, 'imag': np.imag}[mode]
    v = proj(d).astype(np.float32)                   # [f, y, x]
    if flip180:
        v = v[:, ::-1, ::-1]
    vol = np.transpose(v, (2, 1, 0))[:, :, None, :]  # [x, y, 1, f]
    import nibabel as nib
    img = nib.Nifti1Image(vol, mrsi_affine(struct))
    dt = float(struct.get('dwelltime', 1.0) or 1.0)
    img.header['pixdim'][4] = dt
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nib.save(img, path)
    return path


def write_metabolite_4d(conc, crlb, LW, SNR, struct, outdir, flip180=True):
    """create_separate_metabolite_niftis_v2 equivalent: one 4D NIfTI per metabolite
    with channels [conc, CRLB, LW, SNR] ([x,y,1,4]); shared LW/SNR maps too."""
    import nibabel as nib
    os.makedirs(outdir, exist_ok=True)
    def prep(m):
        m = np.asarray(m, float)
        if flip180: m = _flip180(m)
        return np.transpose(m, (1, 0))                # [x,y]
    A = mrsi_affine(struct); paths = []
    for met in conc:
        ch = [prep(conc[met]), prep(crlb.get(met, np.zeros_like(conc[met]))),
              prep(LW), prep(SNR)]
        vol = np.stack(ch, axis=-1)[:, :, None, :]    # [x,y,1,4]
        p = os.path.join(outdir, f'{met.lower()}_4D.nii.gz')
        nib.save(nib.Nifti1Image(vol, A), p); paths.append(p)
    _save(prep(LW)[:, :, None], A, os.path.join(outdir, 'Linewidth_3D.nii.gz'))
    _save(prep(SNR)[:, :, None], A, os.path.join(outdir, 'SNR_3D.nii.gz'))
    return paths


def write_metabolite_niftis(conc, struct, outdir, crlb=None, flip180=True):
    """{met: 2D conc} (+ optional {met: 2D crlb}) -> per-metabolite NIfTIs."""
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for met, m in conc.items():
        paths.append(write_map_nifti(m, struct, os.path.join(outdir, f'{met}_conc.nii.gz'), flip180))
        if crlb and met in crlb:
            paths.append(write_map_nifti(crlb[met], struct,
                                         os.path.join(outdir, f'{met}_crlb.nii.gz'), flip180))
    return paths
