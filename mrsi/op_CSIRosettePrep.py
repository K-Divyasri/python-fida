"""op_CSIRosettePrep.py -- non-cartesian loader-side prep, transpiled from FID-A.

io_CSIload_twix_pair runs these on rosette/concentric data AFTER the raw twix
read, BEFORE op_CSIRecon, to turn the merged interleaved readout into the
[t, coils, (averages), kpts, kshot] layout op_CSIRecon expects:

    op_CSICombineTime1 -> combine_time()     (merge t + extras into one FID)
    op_CSIShift        -> op_CSIShift()       (rosette VOI off-centre k-phase)
    splitReadoutKpts   -> split_readout_kpts()(split merged readout -> t,kpts)

postProcess order (FID-A):
    rosette    : combine_time('extras') -> op_CSIShift(x,y,kFile) -> split_readout_kpts
    concentric : combine_time('extras') ->                          split_readout_kpts

Operates on the FID-A-style struct dict (dims 1-based).  Also provides a bridge
to read a MATLAB-dumped MRSIStruct (.mat) into this Python struct, because the
installed Python twix parsers can't read Siemens XA60 -- load in MATLAB, dump,
process here.
"""
import copy
import numpy as np
from op_CSIRecon import read_kfile, _ax


def _insert_axis_dims(dims, at_axis, new_name, new_axis_1b):
    """After inserting one array axis at 0-based position `at_axis`, bump every
    dim whose 1-based axis > at_axis by +1, then assign new_name -> new_axis_1b."""
    out = {}
    for k, v in dims.items():
        if v and (v - 1) >= at_axis:
            out[k] = v + 1
        else:
            out[k] = v
    out[new_name] = new_axis_1b
    return out


def combine_time(struct, dim_name='extras'):
    """op_CSICombineTime1: collapse the t dim and `dim_name` (e.g. interleaved
    'extras' segments) into a single time dim. Cartesian -> unchanged."""
    if struct['flags'].get('isCartesian'):
        return copy.deepcopy(struct)
    at = _ax(struct, 't')
    ae = _ax(struct, dim_name)
    if ae is None:
        return copy.deepcopy(struct)
    s = copy.deepcopy(struct)
    d = s['data']
    # move extras adjacent AFTER t, merge (t, extras) with t slowest (FID-A keeps
    # t outer, segments inner -> one continuous FID). MATLAB reshape is col-major
    # over (t, extras) but semantics = concatenate segments along time.
    order = [at, ae] + [a for a in range(d.ndim) if a not in (at, ae)]
    A = np.transpose(d, order)                      # [t, extras, rest...]
    nt, nex = A.shape[0], A.shape[1]
    # merge (t, extras) column-major = t fastest (matches FID-A combineTime1)
    A = A.reshape((nt * nex,) + A.shape[2:], order='F')
    s['data'] = A
    s['sz'] = tuple(A.shape)
    # rebuild dims: t now axis1, everything that wasn't t/extras follows
    newdims = {k: 0 for k in s['dims']}
    newdims['t'] = 1
    ax = 2
    for a in order[2:]:
        # find the name at original axis a
        for k, v in struct['dims'].items():
            if v == a + 1:
                newdims[k] = ax; ax += 1
                break
    s['dims'] = newdims
    return s


def op_CSIShift(struct, x_mm, y_mm, kfile):
    """op_CSIShift: rosette VOI off-centre correction. Multiplies each k-sample
    by exp(-1i*2*pi*(kx_norm*dx + ky_norm*dy)) on the still-merged readout.
    x_mm,y_mm = VOI shift (mm) from computeFOVShift. Applied BEFORE kpts split."""
    s = copy.deepcopy(struct)
    traj = read_kfile(kfile)
    kx, ky = traj['kx'].copy(), traj['ky'].copy()
    kp, nshot = traj['kPtsPerCycle'], traj['nShots']
    fovx = s['fov']['x']; fovy = s['fov']['y']
    kxmax = np.max(np.abs(kx)) or 1.0
    kymax = np.max(np.abs(ky)) or 1.0
    # FID-A op_CSIShift renormalisation: (kx/(2*kxmax))/fov
    kxn = (kx / (2 * kxmax)) / fovx
    kyn = (ky / (2 * kymax)) / fovy
    phase = np.exp(-1j * 2 * np.pi * (kxn * x_mm + kyn * y_mm))   # [Nk]
    # tile each shot's kPtsPerCycle phase across the merged time axis
    at = _ax(s, 't')
    tm = s['data'].shape[at]
    # merged readout = nT repeats of the kp-per-shot pattern; phase is per (kpt,shot)
    # build [tm] phase for the current shot layout: readout has kp fastest, repeated
    ph_cycle = phase.reshape(kp, nshot, order='F')  # [kpts, kshot]
    # data axis layout here: [t(=merged kp*nT? no, pre-split t=nT*kp), coils, (avg), kshot]
    # phase depends on (kpt within readout, shot). Apply per kshot slice.
    aks = _ax(s, 'kshot')
    d = s['data']
    nT = tm // kp
    ph_t = np.tile(ph_cycle, (nT, 1))               # [tm, kshot], kpt fastest per cycle
    shp = [1] * d.ndim
    shp[at] = tm; shp[aks] = nshot
    s['data'] = d * ph_t.reshape(shp)
    return s


def split_readout_kpts(struct, kPtsPerCycle):
    """splitReadoutKpts: split merged readout t=(nT*kpts, kpts fastest) into
    separate t (nT) and kpts dims. -> [t, coils, (avg), kpts, kshot]."""
    s = copy.deepcopy(struct)
    d = s['data']
    at = _ax(s, 't')
    kp = kPtsPerCycle
    tm = d.shape[at]
    nT = tm // kp
    # split axis `at` (size tm) into (nT, kp) with kp fastest -> C-order (nT, kp)
    newshape = d.shape[:at] + (nT, kp) + d.shape[at + 1:]
    d2 = np.reshape(d, newshape)                    # C-order: kp is faster (last) -> matches col-major kp-fastest
    s['data'] = d2
    s['sz'] = tuple(d2.shape)
    # t stays at axis `at` (1-based at+1); new kpts axis at `at+1` (1-based at+2)
    s['dims'] = _insert_axis_dims(s['dims'], at + 1, 'kpts', at + 2)
    return s


def prep_noncartesian(struct, kfile, seq_type='rosette', voi_shift=None):
    """Full FID-A postProcess for non-cartesian: combine_time -> (rosette)
    op_CSIShift -> split_readout_kpts. Returns [t,coils,(avg),kpts,kshot] ready
    for op_CSIRecon.

    voi_shift=None (default) uses the loader-computed (xShift_mm, yShift_mm) from
    computeFOVShift; pass an explicit (dx,dy) tuple to override, or (0,0) to skip."""
    if voi_shift is None:
        voi_shift = (struct.get('xShift_mm', 0.0), struct.get('yShift_mm', 0.0))
    traj = read_kfile(kfile)
    s = combine_time(struct, 'extras')
    if seq_type == 'rosette' and (voi_shift[0] or voi_shift[1]):
        s = op_CSIShift(s, voi_shift[0], voi_shift[1], kfile)
    s = split_readout_kpts(s, traj['kPtsPerCycle'])
    return s


# ------------------------------------------------ MATLAB struct bridge (XA60)
def load_struct_mat(path, var='s'):
    """Read a MATLAB-dumped FID-A MRSIStruct (.mat, v7 or v7.3) into this
    Python struct dict. Use when the raw twix is Siemens XA60 that Python twix
    parsers can't read: load with FID-A io_CSIload_twix_pair in MATLAB, save the
    struct, read here, then run op_CSIRecon.

    Expects the .mat to contain fields: data, sz, dims (struct), spectralWidth,
    dwelltime, txfrq, te, tr, fov(struct), voxelSize(struct), flags(struct),
    and (for non-cartesian) dims.kpts / dims.kshot set."""
    try:
        from scipy.io import loadmat
        mat = loadmat(path, simplify_cells=True)
        m = mat[var] if var in mat else mat[[k for k in mat if not k.startswith('__')][0]]
        return _mat_to_struct(m)
    except NotImplementedError:
        import h5py
        with h5py.File(path, 'r') as f:
            if 'data' in f and 'dims' in f:      # MATLAB save('-struct',...) -> top-level fields
                g = f
            else:
                g = f[var] if var in f else f[[k for k in f if not k.startswith('#')][0]]
            return _mat_to_struct_h5(f, g)


def _to_dims(dd):
    out = {}
    for k in dd:
        v = dd[k]
        out[k] = int(np.ravel(v)[0]) if np.size(v) else 0
    return out


def _mat_to_struct(m):
    def xyz(v):
        return {'x': float(v['x']), 'y': float(v['y']), 'z': float(v.get('z', 0))}
    dims = _to_dims(m['dims'])
    d = np.asarray(m['data'])
    return dict(
        data=d, sz=tuple(d.shape), dims=dims,
        spectralWidth=float(m['spectralWidth']), dwelltime=float(m['dwelltime']),
        spectralTime=np.ravel(m.get('spectralTime', np.arange(d.shape[dims['t'] - 1]) * float(m['dwelltime']))),
        adcTime=np.ravel(m.get('adcTime', np.arange(d.shape[dims['t'] - 1]) * float(m['dwelltime']))),
        txfrq=float(m['txfrq']), te=float(m.get('te', 0)), tr=float(m.get('tr', 0)),
        Bo=float(m['txfrq']) / 42.5774e6,
        fov=xyz(m['fov']), voxelSize=xyz(m['voxelSize']),
        imageOrigin=list(np.ravel(m.get('imageOrigin', [0, 0, 0]))),
        normal=list(np.ravel(m.get('normal', [0, 0, 1]))),
        ppm=None, seq=str(m.get('seq', '')),
        flags=_norm_flags({k: int(np.ravel(m['flags'][k])[0]) for k in m['flags']}) if 'flags' in m else
               {'spatialft': 0, 'spectralft': 0, 'isCartesian': 0},
    )


def _norm_flags(fl):
    """FID-A uses camelCase flag names (spatialFT/spectralFT); alias to my
    lowercase (spatialft/spectralft/addedrcvrs) so struct checks work."""
    alias = {'spatialFT': 'spatialft', 'spectralFT': 'spectralft',
             'coilCombined': 'addedrcvrs', 'isCartesian': 'isCartesian'}
    for src, dst in alias.items():
        if src in fl:
            fl.setdefault(dst, fl[src])
    return fl


def _mat_to_struct_h5(f, g):
    def arr(name):
        d = g[name][()]
        if d.dtype.names and 'real' in d.dtype.names:
            d = d['real'] + 1j * d['imag']
        return np.transpose(d, tuple(range(d.ndim)[::-1]))   # F->C
    def scal(name, default=0.0):
        try: return float(np.ravel(g[name][()])[0])
        except Exception: return default
    def sub(name):
        grp = g[name]
        return {k: float(np.ravel(grp[k][()])[0]) for k in grp}
    dims = {k: int(np.ravel(g['dims'][k][()])[0]) for k in g['dims']}
    d = arr('data')
    fl = _norm_flags({k: int(np.ravel(g['flags'][k][()])[0]) for k in g['flags']}) if 'flags' in g else \
         {'spatialft': 0, 'spectralft': 0, 'isCartesian': 0}
    dt = scal('dwelltime')
    return dict(data=d, sz=tuple(d.shape), dims=dims,
                spectralWidth=scal('spectralWidth'), dwelltime=dt,
                spectralTime=np.arange(d.shape[dims['t'] - 1]) * dt,
                adcTime=np.arange(d.shape[dims['t'] - 1]) * dt,
                txfrq=scal('txfrq'), te=scal('te'), tr=scal('tr'),
                Bo=scal('txfrq') / 42.5774e6, fov=sub('fov'), voxelSize=sub('voxelSize'),
                imageOrigin=[0, 0, 0], normal=[0, 0, 1], ppm=None, seq='', flags=fl)
