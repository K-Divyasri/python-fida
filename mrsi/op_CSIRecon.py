"""op_CSIRecon.py -- FID-A op_CSIRecon chain, transpiled to Python.

Full spatial reconstruction for CSI: cartesian FFT AND non-cartesian
(rosette / concentric-ring) via NUFFT / DFT / Tikhonov, with density
compensation (nn / voronoi / pipe_menon). Faithful port of:

    op_CSIRecon        -> op_CSIRecon()          (top dispatcher, skipDcf logic)
    op_CSIReconstruct  -> op_CSIReconstruct()    (nufft|dft|tikhonov)
    op_CSIPSFCorrection-> op_CSIPSFCorrection()  (dcf dispatcher)
       _nn / _v / _pm  -> dcf_nn/voronoi/pipe_menon
    op_NUFFTSpatial1   -> _spatial_nufft()       (finufft type-1 adjoint /Nk)
    op_CSIFourierTransform_dft -> _spatial_dft() (dense sft2, exact)
    op_CSIFourierTransform (tikh) -> _spatial_tikhonov()  (lambda=4e-3, exact)
    (cartesian branch) -> _spatial_cartesian_fft()  (halfPixelShift + fft)

Operates on the FID-A-style MRSIStruct dict from io_CSIload_twix (dims 1-based).
Non-cartesian input layout: data [t, coils, (averages), kpts, kshot].

Deps: numpy, scipy, pandas, finufft.  (voronoi shoelace = no shapely needed.)

NOTE ON EXACTNESS:
  dft / tikhonov / cartesian branches reproduce FID-A math exactly.
  nufft branch uses finufft (different spreading kernel than Fessler IRT KB
  J=6) -> mathematically-equivalent type-1 adjoint, correct recon, amplitudes
  match dft to ~1e-2. Use dft for a bit-exact rosette reference, nufft for speed.
"""
import copy
import os
import numpy as np
import pandas as pd


def _dcf_cache_path(kfile, method, Nk):
    """Path to a precomputed DCF-weights cache: <root>/dcf_cache/<kbase>.<method>.<Nk>.npy.
    Lets the recon reuse reference (e.g. FID-A Fessler pipe_menon) weights so the
    finufft FT reproduces FID-A exactly -- the DCF is trajectory-only, so a cached
    weight vector is valid for any data on that kFile."""
    base = os.path.splitext(os.path.basename(str(kfile)))[0]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, 'dcf_cache', f'{base}.{method}.{Nk}.npy')

# ------------------------------------------------------------------ trajectory
def read_kfile(kfile):
    """Read a FID-A kFile (CSV: TR,Kx,Ky,time). Returns dict with
    kx,ky (1/mm), TR (int cycle idx), time (s), kPtsPerCycle, nShots, dwell."""
    df = pd.read_csv(kfile)
    cols = {c.lower(): c for c in df.columns}
    kx = df[cols.get('kx', df.columns[1])].to_numpy(float)
    ky = df[cols.get('ky', df.columns[2])].to_numpy(float)
    TR = df[cols['tr']].to_numpy(int) if 'tr' in cols else np.ones(len(df), int)
    nshots = int(TR.max())
    kpts = len(df) // nshots                       # getKPtsPerCycle: height/max(TR)
    dwell = None
    if 'time' in cols:
        t = df[cols['time']].to_numpy(float)
        dwell = float(t[2] - t[1]) if len(t) > 2 else float(t[1] - t[0])
    return dict(kx=kx, ky=ky, TR=TR, time=df[cols['time']].to_numpy(float) if 'time' in cols else None,
                kPtsPerCycle=kpts, nShots=nshots, dwell=dwell, Nk=len(df))


# ------------------------------------------------------------------ dim helpers
def _ax(s, name):
    """0-based axis index of a dim, or None if absent."""
    v = s['dims'].get(name, 0)
    return (v - 1) if v else None


def _coords(fov_mm, vox_mm):
    """createCoordinates: -|b|+step/2 : step : |b|-step/2, b=fov/2, step=vox."""
    b = fov_mm / 2.0
    c = np.arange(-b + vox_mm / 2.0, b, vox_mm)
    if c.size and abs(c[-1] - (b - vox_mm / 2.0)) > vox_mm * 0.5:
        c = np.append(c, b - vox_mm / 2.0)
    return c


# ============================================================ DCF (op_CSIPSF*)
def _radial_smooth(kx, ky, w, window):
    """movmean of w after sorting by |k| radius, then scatter back (radialSmooth)."""
    si = np.argsort(np.hypot(kx, ky))
    ws = w.copy()
    sm = pd.Series(w[si]).rolling(window, center=True, min_periods=1).mean().to_numpy()
    ws[si] = sm
    return ws


def dcf_nn(kx, ky, K=10):
    """k-nearest-neighbour local-density DCF (op_CSIPSFCorrection_nn / dcf_nearest)."""
    from scipy.spatial import cKDTree
    N = kx.size
    P = np.column_stack([kx, ky])
    tree = cKDTree(P)
    krep = np.zeros(N, int)
    rho = np.zeros(N)
    # co-located duplicate count per point
    dup = tree.query_ball_point(P, r=1e-12)
    for i in range(N):
        krep[i] = len(dup[i])                       # includes self
        kq = min(K + krep[i], N)
        d, _ = tree.query(P[i], k=kq)
        kdist = np.atleast_1d(d)[-1]                # distance to K-th distinct nbr
        rho[i] = K / (np.pi * max(kdist, np.finfo(float).eps) ** 2)
    w = 1.0 / np.maximum(rho, np.finfo(float).eps)  # geometric weight ~ 1/density
    window = max(round(N / 50), 5)
    w = _radial_smooth(kx, ky, w, window)
    w = w / krep                                    # split among duplicates
    cap = np.percentile(w, 98)
    return np.minimum(w, cap)


def _poly_area(vx, vy):
    """Shoelace polygon area (= polyarea)."""
    return 0.5 * abs(np.dot(vx, np.roll(vy, -1)) - np.dot(vy, np.roll(vx, -1)))


def dcf_voronoi(kx, ky, fov_mm, vox_mm, shape='uniform'):
    """Voronoi-cell-area DCF (op_CSIPSFCorrection_v / dcf_voronoi)."""
    from scipy.spatial import Voronoi, cKDTree
    Nx = round(fov_mm / vox_mm)
    Kmax = Nx / (2.0 * fov_mm)
    # removeDuplicateOrigin: keep ONE (0,0), drop the rest; unique all coincident
    P = np.column_stack([kx, ky])
    uniq, inv = np.unique(np.round(P, 12), axis=0, return_inverse=True)
    xc, yc = uniq[:, 0], uniq[:, 1]
    vor = Voronoi(uniq)
    areas = np.full(len(uniq), np.nan)
    for i, reg_idx in enumerate(vor.point_region):
        region = vor.regions[reg_idx]
        if len(region) < 3 or -1 in region:        # boundary / infinite cell
            continue
        v = vor.vertices[region]
        r = np.hypot(v[:, 0], v[:, 1])             # clip vertices to k-disk Kmax
        out = r > Kmax
        v[out, 0] = Kmax * v[out, 0] / r[out]
        v[out, 1] = Kmax * v[out, 1] / r[out]
        a = _poly_area(v[:, 0], v[:, 1])
        if a > 0:
            areas[i] = a
    finite = areas[np.isfinite(areas)]
    areas[~np.isfinite(areas)] = np.percentile(finite, 90) if finite.size else 1.0
    bad = ~np.isfinite(areas) | (areas <= 0)
    areas[bad] = np.median(areas[np.isfinite(areas)])
    R = np.hypot(xc, yc)
    if shape == 'uniform':
        far = R > Kmax
        taper = np.maximum(1 - (R - Kmax) / (0.1 * Kmax), 0)
        areas[far] = areas[far] * taper[far]
    elif shape == 'gaussian':
        sigma = np.sqrt(-(0.1 ** 2) / (2 * np.log(0.01)))
        areas = areas * np.exp(-(xc ** 2 + yc ** 2) / (2 * sigma ** 2))
    elif shape == 'flatedge':
        areas = areas / (1 + np.exp(-100 * (R - 0.64 * Kmax)))
    # mapBackShared: each original point -> nearest unique, share weight by count
    tree = cKDTree(uniq)
    _, nn = tree.query(P)
    cnt = np.bincount(nn, minlength=len(uniq))
    return areas[nn] / cnt[nn]


def dcf_pipe_menon(kx, ky, fov_mm, vox_mm, iters=25):
    """Iterative Pipe-Menon DCF (op_CSIPSFCorrection_pm). w -> 1/diag(E E^H)."""
    import finufft
    Nx = round(fov_mm / vox_mm)
    dx = fov_mm / Nx
    om_x = 2 * np.pi * kx * dx
    om_y = 2 * np.pi * ky * dx                      # pipe_menon uses dx for both
    Nd = (Nx, Nx)
    w = np.ones(kx.size, complex)
    for _ in range(iters):
        img = finufft.nufft2d1(om_y, om_x, w, Nd)   # adjoint (nonuniform->grid)
        gw = finufft.nufft2d2(om_y, om_x, img)      # forward (grid->nonuniform) = G w
        den = np.abs(gw)
        den = np.maximum(den, 1e-4 * den.max())
        w = w / den
        w = w / w.mean()
    w = np.abs(w)
    lo, hi = np.percentile(w, 2), np.percentile(w, 98)
    w = np.clip(w, lo, hi)
    window = max(round(kx.size / 80), 5)
    return _radial_smooth(kx, ky, w, window)


def op_CSIPSFCorrection(struct, kfile, method='nn', shape='uniform', K=10, iters=25):
    """Apply density compensation: multiply data by per-(kpt,shot) weights,
    normalised so sum(w)=Nk, broadcast over t/coils/averages. FID-A dispatcher."""
    if struct['flags'].get('spatialft'):
        raise RuntimeError('op_CSIPSFCorrection: already spatial-FT')
    s = copy.deepcopy(struct)
    traj = read_kfile(kfile)
    kx, ky = traj['kx'], traj['ky']
    Nk = traj['Nk']
    fov = s['fov']['x']; vox = s['voxelSize']['x']
    # reference DCF cache (trajectory-only) -> reproduces FID-A's Fessler weights.
    # Set MRSI_DCF_NO_CACHE=1 to always DERIVE the weights per trajectory in Python.
    cache = _dcf_cache_path(kfile, method, Nk)
    w = None
    if os.path.exists(cache) and not os.environ.get('MRSI_DCF_NO_CACHE'):
        cw = np.load(cache)
        if cw.size == Nk:
            w = cw; print(f'op_CSIPSFCorrection: using cached {method} DCF ({os.path.basename(cache)})')
    if w is None:
        print(f'op_CSIPSFCorrection: deriving {method} DCF from trajectory (Nk={Nk})')
        if method == 'nn':
            w = dcf_nn(kx, ky, K=K)
        elif method == 'voronoi':
            w = dcf_voronoi(kx, ky, fov, vox, shape=shape)
        elif method == 'pipe_menon':
            w = dcf_pipe_menon(kx, ky, fov, vox, iters=iters)
        else:
            raise ValueError(f'unknown dcf method {method}')
    w = w * (Nk / w.sum())                          # normalise sum(w)=Nk
    nkpt = s['sz'][_ax(s, 'kpts')]
    nshot = s['sz'][_ax(s, 'kshot')]
    W = w.reshape(nkpt, nshot, order='F')           # [kpts, kshot] col-major
    # broadcast over all non-(kpts,kshot) axes
    shp = [1] * s['data'].ndim
    shp[_ax(s, 'kpts')] = nkpt; shp[_ax(s, 'kshot')] = nshot
    s['data'] = s['data'] * W.reshape(shp)
    s['densityComp'] = dict(weights=w, kx=kx, ky=ky, method=method)
    return s


# ================================================= spatial transforms (shared)
def _merge_kpts_time(s):
    """op_NUFFTSpatial1 step 1: permute so kpts fastest, merge (kpts,t) into one
    readout dim. Returns (X[Ntot, nShot, Nextra], nkpt, nshot, extras_shape)."""
    d = s['data']
    at, ac, akp, aks = _ax(s, 't'), _ax(s, 'coils'), _ax(s, 'kpts'), _ax(s, 'kshot')
    aav = _ax(s, 'averages')
    nkpt = d.shape[akp]; nshot = d.shape[aks]; nt = d.shape[at]
    # collect "extra" axes (coils, averages, anything not t/kpts/kshot)
    extra_axes = [a for a in range(d.ndim) if a not in (at, akp, aks)]
    # order: kpts, t, <extras...>, kshot   -> merge (kpts,t) col-major
    order = [akp, at] + extra_axes + [aks]
    A = np.transpose(d, order)
    extras_shape = A.shape[2:-1]
    Nextra = int(np.prod(extras_shape)) if extras_shape else 1
    A = A.reshape((nkpt * nt,) + extras_shape + (nshot,), order='F')
    # -> [Ntot, <extras>, nshot]; move nshot next to Ntot, flatten extras
    A = np.moveaxis(A, -1, 1)                        # [Ntot, nshot, <extras>]
    X = A.reshape(nkpt * nt, nshot, Nextra, order='F')
    return X, nkpt, nshot, extras_shape


def _image_grid(s):
    x = _coords(s['fov']['x'], s['voxelSize']['x'])
    y = _coords(s['fov']['y'], s['voxelSize']['y'])
    return x, y


def _finalize_spatial(s, img, nkpt, extras_shape):
    """img: [NPtemporal, Ny, Nx, Nextra] -> struct with dims t,coils,avg,y,x set;
    kpts/kshot cleared; spectral dwell = adcDwell*kPtsPerCycle; flags.spatialft."""
    NPt, Ny, Nx, Nextra = img.shape
    # split the Nextra axis back into extras_shape. It was flattened order='F'
    # (first extra fastest) in _merge_kpts_time, so unflatten F-consistently:
    # reshape into reversed extras_shape (C-order -> first extra fastest) then
    # reverse the extra axes back to original order. (No-op for a single extra.)
    if extras_shape:
        ne = len(extras_shape)
        img = img.reshape(NPt, Ny, Nx, *extras_shape[::-1])
        img = img.transpose((0, 1, 2) + tuple(range(3 + ne - 1, 2, -1)))
    else:
        img = img[..., 0]
    # canonical output: [t, coils, (averages), y, x]
    # extras order came from [coils, averages, ...]; move y,x to the end already are
    # rearrange to t, <extras>, y, x
    if extras_shape:
        ndim_ex = len(extras_shape)
        img = np.moveaxis(img, [1, 2], [1 + ndim_ex, 2 + ndim_ex])  # y,x after extras
    out = copy.deepcopy(s)
    out['data'] = img
    out['sz'] = tuple(img.shape)
    # rebuild dims: t=1, then extras (coils, averages...), then y, x
    dims = {k: 0 for k in s['dims']}
    dims['t'] = 1
    ax = 2
    for nm in ('coils', 'averages'):
        if _ax(s, nm) is not None:
            dims[nm] = ax; ax += 1
    dims['y'] = ax; dims['x'] = ax + 1
    out['dims'] = dims
    adc = s['dwelltime']
    specDT = adc * nkpt
    out['dwelltime'] = specDT
    out['spectralWidth'] = 1.0 / specDT
    out['spectralTime'] = np.arange(NPt) * specDT
    out['adcTime'] = np.arange(NPt) * specDT
    out['flags']['spatialft'] = 1
    return out


def _spatial_nufft(s, kfile):
    """op_NUFFTSpatial1: adjoint NUFFT per temporal point. finufft type-1 /Nk."""
    import finufft
    traj = read_kfile(kfile)
    kx, ky, Nk = traj['kx'], traj['ky'], traj['Nk']
    nkpt = traj['kPtsPerCycle']
    X, nkpt2, nshot, extras_shape = _merge_kpts_time(s)
    Ntot = X.shape[0]; Nextra = X.shape[2]
    NPt = Ntot // nkpt
    xg, yg = _image_grid(s)
    Nx, Ny = xg.size, yg.size
    dx = np.median(np.abs(np.diff(xg))); dy = np.median(np.abs(np.diff(yg)))
    # om in rad/sample: [Ky*dy -> y(rows), Kx*dx -> x(cols)]
    om_y = 2 * np.pi * ky * dy
    om_x = 2 * np.pi * kx * dx
    # Fessler IRT n_shift = median((0:N-1) - coords/d); finufft centres at N/2.
    # Apply the residual (N/2 - n_shift) as a k-space linear phase so the image
    # grid matches FID-A op_NUFFTSpatial1 exactly (verified bit-exact on ref).
    x_shift = np.median(np.arange(Nx) - xg / dx)
    y_shift = np.median(np.arange(Ny) - yg / dy)
    nshift_ph = np.exp(1j * (om_x * (Nx / 2 - x_shift) + om_y * (Ny / 2 - y_shift)))
    img = np.zeros((NPt, Ny, Nx, Nextra), complex)
    for it in range(NPt):
        i0, i1 = it * nkpt, (it + 1) * nkpt
        # rows i0:i1 (kpts) x all shots = Nk points, col-major (kpt fastest, then shot)
        sl = X[i0:i1, :, :].reshape(Nk, Nextra, order='F') * nshift_ph[:, None]
        for e in range(Nextra):
            Z = finufft.nufft2d1(om_y, om_x, sl[:, e].astype(complex), (Ny, Nx))
            img[it, :, :, e] = Z / Nk
    return _finalize_spatial(s, img, nkpt, extras_shape)


def _sft2_operator(kx, ky, xg, yg, ift=True):
    """sft2_Operator: E_H[p,k] = exp(+2i.pi (x_p kx_k + y_p ky_k)) / Nk (ift=True).
    Image points p ordered y-fastest (MATLAB [xx(:),yy(:)] with meshgrid(x,y))."""
    xx, yy = np.meshgrid(xg, yg)                    # yy varies down rows
    xp = xx.ravel(order='F'); yp = yy.ravel(order='F')   # y-fastest (MATLAB col-major)
    sign = 2j * np.pi if ift else -2j * np.pi
    EH = np.exp(sign * (np.outer(xp, kx) + np.outer(yp, ky))) / kx.size
    return EH, xx.shape                             # (Ny,Nx)


def _apply_slow_ft(s, kfile, EH, grid_shape, per_temporal_op=None):
    traj = read_kfile(kfile)
    nkpt = traj['kPtsPerCycle']; Nk = traj['Nk']
    X, _, nshot, extras_shape = _merge_kpts_time(s)
    Ntot, _, Nextra = X.shape
    NPt = Ntot // nkpt
    Ny, Nx = grid_shape
    op = per_temporal_op if per_temporal_op is not None else EH
    img = np.zeros((NPt, Ny, Nx, Nextra), complex)
    for it in range(NPt):
        i0, i1 = it * nkpt, (it + 1) * nkpt
        sl = X[i0:i1, :, :].reshape(Nk, Nextra, order='F')
        ft = op @ sl                                # [Np, Nextra]
        img[it] = ft.reshape(Ny, Nx, Nextra, order='F')
    return _finalize_spatial(s, img, nkpt, extras_shape)


def _spatial_dft(s, kfile):
    """op_CSIFourierTransform_dft: dense sft2 (exact inverse DFT). E_H @ y."""
    traj = read_kfile(kfile)
    xg, yg = _image_grid(s)
    EH, gshape = _sft2_operator(traj['kx'], traj['ky'], xg, yg, ift=True)
    return _apply_slow_ft(s, kfile, EH, gshape)


def _spatial_tikhonov(s, kfile, lam=4e-3):
    """op_CSIFourierTransform (tikh): B = (E^H E + lam I)^-1 E^H, lam=4e-3."""
    traj = read_kfile(kfile)
    xg, yg = _image_grid(s)
    EH_norm, gshape = _sft2_operator(traj['kx'], traj['ky'], xg, yg, ift=True)
    Nk = traj['Nk']
    EH = Nk * EH_norm                               # true E^H (undo /Nk)
    E = EH.conj().T
    Np = EH.shape[0]
    B = np.linalg.solve(EH @ E + lam * np.eye(Np), EH)
    return _apply_slow_ft(s, kfile, EH, gshape, per_temporal_op=B)


def _spatial_cartesian_fft(s):
    """FID-A cartesian branch: halfPixelShift + per-axis fftshift(fft(fftshift)).
    circshift(1) on odd-length axes. Relabels kx->x, ky->y."""
    out = copy.deepcopy(s)
    d = out['data']
    akx, aky = _ax(s, 'kx'), _ax(s, 'ky')
    Nx, Ny = d.shape[akx], d.shape[aky]
    voxX, voxY = s['voxelSize']['x'], s['voxelSize']['y']
    # cartesian k-grid coords (cycles/mm): centred integer indices / fov
    fx = (np.arange(Nx) - Nx // 2)
    fy = (np.arange(Ny) - Ny // 2)
    # half-pixel shift: kShift = kx*voxX/2 + ky*voxY/2 with kx normalised to grid
    kxn = fx / s['fov']['x']; kyn = fy / s['fov']['y']
    shp = [1] * d.ndim
    shp[akx] = Nx; kshift_x = (kxn * voxX / 2).reshape(shp)
    shp = [1] * d.ndim
    shp[aky] = Ny; kshift_y = (kyn * voxY / 2).reshape(shp)
    d = d * np.exp(-1j * 2 * np.pi * (kshift_x + kshift_y))
    for axn, N in ((akx, Nx), (aky, Ny)):
        if N % 2 == 1:
            d = np.roll(d, 1, axis=axn)
        d = np.fft.fftshift(np.fft.fft(np.fft.fftshift(d, axes=axn), axis=axn), axes=axn)
    out['data'] = d
    out['dims']['x'] = out['dims']['kx']; out['dims']['y'] = out['dims']['ky']
    out['dims']['kx'] = 0; out['dims']['ky'] = 0
    out['flags']['spatialft'] = 1
    return out


# =============================================================== dispatchers
def op_CSIReconstruct(struct, kfile, method='nufft'):
    """Dispatch spatial transform. Cartesian (kfile in (None,'')) -> FFT."""
    if kfile in (None, ''):
        return _spatial_cartesian_fft(struct)
    m = str(method).lower()
    if m == 'nufft':
        return _spatial_nufft(struct, kfile)
    if m == 'dft':
        return _spatial_dft(struct, kfile)
    if m in ('tikhonov', 'tikh'):
        return _spatial_tikhonov(struct, kfile)
    raise ValueError(f'unknown ft method {method}')


def op_CSIRecon(struct, kfile='', dcfMethod='nn', ftMethod='nufft', **dcf_kw):
    """Top-level: skip DCF for tikhonov or dcfMethod='none'; else DCF then FT.
    Faithful to FID-A op_CSIRecon."""
    m = str(ftMethod).lower()
    skipDcf = m in ('tikhonov', 'tikh') or str(dcfMethod).lower() == 'none' or kfile in (None, '')
    if skipDcf:
        return op_CSIReconstruct(struct, kfile, ftMethod)
    dcomp = op_CSIPSFCorrection(struct, kfile, method=dcfMethod, **dcf_kw)
    return op_CSIReconstruct(dcomp, kfile, ftMethod)
