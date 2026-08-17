# python-fida — full pipeline walkthrough

Every function in `mrsi/`, explained in the exact order `run_rosette_pipeline.ipynb`
calls it, with the data structure and the logic in sequence. Read top to bottom;
each stage's output is the next stage's input.

---

## PART 0 — The data structure (`MRSIStruct`)

Everything is one Python **dict** (the port of FID-A's `MRSIStruct`). It flows
through every function; each function deep-copies it, changes `data` + a few
fields, and returns it. Key fields:

| field | meaning |
|---|---|
| `data` | the complex N-D numpy array (k-space early, image+spectrum later) |
| `sz` | `tuple(data.shape)` |
| `dims` | **name → 1-based axis index**, `0` if that axis is absent |
| `spectralWidth` / `dwelltime` | 1/dt and dt (s) of the spectral axis |
| `spectralTime` / `adcTime` | time vectors (s) |
| `txfrq` | transmit frequency (Hz) — sets the ppm scale |
| `te`, `tr`, `Bo` | echo/repetition time (ms), field (T) |
| `ppm` | ppm axis — `None` until the spectral FT fills it |
| `fov` / `voxelSize` | `{x,y,z}` mm |
| `imageOrigin` / `normal` | slice position + orientation (mm) |
| `kPtsPerCycle` / `nShots` | rosette trajectory counts (from the kFile) |
| `xShift_mm` / `yShift_mm` | VOI off-centre shift |
| `flags` | dict of booleans: `spatialft`, `spectralft`, `addedrcvrs`, `averaged`, `leftshifted`, `zerofilled`, `apodized`, `isCartesian`, … |
| `mask` | added at segmentation: `mask['brainmasks']` = `[y,x]` bool |

**`dims` is the heart of it.** Because axes get added/removed/relabelled at every
stage, no function assumes a fixed axis order — it looks the axis up by name.

### Shared helpers — `mrsi_common.py`

- `_ax(s, name)` → 0-based axis of a dim (`dims[name]-1`), or `None` if absent.
- `_faxis(s)` → the spectral axis: `dims.f` if present else `dims.t`, minus 1.
- `_fyx(s)` → a `[f, y, x]` view of `data` + the permutation used.
- `_drop_dim(dims, name)` → zero a dim and shift every higher axis down by 1
  (used after averaging collapses the `averages` axis).
- `_spatial_shape(s, af)` → `data.shape` with the spectral axis removed → `(Ny,Nx)`.
- `_gaussian_negexp(x, sig2)` → FID-A's `gaussian(x,σ) = -exp(x²/(2σ²))`.
- Re-exports `_coords` and `read_kfile` from `op_CSIRecon` so there's one copy.

**Data layout as it evolves** (rosette, subject02):

```
load      [t, coils, averages, kshot, extras]        raw interleaved readout
prep      [t, coils, averages, kpts, kshot]          readout split into time + trajectory pts
recon     [t, coils, averages, y, x]                 image (spatial FT done)
combine   [t, averages, y, x]                         coils merged
average   [t, y, x]                                    averages merged
spectral  [f, y, x]                                    time → spectrum
… B0, water removal, mask, apodize keep [f, y, x] …
```

---

## PART 1 — Step 1: Load  `io_CSIload_twix_pair`

**File:** `io_CSIload_twix.py`

`io_CSIload_twix_pair(metFile, refFile, kfile, 'rosette')` loads the metabolite
`.dat` and the water-reference `.dat`. With a `kfile` it routes both to
`io_CSIload_twix_noncart` (non-cartesian); without one it uses `io_CSIload_twix`
(cartesian).

### `io_CSIload_twix_noncart(filename, kfile, seq_type)`

1. **Read TWIX** with `pymapVBVD` (`mapVBVD`), squeeze singleton dims, get the
   k-space array `data` and its Siemens loop names `sqzDims` (e.g. `Col, Cha,
   Lin, Set, Par, Ave`).
2. **Map Siemens names → FID-A dim names** via `dim_map`:
   `Col→t, Cha→coils, Lin→kshot, Set→kshot, Ave→averages, Par→extras`.
   Builds the `dims` dict (name → 1-based axis).
3. **Trajectory params from the kFile** (authoritative for non-cartesian):
   `nShots = max(TR column)`, `kPtsPerCycle = len(kFile)/nShots`, and
   `dwell = time[2]-time[1]` (the kFile time column overrides the header).
4. **Timing/geometry:** `te`, `tr` from `MeasYaps`; `txfrq` from `hdr.Meas.lFrequency`
   (NOT the MeasYaps transmit freq — that's ~54 kHz higher and would shift ppm);
   `fov` from `sSliceArray.asSlice[0]`.
5. **Recon grid** from the kFile: `Kmax = max|k|`, `Nx = round(2·FOV·Kmax)`,
   `voxelSize = FOV/Nx`.
6. **VOI shift** via `_compute_fov_shift` → `xShift_mm, yShift_mm`.
7. Assemble the `MRSIStruct` dict; `flags.isCartesian = 0`, `ppm = None`.

Helpers: `_yaps` (safe MeasYaps lookup), `_compute_fov_shift` (reads
`sSpecPara.sVoI` normal+position → in-plane VOI offset in mm).

**Output:** two structs laid out `[t, coils, averages, kshot, extras]` with the
trajectory counts attached — ready for prep.

---

## PART 2 — Step 1b: Prep  `prep_noncartesian`

**File:** `op_CSIRosettePrep.py`

The raw readout is one long interleaved FID; recon needs it split into a spectral
time axis `t` and a trajectory-point axis `kpts`. `prep_noncartesian` runs three
FID-A steps in order:

### 1. `combine_time(struct, 'extras')`
Merges the `t` axis and the `extras` (interleave segments) into one continuous
FID. Transpose so `[t, extras, …]`, reshape `(t·extras, …)` **column-major (t
fastest)** so segments concatenate along time. Rebuilds `dims` with `t` at axis 1.
(Cartesian → no-op.)

### 2. `op_CSIShift(struct, x_mm, y_mm, kfile)`  *(skipped when shift = 0)*
Corrects an off-centre VOI. Reads the trajectory, renormalises
`kxn = (kx/(2·kxmax))/FOVx`, builds a linear k-phase
`exp(-i·2π·(kxn·dx + kyn·dy))`, tiles it across the merged readout, multiplies.
For subject02 `xShift = yShift = 0`, so `prep_noncartesian` skips it.

### 3. `split_readout_kpts(struct, kPtsPerCycle)`
Splits the merged readout `t = nT·kpts` (kpts fastest) into two axes `(nT, kpts)`
by reshaping. `_insert_axis_dims` bumps every higher axis by 1 and names the new
axis `kpts`.

**Output:** `[t, coils, averages, kpts, kshot]` — the layout `op_CSIRecon` expects.

*(Also in this file: `load_struct_mat` + `_mat_to_struct*` — a bridge to read a
MATLAB-dumped struct when the raw `.dat` is Siemens XA60 that Python twix parsers
can't open. Not used in the standard-twix notebook path.)*

---

## PART 3 — Step 2: Spatial reconstruction  `op_CSIRecon`

**File:** `op_CSIRecon.py`

`op_CSIRecon(struct, kFile, 'pipe_menon', 'nufft')` turns non-cartesian k-space
into an image. Two phases: **density compensation** then the **Fourier transform**.

### `op_CSIRecon(struct, kfile, dcfMethod, ftMethod)` — top dispatcher
- `skipDcf` if `ftMethod` is tikhonov, or `dcfMethod='none'`, or cartesian.
- Otherwise: `op_CSIPSFCorrection` (DCF) → `op_CSIReconstruct` (FT).

### `read_kfile(kfile)`
Reads the CSV trajectory (`TR, Kx, Ky, time`). Returns `kx, ky` (1/mm), the shot
index `TR`, `kPtsPerCycle = len/nShots`, `nShots`, dwell, `Nk = len`.

### `op_CSIPSFCorrection(struct, kfile, method='pipe_menon')` — DCF
Non-uniform samples are denser near k=0; density compensation weights each sample
by ~1/local-density so the FT isn't dominated by the centre.

- **`dcf_pipe_menon(kx, ky, fov, vox, iters=25)`** (the one used): iterative
  Pipe–Menon. Start `w=1`; each iteration grid the weights (finufft type-1
  adjoint) then sample back (type-2 forward) to get `Gw`, and divide `w ← w/|Gw|`.
  Converges to `w ≈ 1/diag(E Eᴴ)`. Clip 2–98th pct, radial-smooth.
- Alternatives: `dcf_nn` (k-nearest-neighbour local density) and `dcf_voronoi`
  (Voronoi cell area, clipped to the k-disk).
- `_radial_smooth` = moving average of the weights sorted by |k| radius.
- Finally normalise `sum(w) = Nk`, reshape to `[kpts, kshot]`, broadcast-multiply
  into `data`, store under `densityComp`.

### `op_CSIReconstruct(struct, kfile, 'nufft')` → `_spatial_nufft`
Adjoint (type-1) NUFFT per temporal point.
1. `_merge_kpts_time` — permute so kpts is fastest, merge `(kpts, t)` into one
   readout dim, flatten the extra axes (coils, averages) → `X[Ntot, nshot, Nextra]`.
2. `_image_grid` / `_coords` — build the image x/y coordinate grid from
   `fov, voxelSize` (`-b+step/2 : step : b-step/2`).
3. Convert trajectory to radians/sample `om = 2π·k·d`, and add an `n_shift` linear
   phase so finufft's centre matches FID-A's grid exactly.
4. For each spectral point `it`: take that readout's `Nk` samples, apply the shift
   phase, run `finufft.nufft2d1` (nonuniform→grid) per extra channel, divide by `Nk`.
5. `_finalize_spatial` — reshape image `[NPt, Ny, Nx, Nextra]` back to
   `[t, coils, avg, y, x]`, set `spectralDwellTime = adcDwell·kPtsPerCycle`
   (the effective spectral dwell), set `flags.spatialft=1`, relabel dims.

Other FT branches (same file, not used here): `_spatial_dft` (exact dense inverse
DFT — bit-exact to FID-A's dft), `_spatial_tikhonov` (regularised inverse
`B=(EᴴE+λI)⁻¹Eᴴ`), `_spatial_cartesian_fft` (halfPixelShift + FFT).

> **Recon note:** this port's `nufft` uses **finufft**; FID-A uses the **Fessler
> IRT** NUFFT (different spreading kernel), so the two agree ~5% on the final
> spectrum. `dft`/`tikhonov` are bit-exact. Run both in `dft` for an exact match.

**Output:** image `[t, coils, averages, y, x]`, `flags.spatialft = 1`.

---

## PART 4 — Step 3: Coil combination  `op_CSICombineCoils1`

**File:** `op_CSICombineCoils1.py`

Roemer combine — weight each coil by its sensitivity, phase-align, sum. Done in
the **time domain** (before the spectral FT) at the first FID point (highest SNR).

Two-call workflow (like FID-A):
```
cc_w, phase, weights = op_CSICombineCoils1(ft_w)          # derive maps from the water ref
cc                    = op_CSICombineCoils1(ft, 1, phase, weights)  # apply them to metabolite
```

`op_CSICombineCoils1(struct, samplePoint=1, phaseMap=None, weightMap=None)`:
1. Transpose to canonical `[t, coils, (avg), y, x]`.
2. **Phase map** (if not supplied): `angle(data[samplePoint])` per coil (averaged
   over avg) → `[coils,y,x]`; multiply `data · exp(-i·phase)` to align coils.
3. **Weight map** (if not supplied): `|data[samplePoint]|` per coil, normalised to
   unit L2 per voxel → Roemer magnitude weights; multiply.
4. **Sum over the coils axis** → coils gone.

Maps are canonical `[coils,y,x]` so the water-ref maps broadcast onto the
metabolite even though the metabolite still has an `averages` axis.

**Output:** `[t, (averages), y, x]`, `flags.addedrcvrs = 1`.

---

## PART 5 — Step 4: Average + water mask

**Files:** `op_CSIAverage.py`, `op_CSISegment_simple.py`

### `op_CSIAverage(struct)`
Mean over the `averages` axis, then `_drop_dim` removes it (no-op if absent or
already averaged). `flags.averaged = 1`.

### `op_CSISegment_simple(struct, threshold=None, min_size=3)`
Builds the phantom/brain mask:
1. `intensity = |data[0]|` (first point along the leading axis) → 2D `[y,x]`.
2. Threshold at the mean (default) → binary.
3. `bwareaopen` — drop connected components smaller than `min_size`
   (`scipy.ndimage.label` + size filter, 8-connectivity).
4. `binary_fill_holes` → fill interior gaps.
5. Store `mask['brainmasks']` (`[y,x]` bool) + `intensity` + `threshold`.

The notebook computes the mask on the **water reference** (`ccav_w`) then copies
it onto the metabolite: `ccav['mask'] = ccav_w['mask']`.

**Output (met):** `[t, y, x]` + `mask`.

---

## PART 6 — Step 4b: Left-shift  `op_CSIleftshift`

**File:** `op_CSIleftshift.py`

The FID's first sample isn't exactly at t=0 (ADC/echo delay), which puts a
first-order phase across the spectrum. `op_CSIleftshift(struct, ls=None)` drops
the first `ls` time samples (`ls = struct['pointsToLeftshift']`, default 0),
recomputes `spectralTime`/`adcTime`, sets `flags.leftshifted`.

For this rosette data `pointsToLeftshift = 0`, so it's a **no-op** — kept for
FID-A fidelity and other datasets.

---

## PART 7 — Step 5: Spectral FT  `op_CSIFourierTransform`

**File:** `op_CSIFourierTransform.py`

`op_CSIFourierTransform(ccav, spatial=False, spectral=True)` — time → frequency
(the spatial FT already happened in recon).

- `spectral=True`: `data = fftshift(ifft(data, axis=t))`; relabel `dims.t → dims.f`;
  set `flags.spectralft=1`.
- **Fill the ppm axis:** `f = fftshift(fftfreq(N, dwelltime))` (Hz), then
  `ppm = -f/(txfrq/1e6) + 4.65`. The 4.65 references water; the minus flips to the
  MRS convention (high ppm left).
- (`spatial=True` is the cartesian branch: `ifft2` over kx,ky, relabel kx→x, ky→y.)

**Output:** `[f, y, x]`, `ppm` filled, `flags.spectralft = 1`.

---

## PART 8 — Step 6: Lipid/water removal + B0

**Files:** `op_CSIssp.py`, `op_CSIRemoveLipids.py` (+ `make_lipid_basis.py`),
`op_CSIB0Correction_v2.py`

### `op_CSIssp(struct, minppm, maxppm, m=6)` — *skipped for phantom*
SVD subspace lipid suppression. Reshape to `[y·x, spec]`, SVD the lipid-band
slice, project out the top-`m` spatial components: `P = I − Um Umᴴ`, apply to the
full spectrum. Skipped here because the phantom's `0.8–1.88 ppm` band would remove
lactate (`PHANTOM=True`).

### `op_CSIRemoveLipids(struct, lipidPPMRange=(4.5,5.0), lineWidthRange=(1,10))`
L2-regularised removal of everything in a ppm band — here the **residual water**
at 4.5–5.0 ppm.
- `make_lipid_basis` builds `lipidComponents` random spectra in the band (random
  linewidth/ppm/phase Lorentzians; FID → `fftshift(fft)`). *(This basis is
  stochastic in FID-A too; pass a fixed `lipidBasis` for a bit-exact match.)*
- Solve `L2 = inv(I + β·B·Bᴴ)` and apply `data ← L2 @ data` along the f-axis.
  With `β=1e-4` and the band away from the metabolites, this suppresses the band
  and leaves the fit window (0.2–4.25 ppm) essentially untouched.

### `op_CSIB0Correction_v2(met, wat)`
Corrects per-voxel B0 offset using the **water** phase (Klose method), deterministic.
1. Water FID `fidw = fft(ifftshift(water_spectrum))`; phase offset
   `poff = unwrap(angle(fidw))`.
2. Subtract it from **both** metabolite and water FIDs
   (`fid · exp(-i·poff)`), transform back to spectra.
3. Also fits the slope of the unwrapped water phase over a range of endpoints and
   picks the best-R² one → `freqMap` (Hz) + `R2Map` (diagnostic only, not used in
   the correction).

Returns `(met_c, wat_c, freqMap, R2Map)`.

**Output:** B0-corrected `[f, y, x]` for met and water.

---

## PART 9 — Step 6b: Spectral zero-fill  `op_CSIspecZeroFill`  *(optional)*

**File:** `op_CSIspecZeroFill.py`

`op_CSIspecZeroFill(struct, Ntarget)` interpolates the spectrum to a finer grid:
invert the spectral FT (`fft(ifftshift)`), zero-pad the FID to `Ntarget`,
`fftshift(ifft)` back, and recompute `ppm`/time vectors. `flags.zerofilled=1`.

The notebook default is `ZEROFILL=None` (skip → `NUNFIL=576`), which matches the
FID-A subject02 LCModel RAW. Set `4096` to interpolate for finer peak separation.

---

## PART 10 — Step 7: Apply mask + spatial smoothing

**Files:** `op_CSIapplymask.py`, `op_CSIApodize.py`

### `op_CSIapplymask(struct)`
Zero every voxel outside `mask['brainmasks']` (broadcast the `[y,x]` mask across f).

### `op_CSIApodize(struct, 'gaussian', 20)`
Spatial smoothing (Gaussian FWHM = 20 mm). On spatial-FT'd data it works as a
**k-space weighting done as an image-space convolution**:
1. Build 1-D Gaussians on the centred x/y coordinates, FT each → `Xw, Yw`.
2. `weightMatrix = outer(conj(Xw), Yw)` — **note the `conj`**: MATLAB's
   `xWeights' * yWeights` is a *conjugate* transpose. The Gaussian sits half a
   sample off-centre (coords are ±3, ±9, …) so `Xw` is complex; dropping the
   `conj` flips a linear phase and **shifts the whole image one pixel** in x and y.
   This was a real bug — fixed here (verified 0.00% vs FID-A).
3. FT the weight matrix to the image-space kernel, `/numel`, and `conv2('same')`
   each spectral slice.

**Output:** masked, smoothed `[f, y, x]` — the final processed spectrum.

---

## PART 11 — Step 8: LCModel maps  `fit_maps`

**File:** `fit_lcmodel_rosette.py` *(repo root, not `mrsi/` — the fitting driver)*

`fit_maps(met_fyx, wat_fyx, ppm, mask, out_dir)` fits every masked voxel:
1. `spec_to_fid(spec) = fft(fftshift(spec))` → the FID.
2. `write_raw` — LCModel RAW file; writes `[real, -imag]` (conjugate convention,
   matching FID-A `io_writelcm`).
3. `write_control` — LCModel control: `HZPPPM, DELTAT, NUNFIL=len(spec), ECHOT`,
   `DOECC=F, DOWS=T`, basis path, `CHUSE1 = Cr/Cho/Lac`, and the license `KEY`
   (read from the `LCMODEL_KEY` env var).
4. Run LCModel in WSL, one control at a time.
5. `parse_table` → per-metabolite `(conc, %CRLB)`; assemble `conc`/`crlb`/`LW`/`SNR`
   maps; save `maps.npz`; render `lcm_maps.png`.

`to_fyx` (from `viz/_common.py`) reorders the struct `data` to `[f, y, x]` before
fitting.

**Output:** `conc`/`crlb`/`LW`/`SNR` maps + per-voxel `.RAW`/`.control`/`.table`
written to the dataset's `outputs/` folder.

---

## APPENDIX — functions not on the phantom notebook path

- **`op_CSIphase.py`** — per-voxel ACME auto-phasing (entropy minimisation,
  `suspect`). FID-A production has no ACME; used only for a phased view.
- **`hsvd_water_removal.py`** — alternative residual-water removal by HSVD
  (models each FID with damped exponentials, subtracts the near-water ones).
- **`op_CSIFlip180.py`** — rot180 (flip x & y) so the MRSI overlays on a T1;
  reindexes the spatial grid only, spectra unchanged.
- **`op_CSIRecenter.py`** — data-driven recentring of an off-centre VOI
  (`water_centroid` → `compute_roll` → roll). Notebook uses `recenter=False`.
- **Recon variants** in `op_CSIRecon.py`: `dcf_nn`, `dcf_voronoi`, `_spatial_dft`,
  `_spatial_tikhonov`, `_spatial_cartesian_fft` — selectable via the
  `dcfMethod`/`ftMethod` args.

---

## One-line summary of the flow

```
load .dat ──prep──▶ [t,coils,avg,kpts,kshot]
   ──recon(DCF+NUFFT)──▶ image [t,coils,avg,y,x]
   ──coil combine──▶ [t,avg,y,x] ──average──▶ [t,y,x] (+mask)
   ──leftshift(no-op)──▶ ──spectral FT──▶ [f,y,x] (+ppm)
   ──water removal(L2)──▶ ──B0 correction──▶ (──zerofill?──)
   ──apply mask──▶ ──apodize(smooth)──▶ final spectrum
   ──per-voxel LCModel──▶ Cr/Cho/Lac/Act maps
```
