# python-fida — the math, function by function

Every function's **problem, operator/matrix, algorithm, and code mapping**, in
pipeline order. Notation:

- `x, y` image coordinates (mm); `p = 1..Np` indexes image voxels, `Np = Nx·Ny`.
- `k` indexes trajectory samples, `k = 1..Nk`; `kx[k], ky[k]` are 1/mm.
- `ρ` image (what we want); `s` measured k-space signal.
- `Eᴴ` = conjugate transpose (adjoint) of `E`; `⊙` elementwise; `diag(w)` diagonal.
- One **spectral time point** at a time unless noted — the recon is applied to
  each of the `Nt` FID points independently, so the operators below are 2-D
  spatial only.

---

## 0. The encoding model (what the whole recon inverts)

MRSI samples the spatial Fourier transform of the object along a trajectory:

```
s[k] = Σ_{p=1}^{Np} ρ[p] · exp(-2πi (x_p·kx[k] + y_p·ky[k]))      (for each t)
```

In matrix form **`s = E ρ`**, with the `Nk × Np` encoding matrix

```
E[k,p] = exp(-2πi (x_p·kx[k] + y_p·ky[k]))
```

Its adjoint (`Np × Nk`):

```
Eᴴ[p,k] = exp(+2πi (x_p·kx[k] + y_p·ky[k]))
```

`E` is a **non-uniform** Fourier transform (the `k` are scattered, not a grid).
Reconstruction = recover `ρ` from `s`. Three routes are implemented: adjoint +
DCF (nufft/dft), and regularised inverse (tikhonov). All live in `op_CSIRecon.py`.

---

## 1. Load — `io_CSIload_twix.py`

No linear algebra; it's parsing + bookkeeping. The only "math":

- **Dwell / bandwidth:** `spectralWidth = 1/dwell`. For non-cartesian the dwell
  comes from the kFile time column, `dwell = t[2]-t[1]`.
- **Recon grid from Kmax:** the Nyquist relation `Δx = 1/(2·Kmax)` gives
  `Nx = round(FOV / Δx) = round(2·FOV·Kmax)`, `voxelSize = FOV/Nx`.
- **Transmit frequency → ppm scale:** `ν0 = txfrq` (Hz); later `ppm = -f/(ν0/1e6)+4.65`.
- `_compute_fov_shift`: projects the VOI position vector onto the in-plane axes
  using the slice normal's dominant component (Sag/Cor/Tra) and `cos/sin(θ)`,
  `θ = arccos(|normal_max|)` — the VOI centre offset (mm) fed to `op_CSIShift`.

---

## 2. Prep — `op_CSIRosettePrep.py`

Reshapes only; the numbers are unchanged, but the **index algebra** matters.

### `combine_time` — concatenate interleave segments into one FID
The readout is stored as `t` (per-segment) × `extras` (interleave segments). The
true FID is the segments laid end to end. Reshape `(t·extras)` **column-major (t
fastest)** so element `(τ, e)` maps to time index `τ + e·Nt`:

```
FID[τ + e·Nt] = data[τ, e, …]
```

### `op_CSIShift` — off-centre VOI as a k-space linear phase
A spatial shift `(dx, dy)` in the image = a linear phase in k-space (Fourier
shift theorem `ρ(x-dx) ⟷ S(k)·e^{-2πi k·dx}`). With FID-A's renormalisation
`kxn = (kx/(2·kxmax))/FOVx`:

```
data[k] ← data[k] · exp(-2πi (kxn[k]·dx + kyn[k]·dy))
```

Skipped when `dx=dy=0` (subject02).

### `split_readout_kpts` — factor the readout index
`t_merged = nT·kpts`, kpts fastest. Reshape `(nT, kpts)` so
`data[τ, κ] = data_merged[κ + τ·kpts]` → separate spectral axis `t=nT` and
trajectory axis `kpts`. `_insert_axis_dims` renumbers `dims` after inserting the
new axis.

---

## 3. Reconstruction — `op_CSIRecon.py`

### 3a. Density compensation — `op_CSIPSFCorrection` / `dcf_pipe_menon`

**Problem.** The adjoint `Eᴴ s` is a *poor* inverse for non-uniform `k` because
dense regions (near k=0) are counted many times:

```
ρ̂ = Eᴴ s = Eᴴ E ρ,   and   (Eᴴ E) ≠ I  (it's the point-spread operator).
```

Density compensation inserts weights `w` so that `Eᴴ diag(w) E ≈ I`. The
Pipe–Menon choice makes the **gridding kernel** row-sums flat:

```
find w  such that  (G w) = 1,  where G = E Eᴴ  (grid then sample).
```

**Algorithm (fixed-point).** `w₀ = 1`; iterate

```
grid:    img = Eᴴ w      (finufft type-1 adjoint, nufft2d1)
sample:  Gw  = E img = E Eᴴ w   (finufft type-2 forward, nufft2d2)
update:  w  ← w / |Gw|
normalise: w ← w / mean(w)
```

25 iterations. Converges to `w ≈ 1/diag(E Eᴴ)` — the local sample density inverse.
Then clip to the 2–98th percentile and **radial-smooth**.

- `_radial_smooth(kx,ky,w,win)`: sort `w` by radius `|k|`, moving-average, scatter
  back — enforces `w` depends smoothly on `|k|` only.
- Alternatives: `dcf_nn` (w ∝ 1/(K/(π·d_K²)), `d_K` = distance to K-th neighbour —
  local density from a KD-tree); `dcf_voronoi` (w = Voronoi cell area, since a
  cell's area ≈ the k-space "territory" of that sample, clipped to the disk
  `|k| ≤ Kmax`).
- Finally normalise `Σw = Nk` and multiply into `data` per `(kpts, kshot)`.

### 3b. The transform — `op_CSIReconstruct`

**NUFFT branch — `_spatial_nufft` (used).** Adjoint reconstruction:

```
ρ = (1/Nk) · Eᴴ (w ⊙ s)
```

computed with a type-1 NUFFT (`finufft.nufft2d1`) per spectral point. The
trajectory is put in radians/sample `ω = 2π·k·Δ` and a linear **n_shift** phase
`exp(i(ωx·(Nx/2 - x_shift) + ωy·(Ny/2 - y_shift)))` is applied so finufft's centre
(N/2) matches FID-A's grid (`x_shift = median(arange(Nx) - x/Δx)`). Divide by `Nk`.

- `_merge_kpts_time`: permutes/flattens so each spectral point's `Nk` samples are
  contiguous; extras (coils, averages) become one loop axis.
- `_image_grid` / `_coords`: `coords = -b+Δ/2 : Δ : b-Δ/2`, `b = FOV/2` — the
  voxel centres.
- `_finalize_spatial`: reshapes `[NPt,Ny,Nx,Nextra]` → `[t,coils,avg,y,x]`; sets
  the **effective spectral dwell** `= adcDwell · kPtsPerCycle` (because after the
  split, consecutive FID points are `kPtsPerCycle` ADC samples apart).

**DFT branch — `_spatial_dft` (exact, bit-matches FID-A).** Builds the dense
adjoint operator explicitly (`_sft2_operator`):

```
EH[p,k] = exp(+2πi (x_p·kx[k] + y_p·ky[k])) / Nk,   ρ = EH @ (w ⊙ s)
```

Image points ordered y-fastest (MATLAB column-major, `meshgrid(x,y)`).

**Tikhonov branch — `_spatial_tikhonov` (regularised inverse).** Solves the
ridge least-squares problem

```
ρ = argmin_ρ ‖E ρ − s‖²  +  λ‖ρ‖²    ⇒    ρ = (EᴴE + λI)⁻¹ Eᴴ s
```

with `λ = 4e-3`; builds `B = (EᴴE + λI)⁻¹ Eᴴ` (via `np.linalg.solve`) and applies
`B @ s`. No DCF needed — the regulariser handles the conditioning.

**Cartesian branch — `_spatial_cartesian_fft`.** Uniform grid → plain FFT with a
half-pixel phase `exp(-2πi(kx·voxX/2 + ky·voxY/2))` and a `circshift(1)` on
odd-length axes.

---

## 4. Coil combine — `op_CSICombineCoils1.py`

**Problem.** `Nc` coils each measure `s_c = σ_c·ρ + noise`, with unknown complex
sensitivity `σ_c`. The **SNR-optimal (Roemer)** combination is

```
ρ̂ = ( Σ_c conj(σ_c)·s_c ) / sqrt( Σ_c |σ_c|² )
```

FID-A estimates the sensitivity from the signal itself at the first FID point
(highest SNR): `σ_c ≈ s_c(t0)`. Write `s_c(t0) = |s_c(t0)|·e^{iφ_c}`. Then:

```
phase map:   φ_c[y,x] = angle(s_c(t0))          # code: np.angle(ref)
weight map:  w_c[y,x] = |s_c(t0)| / sqrt(Σ_c |s_c(t0)|²)   # unit L2 per voxel
combine:     ρ̂(t) = Σ_c w_c · e^{-iφ_c} · s_c(t)          # phase-align, weight, sum
```

- `conj(σ_c) = |s_c(t0)|·e^{-iφ_c}` → the `e^{-iφ_c}` (phase align) and `|·|`
  (weight) in the code are exactly `conj(σ_c)`; the denominator is the
  normalisation. This is the matched filter that maximises SNR.
- Two-call design: derive `φ, w` from the **water reference** (clean, strong),
  then apply the *same* maps to the metabolite (so both share one sensitivity
  estimate). Maps are `[coils,y,x]` and broadcast over the metabolite's extra
  `averages` axis.

---

## 5. Average + segment

### `op_CSIAverage` — `Σ/N` over the averages axis. Variance ↓ by `N`.

### `op_CSISegment_simple` — threshold + morphology
- `I[y,x] = |data[0]|` (first FID point ≈ total signal energy per voxel).
- `mask = I > mean(I)` (Otsu-free, mean threshold).
- **bwareaopen:** connected-component label (8-neighbour), drop components with
  fewer than `min_size` pixels — removes speckle.
- **fill holes:** `binary_fill_holes` — morphological closing of interior gaps.

No continuous math; it's set operations on a binary image.

---

## 6. Left-shift — `op_CSIleftshift.py`

Dropping `ls` leading FID points removes a **first-order phase** across the
spectrum. A time shift `t0 = ls·dwell` is, by the Fourier shift theorem,

```
fid(t + t0) ⟷ S(f)·e^{+2πi f t0}
```

so the un-shifted FID carries a linear-in-`f` phase ramp; discarding the first
`ls` samples sets the FID origin at t=0 and flattens it. `ls = 0` here → identity.

---

## 7. Spectral FT — `op_CSIFourierTransform.py`

```
S(f) = fftshift( ifft( fid ) )
f    = fftshift( fftfreq(N, dwell) )        # Hz, centred
ppm  = -f/(ν0/1e6) + 4.65
```

- `ifft` (not `fft`) matches FID-A's convention (validated to corr 1.0).
- `fftshift` centres DC (0 Hz) in the middle of the array.
- ppm: divide Hz by the transmitter MHz (`ν0/1e6`), negate for the MRS
  convention (frequency increases right→left), reference water at 4.65 ppm.
- Spatial branch: `fftshift(ifft2(ifftshift(k)))·sqrt(Nx·Ny)` for cartesian data.

---

## 8. Water/lipid removal

### `op_CSIssp` — SVD subspace projection *(skipped for phantom)*
Stack voxels as columns of `D = [Np × Nspec]`. In the lipid band, lipids are a
few **dominant spatial patterns** (bright rim). SVD the band slice
`D_band = U Σ Vᴴ`; the first `m` left-singular vectors `Uₘ` span the lipid spatial
subspace. Project it out of the full data:

```
P = I − Uₘ Uₘᴴ           (Np×Np orthogonal projector)
D_clean = P · D
```

Removes anything spatially correlated with the strong lipid components.

### `op_CSIRemoveLipids` — L2 regularised removal (used, as water removal)
**Problem.** Find `x` close to the data `x₀` but with little energy in the lipid
subspace spanned by basis columns `B = [Nf × Ncomp]`:

```
x = argmin_x  ‖x − x₀‖²  +  β·‖Bᴴ x‖²
```

`Bᴴ x` is the projection of the spectrum onto each lipid template; penalising its
norm suppresses lipid-like structure. Setting the gradient to zero:

```
2(x − x₀) + 2β B Bᴴ x = 0   ⇒   (I + β B Bᴴ) x = x₀
⇒   x = (I + β B Bᴴ)⁻¹ x₀
```

Code: `L2 = inv(eye(Nf) + β·(B @ Bᴴ))`, then `x = L2 @ x₀` along the f-axis.
- `B B ᴴ` is `Nf×Nf`. With `β=1e-4`, `L2 ≈ I − β B Bᴴ` — a gentle notch on the
  band the basis covers.
- **`make_lipid_basis`** fills `B` with `Ncomp` Lorentzian peaks placed randomly
  in the target ppm band (random centre/linewidth/phase), FID
  `exp(2πi f₀ t + iφ)·exp(-π·lw·t)` → `fftshift(fft)`. Here the band is
  `[4.5, 5.0] ppm` → it removes **residual water**, not lipids. Outside the band
  (the 0.2–4.25 fit window) the operator ≈ identity, so metabolites are untouched.

---

## 9. B0 correction — `op_CSIB0Correction_v2.py`

**Problem.** Each voxel has a B0 offset `Δf` and eddy-current phase `φ_ec(t)`. The
**water reference** sees the same distortion but no metabolites:

```
fid_water(t) = A(t) · exp( i·(2π·Δf·t + φ_ec(t)) )
```

The **Klose** correction divides out the water phase from the metabolite FID:

```
φ_w(t) = unwrap( angle( fid_water(t) ) )
fid_met_corrected(t) = fid_met(t) · exp( -i·φ_w(t) )
```

This zeroes the common frequency offset and eddy phase per voxel. Applied to
**both** met and water. Code operates on the masked columns:
`fid = fft(ifftshift(spec))`, `fid ← fid·e^{-iφ_w}`, back with `fftshift(ifft)`.

**freqMap (diagnostic).** Least-squares line fit of `φ_w(t)` vs `t` over a sliding
window `[0:mt]`, `mt = 10..N`; slope `a` gives `Δf = a/(2π)`. The endpoint `mt`
with the best mean `R²` across voxels is chosen. Closed-form OLS slope:

```
a = Σ(t−t̄)(φ−φ̄) / Σ(t−t̄)²,   R² = 1 − SSres/SStot
```

Not used in the correction itself — just a field map to inspect.

---

## 10. Zero-fill — `op_CSIspecZeroFill.py`

Padding the FID with zeros and re-transforming is **sinc interpolation** of the
spectrum (no new information, finer grid):

```
fid   = fft( ifftshift( S ) )            # back to time
fidPad = [fid, 0, 0, …, 0]   length Ntarget
S_new = fftshift( ifft( fidPad ) )       # interpolated spectrum
```

Because `ifft` normalises by `1/Ntarget`, the amplitude scales by `N/Ntarget`, but
met and water scale identically so the water-scaled concentration is unchanged.
ppm/time vectors recomputed for the new length. Skipped by default (`NUNFIL=576`).

---

## 11. Apply mask + apodize

### `op_CSIapplymask` — `data ← data ⊙ mask[y,x]` (broadcast over f). Zeros outside.

### `op_CSIApodize` — Gaussian spatial smoothing
**Key duality:** multiplying k-space by a filter `H(k)` = convolving the image by
`h = FT(H)`. FID-A smooths the **image** with a Gaussian, implemented as a
convolution whose kernel is built by two FTs:

```
1. image-space Gaussian on centred coords:  g(x) = -exp(x²/(2σ²)),
   σ² = (FWHM/2)²/2 · ln(0.5)   (so g has the requested FWHM)
2. Xw = fftshift(fft(fftshift(g_x))),  Yw = fftshift(fft(fftshift(g_y)))   # → "k-space"
3. weightMatrix = conj(Xw) ⊗ Yw          # outer product; MATLAB xWeights' * yWeights
4. weightsFT = FT2(weightMatrix) / numel # back to image space = the conv kernel h
5. out[:,:,f] = conv2( data[:,:,f], weightsFT, 'same' )   # per spectral slice
```

**Why `conj(Xw)`.** The coordinate grid is `±3, ±9, …` — the Gaussian centre sits
**half a sample off** the grid, so `g` is symmetric-but-offset and its FT `Xw` is
**complex** (a linear phase). MATLAB's `xWeights' * yWeights` uses the **conjugate**
transpose. In numpy `np.outer(Xw, Yw)` drops that conjugation → the linear phase
survives → the convolution kernel is shifted by one sample → **the entire smoothed
image shifts one pixel in x and y** (an ~9% error that looked like a scale/shape
mismatch). Using `np.outer(np.conj(Xw), Yw)` reproduces FID-A to 0.00%.

`conv2(A, h, 'same')` returns the central `size(A)` of the full convolution; the
code does `full` then crops `[mb//2 : mb//2+ma]` to match MATLAB's even-kernel
centering.

---

## 12. LCModel fit — `fit_lcmodel_rosette.py` (fitting driver, repo root)

**Model LCModel solves.** For each voxel spectrum `Y(f)` it fits a linear
combination of basis spectra `B_m(f)` (metabolites) plus a smooth spline baseline
`Base(f)`, a global phase, and small per-metabolite frequency/lineshape shifts:

```
Y(f) ≈ e^{i(φ0 + φ1·f)} · [ Σ_m a_m · B_m(f; δ_m, γ_m)  +  Base(f) ]
minimise  ‖Y − model‖²  +  regularisers,   a_m ≥ 0
```

solved by LCModel (constrained non-linear least squares). Outputs the amplitudes
`a_m` → concentrations (water-scaled via `DOWS`, `WCONC`) and `%CRLB`
(Cramér–Rao lower bound = the fit's standard-error estimate).

The Python side only prepares inputs and parses outputs:
- `spec_to_fid(S) = fft(fftshift(S))` — spectrum → FID.
- `write_raw` — writes `[Re(fid), −Im(fid)]` (the **conjugate** convention of
  FID-A `io_writelcm`; getting this sign wrong makes every peak fit as noise).
- `write_control` — `HZPPPM, DELTAT=dwell, NUNFIL=len, ECHOT, DOECC=F, DOWS=T,
  WCONC=55556, CHUSE1={Cr,Cho,Lac}`, license `KEY` from `$LCMODEL_KEY`.
- run in WSL, `parse_table` → `(conc, %CRLB)` per metabolite → maps.

---

## Operator cheat-sheet

| stage | math | code |
|---|---|---|
| encode | `s = E ρ`, `E[k,p]=e^{-2πi(x_p kx+y_p ky)}` | `_sft2_operator` |
| DCF | `w←w/|E Eᴴ w|`, `Eᴴ diag(w) E ≈ I` | `dcf_pipe_menon` |
| nufft recon | `ρ = Eᴴ(w⊙s)/Nk` | `_spatial_nufft` |
| tikhonov | `ρ=(EᴴE+λI)⁻¹Eᴴ s` | `_spatial_tikhonov` |
| coil combine | `ρ̂=Σ conj(σ_c)s_c / √Σ|σ_c|²` | `op_CSICombineCoils1` |
| SSP | `P=I−UₘUₘᴴ` | `op_CSIssp` |
| lipid/water L2 | `x=(I+βBBᴴ)⁻¹x₀` | `op_CSIRemoveLipids` |
| B0 | `fid·e^{-i·unwrap∠fid_water}` | `op_CSIB0Correction_v2` |
| apodize | k-filter `H` ⇔ image conv `h=FT(H)` | `op_CSIApodize` |
| LCModel | `Y≈e^{iφ}(Σa_m B_m+Base)`, `a_m≥0` | `fit_maps` |
