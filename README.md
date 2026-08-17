# python-fida

Pure-Python port of the [FID-A](https://github.com/CIC-methods/FID-A) MRSI
reconstruction + fitting pipeline for Siemens TWIX rosette CSI, validated
stage-by-stage against the MATLAB FID-A reference.

Raw Siemens TWIX → spatial recon → coil combine → spectral FT → water/lipid
removal → B0 correction → spatial smoothing → per-voxel **LCModel** metabolite
maps (Cr / Cho / Lac / Acetate).

## Validation vs FID-A

Every deterministic stage matches FID-A. On the fully-deterministic path
(`dft` + `nn`, no NUFFT) Python reproduces FID-A **end-to-end to <0.25%**
(amplitude scale 1.000):

| stage | shape diff | scale |
|---|---|---|
| spectral FT | 0.17% | 1.000 |
| B0 correction | 0.12% | 1.000 |
| apodize (smoothing) | 0.10% | 1.000 |

The only non-exact stage is the NUFFT recon: this port uses
[finufft](https://finufft.readthedocs.io/) while FID-A uses the Fessler IRT
NUFFT (KB kernel) — same adjoint math, different spreading kernel (~5% on the
final spectrum). `dft` and `tikhonov` recon are bit-exact. See
`op_CSIApodize.py` for a subtle MATLAB→Python bug that mattered: the smoothing
kernel is `xWeights' * yWeights` (**conjugate** transpose); dropping the conj
shifts the whole image by one pixel.

## Layout (FID-A one-function-per-file)

```
mrsi/            one op_ function per file (FID-A style)
  io_CSIload_twix.py        io_CSIload_twix_pair   (pymapVBVD reader)
  op_CSIRosettePrep.py      prep_noncartesian      (reshape readout)
  op_CSIRecon.py            op_CSIRecon            (DCF nn/voronoi/pipe_menon x FT dft/nufft/tikhonov)
  op_CSICombineCoils1.py    op_CSIAverage.py  op_CSISegment_simple.py
  op_CSIleftshift.py        op_CSIFourierTransform.py
  op_CSIRemoveLipids.py     op_CSIB0Correction_v2.py  op_CSIspecZeroFill.py
  op_CSIapplymask.py        op_CSIApodize.py  op_CSIssp.py  op_CSIFlip180.py
  mrsi_common.py            shared dim/axis helpers
viz/             op_plotspec / op_plotfid / imagesc + map/spectrum viewers
fit_lcmodel_rosette.py      per-voxel LCModel fit -> conc/CRLB/LW/SNR maps
run_rosette_pipeline.ipynb  step-by-step notebook (mirror of run_MRSI_Rosette_40x40.m)
run_rosette_pipeline.py     scripted equivalent
```

## Run

Open **`run_rosette_pipeline.ipynb`** (adds `mrsi/` to the path and calls each
function one at a time, like FID-A's run script), or:

```python
python run_rosette_pipeline.py
```

Headless notebook run:

```bash
jupyter nbconvert --to notebook --execute --inplace run_rosette_pipeline.ipynb
```

## LCModel

Fitting shells out to **LCModel** (run via WSL: `~/.lcmodel/bin/lcmodel`). The
proprietary license key is **not shipped** — provide your own:

```bash
export LCMODEL_KEY=<your-lcmodel-key>      # read by fit_lcmodel_rosette.py
```

Basis: a phantom basis at the data dwell (`.basis`), e.g.
`RosettePhantom_LacAceCrCho_TE15_SW1587.basis`.

## Environment

`fida` conda env: numpy, scipy, `finufft`, `mapvbvd` (pymapVBVD), `twixtools`,
`suspect`, nibabel, h5py, pandas, matplotlib. LCModel runs in WSL.

Data files (`.dat`, `.mat`, `.npz`, LCModel outputs) are gitignored — code only.
