# Cartesian MRSI: raw TWIX → metabolite fits, every pathway (NO FID-A)

Model: **RECON** (read→FFT→coil-combine→B0) → **FIT** → **VIEW**.
Recon is the only custom part (off-the-shelf tools can't recon a custom Siemens
sequence). One shared recon core (`recon_core.py`) feeds every fitter.

Env: `FPY="C:\Users\divya\miniconda3\envs\fida\python.exe"`. LCModel + FSL-MRS in WSL.

| file | pathway | recon | fit | status |
|---|---|---|---|---|
| `fit_lcmodel.py`  | 1 | Python (recon_core) | LCModel (DOECC=T) | installed ✓ |
| `fit_fslmrs.py`   | 2 | Python (recon_core) | FSL-MRS `fsl_mrsi` | installed ✓ |
| `fit_tarquin.py`  | 3 | Python (recon_core) | Tarquin | needs Tarquin |
| `export_jmrui.py` | 4 | Python (recon_core) | jMRUI (GUI) | needs jMRUI |
| `recon_cartesian_matlab.m` | 5 | MATLAB (mapVBVD+fft2) | LCModel | needs mapVBVD |
| (below) | 6 | Tarquin direct | Tarquin | single tool |
| (below) | 7 | SIVIC | SIVIC/LCModel | GUI |
| (below) | 8 | BART recon | LCModel | recon-nerd |

---

## Pathway 1 — Python → LCModel   `fit_lcmodel.py`
```
"C:\Users\divya\miniconda3\envs\fida\python.exe" mrsi_pipeline\fit_lcmodel.py
```
recon_core → suspect `.RAW` → LCModel (WSL). Water handled by LCModel:
`DOECC=T` (eddy-current corr from water ref) + `DOWS=T` + `PPMST=4.25`.
Out: `out_pathway1_lcmodel\maps.npz` + per-voxel `.table/.coord/.ps`.

## Pathway 2 — Python → FSL-MRS   `fit_fslmrs.py`
One-time basis convert (LCModel `.basis` → FSL dir):
```
wsl bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate fsl_mrs && \
  basis_tools convert "<lcmodel.basis>" "<...>/fsl_basis"'
```
Then:
```
"C:\Users\divya\miniconda3\envs\fida\python.exe" mrsi_pipeline\fit_fslmrs.py
```
Writes `csi_metab/csi_wref/mask.nii.gz`, runs `fsl_mrsi`. Out: `out_pathway2_fslmrs\fit_out\concs\`.
View: `fsleyes fit_out\concs\raw\Lac.nii.gz`.

## Pathway 3 — Python → Tarquin   `fit_tarquin.py`
Install Tarquin (https://tarquin.sourceforge.net); set `TARQUIN` in the script.
```
"C:\Users\divya\miniconda3\envs\fida\python.exe" mrsi_pipeline\fit_tarquin.py
```
recon_core → `.RAW` (lcm format) → `tarquin --format lcm --water_eddy true`.

## Pathway 4 — Python → jMRUI   `export_jmrui.py`
```
"C:\Users\divya\miniconda3\envs\fida\python.exe" mrsi_pipeline\export_jmrui.py
```
Writes per-voxel jMRUI `.txt`. Open in jMRUI (http://www.jmrui.eu), fit with
**AMARES** (prior knowledge) or **QUEST** (basis). GUI.

## Pathway 5 — MATLAB → LCModel   `recon_cartesian_matlab.m`
Needs `mapVBVD` on the MATLAB path. Run the script → `.RAW` per voxel →
fit with LCModel (same control) or point Tarquin/jMRUI at the RAWs.

---

## Pathway 6 — Tarquin direct (single tool, reads TWIX itself)
No recon code — Tarquin reads Siemens TWIX and does recon+fit:
```
tarquin --input meas_...csi.dat --format siemens \
        --input_w meas_..._w.dat --format_w siemens \
        --water_eddy true --output_csv fit.csv --output_pdf fit.pdf
```
Note: Tarquin's TWIX support targets standard CSI; a very custom sequence may
need the RAW route (pathway 3).

## Pathway 7 — SIVIC (MRSI GUI, DICOM route)
```
# convert Siemens raw/DICOM to SIVIC (DICOM MRS):
svk_file_convert -i input.dcm -o csi -t 3
# GUI: recon, process (ECC/water/apodize), fit (or export to LCModel), overlay on T1
sivic
```
Best if you have Siemens **DICOM** MRS export. Strong grid+anatomy viewer.

## Pathway 8 — BART recon → LCModel   (BART INSTALLED ✓)
BART built from source: `~/tools/bart/bart` (v1.0.00). Run under its deps env:
```
wsl bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate bartdeps && \
  export PATH=$HOME/tools/bart:$PATH && \
  bart fft 6 kspace img'          # bitmask 6 = dims 1,2 (kx,ky)
# then coil-combine (bart cc / own), export FID per voxel -> LCModel RAW.
```
Bridge needed: numpy k-space -> BART `.cfl/.hdr` (`cfl.writecfl`) -> `bart fft`/`bart cc`
-> back to numpy -> suspect `save_raw`. Ask to wire `fit_bart.py`.

---

## Bridges (recon → fitter formats)
- **LCModel**: `.RAW` via `suspect.io.lcmodel.save_raw` (Python) or `write_raw` (MATLAB).
- **FSL-MRS**: NIfTI-MRS via `nifti_mrs.create_nmrs.gen_nifti_mrs`.
- **Tarquin**: reads `.RAW` (`--format lcm`) or TWIX (`--format siemens`).
- **jMRUI**: jMRUI Text format (`export_jmrui.py`).

## Notes learned on this data (subject04 csi_fid_24x24)
- No averages (`NAve=1`). SW=4000 Hz, 2048 pts, TE 2.3, TR 720.
- **B0 spread ~0.45 ppm** across the phantom → per-voxel align is essential (in recon_core).
- Water is broad + eddy-current-distorted → **`DOECC=T`** matters for LCModel.
- Basis TE **must** equal data TE (2.3 ms) or LCModel errors `MYBASI 10`.
- Absolute mM needs relaxation correction (`apply_water_scaling_all` — not yet ported).
