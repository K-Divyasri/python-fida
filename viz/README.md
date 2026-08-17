# mrsi viz — MRSI visualisers (FID-A-style, Python)

Visualisers for the rosette MRSI pipeline output, mirroring FID-A's plotting.
matplotlib for `imagesc`/`plot`, nibabel for NIfTI (fsleyes / nii_viewer).

## Modules
| module | FID-A equivalent | what |
|---|---|---|
| `spectrum.py` | `op_plotspec` / `op_CSIPlotVoxelSpec` | single-voxel spectrum, real, ppm high→left |
| `spectral_grid.py` | `op_CSIPlot1` | grid of voxel spectra over the FOV (row 1 bottom, y-up) |
| `maps.py` | `op_CSIintegrate`, `op_CSILCModelMaps` | water map, peak-integration maps, conc+CRLB panels |
| `voxel_viewer.py` | `op_CSIPlotVoxelSpec` (interactive) | click a voxel → its spectrum |
| `nifti_out.py` | `create_separate_metabolite_niftis_v2` | 4D MRSI NIfTI, map NIfTI, per-met 4D [conc,CRLB,LW,SNR] |
| `mrsi_on_t1.py` | `mrsi_on_t1_map` / `nii_viewer_mrsi` | MRSI overlay on T1 (static + NIfTI), rot180 |
| `lcmodel_read.py` | `op_CSILCModelMaps` (parser) | parse LCModel `.table` files → conc/CRLB/LW/SNR |

## Run
```
python run_viz.py                              # defaults: s11_smooth_ref -> viz_out/
python run_viz.py <stage.npz> <out> [t1.nii.gz] [lcm_dir] [empty_template.nii]
```
In Python:
```python
from viz._common import load_stage
from viz.spectral_grid import spectral_grid
s = load_stage('rosette_py_stages/s11_smooth_ref.npz')
spectral_grid(s, mode='real', ppm_range=(0.2,5.2), out_path='grid.png')
```
Interactive voxel viewer (needs a GUI backend — run from a normal session, not headless):
```python
import matplotlib; matplotlib.use('TkAgg')
from viz.voxel_viewer import launch_viewer
fig = launch_viewer(load_stage('rosette_py_stages/s11_smooth_ref.npz')); import matplotlib.pyplot as plt; plt.show()
```

## Orientation notes (from the FID-A survey)
- Grid uses **op_CSIPlot1** convention: y increases **upward** (`origin='lower'`), matches nii_viewer.
- ppm axis reversed (high ppm on the **left**), real part by default (`op_plotspec`).
- `op_CSIintegrate`: real-part sum, **strict** ppm bounds, returns `map[y,x]`.
- Overlay/NIfTI use **rot180** (flip both spatial axes) — the correct overlay flip per `test_overlay_flip` (FID-A's `mrsi_on_t1_map` hardcodes `lr`).
- NIfTI affine: pass a **spec2nii empty template** to copy its affine verbatim (as FID-A does); else a simple geometry affine is used.

## Status
Built + tested on the water-ref + metabolite-path pipeline stages. LCModel map panel
+ per-metabolite 4D NIfTI verified with synthetic fit data (drop in a real LCModel
output folder via `run_viz.py <stage> <out> "" <lcm_dir>`).
