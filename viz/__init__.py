"""mrsi.viz -- visualisers for the rosette MRSI pipeline output.

FID-A-style plotting ported to Python (matplotlib for imagesc/plot, nibabel for
NIfTI/overlay). Operates on the pipeline structs / saved stage .npz files.

Modules
  _common       shared loading, [f,y,x] reorder, ppm handling, band integration, masks
  spectrum      single-voxel spectrum (MRS ppm axis)
  op_CSIPlot    faithful FID-A op_CSIPlot: grid of voxel spectra + labels/ranges/yMul
  spectral_grid simpler grid-of-voxel-spectra (imagesc layout)
  maps          water map, metabolite peak-integration map, metabolite+CRLB maps (imagesc)
  voxel_viewer  interactive: click a voxel -> its spectrum
  nifti_out     write metabolite maps to NIfTI (create_separate_metabolite_niftis equiv)
  mrsi_on_t1    overlay MRSI on a T1 NIfTI (mrsi_on_t1_map equiv; applies the 180 flip)
"""
