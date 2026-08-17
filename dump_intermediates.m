% dump_intermediates.m
% Runs the FID-A rosette pipeline (met=ref=water proxy) and dumps every
% deterministic intermediate to rosette_matlab_stages\*.mat (-v7.3), matching the
% Python run_rosette_pipeline.py stage keys, for compare_intermediates.py.
% Uses op_CSIRecon('dft') (matches the Python recon) and SKIPS the stochastic
% water-removal (op_CSIRemoveLipids) so every dumped stage is deterministic.

set(0,'DefaultFigureVisible','off');
addpath(genpath('C:\Users\divya\Downloads\fida codes\fid_a'));
REF='F:\fida\divya\28thJULYpHANTOM40X40\subject01\mrs_ref\meas_MID01095_FID59456_XA60_RosetteSpinEcho_2_avg_8mste_4sTR_w.dat';
kFile='C:\Users\divya\Downloads\fida codes\fid_a\processingTools\MRSI\kFiles\Rosette_traj_40x40.txt';
outDir='C:\Users\divya\Downloads\mrsi_pipeline\rosette_matlab_stages';
if ~exist(outDir,'dir'); mkdir(outDir); end

function dump(s,outDir,key)
  o=struct(); o.data=s.data; o.sz=s.sz; o.dims=s.dims;
  o.spectralWidth=gf(s,'spectralWidth',0); o.dwelltime=gf(s,'dwelltime',gf(s,'adcDwellTime',0));
  o.txfrq=gf(s,'txfrq',0);
  if isfield(s,'ppm'); o.ppm=s.ppm; end
  if isfield(s,'fov'); o.fov=s.fov; end
  if isfield(s,'voxelSize'); o.voxelSize=s.voxelSize; end
  save(fullfile(outDir,[key '.mat']),'-v7.3','-struct','o');
  fprintf('  dumped %s [%s]\n', key, mat2str(s.sz));
end
function v=gf(s,f,d); if isfield(s,f); v=s.(f); else; v=d; end; end

fprintf('1 load...\n'); [met,~]=io_CSIload_twix_pair(REF,REF,kFile);
dump(met,outDir,'s01_load');
fprintf('3 recon dft...\n');
ftS=op_CSIRecon(met,kFile,'nn','dft'); dump(ftS,outDir,'s03_recon');
fprintf('4 combine...\n');
[ccw,phase,weights]=op_CSICombineCoils1(ftS);
cc=op_CSICombineCoils1(ftS,1,phase,weights); dump(cc,outDir,'s04_combine');
fprintf('5 avg+mask...\n');
ccav=op_CSIAverage(cc); ccav_w=op_CSIAverage(ccw); ccav_w=op_CSISegment_simple(ccav_w);
ccav.mask=ccav_w.mask; dump(ccav,outDir,'s05_ccav');
fprintf('6 spectral FT...\n');
ftSpec=op_CSIFourierTransform(ccav); ftSpec_w=op_CSIFourierTransform(ccav_w);
dump(ftSpec,outDir,'s06_spec');
fprintf('7 SSP...\n'); rmlip=op_CSIssp(ftSpec,0.8,1.88); dump(rmlip,outDir,'s07_ssp');
% 8 water removal SKIPPED (stochastic)
fprintf('9 B0...\n');
[b0,b0w,~,~]=op_CSIB0Correction_v2(rmlip,ftSpec_w); dump(b0,outDir,'s09_b0');
fprintf('10 mask...\n');
b0.mask=ccav_w.mask; masked=op_CSIapplymask(b0); dump(masked,outDir,'s10_masked');
fprintf('11 apodize...\n');
smooth=op_CSIApodize(masked,'functionType','gaussian','fullWidthHalfMax',20);
dump(smooth,outDir,'s11_smooth');
fprintf('DONE -> %s\n', outDir);
