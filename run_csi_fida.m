addpath(genpath('C:\Users\divya\Downloads\fida codes\fida27May\fida-main'));
addpath(genpath('C:\Users\divya\Downloads\fida codes\fid_a'));
metFile = 'C:\Users\divya\Downloads\mrsi_pipeline\met.dat';
refFile = 'C:\Users\divya\Downloads\mrsi_pipeline\ref.dat';
try
  [met, ref] = io_CSIload_twix_pair(metFile, refFile, '');
  ft   = op_CSIRecon(met, '', 'pipe_menon', 'nufft');
  ft_w = op_CSIRecon(ref, '', 'pipe_menon', 'nufft');
  [cc_w, phase, weights] = op_CSICombineCoils1(ft_w);
  cc = op_CSICombineCoils1(ft, 1, phase, weights);
  ccav = op_CSIAverage(cc); ccav_w = op_CSIAverage(cc_w);
  ccav_w = op_CSISegment_simple(ccav_w); ccav.mask = ccav_w.mask;
  ftSpec = op_CSIFourierTransform(ccav); ftSpec_w = op_CSIFourierTransform(ccav_w);
  ftSpec_rmw = op_CSIRemoveLipids(ftSpec, 'lipidPPMRange',[4.5 5.0], 'linewidthRange',[1 10]);
  [ftSpec_B0, ftSpec_B0_w] = op_CSIB0Correction_v2(ftSpec_rmw, ftSpec_w);
  ftSpec_masked = op_CSIapplymask(ftSpec_B0);
  ftSpec_smooth   = op_CSIApodize(ftSpec_masked,  'functionType','gaussian','fullWidthHalfMax',20);
  ftSpec_smooth_w = op_CSIApodize(ftSpec_B0_w,    'functionType','gaussian','fullWidthHalfMax',20);
  disp('FIELDS:'); disp(fieldnames(ftSpec_smooth));
  fprintf('sz=%s\n', mat2str(ftSpec_smooth.sz)); disp(ftSpec_smooth.dims);
  save('C:\Users\divya\Downloads\mrsi_pipeline\fida_ftspec.mat', 'ftSpec_smooth','ftSpec_smooth_w','-v7');
  disp('FIDA_DONE');
catch e
  disp(['FIDA_ERR: ' e.message]); disp(e.getReport);
end
