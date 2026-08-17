% export_rosette_ref_stages.m
% Dump FID-A rosette stages for the WATER REF (standard twix) so Python can
% diff/corr step-by-step:  raw read -> prep(load pair) -> recon.
% The met is a corrupted INDX container, so pass the ref as BOTH met and ref
% (io_CSIload_twix_pair needs a pair; both being the same standard-twix ref is
% fine for a per-stage comparison against the Python ref pipeline).

clear; clc;
addpath(genpath('C:\Users\divya\Downloads\fida codes\fid_a'));

REF   = 'F:\fida\divya\28thJULYpHANTOM40X40\subject01\mrs_ref\meas_MID01095_FID59456_XA60_RosetteSpinEcho_2_avg_8mste_4sTR_w.dat';
kFile = 'C:\Users\divya\Downloads\fida codes\fid_a\processingTools\MRSI\kFiles\Rosette_traj_40x40.txt';
outDir= 'C:\Users\divya\Downloads\mrsi_pipeline\rosette_ref_stages';
if ~exist(outDir,'dir'); mkdir(outDir); end

% --- STAGE 0: raw mapVBVD squeezed image (compare the read itself) ---
tw = mapVBVD(REF);
if iscell(tw); tw = tw{end}; end
tw.image.flagRemoveOS = false;   tw.image.squeeze = true;
raw = tw.image();
sqzDims = tw.image.sqzDims;
save(fullfile(outDir,'s0_raw.mat'),'raw','sqzDims','-v7.3');
fprintf('s0 raw: %s  dims=%s\n', mat2str(size(raw)), strjoin(sqzDims,','));

% --- STAGE 1: FID-A load+prep (io_CSIload_twix_pair does combineTime+shift+split) ---
[refS, ~] = io_CSIload_twix_pair(REF, REF, kFile);
fprintf('s1 prep: sz=%s dims:\n', mat2str(refS.sz)); disp(refS.dims);
dumpStruct(refS, fullfile(outDir,'s1_prep.mat'));

% --- STAGE 2: recon (dft = exact, no IRT needed; also try nufft/tikhonov) ---
try
    rec_dft = op_CSIRecon(refS, kFile, 'nn', 'dft');
    dumpStruct(rec_dft, fullfile(outDir,'s2_recon_dft.mat'));
    fprintf('s2 dft recon OK sz=%s\n', mat2str(rec_dft.sz));
catch e; fprintf('dft FAILED: %s\n', e.message); end
try
    rec_nu = op_CSIRecon(refS, kFile, 'nn', 'nufft');
    dumpStruct(rec_nu, fullfile(outDir,'s2_recon_nufft.mat'));
    fprintf('s2 nufft recon OK sz=%s\n', mat2str(rec_nu.sz));
catch e; fprintf('nufft FAILED (IRT?): %s\n', e.message); end
try
    rec_tk = op_CSIRecon(refS, kFile, 'none', 'tikhonov');
    dumpStruct(rec_tk, fullfile(outDir,'s2_recon_tikhonov.mat'));
    fprintf('s2 tikhonov recon OK sz=%s\n', mat2str(rec_tk.sz));
catch e; fprintf('tikhonov FAILED: %s\n', e.message); end

fprintf('DONE -> %s\n', outDir);

function dumpStruct(s, path)
    o = struct();
    o.data = s.data; o.sz = s.sz; o.dims = s.dims;
    o.spectralWidth = gf(s,'spectralWidth',0);
    o.dwelltime = gf(s,'dwelltime', gf(s,'adcDwellTime',0));
    o.txfrq = gf(s,'txfrq',0); o.te = gf(s,'te',0); o.tr = gf(s,'tr',0);
    o.fov = s.fov; o.voxelSize = s.voxelSize; o.flags = s.flags;
    save(path,'-struct','o','-v7.3');
    fprintf('  dumped %s\n', path);
end
function v = gf(s,f,d); if isfield(s,f); v=s.(f); else; v=d; end; end
