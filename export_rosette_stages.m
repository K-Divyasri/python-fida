% export_rosette_stages.m
% Bridge for XA60 rosette: Python twix parsers (pymapVBVD 0.6.1, twixtools)
% can't read Siemens XA60 headers, but FID-A's MATLAB mapVBVD can. This loads
% the rosette pair with FID-A, then dumps (a) the loaded struct and (b) the
% op_CSIRecon output for each method, as -v7.3 .mat, so op_CSIRecon.py /
% op_CSIRosettePrep.load_struct_mat can read + compare them in Python.
%
% Run in MATLAB (R2024a).  ~1.5 GB read from F: -- ensure F: is mounted.

clear; clc;
addpath(genpath('C:\Users\divya\Downloads\fida codes\fid_a'));

datasetFolder = 'F:\fida\divya\28thJULYpHANTOM40X40\subject01';
kFile = 'C:\Users\divya\Downloads\fida codes\fid_a\processingTools\MRSI\kFiles\Rosette_traj_40x40.txt';
outDir = 'C:\Users\divya\Downloads\mrsi_pipeline\rosette_stages';
if ~exist(outDir,'dir'); mkdir(outDir); end

% --- locate the .dat files ---
md = dir(fullfile(datasetFolder,'met','*.dat'));      metFile = fullfile(md(1).folder, md(1).name);
rd = dir(fullfile(datasetFolder,'mrs_ref','*.dat'));  refFile = fullfile(rd(1).folder, rd(1).name);
fprintf('met: %s\nref: %s\n', metFile, refFile);

% --- 1) load pair (this is the io_CSIload_twix_pair the user asked about) ---
[met, ref] = io_CSIload_twix_pair(metFile, refFile, kFile);
fprintf('loaded. met dims:\n'); disp(met.dims);
fprintf('met.sz = %s\n', mat2str(met.sz));

dumpStruct(met, fullfile(outDir,'s1_rosette_load_met.mat'));
dumpStruct(ref, fullfile(outDir,'s1_rosette_load_ref.mat'));

% --- 2) recon each method (compare against op_CSIRecon.py) ---
try
    rec_nufft = op_CSIRecon(met, kFile, 'nn', 'nufft');
    dumpStruct(rec_nufft, fullfile(outDir,'s2_rosette_nufft.mat'));
    fprintf('nufft recon OK, sz=%s\n', mat2str(rec_nufft.sz));
catch e; fprintf('nufft FAILED: %s\n', e.message); end

try
    rec_dft = op_CSIRecon(met, kFile, 'nn', 'dft');
    dumpStruct(rec_dft, fullfile(outDir,'s2_rosette_dft.mat'));
    fprintf('dft recon OK, sz=%s\n', mat2str(rec_dft.sz));
catch e; fprintf('dft FAILED: %s\n', e.message); end

try
    rec_tikh = op_CSIRecon(met, kFile, 'none', 'tikhonov');
    dumpStruct(rec_tikh, fullfile(outDir,'s2_rosette_tikhonov.mat'));
    fprintf('tikhonov recon OK, sz=%s\n', mat2str(rec_tikh.sz));
catch e; fprintf('tikhonov FAILED: %s\n', e.message); end

fprintf('DONE -> %s\n', outDir);

% ---- helper: save the fields op_CSIRosettePrep.load_struct_mat expects ----
function dumpStruct(s, path)
    o = struct();
    o.data = s.data;
    o.sz = s.sz;
    o.dims = s.dims;
    o.spectralWidth = getfielddef(s,'spectralWidth',0);
    o.dwelltime = getfielddef(s,'dwelltime', getfielddef(s,'adcDwellTime',0));
    o.txfrq = getfielddef(s,'txfrq',0);
    o.te = getfielddef(s,'te',0); o.tr = getfielddef(s,'tr',0);
    o.fov = s.fov; o.voxelSize = s.voxelSize;
    o.flags = s.flags;
    if isfield(s,'spectralTime'); o.spectralTime = s.spectralTime; end
    save(path, '-struct', 'o', '-v7.3');
    fprintf('  dumped %s\n', path);
end

function v = getfielddef(s, f, d)
    if isfield(s,f); v = s.(f); else; v = d; end
end
