% TWIX -> FID-A struct via mapVBVD (MATLAB).  Run in MATLAB.
% Reads k-space, arranges as FID-A [t, coils, x, y], writes a FID-A-compatible struct.
addpath(genpath('C:\Users\divya\Downloads\mrsi_pipeline\matlab_deps\mapVBVD'));

dat = 'F:\fida\divya\20260605_phantom_test\subject04\met\meas_MID00151_FID48095_csi_fid_24x24_isoctr.dat';
out = 'C:\Users\divya\Downloads\mrsi_pipeline\readers\out\fida_mapvbvd.mat';

tw = mapVBVD(dat, 'removeOS', false);
if iscell(tw); tw = tw{end}; end
tw.image.flagRemoveOS = false;
k = squeeze(tw.image());                 % [Col, Cha, Lin, Seg]
fids = permute(k, [1 2 4 3]);            % -> [t, coils, x(=Seg), y(=Lin)]

dt = tw.hdr.MeasYaps.sRXSPEC.alDwellTime{1} * 1e-9;
te = tw.hdr.MeasYaps.alTE{1} / 1000;     % ms
try; tr = tw.hdr.MeasYaps.alTR{1} / 1000; catch; tr = NaN; end
txfrq = tw.hdr.MeasYaps.sTXSPEC.asNucleusInfo{1}.lFrequency;

n = size(fids,1); sw = 1/dt;
t = (0:n-1)*dt;  ppm = ((0:n-1)-floor(n/2))/(n*dt) / -(txfrq/1e6) + 4.65;
fida = struct();
fida.fids = fids;
fida.specs = fftshift(fft(fids,[],1),1);
fida.sz = size(fids);
fida.dims = struct('t',1,'coils',2,'x',3,'y',4,'z',0,'averages',0,'subSpecs',0,'extras',0);
fida.spectralwidth = sw; fida.dwelltime = dt;
fida.txfrq = txfrq; fida.Bo = txfrq/42.577478e6;
fida.te = te; fida.tr = tr; fida.n = n; fida.t = t; fida.ppm = ppm;
fida.seq = 'csi_fid'; fida.date=''; fida.sim=''; fida.pointsToLeftshift=0;
fida.flags = struct('writtentostruct',1,'gotparams',1,'leftshifted',0,'filtered',0,...
    'zeropadded',0,'freqcorrected',0,'phasecorrected',0,'averaged',0,'addedrcvrs',0,...
    'subtracted',0,'writtentotext',0,'downsampled',0,'isFourSteps',0);

if ~exist(fileparts(out),'dir'); mkdir(fileparts(out)); end
save(out, 'fida', '-v7');
fprintf('wrote FID-A struct %s  fids [%s]  SW=%.0f txfrq=%.4f MHz TE=%g\n', ...
    out, num2str(size(fids)), sw, txfrq/1e6, te);
