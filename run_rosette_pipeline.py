"""run_rosette_pipeline.py -- full rosette MRSI pipeline (pure-Python FID-A port).

Mirrors FID-A run_rosette_pipeline.m step-for-step:

  1 load pair              io_CSIload_twix_pair
  2 prep (reshape)         prep_noncartesian  (combine_time + op_CSIShift + split)
  3 spatial recon          op_CSIRecon
  4 coil combine           op_CSICombineCoils1
  5 average + water mask    op_CSIAverage + op_CSISegment_simple
  6 spectral FT            op_CSIFourierTransform
  7 SSP lipid suppression  op_CSIssp (0.8-1.88 ppm)
  8 water removal (L2)     op_CSIRemoveLipids (4.5-5.0 ppm)   [stochastic basis]
  9 B0 correction          op_CSIB0Correction_v2
 10 apply mask             op_CSIapplymask
 11 spatial smoothing      op_CSIApodize (gaussian, FWHM 20 mm)
 12 180 flip (overlay)     op_CSIFlip180

Returns a dict of every stage. With save_dir set, dumps each stage to
<save_dir>/<key>.npz (data + dims + ppm + fov + vox + flags) for cross-checking
against the MATLAB dump via compare_intermediates.py.

NOTE: the met .dat in 28thJULYpHANTOM40X40\subject01 is a corrupted INDX
container; MET defaults to the water ref so the pipeline runs end-to-end now.
Point MET at a re-copied standard-twix metabolite file for real fitting.

Run:  python run_rosette_pipeline.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'mrsi'))
import os, json
import numpy as np

from io_CSIload_twix import io_CSIload_twix_pair
from op_CSIRosettePrep import prep_noncartesian
from op_CSIRecon import op_CSIRecon
from op_CSIFourierTransform import op_CSIFourierTransform
from op_CSICombineCoils import op_CSICombineCoils1, op_CSIAverage, op_CSISegment_simple
from op_CSIspectralproc import (op_CSIB0Correction_v2, op_CSIRemoveLipids,
                                hsvd_water_removal, op_CSIphase, op_CSIspecZeroFill)
from op_CSIpostproc import op_CSIssp, op_CSIapplymask, op_CSIApodize, op_CSIFlip180
from op_CSIRecenter import recenter_pair

# ============================ CONFIG ============================
REF   = r'F:\fida\divya\28thJULYpHANTOM40X40\subject01\mrs_ref\meas_MID01095_FID59456_XA60_RosetteSpinEcho_2_avg_8mste_4sTR_w.dat'
MET   = REF     # corrupted INDX met -> use ref as proxy; swap for re-copied met
KFILE = r'C:\Users\divya\Downloads\fida codes\fid_a\processingTools\MRSI\kFiles\Rosette_traj_40x40.txt'
SAVE  = r'C:\Users\divya\Downloads\mrsi_pipeline\rosette_py_stages'
DCF, METHOD, SEQ = 'nn', 'dft', 'rosette'
LIPID_SEED = 0            # for the stochastic water-removal basis (reproducible)
WATER_BASIS = r'C:\Users\divya\Downloads\mrsi_pipeline\lipid_basis_te15_water.npy'  # FID-A reference basis -> exact match
# ===============================================================


def save_stage(save_dir, key, s):
    if save_dir is None:
        return
    os.makedirs(save_dir, exist_ok=True)
    meta = dict(dims=s.get('dims', {}), flags=s.get('flags', {}),
                fov=s.get('fov', {}), voxelSize=s.get('voxelSize', {}))
    np.savez(os.path.join(save_dir, key + '.npz'),
             data=s['data'],
             ppm=np.asarray(s['ppm']) if s.get('ppm') is not None else np.array([]),
             meta=json.dumps(meta, default=float))


def run_pipeline(metFile=MET, refFile=REF, kfile=KFILE, seq_type=SEQ,
                 method=METHOD, dcf=DCF, save_dir=None, lipid_seed=LIPID_SEED,
                 skip_water_removal=False, water_removal='hsvd', do_phase=True,
                 do_lcmodel=False, lcm_out=None, recenter=True, phantom=True,
                 zerofill=4096):
    st = {}
    def keep(k, s): st[k] = s; save_stage(save_dir, k, s); return s

    print('1 load...');   met, ref = io_CSIload_twix_pair(metFile, refFile, kfile, seq_type)
    keep('s01_load_met', met); keep('s01_load_ref', ref)
    print('2 prep...');   mp = keep('s02_prep_met', prep_noncartesian(met, kfile, seq_type))
    wp = keep('s02_prep_ref', prep_noncartesian(ref, kfile, seq_type))
    print('3 recon...');  ftS = keep('s03_recon_met', op_CSIRecon(mp, kfile, dcf, method))
    ftSw = keep('s03_recon_ref', op_CSIRecon(wp, kfile, dcf, method))
    print('4 combine...')
    ccw, phase, weights = op_CSICombineCoils1(ftSw); keep('s04_combine_ref', ccw)
    cc = op_CSICombineCoils1(ftS, 1, phase, weights)[0]; keep('s04_combine_met', cc)
    if recenter:
        cc, ccw, roll = recenter_pair(cc, ccw)         # centre off-centre VOI (data-driven)
        print(f'4b recenter roll (dy,dx)={roll}')       # no-op (0,0) for iso-centred data
        keep('s04b_recenter_met', cc); keep('s04b_recenter_ref', ccw)
    print('5 avg+mask...')
    ccav = op_CSIAverage(cc); ccav_w = op_CSIAverage(ccw)
    ccav_w = op_CSISegment_simple(ccav_w)
    ccav['mask'] = ccav_w['mask']
    keep('s05_ccav_met', ccav); keep('s05_ccav_ref', ccav_w)
    print('6 spectral FT...')
    ftSpec = keep('s06_spec_met', op_CSIFourierTransform(ccav, spatial=False, spectral=True))
    ftSpec_w = keep('s06_spec_ref', op_CSIFourierTransform(ccav_w, spatial=False, spectral=True))
    if phantom:
        print('7 SSP SKIPPED (phantom: 0.8-1.88 ppm band removes Lac)')
        rmlip = keep('s07_ssp_met', ftSpec)
    else:
        print('7 SSP lipid...')
        rmlip = keep('s07_ssp_met', op_CSIssp(ftSpec, 0.8, 1.88))
    wm = ccav['mask']['brainmasks']
    if skip_water_removal or water_removal == 'none':
        print('8 water removal SKIPPED')
        rmw = keep('s08_rmwater_met', rmlip)
    elif water_removal == 'hsvd':
        print('8 water removal (HSVD)...')
        rmw = keep('s08_rmwater_met', hsvd_water_removal(rmlip, rank=30, mask=wm))
    else:                                             # L2 water removal
        wbasis = np.load(WATER_BASIS) if os.path.exists(WATER_BASIS) else None
        if wbasis is not None:
            print(f'8 water removal (L2, FID-A reference basis {wbasis.shape})...')
        else:
            print('8 water removal (L2, stochastic basis)...')
        rmw = keep('s08_rmwater_met', op_CSIRemoveLipids(rmlip, lipidPPMRange=(4.5, 5.0),
                                                         lineWidthRange=(1, 10), lipidBasis=wbasis,
                                                         rng=np.random.default_rng(lipid_seed)))
    print('9 B0...')
    b0, b0w, fmap, r2 = op_CSIB0Correction_v2(rmw, ftSpec_w)
    keep('s09_b0_met', b0); keep('s09_b0_ref', b0w)
    if zerofill and zerofill > b0['data'].shape[b0['dims'].get('f', b0['dims'].get('t', 1)) - 1]:
        print(f'9c zero-fill spectral -> {zerofill} ...')
        b0 = op_CSIspecZeroFill(b0, zerofill); b0w = op_CSIspecZeroFill(b0w, zerofill)
        keep('s09c_zf_met', b0); keep('s09c_zf_ref', b0w)
    b0_raw = b0                                    # pre-ACME, zero-filled (FID-A LCModel input)
    if do_phase:
        print('9b phasing (ACME)...')
        b0 = keep('s09b_phased_met', op_CSIphase(b0, range_ppm=(0.5, 4.0), mask=wm))
    print('10 mask...');  masked = keep('s10_masked_met', op_CSIapplymask(b0))
    print('11 apodize...')
    smooth = keep('s11_smooth_met', op_CSIApodize(masked, 'gaussian', 20))
    smooth_w = keep('s11_smooth_ref', op_CSIApodize(b0w, 'gaussian', 20))
    print('12 flip180...'); keep('s12_flip_met', op_CSIFlip180(smooth))
    st['freqMap'] = fmap; st['R2Map'] = r2

    # ---- 13: LCModel per-voxel fit -> conc/CRLB maps (FID-A-faithful input) ----
    if do_lcmodel:
        print('13 LCModel per-voxel fit...')
        from fit_lcmodel_rosette import fit_maps
        from viz._common import to_fyx
        b0_raw['mask'] = ccav['mask']              # no-ACME met, apodized + masked (= FID-A)
        met_fit = op_CSIApodize(op_CSIapplymask(b0_raw), 'gaussian', 20)
        met_fyx = to_fyx(met_fit['data'], met_fit['dims'])
        wat_fyx = to_fyx(smooth_w['data'], smooth_w['dims'])
        out = lcm_out or ((save_dir.rstrip('\\/') + '_lcm') if save_dir else 'lcm_out')
        st['lcm'] = fit_maps(met_fyx, wat_fyx, np.asarray(smooth['ppm']), wm, out)
    print('done. stages:', len(st))
    return st


if __name__ == '__main__':
    run_pipeline(save_dir=SAVE)
    print('saved stages ->', SAVE)
