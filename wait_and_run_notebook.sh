#!/usr/bin/env bash
# Wait for the intermittent F: drive to return, then execute the notebook and
# confirm the maps landed in the FID-A outputs folder.
FPY="/c/Users/divya/miniconda3/envs/fida/python.exe"
MET="F:/fida/divya/20260605_phantom_test/subject02/met/meas_MID00138_FID48082_Rosette_40x40_isoctr.dat"
OUT="F:/fida/divya/20260605_phantom_test/subject02/outputs/lcm_out_py"
cd "/c/Users/divya/Downloads/mrsi_pipeline" || exit 1

echo "waiting for F: ..."
for i in $(seq 1 120); do          # up to ~60 min (120 x 30s)
  if ls "$MET" >/dev/null 2>&1; then echo "F: back after $((i*30))s"; break; fi
  sleep 30
done
if ! ls "$MET" >/dev/null 2>&1; then echo "F: never returned; aborting."; exit 2; fi

echo "executing notebook ..."
"$FPY" -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=2400 --ExecutePreprocessor.kernel_name=fida \
  run_rosette_pipeline.ipynb
rc=$?
echo "nbconvert rc=$rc"
echo "=== outputs in FID-A location ==="
ls "$OUT/maps.npz" "$OUT/lcm_maps.png" 2>/dev/null
echo "tables: $(ls "$OUT"/*.table 2>/dev/null | wc -l)"
