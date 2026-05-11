#!/usr/bin/env bash
# exp004 full pipeline: HPO LGB -> HPO CB -> CV ablation -> kernel patch -> push.
# Assumes outputs/exp004/features.parquet + features_test.parquet already exist.
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH=src
PY=.venv/bin/python

# --- HPO (M2) -----------------------------------------------------------------
if [[ ! -f outputs/exp004/hpo_lgb.json ]]; then
  echo "[run_pipeline] HPO LGB 100 trials..."
  $PY -m rogii.hpo --target lgb --trials 100 --features outputs/exp004/features.parquet
fi
if [[ ! -f outputs/exp004/hpo_cb.json ]]; then
  echo "[run_pipeline] HPO CB 100 trials..."
  $PY -m rogii.hpo --target cb --trials 100 --features outputs/exp004/features.parquet
fi

# --- Loss ablation (M3) -------------------------------------------------------
echo "[run_pipeline] Loss ablation 3 variants..."
for OBJ in regression huber regression_l1; do
  TAG=$(echo "$OBJ" | tr -d '_')
  OUTLOG=outputs/exp004/cv_${TAG}.log
  if [[ ! -f $OUTLOG ]]; then
    $PY -m rogii.train_v3 --exp exp004 --cv-only --objective "$OBJ" --quiet > "$OUTLOG" 2>&1
  fi
done

# --- D1 ablation (M4) ---------------------------------------------------------
echo "[run_pipeline] D1 cluster ablation (OFF)..."
[[ -f outputs/exp004/cv_nod1.log ]] || $PY -m rogii.train_v3 --exp exp004 --cv-only --no-d1 --quiet > outputs/exp004/cv_nod1.log 2>&1

# --- Default (D1 ON, regression) full run -------------------------------------
echo "[run_pipeline] CV main run (D1 ON, regression)..."
[[ -f outputs/exp004/cv_main.log ]] || $PY -m rogii.train_v3 --exp exp004 --cv-only --quiet > outputs/exp004/cv_main.log 2>&1

# --- Kernel patch + ruff ------------------------------------------------------
echo "[run_pipeline] Patch kernel script..."
$PY scripts/patch_kernel.py
$PY -m ruff check kaggle_kernels/exp004_numba_beam_pf/exp004_numba_beam_pf.py

# --- Kaggle push --------------------------------------------------------------
echo "[run_pipeline] Push kernel to Kaggle..."
.venv/bin/kaggle kernels push -p kaggle_kernels/exp004_numba_beam_pf/

echo "[run_pipeline] DONE."
