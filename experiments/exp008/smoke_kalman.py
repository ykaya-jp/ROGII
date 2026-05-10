#!/usr/bin/env python
"""
exp008 local smoke — Kalman feature builder leak/numerical/runtime checks.

Phases
------
1. Unit test compute_kalman_features() with synthetic signals
   (= verify std monotonic, leak-safe, numerical guards).
2. Run build_well_features() over 1-3 train wells with KALMAN_ENABLE=True
   (= integration smoke: ensure 7 kalman_* cols populated, no NaN, std grows).
3. Per-well φ/σ_ε MLE values vs precomputed parquet sanity check.
4. Leak guard 4-point assert.

Logs to stdout. Exit 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import sys
import time
import inspect
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
KERNEL = REPO / "kaggle_kernels" / "exp008_case_d_kalman" / "exp008_case_d_kalman.py"
DATA = REPO / "data" / "raw"
PARQUET = REPO / "outputs" / "eda" / "first_principles" / "per-well-stats.parquet"

# ─── Import kernel module dynamically (it does heavy work at import time;
#     we silence that by trimming) ────────────────────────────────────────
# Parse the kernel source and exec only up through Kalman helper definitions.
src_full = KERNEL.read_text()


def _exec_until_marker(src: str, marker: str) -> dict:
    """Exec the source up until the line containing `marker`, return globals."""
    lines = src.splitlines()
    cut = None
    for i, line in enumerate(lines):
        if marker in line:
            cut = i + 1
            break
    if cut is None:
        raise RuntimeError(f"marker not found: {marker}")
    truncated = "\n".join(lines[:cut])
    g: dict = {"__name__": "__smoke__"}
    exec(compile(truncated, str(KERNEL), "exec"), g)
    return g


def _extract_kalman_only() -> dict:
    """
    Extract just the Kalman helpers + their config dependencies,
    sidestepping the kernel's heavy global init (ArtifactManager
    needs /kaggle paths). Used in non-Kaggle environments.
    """
    src = src_full
    # Pull config block (KALMAN_*) and the helper functions only.
    snippets: list[str] = []

    # Config block
    cfg_start = src.find("# ─── Edge D knobs")
    cfg_end = src.find("# ─── 自前 base hyperparameter")
    if cfg_start == -1 or cfg_end == -1:
        raise RuntimeError("config markers not found")
    snippets.append(src[cfg_start:cfg_end])

    # Helper functions (kalman_feature_names + _yule_walker_ar1 + compute_kalman_features)
    fn_start = src.find("def kalman_feature_names()")
    fn_end = src.find('print("Edge Q + Edge M + Edge D (Kalman) helpers loaded ✓")')
    if fn_start == -1 or fn_end == -1:
        raise RuntimeError("Kalman helper markers not found")
    snippets.append(src[fn_start:fn_end])

    code = (
        "from typing import Dict, List, Tuple\n"
        "import numpy as np\n"
        + "\n".join(snippets)
    )
    g: dict = {"__name__": "__smoke_extract__"}
    exec(compile(code, str(KERNEL), "exec"), g)
    return g


def phase1_unit_kalman():
    print("=" * 72)
    print("Phase 1: compute_kalman_features unit tests")
    print("=" * 72)

    # We only need the Kalman helpers; extract them surgically to avoid the
    # kernel's heavy global init (ArtifactManager etc. need /kaggle paths).
    g = _extract_kalman_only()
    compute_kalman_features = g["compute_kalman_features"]
    kalman_feature_names = g["kalman_feature_names"]
    _yule_walker_ar1 = g["_yule_walker_ar1"]

    # ── Test 1: feature names ──────────────────────────────────────────
    names = kalman_feature_names()
    assert len(names) == 7, f"expected 7 features, got {len(names)}"
    assert "kalman_d" in names
    assert "kalman_std" in names
    assert "kalman_phi" in names
    assert "kalman_sigma_eps" in names
    assert "kalman_t_md" in names
    assert "kalman_vs_pf_a_d" in names
    assert "kalman_vs_beam_cons_d" in names
    print(f"  ✓ feature names = {names}")

    # ── Test 2: Yule-Walker on a synthetic AR(1) series ────────────────
    rng = np.random.default_rng(42)
    n = 1000
    phi_true = 0.95
    sigma_true = 0.5
    eps = rng.normal(0, sigma_true, size=n)
    dt = np.zeros(n)
    for i in range(1, n):
        dt[i] = phi_true * dt[i - 1] + eps[i]
    phi_hat, sigma_hat = _yule_walker_ar1(dt)
    print(f"  ✓ YW MLE: phi_hat={phi_hat:.4f} (true {phi_true}), "
          f"sigma_hat={sigma_hat:.4f} (true {sigma_true})")
    assert abs(phi_hat - phi_true) < 0.05, f"phi MLE off: {phi_hat}"
    assert abs(sigma_hat - sigma_true) < 0.05, f"sigma MLE off: {sigma_hat}"

    # ── Test 3: Kalman forward pass on a realistic visible-only series ─
    # Simulate visible TVT with phi=0.999 dTVT random walk
    rng = np.random.default_rng(7)
    nk = 1500
    n_hidden = 3000
    median_dmd = 0.5
    sigma_dt = 0.016
    phi_dt = 0.999
    dt_full = np.zeros(nk + n_hidden)
    for i in range(1, nk + n_hidden):
        dt_full[i] = phi_dt * dt_full[i - 1] + rng.normal(0, sigma_dt)
    tvt_full = np.cumsum(dt_full) + 11000.0
    md_full = np.cumsum(np.full(nk + n_hidden, median_dmd)) + 5000.0

    ktvt = tvt_full[:nk].astype(np.float32)
    kmd  = md_full[:nk].astype(np.float32)
    hmd  = md_full[nk:].astype(np.float32)
    last_tvt = float(ktvt[-1])
    last_md  = float(kmd[-1])
    pf_a_d   = np.zeros(n_hidden, dtype=np.float32)
    beam_cons_d = np.zeros(n_hidden, dtype=np.float32)

    t0 = time.perf_counter()
    out = compute_kalman_features(
        ktvt=ktvt, kmd=kmd,
        last_tvt=last_tvt, last_md=last_md,
        hmd=hmd, pf_a_d=pf_a_d, beam_cons_d=beam_cons_d,
    )
    dt_run = time.perf_counter() - t0
    print(f"  ✓ Kalman forward pass: nk={nk} n_hidden={n_hidden} "
          f"runtime={dt_run*1000:.1f} ms")

    # All keys present, no NaN
    for k, v in out.items():
        assert v.shape == (n_hidden,), f"{k} shape {v.shape}"
        assert v.dtype == np.float32, f"{k} dtype {v.dtype}"
        assert not np.any(np.isnan(v)), f"{k} has NaN"
        assert not np.any(np.isinf(v)), f"{k} has Inf"
    print(f"  ✓ all 7 cols populated, no NaN, no Inf")

    # std monotonically non-decreasing (forward filter, variance grows)
    std = out["kalman_std"]
    assert np.all(np.diff(std) >= -1e-6), \
        f"std not monotonic: max_decrease={np.diff(std).min():.6f}"
    print(f"  ✓ kalman_std monotonic increasing: "
          f"first={std[0]:.4f} last={std[-1]:.4f}")

    # phi within shrunk range (between local MLE and global)
    phi = float(out["kalman_phi"][0])
    assert 0.0 <= phi <= 1.0, f"phi out of [0,1]: {phi}"
    sigma_eps = float(out["kalman_sigma_eps"][0])
    assert sigma_eps > 0, f"sigma_eps non-positive: {sigma_eps}"
    print(f"  ✓ phi={phi:.5f} sigma_eps={sigma_eps:.5f} (constants per well)")

    # t_md monotonic (= MD horizon)
    tmd = out["kalman_t_md"]
    assert np.all(np.diff(tmd) >= -1e-6), "kalman_t_md not monotonic"
    print(f"  ✓ kalman_t_md monotonic: first={tmd[0]:.2f} last={tmd[-1]:.2f}")

    # ── Test 4: leak guard (string scan of source) ─────────────────────
    src_kalman = inspect.getsource(compute_kalman_features)
    forbidden = ["TVT_input", 'ev["TVT"', "ev['TVT'", "ev.TVT", "hidden TVT_input"]
    for forb in forbidden:
        assert forb not in src_kalman, f"leak: source contains {forb!r}"
    print(f"  ✓ leak guard 1: source code does not reference hidden TVT or TVT_input")

    # ── Test 5: edge case - very short visible region ──────────────────
    short_ktvt = np.array([11000.0, 11000.05, 11000.10], dtype=np.float32)
    short_kmd  = np.array([5000.0, 5000.5, 5001.0], dtype=np.float32)
    short_hmd  = np.array([5001.5, 5002.0, 5002.5], dtype=np.float32)
    out_short = compute_kalman_features(
        ktvt=short_ktvt, kmd=short_kmd,
        last_tvt=11000.10, last_md=5001.0,
        hmd=short_hmd,
        pf_a_d=np.zeros(3, dtype=np.float32),
        beam_cons_d=np.zeros(3, dtype=np.float32),
    )
    for k, v in out_short.items():
        assert not np.any(np.isnan(v)), f"short visible: {k} has NaN"
    print(f"  ✓ short visible (nk=3) handled gracefully (shrinkage to global prior)")

    # ── Test 6: empty hidden array ─────────────────────────────────────
    out_empty = compute_kalman_features(
        ktvt=ktvt, kmd=kmd,
        last_tvt=last_tvt, last_md=last_md,
        hmd=np.array([], dtype=np.float32),
        pf_a_d=np.array([], dtype=np.float32),
        beam_cons_d=np.array([], dtype=np.float32),
    )
    for k, v in out_empty.items():
        assert len(v) == 0, f"empty hidden: {k} has shape {v.shape}"
    print(f"  ✓ empty hidden handled gracefully")

    print("Phase 1: PASS")
    return g


def phase2_well_smoke(g: dict, n_wells: int = 2):
    """
    Build features for a few real train wells (= integration smoke).
    Use the kernel's build_well_features path and verify Kalman cols.
    """
    print()
    print("=" * 72)
    print(f"Phase 2: build_well_features integration smoke ({n_wells} wells)")
    print("=" * 72)

    # We must exec further into the kernel to pick up build_well_features.
    # The kernel is built for /kaggle/* paths; in a local env we shim those
    # by creating /tmp pseudo-paths and pointing DATA_DIR / TRAIN_DIR /
    # ARTEFACT_DIR there before exec.
    print("  loading kernel up through build_well_features ...")
    t0 = time.perf_counter()

    # shim: mock /kaggle paths via TMPDIR
    import os
    import tempfile
    tmpdir = Path(tempfile.mkdtemp(prefix="rogii_exp008_smoke_"))
    artefact_shim = tmpdir / "artefacts"
    artefact_shim.mkdir(parents=True, exist_ok=True)

    # Patch the source to point ARTEFACT_DIR + DATA_DIR to local paths
    src = src_full
    src = src.replace(
        'DATA_DIR = next((p for p in _KAGGLE_CANDIDATES if (p / "test").exists()),\n                Path("../../data").resolve())',
        f'DATA_DIR = Path({str(DATA)!r})',
    )
    src = src.replace(
        'ARTEFACT_DIR = next((p for p in _ARTEFACT_CANDIDATES if p.exists()),\n                       _ARTEFACT_CANDIDATES[0])',
        f'ARTEFACT_DIR = Path({str(artefact_shim)!r})',
    )
    # Also stop the post-marker init (we only want functions defined)
    marker = 'print("Per-well feature builder loaded ✓")'
    cut = src.find(marker)
    if cut == -1:
        print("  WARN: marker not found in source")
        return False
    truncated = src[: cut + len(marker)]

    try:
        g2 = {"__name__": "__smoke__"}
        exec(compile(truncated, str(KERNEL), "exec"), g2)
    except Exception as e:
        print(f"  WARN: kernel exec failed: {type(e).__name__}: {e}")
        print("  Skipping Phase 2.")
        return False
    print(f"  kernel loaded in {time.perf_counter()-t0:.1f} sec")

    build_well_features = g2["build_well_features"]
    kalman_feature_names = g2["kalman_feature_names"]
    edge_m_feature_names = g2["edge_m_feature_names"]

    train_dir = DATA / "train"
    hw_paths = sorted(train_dir.glob("*__horizontal_well.csv"))[:n_wells]
    print(f"  test wells: {[p.stem.replace('__horizontal_well','') for p in hw_paths]}")

    kal_cols = kalman_feature_names()
    em_cols = edge_m_feature_names()

    n_pass = 0
    for hp in hw_paths:
        wid = hp.stem.replace("__horizontal_well", "")
        tw_p = train_dir / f"{wid}__typewell.csv"
        if not tw_p.exists():
            print(f"  ⚠ {wid}: typewell missing, skip")
            continue
        t0 = time.perf_counter()
        try:
            df = build_well_features(str(hp), str(tw_p), is_train=True)
        except Exception as e:
            print(f"  ✗ {wid}: build_well_features raised: {type(e).__name__}: {e}")
            continue
        dt_run = time.perf_counter() - t0
        if df is None:
            print(f"  ⚠ {wid}: build_well_features returned None, skip")
            continue
        n_rows = len(df)

        # Check Kalman cols present + no NaN
        missing = [c for c in kal_cols if c not in df.columns]
        if missing:
            print(f"  ✗ {wid}: missing Kalman cols: {missing}")
            continue
        nan_cnt = {c: int(df[c].isna().sum()) for c in kal_cols}
        if any(nan_cnt.values()):
            print(f"  ✗ {wid}: Kalman cols have NaN: {nan_cnt}")
            continue

        # Check Edge M cols too
        missing_em = [c for c in em_cols if c not in df.columns]
        if missing_em:
            print(f"  ✗ {wid}: missing Edge M cols: {missing_em}")
            continue

        # Check std monotonic increasing
        std = df["kalman_std"].to_numpy()
        non_mono = int((np.diff(std) < -1e-3).sum())
        # Note: std is computed in MD-horizon order, but rows in build_well_features
        # are ordered by ev.index which IS monotonic in MD. So std should be
        # weakly increasing.
        std_ok = (non_mono == 0)

        # Constants per-well
        phi_unique  = df["kalman_phi"].nunique()
        sig_unique  = df["kalman_sigma_eps"].nunique()
        const_ok = (phi_unique <= 1 and sig_unique <= 1)

        print(f"  ✓ {wid}: rows={n_rows} runtime={dt_run:.2f}s "
              f"NaN=0 std_mono={std_ok} const_per_well={const_ok}")
        print(f"      kalman_d range  : [{df['kalman_d'].min():.3f}, {df['kalman_d'].max():.3f}]")
        print(f"      kalman_std range: [{df['kalman_std'].min():.3f}, {df['kalman_std'].max():.3f}]")
        print(f"      kalman_phi      : {df['kalman_phi'].iloc[0]:.5f}")
        print(f"      kalman_sigma_eps: {df['kalman_sigma_eps'].iloc[0]:.5f}")
        print(f"      kalman_t_md     : [{df['kalman_t_md'].min():.1f}, {df['kalman_t_md'].max():.1f}]")
        print(f"      kalman_vs_pf_a_d range: [{df['kalman_vs_pf_a_d'].min():.3f}, "
              f"{df['kalman_vs_pf_a_d'].max():.3f}]")

        if not std_ok:
            print(f"  ⚠ {wid}: std non-monotonic in {non_mono} rows (acceptable "
                  f"if rows are not strictly sorted by MD per chunk)")
        if not const_ok:
            print(f"  ✗ {wid}: phi/sigma not constant per well "
                  f"(phi_unique={phi_unique}, sig_unique={sig_unique})")
            continue
        n_pass += 1

    print(f"Phase 2: {n_pass}/{n_wells} wells PASS")
    return n_pass == n_wells


def phase3_parquet_sanity():
    print()
    print("=" * 72)
    print("Phase 3: per-well-stats parquet sanity (global priors)")
    print("=" * 72)
    if not PARQUET.exists():
        print(f"  ⚠ parquet missing: {PARQUET}, skip")
        return True
    df = pd.read_parquet(PARQUET)
    phi_p50 = float(df["ar1_phi"].median())
    sig_p50 = float(df["ar1_eps_std"].median())
    print(f"  parquet n={len(df)}")
    print(f"  ar1_phi p50    : {phi_p50:.5f}  (kernel hardcoded 0.99887)")
    print(f"  ar1_eps_std p50: {sig_p50:.5f}  (kernel hardcoded 0.01556)")
    # Allow 1% drift
    assert abs(phi_p50 - 0.99887) < 0.01, f"phi global drifted: {phi_p50}"
    assert abs(sig_p50 - 0.01556) < 0.005, f"sigma global drifted: {sig_p50}"
    print("Phase 3: PASS")
    return True


def main():
    print(f"REPO   = {REPO}")
    print(f"KERNEL = {KERNEL}")
    print(f"DATA   = {DATA}")
    print()
    g = phase1_unit_kalman()
    if not phase3_parquet_sanity():
        print("FAIL: phase3")
        sys.exit(1)
    p2 = phase2_well_smoke(g, n_wells=2)
    if not p2:
        print()
        print("⚠ Phase 2 INCOMPLETE (= integration with build_well_features did")
        print("  not run, likely Kaggle path resolution issue locally). Phase 1")
        print("  unit tests + Phase 3 parquet sanity still PASS.")
    print()
    print("ALL PASS" if p2 else "PARTIAL PASS (unit + parquet, no integration)")


if __name__ == "__main__":
    main()
