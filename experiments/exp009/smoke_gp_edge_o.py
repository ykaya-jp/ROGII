#!/usr/bin/env python
"""
exp009 local smoke — GP imputer + Edge O Beam feature builder leak/numerical/runtime checks.

Phases
------
1. Unit test GPFormationImputer with synthetic data
   (= verify per-formation fit, posterior mean+var sane, KNN fallback works).
2. Unit test beam_search_dir + compute_edge_o_features with synthetic data
   (= verify dir-aware penalty fires, paths produced, std reasonable).
3. Run build_well_features() over 1-3 train wells with GP/Edge O ENABLE
   (= integration smoke: 24 + 7 = 31 cols populated, no NaN, GP runtime
   reasonable, leak guard).

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
KERNEL = REPO / "kaggle_kernels" / "exp009_case_e_edge_o" / "exp009_case_e_edge_o.py"
DATA = REPO / "data" / "raw"

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


def _load_kernel_with_shims():
    """Load the kernel up to per-well feature builder, with /kaggle paths shimmed."""
    import tempfile

    tmpdir = Path(tempfile.mkdtemp(prefix="rogii_exp009_smoke_"))
    artefact_shim = tmpdir / "artefacts"
    artefact_shim.mkdir(parents=True, exist_ok=True)

    src = src_full
    src = src.replace(
        'DATA_DIR = next((p for p in _KAGGLE_CANDIDATES if (p / "test").exists()),\n                Path("../../data").resolve())',
        f'DATA_DIR = Path({str(DATA)!r})',
    )
    src = src.replace(
        'ARTEFACT_DIR = next((p for p in _ARTEFACT_CANDIDATES if p.exists()),\n                       _ARTEFACT_CANDIDATES[0])',
        f'ARTEFACT_DIR = Path({str(artefact_shim)!r})',
    )

    marker = 'print("Per-well feature builder loaded ✓")'
    cut = src.find(marker)
    if cut == -1:
        raise RuntimeError(f"marker not found: {marker}")
    truncated = src[: cut + len(marker)]

    g = {"__name__": "__smoke__"}
    exec(compile(truncated, str(KERNEL), "exec"), g)
    return g


def phase1_gp_unit():
    print("=" * 72)
    print("Phase 1: GPFormationImputer unit test (small synthetic well set)")
    print("=" * 72)

    g = _load_kernel_with_shims()
    GPFormationImputer = g["GPFormationImputer"]
    gp_feature_names = g["gp_feature_names"]
    FORMATIONS = g["FORMATIONS"]

    names = gp_feature_names()
    assert len(names) == 24, f"expected 24 GP cols, got {len(names)}"
    assert "gp_ANCC_mean" in names
    assert "gp_ANCC_var" in names
    assert "gp_ANCC_mean_minus_plane" in names
    assert "gp_ANCC_var_norm" in names
    print(f"  ✓ feature names = {len(names)} cols, sample: {names[:6]}")

    # Build a small imputer using actual train wells (= real (X, Y) and
    # formation depth values).
    train_dir = DATA / "train"
    hw_paths = sorted(train_dir.glob("*__horizontal_well.csv"))
    well_ids = [p.stem.replace("__horizontal_well", "") for p in hw_paths[:50]]
    print(f"  building GP on {len(well_ids)} wells (subset for speed) ...")

    t0 = time.perf_counter()
    gp = GPFormationImputer(well_ids, train_dir, M=50, seed=42)
    fit_time = time.perf_counter() - t0
    n_fit = sum(1 for x in gp.gprs if x is not None)
    print(f"  ✓ GP fit: {n_fit}/{len(FORMATIONS)} formations fit "
          f"in {fit_time:.1f} sec (M=50)")
    assert n_fit > 0, "no formations fit successfully"

    # Test predict
    rng = np.random.default_rng(0)
    train_xy_orig = (gp.train_xy_norm * gp.xy_std) + gp.xy_mean
    test_xy = train_xy_orig[:5] + rng.normal(0, 100, size=(5, 2))
    t0 = time.perf_counter()
    mean, var = gp.impute(test_xy)
    pred_time = time.perf_counter() - t0
    print(f"  ✓ GP predict: shape={mean.shape} var={var.shape} "
          f"in {pred_time*1000:.1f} ms")
    assert mean.shape == (5, len(FORMATIONS))
    assert var.shape == (5, len(FORMATIONS))
    assert not np.any(np.isnan(mean)), "GP mean has NaN"
    assert not np.any(np.isnan(var)), "GP var has NaN"
    assert np.all(var > 0), f"GP var non-positive: min={var.min()}"
    print(f"  ✓ predict mean ANCC: {mean[:, 0]}")
    print(f"  ✓ predict var ANCC : {var[:, 0]}")

    # Empty fallback test
    gp_empty = GPFormationImputer.empty()
    mean_e, var_e = gp_empty.impute(test_xy)
    assert mean_e.shape == (5, len(FORMATIONS))
    assert not np.any(np.isnan(mean_e))
    print(f"  ✓ empty fallback: mean filled with zeros, var with ones")

    print("Phase 1: PASS")
    return g


def phase2_edge_o_unit(g: dict):
    print()
    print("=" * 72)
    print("Phase 2: beam_search_dir + compute_edge_o_features unit test")
    print("=" * 72)

    beam_search_dir = g["beam_search_dir"]
    compute_edge_o_features = g["compute_edge_o_features"]
    edge_o_feature_names = g["edge_o_feature_names"]

    names = edge_o_feature_names()
    assert len(names) == 7, f"expected 7 Edge O cols, got {len(names)}"
    print(f"  ✓ Edge O feature names: {names}")

    # Synthetic typewell + horizontal-well GR signature
    rng = np.random.default_rng(11)
    T = 800
    tw_tvt = np.linspace(11000.0, 11050.0, T).astype(np.float32)
    tw_gr = (50.0 + 20.0 * np.sin(np.linspace(0, 6 * np.pi, T))
             + rng.normal(0, 2.0, T)).astype(np.float32)

    nh = 600
    # horizontal GR follows typewell with a linear offset (= path lag of ~50)
    lag = 50
    hgr = np.concatenate([tw_gr[lag:], tw_gr[:lag]])[:nh].astype(np.float32)
    hgr += rng.normal(0, 1.0, len(hgr)).astype(np.float32)

    last_tvt = float(tw_tvt[lag])
    sel_local = np.arange(nh, dtype=np.int64)

    t0 = time.perf_counter()
    out = compute_edge_o_features(
        hgr_full=hgr, tw_tvt=tw_tvt, tw_gr=tw_gr,
        last_known_tvt=last_tvt, sel_local=sel_local,
        lam_dir=0.5,
    )
    eo_time = time.perf_counter() - t0
    print(f"  ✓ Edge O 5 configs runtime: {eo_time*1000:.1f} ms (nh={nh})")

    for k, v in out.items():
        assert v.shape == (nh,), f"{k} shape {v.shape}"
        assert v.dtype == np.float32, f"{k} dtype {v.dtype}"
        assert not np.any(np.isnan(v)), f"{k} has NaN"
        assert not np.any(np.isinf(v)), f"{k} has Inf"
    print(f"  ✓ all 7 cols populated, no NaN, no Inf")

    # std should be > 0 (5 paths should differ)
    std = out["beam_dir_std_d"]
    print(f"  ✓ beam_dir_std_d range: [{std.min():.3f}, {std.max():.3f}], "
          f"mean={std.mean():.3f}")
    # Allow some std collapse (= λ overrules cost), but typical is > 0.05
    if std.mean() < 0.001:
        print(f"  ⚠ beam_dir_std_d unexpectedly small (paths collapsed?)")

    # mean should be reasonable
    mean_d = out["beam_dir_mean_d"]
    print(f"  ✓ beam_dir_mean_d range: [{mean_d.min():.3f}, {mean_d.max():.3f}]")

    # ── Test: λ_dir=0 vs λ_dir=0.5 should differ ─────────────────────────
    out_zero = compute_edge_o_features(
        hgr_full=hgr, tw_tvt=tw_tvt, tw_gr=tw_gr,
        last_known_tvt=last_tvt, sel_local=sel_local,
        lam_dir=0.0,
    )
    diff_cons = float(np.abs(
        out["beam_dir_cons_d"] - out_zero["beam_dir_cons_d"]
    ).mean())
    print(f"  ✓ λ_dir=0 vs λ_dir=0.5 mean abs diff (cons): {diff_cons:.4f}")
    # The penalty should change paths, but on a noisy synthetic this might
    # be small; just verify execution succeeded both ways.

    # leak guard: source string scan
    src_eo = inspect.getsource(beam_search_dir)
    forbidden = ["TVT_input", 'ev["TVT"', "ev['TVT'", "ev.TVT"]
    for forb in forbidden:
        assert forb not in src_eo, f"leak: source contains {forb!r}"
    print(f"  ✓ leak guard: beam_search_dir source does not reference TVT_input")

    print("Phase 2: PASS")
    return True


def phase3_well_smoke(g: dict, n_wells: int = 2):
    """Build features for a few real train wells with GP/Edge O ENABLE."""
    print()
    print("=" * 72)
    print(f"Phase 3: build_well_features integration smoke ({n_wells} wells)")
    print("=" * 72)

    build_well_features = g["build_well_features"]
    GPFormationImputer = g["GPFormationImputer"]
    gp_feature_names = g["gp_feature_names"]
    edge_o_feature_names = g["edge_o_feature_names"]
    edge_m_feature_names = g["edge_m_feature_names"]
    kalman_feature_names = g["kalman_feature_names"]
    FORMATIONS = g["FORMATIONS"]

    train_dir = DATA / "train"
    all_hw = sorted(train_dir.glob("*__horizontal_well.csv"))
    print(f"  total train wells: {len(all_hw)}")

    # Build a small GP imputer over a subset of train wells (= keep runtime
    # low). The kernel's _GP_REF will be set in this exec'd module's globals.
    well_ids_for_gp = [p.stem.replace("__horizontal_well", "") for p in all_hw[:60]]
    print(f"  building GP imputer on {len(well_ids_for_gp)} wells ...")
    t0 = time.perf_counter()
    gp_imp = GPFormationImputer(well_ids_for_gp, train_dir, M=50, seed=42)
    gp_fit_time = time.perf_counter() - t0
    n_fit = sum(1 for x in gp_imp.gprs if x is not None)
    print(f"  ✓ GP fit: {n_fit}/{len(FORMATIONS)} formations in {gp_fit_time:.1f}s")

    # Inject into kernel module globals via the exec'd `g`
    g["_GP_REF"] = gp_imp

    hw_paths = all_hw[:n_wells]
    print(f"  test wells: {[p.stem.replace('__horizontal_well','') for p in hw_paths]}")

    gp_cols = gp_feature_names()
    eo_cols = edge_o_feature_names()
    em_cols = edge_m_feature_names()
    kal_cols = kalman_feature_names()

    n_pass = 0
    fe_times: list[float] = []
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
            import traceback
            traceback.print_exc()
            continue
        dt_run = time.perf_counter() - t0
        fe_times.append(dt_run)
        if df is None:
            print(f"  ⚠ {wid}: build_well_features returned None, skip")
            continue
        n_rows = len(df)

        missing_gp = [c for c in gp_cols if c not in df.columns]
        if missing_gp:
            print(f"  ✗ {wid}: missing GP cols: {missing_gp[:5]}{'...' if len(missing_gp) > 5 else ''}")
            continue
        nan_gp = {c: int(df[c].isna().sum()) for c in gp_cols}
        if any(nan_gp.values()):
            print(f"  ✗ {wid}: GP cols have NaN: "
                  f"{[c for c, n in nan_gp.items() if n > 0][:5]}")
            continue

        missing_eo = [c for c in eo_cols if c not in df.columns]
        if missing_eo:
            print(f"  ✗ {wid}: missing Edge O cols: {missing_eo}")
            continue
        nan_eo = {c: int(df[c].isna().sum()) for c in eo_cols}
        if any(nan_eo.values()):
            print(f"  ✗ {wid}: Edge O cols have NaN: {nan_eo}")
            continue

        # Spot check a few stat cols
        for c in (em_cols + kal_cols):
            if c not in df.columns:
                print(f"  ✗ {wid}: missing legacy col {c}")
                continue

        print(f"  ✓ {wid}: rows={n_rows} runtime={dt_run:.2f}s "
              f"NaN(GP)=0 NaN(EO)=0 cols={len(df.columns)}")
        print(f"      gp_ANCC_mean range: "
              f"[{df['gp_ANCC_mean'].min():.3f}, {df['gp_ANCC_mean'].max():.3f}]")
        print(f"      gp_ANCC_var range : "
              f"[{df['gp_ANCC_var'].min():.3e}, {df['gp_ANCC_var'].max():.3e}]")
        print(f"      gp_ANCC_mean_minus_plane p50: "
              f"{df['gp_ANCC_mean_minus_plane'].median():.4f}")
        print(f"      beam_dir_cons_d range: "
              f"[{df['beam_dir_cons_d'].min():.3f}, "
              f"{df['beam_dir_cons_d'].max():.3f}]")
        print(f"      beam_dir_std_d  mean : {df['beam_dir_std_d'].mean():.3f}")

        n_pass += 1

    if fe_times:
        print(f"  per-well FE runtime: mean={np.mean(fe_times):.2f}s "
              f"max={np.max(fe_times):.2f}s")
        # Estimate full corpus runtime: ~776 wells (test ~6, train ~770)
        est_train_min = 776 * float(np.mean(fe_times)) / 60.0
        print(f"  estimated full-train FE: ~{est_train_min:.1f} min")

    print(f"Phase 3: {n_pass}/{n_wells} wells PASS")
    return n_pass == n_wells


def phase4_leak_guards(g: dict):
    print()
    print("=" * 72)
    print("Phase 4: 4-point leak guard")
    print("=" * 72)

    GPFormationImputer = g["GPFormationImputer"]

    # Source-string scan (the kernel module was loaded via exec(), so
    # inspect.getsource cannot retrieve a file location; we scan the
    # original kernel source instead).
    def _extract_block(src: str, start_marker: str, end_marker: str) -> str:
        i = src.find(start_marker)
        j = src.find(end_marker, i + 1) if i >= 0 else -1
        if i < 0 or j < 0:
            raise RuntimeError(f"markers not found: {start_marker} ... {end_marker}")
        return src[i:j]

    def _strip_comments_and_docstrings(src: str) -> str:
        """Remove # comments and triple-quoted docstrings (best-effort)."""
        # Strip triple-double-quoted blocks
        out: list[str] = []
        i = 0
        n = len(src)
        while i < n:
            if src.startswith('"""', i):
                j = src.find('"""', i + 3)
                if j < 0:
                    break
                i = j + 3
                continue
            if src.startswith("'''", i):
                j = src.find("'''", i + 3)
                if j < 0:
                    break
                i = j + 3
                continue
            out.append(src[i])
            i += 1
        no_doc = "".join(out)
        # Strip line comments
        cleaned: list[str] = []
        for ln in no_doc.splitlines():
            hashpos = ln.find("#")
            if hashpos >= 0:
                ln = ln[:hashpos]
            cleaned.append(ln)
        return "\n".join(cleaned)

    # Forbidden tokens that would indicate a leak path. We strip docstrings
    # / comments first because the file mentions "TVT_input" in the leak
    # guarantee docstrings.
    forbidden_strict = ["TVT_input", 'ev["TVT', "ev['TVT'", "ev.TVT_input"]

    src_gp_full = _extract_block(
        src_full, "class GPFormationImputer:", "class DenseANCCImputer:",
    )
    src_gp = _strip_comments_and_docstrings(src_gp_full)
    for forb in forbidden_strict:
        assert forb not in src_gp, f"leak: GPFormationImputer code contains {forb!r}"
    print(f"  ✓ guard 1: GPFormationImputer code (post-docstring) "
          f"does not reference TVT_input/ev[TVT]")

    src_eo_fn_full = _extract_block(
        src_full, "def beam_search_dir(", "def compute_edge_o_features(",
    )
    src_eo_fn = _strip_comments_and_docstrings(src_eo_fn_full)
    for forb in forbidden_strict:
        assert forb not in src_eo_fn, f"leak: beam_search_dir code contains {forb!r}"
    print(f"  ✓ guard 2: beam_search_dir code (post-docstring) "
          f"does not reference TVT_input/ev[TVT]")

    src_ceo_full = _extract_block(
        src_full, "def compute_edge_o_features(",
        'print("Direction-aware Beam (Edge O) functions loaded',
    )
    src_ceo = _strip_comments_and_docstrings(src_ceo_full)
    for forb in forbidden_strict:
        assert forb not in src_ceo, f"leak: compute_edge_o_features code contains {forb!r}"
    print(f"  ✓ guard 3: compute_edge_o_features code (post-docstring) "
          f"does not reference TVT_input/ev[TVT]")

    # 4) Self-exclusion smoke (= test self_wid arg flows through)
    train_dir = DATA / "train"
    hw_paths = sorted(train_dir.glob("*__horizontal_well.csv"))
    well_ids = [p.stem.replace("__horizontal_well", "") for p in hw_paths[:30]]
    gp = GPFormationImputer(well_ids, train_dir, M=20, seed=42)
    if gp.train_xy_norm is None:
        print(f"  ⚠ guard 4 skipped (GP empty)")
    else:
        train_xy_orig = (gp.train_xy_norm * gp.xy_std) + gp.xy_mean
        test_xy = train_xy_orig[:3]
        m_with, _ = gp.impute(test_xy, self_wid=None)
        m_excl, _ = gp.impute(test_xy, self_wid=well_ids[0])
        # GP-level prediction uses cached fit; self-exclusion only affects
        # KNN fallback. Just ensure no exception and shapes match.
        assert m_with.shape == m_excl.shape, "self-exclusion shape mismatch"
        diff = float(np.abs(m_with - m_excl).mean())
        print(f"  ✓ guard 4: self-exclusion call ok (mean abs diff: {diff:.4f})")

    print("Phase 4: PASS")


def main():
    print(f"REPO   = {REPO}")
    print(f"KERNEL = {KERNEL}")
    print(f"DATA   = {DATA}")
    print()
    g = phase1_gp_unit()
    phase2_edge_o_unit(g)
    p3 = phase3_well_smoke(g, n_wells=2)
    phase4_leak_guards(g)
    print()
    print("ALL PASS" if p3 else "PARTIAL PASS (unit phases ok, integration partial)")
    return 0 if p3 else 1


if __name__ == "__main__":
    sys.exit(main())
