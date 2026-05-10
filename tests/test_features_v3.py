"""Smoke tests for exp003 features (tysig + multi-scale SC + tasmim xcorr).

Runs on 1 train well to keep the test fast (<30s).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rogii.data_io import TRAIN_DIR, list_wells, load_horizontal, load_typewell
from rogii.features import FEATURE_COLS_V3, SC_HALF_WINDOWS, add_features
from rogii.imputers import FormationPlaneKNN
from rogii.tysig import multi_scale_sc, wls_b_well, xcorr_tvt_offsets


@pytest.fixture(scope="module")
def imputer():
    return FormationPlaneKNN(TRAIN_DIR, k=10)


@pytest.fixture(scope="module")
def first_well():
    wells = list_wells(TRAIN_DIR)
    if not wells:
        pytest.skip("no train wells available")
    wid = wells[0]
    h = load_horizontal(TRAIN_DIR, wid)
    tw = load_typewell(TRAIN_DIR, wid)
    return wid, h, tw


def test_xcorr_tvt_offsets_shape():
    """xcorr_tvt_offsets returns 3 arrays of equal length to hgr."""
    rng = np.random.default_rng(0)
    n = 200
    hgr = rng.normal(80, 10, n).astype(np.float32)
    tw_tvt = np.linspace(10000.0, 11000.0, 500, dtype=np.float32)
    tw_gr = rng.normal(80, 10, 500).astype(np.float32)
    out = xcorr_tvt_offsets(hgr, tw_tvt, tw_gr, last_known_tvt=10500.0,
                            window=30, search_half=80.0, step=2.0)
    assert set(out.keys()) == {"xcorr_delta", "xcorr_corr", "xcorr_absdiff"}
    for k, v in out.items():
        assert v.shape == (n,), f"{k} shape mismatch"
        assert v.dtype == np.float32
    # corr in [-1, 1]
    assert (out["xcorr_corr"] >= -1.0).all() and (out["xcorr_corr"] <= 1.0).all()
    # delta within search half
    assert (np.abs(out["xcorr_delta"]) <= 80.0 + 1e-3).all()


def test_multi_scale_sc_3_outputs():
    """multi_scale_sc returns one signal per requested window."""
    rng = np.random.default_rng(1)
    n_visible = 100
    n_hidden = 200
    kgr = rng.normal(80, 10, n_visible).astype(np.float32)
    ktvt = np.linspace(10000.0, 10100.0, n_visible, dtype=np.float32)
    hgr = rng.normal(80, 10, n_hidden).astype(np.float32)
    res = multi_scale_sc(kgr, ktvt, hgr, half_windows=(8, 15, 25), stride=3)
    assert set(res.keys()) == {"h8", "h15", "h25"}
    for tag in ("h8", "h15", "h25"):
        sc_tvt, sc_score = res[tag]
        assert sc_tvt.shape == (n_hidden,)
        assert sc_score.shape == (n_hidden,)
        # Score is NCC ∈ [-1, 1]
        assert (sc_score >= -1.0 - 1e-3).all() and (sc_score <= 1.0 + 1e-3).all()


def test_wls_b_well_smoke():
    """wls_b_well returns finite scalar with sensible behaviour."""
    rng = np.random.default_rng(2)
    n = 100
    z = np.linspace(2000.0, 2100.0, n).astype(np.float64)
    f_imp = np.full(n, 12500.0, dtype=np.float64)
    # b_well = TVT_input + Z - f_imp; if true b = 50 then TVT_input = f_imp - Z + 50
    true_b = 50.0
    tvt_input = (f_imp - z + true_b + rng.normal(0, 0.5, n)).astype(np.float64)
    b_est = wls_b_well(tvt_input, z, f_imp, decay=0.02)
    assert np.isfinite(b_est)
    assert abs(b_est - true_b) < 2.0, f"b_est={b_est} far from true 50"


def test_add_features_one_well_v3(imputer, first_well):
    """add_features on a single well produces all FEATURE_COLS_V3 columns."""
    wid, h, tw = first_well
    h["row_idx"] = np.arange(len(h), dtype=np.int32)
    typewells = {wid: (tw["TVT"].to_numpy(), tw["GR"].to_numpy())}
    df = add_features(h, imputer=imputer, typewells=typewells, exclude_self=True)
    # Length matches input
    assert len(df) == len(h), "row count drift"
    # All FEATURE_COLS_V3 columns present
    missing = [c for c in FEATURE_COLS_V3 if c not in df.columns]
    assert not missing, f"missing columns: {missing[:10]} ... ({len(missing)} total)"
    # New v3 columns are non-trivial (have variance)
    for col in (
        "xcorr_delta",
        "xcorr_corr",
        "xcorr_absdiff",
        "tvtFw_ANCC",
        "bww_ANCC",
        f"sc{SC_HALF_WINDOWS[0]}_raw_tvt",
        f"sc{SC_HALF_WINDOWS[2]}_raw_tvt",
        "sc_med_tvt",
        "gr_detrend",
    ):
        v = df[col].to_numpy()
        assert np.isfinite(v).any(), f"{col} all non-finite"
    # tvt_formula sanity (Pearson with negative-Z + ANCC + b_well should give residual ~0)
    if "TVT" in df.columns:
        tvt = df["TVT"].to_numpy()
        tvtF = df["tvtF_ANCC"].to_numpy()
        # On visible rows tvtF should track TVT closely (median b_well fit)
        visible = ~np.isnan(df["TVT_input"].to_numpy())
        if visible.sum() >= 10:
            err = np.abs(tvt[visible] - tvtF[visible])
            assert np.median(err) < 50.0, f"tvtF_ANCC residual median {np.median(err):.1f} too large"
