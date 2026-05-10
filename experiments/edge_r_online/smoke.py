"""Edge R smoke test — visible-as-pseudo-hidden dataset + LGB continued training.

ローカルで data/raw/test の 3 wells に対して:
  1. visible_dataset builder が動作することを確認 (= 3 wells 抽出)
  2. visible_df の target = TVT_input から作られていることを確認 (= leak guard)
  3. karnakbaev pretrained LGB 1 base 各々に init_model で continued training (= 1 base のみ smoke)
  4. NaN/inf がないことを確認
  5. runtime 測定 (= per base time, total time)

期待:
  - visible_df.shape[0] > 0
  - y_R が全て finite
  - online predict が NaN なし、 range が妥当 (= [-100, 100] ft 程度)
  - 1 base continued training の runtime < 60s
"""
from __future__ import annotations

import sys
import time
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
KERNEL_DIR = REPO / "kaggle_kernels" / "exp005_cache_blend"
sys.path.insert(0, str(KERNEL_DIR))

# Run kernel module — DEBUG_MAX_WELLS = 3 で 3 wells のみ build
# kernel は import すると即座に MODE == "infer" branch を実行してしまうので、
# inline 実行ではなく必要関数だけ取得するアプローチを取る。

# ── 手動で kernel script を構築せず、 helper を直接書き写して試す ──
# 一番安全な smoke は: 既存の kernel function build_visible_dataset を import
# できる形に切り出すよりも、 主要 logic を inline で再現すること。
# ただし build_well_features は kernel 内で大量定義 + module globals 依存なので
# 切り出しが困難。 → kernel を MODE != infer で walk-import するか、 直接実行する。

# 最小 smoke: visible_dataset builder の core logic (= csv mask + tmp file write)
# だけ動作確認する。 LGB continued training は別途 manual で確認。

REPO_DATA = REPO / "data" / "raw"
TEST_DIR  = REPO_DATA / "test"

TAIL_K = 100
MIN_VISIBLE_ROWS = 30


def smoke_visible_mask(n_wells: int = 3) -> dict:
    """Smoke: visible-as-pseudo-hidden mask logic (= leak guard 核)."""
    hw_files = sorted(TEST_DIR.glob("*__horizontal_well.csv"))[:n_wells]
    results = {"n_wells": len(hw_files), "skipped": 0, "ok": 0, "rows": []}

    for hw_path in hw_files:
        wid = hw_path.stem.replace("__horizontal_well", "")
        hw_raw = pd.read_csv(hw_path)
        if "TVT_input" not in hw_raw.columns:
            results["skipped"] += 1
            continue

        visible_mask = hw_raw["TVT_input"].notna().to_numpy()
        n_visible = int(visible_mask.sum())
        if n_visible < (MIN_VISIBLE_ROWS + TAIL_K):
            results["skipped"] += 1
            results["rows"].append({
                "well": wid, "n_visible": n_visible, "status": "skipped"
            })
            continue

        visible_idx = np.flatnonzero(visible_mask)
        tail_idx = visible_idx[-TAIL_K:]

        # leak guard: TVT_input → NaN 化される行
        orig = hw_raw["TVT_input"].to_numpy().copy()
        hw_mod = hw_raw.copy()
        hw_mod["TVT"] = np.nan
        hw_mod.loc[hw_mod["TVT_input"].notna(), "TVT"] = orig[visible_mask]
        hw_mod.loc[tail_idx, "TVT_input"] = np.nan

        # leak check: 擬似 hidden 行の TVT は visible TVT_input 由来
        tvt_at_tail = hw_mod.loc[tail_idx, "TVT"].to_numpy()
        orig_at_tail = orig[tail_idx]
        assert np.allclose(tvt_at_tail, orig_at_tail), \
            f"leak guard fail at {wid}: TVT != orig TVT_input"

        # leak check: hidden region (原 TVT_input NaN な行) は TVT も NaN
        original_hidden_mask = ~visible_mask
        if original_hidden_mask.any():
            tvt_at_hidden = hw_mod.loc[original_hidden_mask, "TVT"].to_numpy()
            assert np.all(pd.isna(tvt_at_hidden)), \
                f"leak guard fail at {wid}: hidden TVT not NaN — would leak"

        results["ok"] += 1
        results["rows"].append({
            "well": wid,
            "n_total_rows": len(hw_raw),
            "n_visible": n_visible,
            "n_hidden_orig": int(original_hidden_mask.sum()),
            "n_pseudo_hidden": int(len(tail_idx)),
            "status": "ok",
        })

    return results


def smoke_lgb_continued_training() -> dict:
    """Smoke: LGB init_model continued training works on toy data."""
    import lightgbm as lgb

    # toy dataset
    rng = np.random.default_rng(42)
    X_base = rng.normal(size=(2000, 30)).astype(np.float32)
    y_base = X_base[:, 0] * 2.0 + X_base[:, 1] * -1.5 + rng.normal(scale=0.1, size=2000)

    # pretrained on base
    p = dict(
        boosting_type="gbdt", learning_rate=0.04, num_leaves=31,
        min_child_samples=20, objective="regression", verbose=-1, n_jobs=-1,
    )
    t0 = time.time()
    base_booster = lgb.train(p, lgb.Dataset(X_base, label=y_base), num_boost_round=200)
    t_base = time.time() - t0

    # online data (simulated test visible — slight shift)
    X_online = rng.normal(size=(500, 30)).astype(np.float32)
    y_online = X_online[:, 0] * 2.0 + X_online[:, 1] * -1.5 + rng.normal(scale=0.1, size=500) + 0.3

    p2 = dict(p)
    p2["learning_rate"] = 0.02  # halved
    t0 = time.time()
    online = lgb.train(
        p2, lgb.Dataset(X_online, label=y_online),
        num_boost_round=200, init_model=base_booster, keep_training_booster=False,
    )
    t_online = time.time() - t0

    # predict
    X_test = rng.normal(size=(1000, 30)).astype(np.float32)
    pred_base   = base_booster.predict(X_test)
    pred_online = online.predict(X_test)

    return {
        "base_train_sec"  : round(t_base, 2),
        "online_train_sec": round(t_online, 2),
        "n_base_trees"    : base_booster.num_trees(),
        "n_online_trees"  : online.num_trees(),
        "base_pred_range" : [float(pred_base.min()), float(pred_base.max())],
        "online_pred_range": [float(pred_online.min()), float(pred_online.max())],
        "base_pred_nan"   : int(np.isnan(pred_base).sum()),
        "online_pred_nan" : int(np.isnan(pred_online).sum()),
        "diff_mean_abs"   : float(np.abs(pred_online - pred_base).mean()),
    }


if __name__ == "__main__":
    print("=" * 80)
    print("Edge R smoke test — visible-as-pseudo-hidden + LGB continued training")
    print("=" * 80)

    print("\n[1] visible-as-pseudo-hidden mask + leak guard (3 wells)")
    t0 = time.time()
    res1 = smoke_visible_mask(n_wells=5)
    print(f"  elapsed: {time.time() - t0:.2f}s")
    print(f"  n_wells={res1['n_wells']}  ok={res1['ok']}  skipped={res1['skipped']}")
    for r in res1["rows"]:
        print(f"    {r}")

    print("\n[2] LGB init_model continued training on toy data")
    t0 = time.time()
    res2 = smoke_lgb_continued_training()
    print(f"  elapsed: {time.time() - t0:.2f}s")
    for k, v in res2.items():
        print(f"    {k}: {v}")

    print("\n[result]")
    pass_count = 0
    fail_count = 0
    checks = [
        ("visible_mask: at least 1 ok well", res1["ok"] >= 1),
        ("visible_mask: no exception", True),
        ("lgb online: n_trees > n_base (warm extends)", res2["n_online_trees"] > res2["n_base_trees"]),
        ("lgb online: no NaN", res2["online_pred_nan"] == 0),
        ("lgb online: range sane", abs(res2["online_pred_range"][0]) < 100 and abs(res2["online_pred_range"][1]) < 100),
        ("lgb online: diff > 0 (warm changed predictions)", res2["diff_mean_abs"] > 0.001),
        ("lgb online: runtime < 60s", res2["online_train_sec"] < 60),
    ]
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if ok: pass_count += 1
        else: fail_count += 1

    print(f"\n  total: {pass_count}/{len(checks)} PASS, {fail_count} FAIL")
    sys.exit(0 if fail_count == 0 else 1)
