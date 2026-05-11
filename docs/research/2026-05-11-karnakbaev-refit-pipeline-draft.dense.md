# karnakbaev pretrained re-fit pipeline (= Phase 2.1 extension、 別タスク draft)

> 起源: docs/dev/2026-05-11-cv-lb-correlation.md § 2.1.f finding
> 親タスク: kaggle-rogii-cv-strategies-2026-05-11 (= self-base 4 kernel 限定)
> 拡張タスク (= 本 doc が draft): kaggle-rogii-karnakbaev-refit-2026-05-12
> 目的: exp005 / exp005 v2 / exp005 v3 / exp006 の OOF を 4 CV 戦略で再生成、 correlation table の sample size を 4 → 8 件に拡大、 AC-7 multi-metric pass の statistical power を強化

---

## 1. 問題: なぜ karnakbaev only kernel は本タスクで OOF 取得できないか

exp005 系 (= karnakbaev pretrained LGB3+XGB+CB blend) は、 karnakbaev が **公開** している `oof_predictions.parquet` (= karnakbaev 自身の元 fold で生成) を kernel 内で **read** して blend に使用。 = 我々は karnakbaev base を **再 fit** しないため、 fold を override しても karnakbaev OOF は **不変**。

つまり:
- exp005 系 kernel に FOLD_OVERRIDE_PARQUET を inject しても、 karnakbaev published OOF は同じ
- 自前 base 部分 (= exp005 v2 の Edge S round-to-grid は post-proc、 fold 非依存) はそもそも fold 設計に依らない
- = **fold strategy 変えても exp005 系 の OOF は変わらない、 LB との correlation 測定不能**

これが「self-base 4 kernel 限定」 (= AC-5/6/7 で n=4 制約) の根本原因。

---

## 2. 解決 path = karnakbaev 5 base を自分で各 fold で re-fit

### 2.1 必要なもの

karnakbaev public dataset (`karnakbaevarthur/rogii-code-helper-dataset` 1.36 GB Apache-2.0) には:
- `train_df.parquet` (= 3.78M rows × 160 cols、 全 features 計算済)
- 5 base hyperparameters (= lgb_X3 + xgb + cb)
- preprocessing pipeline (= y_kb = target, formation imputer 等)

これらを用いて、 **各 fold の train side で 5 base を fit → val side で predict** = 真の fold-aware OOF。

### 2.2 推定 compute

- 1 base × 1 fold ≈ 5-15 min (= LGB on 3M rows、 GPU 推奨)
- 5 base × 5 fold = 25 fold-train = 125-375 min ≈ 2-6 hr/CV strategy
- 4 CV × 4 SCORED (= exp005 + exp005 v2 + exp005 v3 + exp006、 すべて karnakbaev base 流用) = 16 OOF set
- 合計 wall clock: 16 × (2-6 hr) = 32-96 hr local CPU
- Colab/Kaggle Notebook 並列 (= 4 server) で wall 8-24 hr

これは exp008 v2 base re-fit と同程度の規模。 本 task 完了後の妥当な拡張。

### 2.3 実装 sketch

```python
# scripts/karnakbaev_refit.py (= 新規)
"""karnakbaev 5 base を各 fold で再 fit、 fold-aware OOF parquet 出力。"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# karnakbaev train_df + hyperparams を /kaggle/input/rogii-code-helper-dataset/
# から load (= local 実行時は kaggle datasets download で手動配置)

KARN_BASES = {
    "lgb1": dict(model="lgb", params={...}, n_estimators=8000),
    "lgb2": dict(model="lgb", params={...}, n_estimators=4000),
    "lgb3": dict(model="lgb", params={...}, n_estimators=2000),
    "xgb":  dict(model="xgb", params={...}),
    "cb":   dict(model="cb",  params={...}),
}


def refit_one_fold(train_df, y_kb, base_key, fold_id, val_fold):
    """Return (OOF predictions for val_fold rows)."""
    base_cfg = KARN_BASES[base_key]
    tr_mask = (fold_id != val_fold) & (fold_id != -1)
    va_mask = (fold_id == val_fold)
    X_tr, y_tr = train_df.loc[tr_mask, features], y_kb[tr_mask]
    X_va = train_df.loc[va_mask, features]
    model = fit_model(base_cfg, X_tr, y_tr)
    return va_mask, model.predict(X_va)


def refit_all_folds(train_df, y_kb, fold_id, n_splits=5):
    oof = {k: np.full(len(train_df), np.nan) for k in KARN_BASES}
    for base_key in KARN_BASES:
        for k in range(n_splits):
            va_mask, pred = refit_one_fold(train_df, y_kb, base_key, fold_id, k)
            oof[base_key][va_mask] = pred
    return oof  # {base_key: per-row OOF predictions}


def main():
    # for each CV strategy in [baseline, C1, C2, C3, C4]:
    #   for each SCORED exp in [exp005, exp005_v2, exp005_v3, exp006]:
    #     fold_parquet = outputs/folds/<cv>.parquet
    #     karn_oof = refit_all_folds(train_df, y_kb, fold_id)
    #     # exp-specific post-proc:
    #     if exp == 'exp005':       post_proc_oof = simple_blend(karn_oof)
    #     elif exp == 'exp005_v2':  post_proc_oof = simple_blend + edge_s_round_to_grid(...)
    #     elif exp == 'exp005_v3':  post_proc_oof = simple_blend + edge_s + edge_r_online(...)
    #     elif exp == 'exp006':     post_proc_oof = ridge_6_base_with_tabicl(karn_oof)
    #     save oof to outputs/oof/cv-lb-correlation/<exp>__cv-<cv>/oof_predictions_self.parquet
```

### 2.4 検証

- karnakbaev 元 OOF (= public) と本 pipeline の **karnakbaev 元 fold での re-fit OOF** が一致 (= sanity check)
- 4 fold strategy 横断で OOF が変化 (= fold override が機能している evidence)

---

## 3. 本タスクとの分業

| layer | 本タスク (= cv-strategies-2026-05-11) | 拡張タスク (= karnakbaev-refit-2026-05-12) |
|---|---|---|
| 4 CV 戦略の **fold 配分** | ✓ outputs/folds/ で確定 | (流用) |
| **self-base 4 kernel** の OOF | ✓ 4 kernel patch + scripts/regenerate_oof.py | (本タスク成果を使用) |
| **karnakbaev only kernel** の OOF | (限定 = AC-5 で 4 件) | ✓ refit pipeline で 4 件追加、 AC-5 を 8 件に拡大 |
| correlation 計算 | ✓ tools/measure_cv_lb_correlation.py | (流用、 oof_table 拡大で再計算) |
| 最良 CV 確定 | n=4-5 で暫定 | n=8-9 で robust 確定 (= AC-7 確実達成) |

---

## 4. 想定 acceptance criteria (= 別 plan の核)

```yaml
task_id: kaggle-rogii-karnakbaev-refit-2026-05-12
goal: |
  karnakbaev 5 base を 各 fold (= 4 CV strategy) で再 fit、 exp005 / exp005 v2 /
  exp005 v3 / exp006 の OOF を 4 CV × 4 SCORED = 16 件追加して correlation
  table を 4 → 8 件サンプルに拡張、 最良 CV の robust 確定 (= AC-7 multi-metric
  全 pass at n=8) を達成する。

deliverables:
  - path: scripts/karnakbaev_refit.py  (= 5 base × 5 fold × 4 CV × 4 SCORED の OOF pipeline)
  - path: kaggle_kernels/karnakbaev_refit/  (= GPU kernel として push 用)
  - 出力: outputs/oof/cv-lb-correlation/exp005__cv-<cv>/oof_predictions_self.parquet (= 4 × 5 = 20 件追加)
  - update: docs/dev/2026-05-11-cv-lb-correlation.md § 3 「最良 CV 確定」 を n=8 で update

baseline:
  - existing AC-7 = n=4 で multi-metric pass
  - target AC-7 = n=8 で multi-metric pass (= more robust)
```

---

## 5. 関連

- `docs/dev/2026-05-11-cv-lb-correlation.md` § 2.1.f
- `kaggle_kernels/exp005_cache_blend/` (= karnakbaev OOF を read している kernel)
- `karnakbaevarthur/rogii-code-helper-dataset` (= 1.36 GB Apache-2.0、 train_df + base configs)
- 本 doc は draft、 別タスク plan 起票時に `.criteria/kaggle-rogii-karnakbaev-refit-2026-05-12.yaml` で確定
