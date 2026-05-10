# exp006 — TabICL + PF-lite blend (LB 9 帯狙い)

> **目的**: LB 9.x 帯到達。exp005 (LB 10.x 想定) を base に、TabICL を 6th base として追加 + Ridge meta を 6-base 構成で再 fit する。PF-lite features は v1 では dormant 配置 (= 関数だけ用意、本番 path には未接続)。
> **ベース branch**: `feat/phase-3-exp006-tabicl-pflite` (from `feat/phase-3-exp005-cache-blend`)
> **構造**: exp005 kernel 拡張 + `MODE='infer_tabicl'` 新設 + GPU 必須 (= `enable_gpu=true`)
> **設計 doc**: `experiments/exp006/design.md`

## 0. 状況 (2026-05-11 01:40 JST 時点)

- 我々の現状 LB: exp002 = **14.695** / exp003 = PENDING (期待 11-12) / exp005 = PENDING (期待 10.x)
- 公開 LB Gold 圏: 9.256 - 9.919 (Top 20)
- 必要改善 vs exp005: -0.5〜-1.0 ft で 9.x 帯到達
- submit 残数: 5/10 分は 1 回使用済 (= exp005)、5/11 分は **5 回 fresh** (我々はまだ何も submit していない)

## 1. 採った Approach (= Approach A: フル統合)

### 候補比較 (= 設計 doc §1)

| Approach | 概要 | 期待 LB | リスク |
|---|---|---|---|
| **A: フル統合 (採用)** | exp005 kernel base + TabICL 5-fold OOF on train_df + 6-base Ridge meta 再 fit | **9.x 帯到達狙い** | TabICL = GPU 必須、8 hr cap 余裕あり (1-1.5 hr 想定) |
| B: TabICL 軽量 (= no fold) | TabICL を 1-fold 全 train fit、固定 blend weight | -0.3 ft 程度 | OOF 経由じゃないので 9.x への寄与小 |
| C: PF-lite のみ | TabICL 捨てて PF-lite features のみ | -0.2〜-0.4 ft | 9 帯到達は薄い |

### 選定理由 (A)

1. **TabICL OOF + Ridge re-fit が必須**: needless090 LB 10.081 は 8 base 構成内で TabICL が貢献。固定 weight blend では効果激減
2. **train_df.parquet (1.47 GB)** を karnakbaev dataset 経由で load → feature engineering 不要
3. **TabICL = GPU 必須**だが Kaggle T4 で 1 fold ~5-10 min × 5 fold + test infer ~5-10 min = 約 1 hr で完了見込み
4. **PF-lite を v1 で連携しない理由**: karnakbaev train_df.parquet には PF-lite columns がない → train side を再生成する work が必要 (= per-well ループで PF-lite だけ計算するヘルパーが必要)。複雑度抑制のため v1 では dormant。kernel script に関数群は実装済 (= v2 で接続するだけで活性化)

## 2. 必要 host dataset

| Slug | サイズ | 中身 | 使い方 |
|---|---|---|---|
| `karnakbaevarthur/rogii-code-helper-dataset` | 1.36 GB | LGB×3 / XGB / CB artifacts + train_df.parquet (1.47 GB) + oof_predictions.parquet (149 MB) + features.json + sample_submission | `train_df` から TabICL fit、`oof_predictions` から既存 5 base OOF を取得 |
| `thermostatic/rogii-tabicl-v2-public-assets` | **107 MB** (新規追加) | `tabicl-2.1.1-py3-none-any.whl` (= TabICL package) + `tabicl-regressor-v2-20260212.ckpt` (= TabICL pretrained regressor 114 MB) | `pip install --no-index --no-deps <wheel>` → `from tabicl import TabICLRegressor` → `device="cuda"` で fit/predict |

License: `karnakbaev = apache-2.0`、`thermostatic TabICL = apache-2.0` (出典: https://github.com/soda-inria/tabicl LICENSE、`docs/research/host-datasets.dense.md` §A2)。

## 3. 計画 (TODO)

- [x] branch `feat/phase-3-exp006-tabicl-pflite` 作成
- [x] 設計 doc (`experiments/exp006/design.md`) 作成 → 1st commit + push (`8b505ef`)
- [x] kernel script (`exp006_tabicl_pflite.py`, 2426 行) 作成 → 2nd commit + push (`c4ba101`)
  - exp005 base + `MODE='infer_tabicl'` 新設
  - PF-lite helpers 追加 (dormant)
  - TabICL 統合 (load train_df + OOF → LGB top-50 → 5-fold TabICL + Ridge re-fit)
- [x] kernel-metadata.json 作成 (`enable_gpu=true`、`dataset_sources` に TabICL 追加)
- [x] ローカル smoke (= 3 test wells、TABICL_ENABLE=False で fallback path 動作確認)
- [x] PF-lite helpers の standalone test (= 100 行 synthetic input で 10 features 出力、NaN handling OK)
- [x] notes.md 完成 (= 本ファイル) → 3rd commit + push
- [ ] **kernel push は実行しない** (= 中央指示待ち、exp005 LB 確定後に判断)

## 4. 実装の中身

### 4.1 kernel script の 4 大変更

| # | 場所 | 内容 |
|---|---|---|
| 1 | docstring (line 1-40) | exp006 用 4-step pipeline + TabICL/PF-lite license attribution |
| 2 | config (line 76-110) | `MODE = 'infer_tabicl'` + TabICL 9 種 knob (CTX/CHUNK/SEEDS/USE_AMP/...) + `STACK_REFIT_RIDGE` |
| 3 | helpers (line 1006-1112) | PF-lite 3 関数 (`pf_lite_feature_names`, `typewell_gr_at_tvt`, `weighted_candidate_tvt_features`) を `affine_cal` 直後に dormant 追加 |
| 4 | dispatcher (line 2098-2299) | `elif MODE == 'infer_tabicl':` block 約 200 行 — load karnakbaev train_df + OOF, install TabICL wheel, LGB top-50 quick-fit, 5-fold TabICL OOF + test pred, 6-base Ridge re-fit, fallback to NM |

### 4.2 失敗時の挙動 (= fallback path)

`infer_tabicl` block は **try/except で全体を包んだ TabICL phase** + **無条件 fallback to karnakbaev 5-base NM blend**。失敗パターン:
- TabICL dataset 未 attach → `FileNotFoundError` → fallback (= exp005 と同じ blend、LB は exp005 と同等)
- `train_df.parquet` 不在 / 列 mismatch → `RuntimeError` → fallback
- TabICL fit OOM → exception → fallback
- `TABICL_HARD_TIMEOUT_S` (6 hr) 超過 → 残 fold break、test_pred は既出 fold 平均で続行

### 4.3 GPU / runtime 見積

| 段階 | 想定時間 |
|---|---|
| test feature build (= 自前計算、~6 wells × 14k rows) | ~30 sec |
| 5 base load + predict | ~1-3 min |
| train_df.parquet load (1.47 GB) | ~30 sec |
| oof_predictions.parquet load + merge | ~5 sec |
| TabICL wheel install + ckpt load | ~30 sec |
| LGB quick-fit (200k rows × 300 trees, GPU) | ~2-3 min |
| TabICL 5-fold OOF (4096 ctx, n_est=4, 1 seed) | ~30-50 min |
| TabICL test prediction (per fold avg) | (上記に含む) |
| 6-base Ridge re-fit + meta predict | <1 sec |
| postproc + SG smooth + submission | ~5 sec |
| **合計 (中央値)** | **~45-75 min** |
| 8 hr cap 余裕 | **6.7+ hr** |

## 5. ローカル smoke 結果 (2026-05-11 01:40 JST)

### 5.1 設定 (`/tmp/rogii_smoke6/smoke.py`)

- kernel script を `/tmp/rogii_smoke6/smoke.py` にコピー
- 変更: `DEBUG_MAX_WELLS=5`、`TABICL_ENABLE=False` (= GPU TabICL skip、fallback path テスト)
- `DATA_DIR=/home/yusuke_kaya/.../data/raw`、`ARTEFACT_DIR=/home/.../data/external/host/karnakbaev`
- `OUTPUT_DIR=/tmp/rogii_smoke6_out`

### 5.2 結果

| 段階 | 観測値 |
|---|---|
| test feature build | 20.87 sec for 3 test wells (000d7d20, 00bbac68, 00e12e8b) → 14,151 rows × 160 cols |
| 5 base predict | 全 5 model OK、delta range: lgb0=[-15.29, 14.11] / lgb1=[-15.62, 15.43] / lgb2=[-16.10, 14.81] / xgb=[-13.92, 15.28] / cb=[-15.17, 14.62] |
| TabICL phase | **skip** (TABICL_ENABLE=False) — fallback path 起動 |
| NM 5-base blend (fallback) | test_delta range=[-15.0, 14.16]、これは exp005 と同等 |
| postproc + SG smooth | range=[-14.96, 14.10]、clip ±60 ft (train_df 未 load 時の保守 clip) |
| submission.csv 出力 | shape (14151, 2)、NaN 0、id 全 sample submission と一致、tvt range 11592-12238 ft (last_known_tvt 周辺で物理的に妥当)、duplicates 0 |
| 全パイプライン | < 25 秒 |

### 5.3 PF-lite helpers standalone test

別途 100 行 synthetic input (4 candidate TVT 列、20 行 NaN injected) で `weighted_candidate_tvt_features` を直接呼出:
- 出力 10 features 全て shape=(100,) で生成
- `pf_lite_tvt` range=[11806, 11994] ft、`pf_lite_delta` range=[6, 194] ft (= last_known=11800 から)
- `pf_lite_std` range=[1.2, 12.6] ft、5 NaN 行 (= beam_cons NaN injection 行)
- `pf_lite_vs_beam_cons` で同じ 5 NaN を継承 (= 期待通り)
- 他 feature 全て NaN 0、無限大なし

## 6. 想定 LB と 9 帯到達への path

### 6.1 LB シナリオ

- **Best case**: TabICL OOF が需要通り -0.3〜-0.5 ft 改善 + 6-base Ridge re-fit で +0.1〜-0.2 ft → exp005 が 10.5 だったとして **10.0〜9.7 ft 着地**、Gold 圏 (= 9.919 が 20 位) ギリギリ到達
- **Mid case**: TabICL effective だが Ridge wts が tabicl < 0.05 で blend 効果薄い → exp005 と同水準 (10.x)
- **Worst case (fallback)**: TabICL crashes → fallback で exp005 と完全に同じ submission (= LB は exp005 と同点)

### 6.2 9 帯到達のキーリスク

1. **karnakbaev OOF と TabICL OOF の fold 不整合** = leak。これは `train_df` の row order を保ちつつ groupkfold(5).split(X, y, well) で **同じ random_state なら同じ fold 配分が取れる**前提だが、**karnakbaev kernel が GroupKFold の random_state をどこで指定したか不明**。最悪 leak が出ると Ridge wts が **fold-leak で TabICL に過大重み付け** → val 上では低 RMSE、LB では悪化。**対策**: kernel run 後の `Ridge weights` ログを必ず見て、tabicl の weight が 0.4+ なら異常と判断、submit 前に 5-base only の fallback も試す
2. **TabICL feature top-50 が train side だけでバイアス**: train で重要だが test で意味薄い feature を選ぶと TabICL が test で精度落とす可能性。**対策**: feature 選定で `karnakbaev features.json` を base にしているので train/test schema 整合は OK
3. **TabICL の n_estimators=4 で OOF が単一 estimator 並み**: needless090 は 4 だが seeds 5 で stabilize。我々 1 seed なので variance 高い。**対策**: v2 で `TABICL_SEEDS=[0,1,2,3,4]` に拡張 (runtime 5 倍 ≈ 4-5 hr、cap 内)、v1 では single seed の risk 受容

## 7. 残課題 / blocker

- **kernel push は実行しない**: 中央が exp005 LB を確認してから判断。exp005 が 10.x 着地 → exp006 push、exp005 が既に 9.x → exp006 不要
- **karnakbaev OOF の fold 整合性検証は kernel run 中ログでしか確認できない**: ローカルでは `oof_predictions.parquet` (149 MB) を DL していない (= disk 節約)。Kaggle 上で初回 run 時に `OOF NaN counts` ログを必ず見る
- **TabICL wheel install が deps 解決失敗の可能性**: needless090 では `--no-deps` で動いているので Kaggle 標準環境 (torch>=2) が `tabicl-2.1.1` の peer requirements を満たしている前提。失敗時は kernel run log の `pip install` セクションで判断
- **v2 への伸びしろ**: PF-lite を train side で計算 + TabICL の seeds 5 拡張 + tabicl_B (8192 ctx, 1 seed) variant 追加。effort 1-2 日、effect -0.2〜-0.5 ft 想定

## 8. References

- 設計 doc: `experiments/exp006/design.md`
- exp005 notes: `experiments/exp005/notes.md`
- needless090 kernel (TabICL ref, LB 10.081): https://www.kaggle.com/code/needless090/score-10-081-score-lb-32-rank
- TabICL package (Apache 2.0): https://github.com/soda-inria/tabicl
- TabICL dataset: https://www.kaggle.com/datasets/thermostatic/rogii-tabicl-v2-public-assets
- karnakbaev artifacts (LB 10.784): https://www.kaggle.com/datasets/karnakbaevarthur/rogii-code-helper-dataset
- pilkwang PF-lite ref: https://www.kaggle.com/code/pilkwang/rogii-eda-v4-same-matrix-super-stack
- top3 distill (TabICL section): `docs/research/top3-distill.dense.md` §6 + §3
- pilkwang distill (PF-lite section): `docs/research/pilkwang-distill.dense.md` §2.2.2 + §7.1

## 9. Smoke ファイル

- 入力: `/tmp/rogii_smoke6/smoke.py` (= kernel script の DEBUG_MAX_WELLS=5 + paths を local 化 + TABICL_ENABLE=False)
- 出力: `/tmp/rogii_smoke6_out/submission.csv` (14151 rows)
- log: smoke の stdout は本 notes §5.2 に要約済 (full log は再実行で取得可)
