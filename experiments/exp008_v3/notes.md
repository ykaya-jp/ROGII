# exp008 v3 — 開発ノート (Phase 5 ROI 3 layer + fold-misalign 解消)

## 0. ブランチ + コミット履歴

- ベース: `feat/phase-4-edge-s-roundgrid` (Edge S round-to-grid 注入済 = v2)
- 新規ブランチ: `feat/phase-5-roi3-v3`
- コミット系列:
  1. `c4b79d1 docs(phase-5-roi3): exp008_v3 + exp009_v3 設計書` (= 30-60 分目標、完了)
  2. `01a4d83 feat(exp008-v3): inject 4 改修` (= kernel 修正完了)
  3. `72db496 feat(exp009-v3): inject 4 改修`
  4. (本コミット) `test(exp008-v3): local smoke 4/4 PASS + notes.md`

## 1. 4 改修の inject 位置 (= exp008_case_d_kalman.py 修正後 line 番号)

| 改修 | 内容 | 修正 line |
|---|---|---|
| 1 | LGB `objective="huber"` + `alpha=0.9` | `LGB_BASE` dict, 304-305 |
| 1 | CB `loss_function="Huber:delta=0.5"` | `CB_PARAMS` dict, 339 |
| 2 | hetero sample_weight 構築 (per-well-stats.parquet) | own train Step E.1, 3265-3320 |
| 2 | `lgb.LGBMRegressor.fit(... sample_weight=w_tr, eval_sample_weight=[w_va])` | own train, 3380-3398 |
| 2 | `Pool(... weight=w_tr)` (CB) | own train, 3439-3441, 3448-3450 |
| 3 | LGB schedule = LR_MED × SEEDS_MED (= 3 seed) | own train, 3343-3349 |
| 3 | LGB seed loop + MEDIAN aggregate | own train, 3375-3422 |
| 3 | CB seed loop + MEDIAN aggregate | own train, 3424-3469 |
| 4 | path b blend = kb simple-avg + own Ridge + 1D grid | Step F, 3475-3550 |
| 4 | legacy n-base Ridge fallback (= STACK_PATH_B_ENABLE=False) | Step F, 3552-3597 |

## 2. ローカル smoke 結果 (= 5 wells × 200 row 合成データ + 実 per-well-stats.parquet)

`experiments/exp008_v3/smoke_phase5_v3.py` を `uv run` 実行、4 件すべて PASS:

| smoke | 内容 | 結果 |
|---|---|---|
| 1 | Huber loss: LGB huber + CB Huber:delta=0.5 fit | LGB val RMSE 2.82 / CB val RMSE 2.77、NaN なし |
| 2 | hetero sample_weight from per-well-stats.parquet (5 real wells) | raw sigma range [0.00742, 0.01018] → w range [0.841, 1.154]、mean=1.000、clip 内 |
| 3 | Multi-seed MEDIAN (3 seed × LGB + 3 seed × CB) | LGB MEDIAN 2.81 (worst single 2.81)、CB MEDIAN 2.76、shape 全一致 |
| 4 | path b: kb-avg + own-Ridge + 1D grid | w_kb*=0.80、blend RMSE 0.205 < max(kb_only 0.228, own_only 0.501) |

詳細:
- 改修 1: Huber 動作確認 OK。`LGBMRegressor` の `objective="huber"` で fit + predict が NaN 出さず、CB `Huber:delta=0.5` も best_iteration OK
- 改修 2: 実 parquet (= 773 well) の 5 well から sigma 取得、`1/sigma` を正規化 + clip [0.5, 2.0] で範囲内に収束
- 改修 3: MEDIAN shape (200,) = val 数と一致、NaN なし、3 seed の平均より MEDIAN が安定
- 改修 4: own Ridge coef が positive、grid search で 0 ≤ w_kb ≤ 1、blend が両端よりも RMSE 小

## 3. Multi-seed MEDIAN 戦略 + 9 hr cap 検証

- 構成: **LGB lr=0.05 × 1 + CB lr=0.05 × 1 = 2 model × 3 seed (42, 123, 2024) × 5 fold = 30 train run**
- runtime 想定:
  - 1 train run = 5-10 min (= subagent K exp007 実測の half、Huber は RMSE と同等)
  - 30 run = 150-300 min = **2.5-5 hr**
  - test FE + train FE rebuild + kb predict + post-proc を加えても **6-8 hr** = **9 hr cap 内**
- fail-safe:
  - `OWN_TRAIN_HARD_TIMEOUT_S = 6 * 3600` (= 6 hr) で early break、partial seed MEDIAN を使用
  - GPU CB が OOM の場合 `cb_params["task_type"] = "CPU"` fallback (各 seed 試行で個別 try/except)
  - 各 seed 完了後 `del cb_model; gc.collect()` で memory pressure 緩和

## 4. 想定 LB (= 4 改修累積、mid case)

| stage | exp008 v2 base | + Huber | + hetero | + MEDIAN | + path b | 累積 mid |
|---|---|---|---|---|---|---|
| CV | 9.6 (推定) | -0.30 | -0.20 | -0.30 | -0.30 | **8.5** |
| LB | 未確定 | -0.30 | -0.20 | -0.30 | -0.30 | **8.7-9.0** |

= **9 切り射程内 (= ユーザー要求の 8 台達成は exp009 v3 で狙う)**

## 5. 残課題

- Kaggle 上で実 run は中央が判断 (= exp007 LB / exp008 v2 LB / exp009 v2 LB 結果待ち)
- kernel push は本 task 範囲外 (= 中央指示待ち)
- v4 候補:
  - path a (= karnakbaev pretrained を Edge Q fold で真の OOF 再生成) で fold-misalign を完全解消
  - Adversarial Validation (= GM §1.1) で test 分布 shift driver 特定
  - Discrete dTVT classification + cumsum (= 数理 §1.3 C) で Track 4 構造変革

## 6. kernel push 直前チェックリスト (= 中央向け)

- [ ] `OWN_USE_MEDIAN_SEEDS = True` (= 改修 3 enable) を確認
- [ ] `HETERO_W_ENABLE = True` (= 改修 2 enable) を確認
- [ ] `STACK_PATH_B_ENABLE = True` (= 改修 4 enable) を確認
- [ ] `LGB_BASE["objective"] == "huber"` と `CB_PARAMS["loss_function"] == "Huber:delta=0.5"` を kernel ログで確認
- [ ] per-well-stats.parquet を `/kaggle/input/rogii-per-well-stats/` データセットとして attach (= 不在なら HETERO_W_FILL_P50 fallback だが望ましくない)
- [ ] Kaggle 実 run で `hetero_w: min=0.50, max=2.00, mean=1.000` のログを確認
- [ ] Kaggle 実 run で `Multi-seed MEDIAN: 2 model × 3 seed × 5 fold = 30 train runs` を確認
- [ ] Kaggle 実 run で `path b grid search: w_kb*=X.XX blend OOF RMSE=X.XX` を確認
- [ ] runtime が 9 hr cap 内 (= ≤ 8.5 hr) で完走することを GPU log で確認
