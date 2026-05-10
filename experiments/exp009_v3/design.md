# exp009 v3 設計書 — ROI 3 layer + fold-misalign 解消 (Phase 5)

## 0. 目的

`feat/phase-4-edge-s-roundgrid` の `kaggle_kernels/exp009_case_e_edge_o/exp009_case_e_edge_o.py` (= exp009 v2 base、Edge S 注入済、Case E Sparse GP + Edge O dir-aware Beam + Edge D Kalman + Edge Q + Edge M 全部入り) に対し、
**exp008 v3 と同じ 4 改修**を inject し、CV 9.1 → 8.0 帯、LB 8.2-8.5 帯まで押し込む。

`exp008_v3/design.md` の論拠・数理・失敗 mode はそのまま適用 (= 重複を避けるため省略、相違点のみ記述)。

## 1. 出典の論拠

`exp008_v3/design.md` §1 と同一。追加:

| doc | section | claim |
|---|---|---|
| `docs/research/independent-edges.dense.md` | §2 Case E | exp009 GP 24 features の数理根拠 |
| 〃 | §3 Edge O | dir-aware Beam 7 features の数理根拠 |

## 2. 4 改修の inject 位置

base: `kaggle_kernels/exp009_case_e_edge_o/exp009_case_e_edge_o.py` (= 4019 行)

### 改修 1: Huber loss + δ tuning

- 対象 line: `LGB_BASE` dict (`352: objective = "regression"`), `CB_PARAMS` dict (`386: loss_function = "RMSE"`)
- 変更内容は exp008 v3 と同一 (LGB → `huber`, CB → `Huber:delta=0.5`、LGB `alpha=0.9`)
- 注: exp009 は **GP 24 features + Edge O 7 features** が追加されるため、自前 base の input dimension が 154 (kb base) + 4 (Edge M) + 7 (Kalman) + 24 (GP) + 7 (Edge O) ≈ 196。Huber loss の挙動は feature 数増加で変わらない (= per-sample loss)

### 改修 2: Heteroscedastic sample_weight

- 対象 line: 自前 4 base train の `model.fit()` (= line 3845-3862) + `Pool(Xt.values, yt)` (= line 3878)
- 変更は exp008 v3 と同一
- 注: exp009 は GP 24 features + Edge O が input、これらは per-row noise structure とは無関係 = per-well sigma 由来の sample_weight をそのまま適用

### 改修 3: Multi-seed × 5-fold MEDIAN

- 対象 line: `OWN_LGB_LRS` (= line 220), `OWN_LGB_N_ESTS`, `OWN_LGB_SEEDS`, `OWN_CB_SEED` + own train loop (= line 3835-3894)
- 変更:
  - `OWN_LGB_LRS = (0.05,)`, `OWN_LGB_N_ESTS = (4000,)`, `OWN_LGB_SEEDS = (42, 123, 2024)`
  - `OWN_CB_SEEDS = (42, 123, 2024)` (新規定数追加)
  - own train loop refactor: seed loop を内側に展開、 6 model × 5 fold = 30 train run
  - aggregate: `lgb_own_med`, `cb_own_med` の 2 own_key、own_oof[k] = MEDIAN({3 seed})
- runtime 注意: exp009 は GP fit (= sklearn GaussianProcessRegressor with 200 inducing, n_restarts=3) が **追加で 1-2 hr** 消費する。MEDIAN による own train 短縮 (2.5-5 hr) と相殺、合計 **6-8 hr** 想定 = 9 hr cap ギリギリだが収まる
- CB は GPU で 3 seed concurrent fit → memory pressure 高、`gc.collect()` per seed 必須

### 改修 4: fold-misalign 解消 (path b)

- 対象 line: 9-base Ridge meta (= line 3910-3978)
- 変更は exp008 v3 と同一
- 注: exp009 では own base diversity が exp008 より高い (= +31 features) ので Ridge own_total_w が degenerate しやすい (= GP + Edge O が dominate) → STACK_OWN_WEIGHT_DEGRADE_LIMIT 0.5 → 0.8 緩和必須

## 3. 累積期待効果

| 改修 | CV Δ | LB Δ |
|---|---|---|
| 1. Huber | -0.30 | -0.30 |
| 2. Hetero | -0.20 | -0.20 |
| 3. MEDIAN | -0.30 | -0.30 |
| 4. path b | -0.30 | -0.30 |
| 合計 (mid) | -1.10 | -1.10 |
| **想定 CV** | exp009 v2 CV 9.1 → **8.0** | exp009 v2 LB ?.? → **8.2-8.5 帯** |

## 4. 9 hr cap risk

exp009 v3 は exp008 v3 より 1-2 hr 多い (= GP fit)。**最 worst case**:

- test FE build: 60 min (kalman + GP + Edge O)
- train FE rebuild: 60 min (= Edge M + Kalman + GP + Edge O over TRAIN_DIR)
- kb 5 base test predict: 5 min
- 自前 train Multi-seed MEDIAN: 5 hr
- Ridge / path b blend: 5 min
- post-proc + submission: 10 min
- 合計: **8 hr** = cap 内 (= 60 min buffer)

cap 超過時の fallback:
- CB seed 数を 3 → 2 に下げる (= 6 → 5 model × 5 fold = 25 run)
- それでも超過したら GP の `n_restarts=3` を 1 に下げる (= GP 精度低下と引き換え)

## 5. 残課題

`exp008_v3/design.md` §5 と同一 + 追加:
- exp009 は GP + Edge O により feature dim が大きく、Huber gradient computation が CPU では遅い → GPU で動作確認必須
