# exp004 実装ノート — Numba Beam + PF + LGB×3 + CB stack (Approach A)

> 位置づけ: 9 切り戦略 (= public LB 8.x 帯到達) の **Approach A diversity 源**。
> 現在の Approach B (= karnakbaev artifact blend exp005 v2 = LB 10.203 Silver) と
> 独立 stack を作り、exp012 で両 Approach を Ridge meta blend → 9.0 帯到達を狙う。
> 単独 LB は 10.2 (= B) より悪化する可能性が高いが、それは設計通り (= diversity が目的)。
>
> 出典 (license compliance): `_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py`
> (Apache 2.0 default Kaggle kernel license)、アルゴリズムを自前再実装し、30 行以上の連続 copy はしていない。
> Beam 移植元 = line 108-144、PF 移植元 = line 223-313、Plane KNN 移植元 = line 319-394。

---

## Section 1: 実装した features と移植元 file:line

### 1.1 Numba JIT Beam Search (= 7 configs、delta +/-2)

- 実装: `src/rogii/beam.py:42-149` (`_beam_kernel` njit + cache=True)
- 7 config 定義: `src/rogii/beam.py:24-32` (BEAM_CONFIGS)
  - cons / loose / vcons / sm5 / vloose / mid / stiff
- 移植元: `rogii-super-solution-lb-top-3.py:108-144` (Apache 2.0)
- 改良点:
  - public API に explicit type hints
  - pure-numerical kernel (= Python object 混入なし)
  - leak-safe (= 関数 args 経由のみ、global typewell store なし)
  - warmup helper で AOT cold start 緩和

### 1.2 Particle Filters (= ANCC + Z、 N=500)

- ANCC PF: `src/rogii/pf.py:71-153` (`run_pf_ancc`)
- Z PF: `src/rogii/pf.py:156-243` (`run_pf_z`)
- 移植元: `rogii-super-solution-lb-top-3.py:223-313` (Apache 2.0)
- 改良点:
  - numerical floor `lk = max(lk, 1e-300)` で particle collapse 防止
  - position clip `tmin - 50, tmax + 50` (50 ft buffer、 top3-distill §8.2)
  - `_safe_normalize` で weight collapse fallback
  - rng_seed 引数で deterministic (= np.random.default_rng 使用)
  - PFConfig dataclass (frozen) でハイパーパラメータ凍結

### 1.3 Spatial Imputers (= 自己除外付き)

- FormationPlaneKNN: `src/rogii/imputers.py:23-109` (= 既存実装、 self_wid leak-safe)
- DenseANCCImputer: `src/rogii/imputers.py:113-179` (= 本 exp004 で新規追加)
- 移植元: `rogii-super-solution-lb-top-3.py:319-394`
- 改良点:
  - `self_wid` 引数で訓練時の OOF leak 防止
  - linalg.solve 失敗時 pinv fallback

### 1.4 Per-Well Feature Builder

- 実装: `src/rogii/features_v4.py:122-433` (build_well)
- 全 features = 171 列 (smoke run で確認)
- 主要 family:
  - PF (pf_ancc / pf_ancc_std / pf_ancc_d / pf_z / pf_z_d / pf_vs_z)
  - Beam 7 configs (beam_{tag}_d、 beam_mean_d / std_d / med_d)
  - Multi-scale NCC 3 windows (sc8 / sc15 / sc25 + scores)
  - Formation TVT (= 6 formations × median+wls) + form consensus
  - Dense ANCC (= 3 baselines + std + dist)
  - Inter-signal consensus (= signal_std / signal_mean_d)
  - GR rolling (= 4 window × 4 lag × 4 stat = grm/grs/glag/glead/gr_d1/gr_d2/gr_env/gr_nrg)
  - Trajectory (= dz/dx/dy/dxy/dzdmd 等)
  - 4 anchor families × 11 offsets (= tda / tdbc / tdsc / tdpf)
  - **D1 cluster id** (= 本 exp004 新規追加、 AC-13)

### 1.5 D1 cluster encoding (AG deep-EDA finding)

- 実装: `src/rogii/features_d1.py:30-50` (`build_cluster_map`)
- 入力: `outputs/eda/deepest_eda/per-well-outlier.parquet` (b_med round 2 decimals)
- 出力: 766 train wells -> 66 distinct cluster ids (zero-indexed, dense)
- test 3 wells: 000d7d20 -> 37, 00e12e8b -> 37, 00bbac68 -> 52 (= all seen, AC-13 OK)
- LGB / CB へは `categorical_feature=["d1_cluster_id"]` で投入

---

## Section 2: CV TVT hidden RMSE (per-fold + overall)

(M4 完了時に追記、 train_v3.py 実行結果から自動 import 予定)

CV strategy:
- 5-fold GroupKFold by `typewell_groups.parquet:group_id` (= 752 unique groups で leak 隔離)
- post-grid alpha / tau / w_pf search で abs TVT RMSE 最小化
- 報告 metric = "CV TVT hidden RMSE" (= AC-3 抽出 pattern と一致)

| fold | LGB0 | LGB1 | LGB2 | CB | Ridge stack | Post-grid (abs RMSE) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | (M4 で記載) |  |  |  |  |  |
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| overall |  |  |  |  |  | **TBD** |

---

## Section 3: HPO best params (LGB + CB) + Optuna trial history 要約

(M2 完了時 = `outputs/exp004/hpo_lgb.json` + `hpo_cb.json` から自動 import 予定)

### 3.1 LGB best params

(TBD: M2 後)

| param | range | best | trial |
|:--|:--|:--|:--|
| learning_rate | [0.01, 0.10] (log) |  |  |
| num_leaves | [15, 255] |  |  |
| min_data_in_leaf | [5, 100] |  |  |
| lambda_l1 | [0, 10] |  |  |
| lambda_l2 | [0, 10] |  |  |
| feature_fraction | [0.5, 1.0] |  |  |
| bagging_fraction | [0.5, 1.0] |  |  |

### 3.2 CB best params

| param | range | best | trial |
|:--|:--|:--|:--|
| learning_rate | [0.01, 0.10] (log) |  |  |
| depth | [4, 10] |  |  |
| l2_leaf_reg | [1, 10] |  |  |

### 3.3 Optuna trial history 要約

- sampler = TPE (seed=42)
- 100 trial 走行時間 = TBD
- best CV (LGB) = TBD ft、 hard-code baseline (= exp007 LGB lr=0.025/nl=255 = 10.66-10.93 ft) 比較 = TBD ft 改善

---

## Section 4: Loss ablation 結果 (= MSE / Huber / MAE per-fold CV)

(M3 完了時に追記)

3 種の loss を同 best params で CV 比較:

| loss | obj name | per-fold CV | overall | vs baseline (= exp008 v3 Huber + hetero) |
|:--|:--|:--|:--|:--|
| MSE | regression | TBD | TBD | — |
| Huber δ=1.0 | huber (alpha=1.0) | TBD | TBD | TBD |
| MAE | regression_l1 | TBD | TBD | — |

best loss = TBD → AC-3 / AC-7 の対象として採用。

考察:
- (M3 後に記載)
- 既存 exp008 v3 (= karnakbaev + Huber + hetero weighting) との比較 (TBD)。

---

## Section 5: D1 cluster encoding ablation (= ON / OFF CV delta)

(M4 完了時に追記)

| 設定 | overall CV | per-fold | delta vs baseline | 備考 |
|:--|:--|:--|:--|:--|
| OFF (= d1_cluster_id 削除) | TBD | TBD | (基準) | `train_v3 --no-d1` |
| ON (= d1_cluster_id ありで LGB categorical) | TBD | TBD | TBD | `train_v3` (default) |

期待: -0.1 〜 -0.5 ft 改善 (AC-13 rubric)。

---

## Section 6: 失敗 pattern と回避ログ + 次の variants

### 6.1 着手前の risk (= .criteria/...yaml § risks)

| risk | 対策 | 結果 |
|:--|:--|:--|
| Numba JIT cold start | `cache=True` + `warmup()` で AOT 化 | TBD (kernel run 確認) |
| Numba random seed mis-align | `np.random.default_rng(seed)` 統一、LGB seed=42 / CB seed=42 | OK (smoke test で deterministic 確認) |
| Kaggle 9 hr runtime overflow | `time.perf_counter()` で 8 hr hard timeout + fallback | OK (kernel 内に組み込み) |
| Self-NCC + Beam + PF OOM | joblib `prefer='threads'` (top3-distill §8.5) | TBD (kernel 実行で確認) |
| CV-LB 乖離 (= overfit) | CV ≤ 10.8 で AC-3 通過、LB-CV diff > 0.5 で postmortem | TBD |
| leak (= self exclusion 漏れ) | `tests/test_beam.py::test_leak_smoke_no_typewell_cross_contamination` で検査 | OK (21 tests pass) |
| numerical instability | `lk = max(lk, 1e-300)` + `pos clip ±50 ft` + `solve -> pinv fallback` | OK (test で確認) |
| D1 unseen cluster | `UNKNOWN_CLUSTER_ID = -1`、 test 3 wells はすべて seen | OK (test_features_d1) |
| Optuna HPO overfit | trial 100 max、 final model は best params 5 seed average | TBD |

### 6.2 開発中の失敗

- (M1-M8 進行中に追記)
- 例: smoke run 8 wells で LGB が 5000 iter まで走って timeout → 本実 776 wells では early-stop 安定動作確認

### 6.3 次の variants (= exp005 系列改善案)

- (M7 完了時に記載)
- 候補:
  - PF particle 数を N=500 から N=1000 へ
  - Beam config 7 から 14 へ (= delta ±3 系)
  - LGB seed average 5 -> 10
  - exp012 で Approach B (karnakbaev blend) と Ridge meta blend
