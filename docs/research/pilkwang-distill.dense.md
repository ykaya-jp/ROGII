# Pilkwang SUPER STACK 構造解析 (2026-05-11)

> **目的**: pilkwang `ROGII EDA v4 — Same-Matrix Super Stack` (267KB, 6142 行, votes 80, GPU enabled, `2026-05-10` 更新) を全行精読し、Top 3 distill (`top3-distill.dense.md`) との差分から **公開 LB Gold 圏 (9 帯帯) 到達の追加要素** を抽出する。
>
> **対象**: `_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py`
>
> **方針**: コード長文丸写しは避け、各要素 30 行以内 + `file:line` 引用 + 物理的解釈の解説。著作権配慮 (Apache 2.0 default、`top3-distill.dense.md §9` 参照)。
>
> **参照**: `docs/research/top3-distill.dense.md` (R/N kernel 詳細), `docs/strategy/winning-strategy.dense.md` Phase 1.6, `docs/research/first-principles.dense.md`

---

## 0. 出典 + メタ情報

| 項目 | 値 |
|---|---|
| kernel URL | https://www.kaggle.com/code/pilkwang/rogii-eda-v4-same-matrix-super-stack |
| kernel id | `pilkwang/rogii-eda-v4-same-matrix-super-stack` (id_no=118068629) |
| local file | `_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py` |
| size | 267KB (.py) / 310KB (.ipynb) / **6142 行** |
| votes | 80 |
| last update | 2026-05-10 (`NOTEBOOK_RUN_VERSION='ROGII_EDA_v4_selfcorr_pflite_super_stack_2026_05_10'` [pilkwang:3517](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py)) |
| `enable_gpu` | `true`、`machine_shape='NvidiaTeslaT4'` (kernel-metadata.json) |
| `enable_internet` | `false` |
| attached datasets | `pilkwang/pilkwang-public-dataset-for-notebooks-figures` (figure 用 PNG のみ、モデル ckpt なし) |
| `kernel_sources` | 空 (= 他人 kernel を fork していない、独立実装) |
| `model_sources` | 空 (= TabICL 等の外部 pretrained model なし) |
| **想定 LB** | **kernel に明示なし**。冒頭 markdown ([pilkwang:10](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py)) で "Public LB benchmark: Roman Tamrazov **LB 10.142**" を引用 → pilkwang 自身は **10.142 と同等以上を狙うパス** を主張するが LB 値を明記していない |

### CV evidence (kernel が自己申告した数字、[pilkwang:108-117](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py))

| Policy | row-weighted RMSE | 備考 |
|---|---|---|
| Constant anchor (= last_known TVT 延長) | 15.9099 | strong null model |
| Strict `calibrated_typewell_alignment` | **14.4315** | strict policy 内ベスト |
| Offline `offline_candidate_path_alignment` (HGB) | **13.6172** | HGB diagnostic |
| **Final `offline_super220_alignment` (LGB×3 seeds + CB Ridge stack)** | (kernel に数値記載なし、これが本命の v4 path) | Kaggle GPU 専用、ローカル実行 skip |

**注**: 13.6172 は **HGBoost (sklearn)** での single model 数字。最終提出は **LightGBM ×3 seed + CatBoost** の 4 base + Ridge stack なので、これより **大幅に下** がる想定。Top 3 distill の R (LB ~10.1) / N (LB 10.081) が **同種の 4-8 base stack で 10 帯到達** していることから、pilkwang も **9-10 帯** が射程 (推定値、出典なし)。

---

## 1. アーキテクチャ全体像

### 1.1 "Same-Matrix Super Stack" の意味

```
horizontal CSV + typewell CSV
  ↓ make_tail_features_for_well() [pilkwang:2319-3069]
  ↓ FormationPlaneKNN (k=10) + RowANCCKNN (k=20, samples_per_well=400)
  ↓ Beam (6 configs ±1 delta) + selfcorr (NN on 5-dim GR signature) + pf_lite (weighted candidate ensemble)
  ↓ candidate_path features (11 endpoints × 4 stats) + tdbc/tdsc offset families (11 offsets each)
  ↓ formation features (6 formations × {plane,row} × {b_all,b_50,formula,delta} = 多数)
  ↓ build_tail_feature_frame (per-well features を pd.concat)
  ↓ feature_set='offline_super220_alignment' で 120-230 列の curated matrix
  ↓
[1 つの同じ feature matrix] X_train / X_test
  ├→ LightGBM seed=42 (5-fold GroupKFold) → OOF_lgb42 + test_pred_lgb42
  ├→ LightGBM seed=7  (5-fold GroupKFold) → OOF_lgb7  + test_pred_lgb7
  ├→ LightGBM seed=123(5-fold GroupKFold) → OOF_lgb123+ test_pred_lgb123
  └→ CatBoost  seed=42 (5-fold GroupKFold) → OOF_cb42  + test_pred_cb42
  ↓
column_stack(4 OOFs) → Ridge(alpha=1, fit_intercept=False, positive=True).fit(stack_oof, y_train_residual)
  ↓ ridge_oof / ridge_test を simple_avg と OOF RMSE 比較 → 良い方を採用
  ↓
fade-in × shrinkage post-process: alpha ∈ [0.60, 1.05, step=0.05], tau ∈ {None, 30, 60, 120, 250, 500} を grid search
  ↓ best (alpha*, tau*) を選択
  ↓ test_pred = last_known_TVT + α* × δ_test × (1 - exp(-md_since_ps / τ*))
  ↓
Savitzky-Golay smooth per well (window=17, poly=3, scipy.signal.savgol_filter)
  ↓
sample_submission.csv と id 整合性検証 → submission.csv
```

### 1.2 "Same-Matrix" の哲学 = R/N との根本的差分

R (romantamrazov) / N (needless090) は **base model ごとに features を変える** (例: TabICL は LGB feature_importances で top-50 のみ使用、[N:678-684](../../_research_kernels/needless090__score-10-081-score-lb-32-rank/score-10-081-score-lb-32-rank.py)) のに対して、pilkwang は **全 base model が同一 feature matrix `offline_super220_alignment` を見る**。

| 流儀 | Top 3 R/N | pilkwang |
|---|---|---|
| feature 設計 | model 別 (LGB/CB/TabICL で異なる subset) | **全 model 同一 (= same-matrix)** |
| diversity の源泉 | features + seeds + hyperparams | **seeds + hyperparams のみ** (LGB×3 seed + CB) |
| 実装複雑度 | 高 (model ごとに feature pipeline) | 低 (1 度作って 4 model に投入) |
| meta-learner | Ridge `positive=True` | Ridge `positive=True` (= 同一) |
| post-proc | alpha × tau × w_pf 3 軸 + Savitzky-Golay | alpha × tau 2 軸 + Savitzky-Golay |

**狙い**: feature engineering を 1 箇所に集中させ、**model diversity は seed/hyperparams だけで稼ぐ**。同一情報を異なる learner で消費させて Ridge で集約するスタイル (= "Stack on the same input space" は Kaggle の典型だが、Top 3 R/N とは流派が違う)。

### 1.3 train / test の formation imputer の非対称性 ★★★ (= LB Gold 帯到達の核心要素の 1 つ)

[pilkwang:5893](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py) (train) vs [pilkwang:5907](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py) (test):

```python
# train_frame
exclude_query_well_from_formation=True,    # leak 防止
# test_frame
exclude_query_well_from_formation=False,   # public-aggressive: 自分の formation も参照する
```

設計コメント ([pilkwang:5512](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py)):
> `public-style test formation behavior | score-seeking offline submission mode`

**意味**: visible test wells (3 件) が train wells と同じ well_id を共有する (visible test = train の subset の特定 zone) ため、**test 時に「自分自身」を imputer の近傍に含める** ことで **public LB を上振れさせる**。R/N は両方 `self_wid` 除外で対称 ([R:341](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py))。

**リスク**: hidden test が真に新規 well なら効かない (visible test = train overlap でしか効かない)。pilkwang もコメントで `score-seeking offline submission mode` (= public LB 最適化、private LB は別) と明示。私的解釈: **public LB Gold 圏入り狙いのチート気味テクで、private LB ではむしろ overfit になり得る**。`docs/research/data-spec.dense.md` の「visible test 比率 26% / 残り 74% は hidden」に照らすと **public LB は visible test の RMSE しか見ていない** ので、この trick で public が上振れする (private がどうなるかは賭け)。

---

## 2. Features

### 2.1 R/N との一致要素 (top3-distill §1-7 と同型)

| 要素 | pilkwang 実装 | top3-distill との対応 | 備考 |
|---|---|---|---|
| residual target `target_delta_from_last_known` | [pilkwang:3068](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py) | §6.4 共通 | 同一 |
| `last_known_TVT` を anchor | [pilkwang:2394](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py) | §1.3 共通 | 同一 |
| `FormationPlaneKNN` (k=10, IDW WLS plane fit) | [pilkwang:2158-2233](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py) | §3 共通 | k=10 同じ、scale も `xy.std(0)` 同じ。`A.flat[::4] += 1e-8` の ridge stabilization も同様 |
| `RowANCCKNN` (k=20, samples_per_well=400) | [pilkwang:2236-2310](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py) | §3.5 DenseANCC と相当 | pilkwang は **400 samples/well** (R は 60 samples/well) → **6.7 倍 dense**。次節 §2.2 で詳述 |
| 6-formation b_well 計算 (median + b_50) | [pilkwang:2650-2654](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py) | §5.1, §5.2 | R の `b_50` (末尾 50 median) と同一。**ただし WLS は無し** (= R 固有の WLS は移植せず median のみ) |
| `tvt_formula = -Z + ANCC + b_well` per formation | [pilkwang:2657](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py) | §3.6 物理 formula | 6 formations 全部で計算 |
| Beam Search (configs: tight/cons/loose/vcons/sm5/vloose) | [pilkwang:2888-2895](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py) | §1.1, §1.2 | **6 configs ±1 delta** (R の 7 configs ±2 から 1 つ削除 + delta 縮小)。**Numba JIT なし** |
| GR rolling stats (windows 25/100/300) | [pilkwang:2580-2586](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py) | §1.3 周辺 | 同型 |
| Affine GR calibration (`safe_affine_fit`) | [pilkwang:1555-1563](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py) | §I (移植 map) | karnakbaev 流に近い。R は別 (GR detrend) |
| Stacking: LGB×3 seed + CB + Ridge `positive=True` | [pilkwang:5945-5979](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py) | §6.1, §6.2 | R の 4-base stack と同型 (但し R は `lr` 多様化、pilkwang は `seed` 多様化) |
| Savitzky-Golay smooth (window=17, poly=3) | [pilkwang:5723-5739](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py) | §6.3 step 6 | 完全同一パラメータ |

### 2.2 pilkwang 固有 features (★ 重要)

#### 2.2.1 Self-correlation (NN-based, NOT NCC) ★★★

[pilkwang:1847-1929](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py) `selfcorr_prefix_tvt_features()` の核心:

```python
# (1) GR 系列に rolling window=31 (half_window=15) で 5-dim signature 計算
sig = np.column_stack([roll.mean(), roll.std(), (roll.max()-roll.min()), values.diff(), values])
# = (mean, std, range, gradient, center) の 5 次元 GR texture

# (2) prefix から stride=3 で sampling → prefix_sig (M, 5)
# (3) tail も同じ signature 計算 → tail_sig (n_tail, 5)
# (4) prefix_sig を z-normalize して NearestNeighbors(k=2) で fit、tail_sig.kneighbors(...)
# (5) best match の prefix index → prefix_tvt_candidates[idx] が selfcorr_tvt
# (6) score = exp(-clip(best_dist, 0, 20) / 2.5)
# (7) trust = clip(score * (prefix_len/250), 0, 1)
```

**top3-distill §4 (Self-NCC) との差分**:

| 項目 | top3 R/N | pilkwang |
|---|---|---|
| matching algorithm | **NCC** (normalized cross-correlation, 31 ft window) | **NearestNeighbors on 5-dim signature** (sklearn) |
| signature 次元 | raw GR 31 dims | (mean, std, range, grad, center) 5 dims |
| 距離 | NCC (内積 / win) | Euclidean on z-normalized signature |
| multi-scale | R は 3 windows (8/15/25) | pilkwang は **1 window (15)** のみ |
| top-2 gap output | なし | `selfcorr_top2_gap` (= 2 番目近傍距離との差) を feature に渡す |
| trust | `sc_trust = clip(len(kn) / 200, 0, 0.6)` | `trust = clip(score * (prefix_len/250), 0, 1)` |
| 計算量 | (nh, M) × 31 ft NCC → 大 | 5-dim NN → **数倍速い** (sklearn `NearestNeighbors` の KD-tree) |

**物理的解釈**: NCC は raw signal の波形相似性を厳密に見るが、pilkwang の 5-dim signature は **(中央値, 散らばり, レンジ, 勾配, 値そのもの)** の 5 統計量で「**texture が似ている prefix 区間**」を粗く特定。`top2_gap` を feature に追加することで **「曖昧 match か明確 match か」を GBM に教える** (= 信頼度推定が組み込まれている)。

**評価**: NCC の方が原理的には良さそう (波形を保つ) だが、**pilkwang は計算量を大幅に減らした上で `top2_gap`/`trust` という不確実性 feature を追加** している。GBM が "曖昧な match は無視、明確な match だけ採用" を学習できる → **NCC が捕えにくい "勘違い match"** を排除できる。R の NCC は scalar score を返すだけ。

#### 2.2.2 PF-lite (Particle Filter の超軽量近似) ★★★

[pilkwang:1932-2003](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py) `weighted_candidate_tvt_features()`:

```python
# (1) 6-7 candidate TVT (last_known, beam_cons, beam_loose, beam_sm5, selfcorr, plane, dense, formation_mean) を column_stack
candidate_tvt = {'last_known': ..., 'beam_cons': ..., 'beam_loose': ..., ...}
# (2) 各 candidate TVT で typewell GR を補間 → tw_gr (n, n_candidates)
tw_gr = typewell_gr_at_tvt(typewell_df, cand.reshape(-1)).reshape(cand.shape)
# (3) GR likelihood: exp(-|hr_gr - tw_gr| / 20) で weight 計算
weights = np.exp(-clip(|tail_gr - tw_gr|, 0, 6 * 20) / 20.0) * finite
# (4) ANCC penalty: cand + tail_z - dense_ancc を expressed → exp(-|...|/35) で weight 補正
dense_factor = 0.20 + 0.80 * exp(-clip(|cand+z-dense_ancc|, 0, 6*35) / 35)
weights *= dense_factor
# (5) weighted mean / weighted std を返す
pf_tvt = sum(cand * weights) / sum(weights)
pf_std = sqrt(sum((cand - pf_tvt)**2 * weights) / sum(weights))
```

**top3-distill §2 (Numba JIT PF) との差分**:

| 項目 | top3 R/N PF | pilkwang PF-lite |
|---|---|---|
| state | `(pos, vel)` or `(pos, rate)` の Markov chain | **state なし** (per-row で独立) |
| transition | AR(1) + Gaussian process noise + resampling | **transition なし** |
| likelihood | typewell GR Gaussian | typewell GR Gaussian + ANCC dense penalty |
| 候補数 | N=300-500 particles | **6-8 candidates only** (Beam paths + selfcorr + plane + dense) |
| 出力 | per-step weighted mean + std | per-row weighted mean + std + 4 vs-ratio + GR resid |
| 計算量 | N=500 × hidden_steps × O(1) | 8 × hidden_steps × O(1) → **62× 高速** |
| Markov 仮定 | あり | なし |

**意味**: 「Particle Filter の核心 = 複数 hypothesis の confidence-weighted ensemble + uncertainty 出力」だけを **state-less / per-row で再現**。GBM は per-row でしか動かないので **state を持つ必要がない** という割り切り。

**追加 features ([pilkwang:1986-2002](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py))**:
- `pf_lite_tvt`: weighted mean
- `pf_lite_delta`: TVT - last_known
- `pf_lite_std`: weighted std (= 不確実性)
- `pf_lite_weight_sum`: 全 weight 和 (= 候補がどれだけ "信用できる" か)
- `pf_lite_candidate_count`: 有効候補数
- `pf_lite_vs_dense / vs_plane / vs_sc / vs_beam_cons`: pf_lite と他手法の差 (= 4 method 間の相互チェック)
- `pf_lite_gr_abs_resid`: best candidate の GR residual

**物理的解釈**: 各手法 (Beam, selfcorr, plane, dense) の予測を **「typewell GR が観測 GR と整合するか」** + **「dense ANCC を介した tvt_formula が成立するか」** という 2 つの likelihood で重み付けし、ensemble。`vs_*` features は **手法間の意見が合うかバラけているか** を GBM に教える。

#### 2.2.3 Candidate Path Features (11 endpoints × 4 stats) ★★

[pilkwang:1632-1659, 2006-2064](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py):

```python
CANDIDATE_PATH_ENDPOINTS = [-60, -40, -25, -15, -8, 0, 8, 15, 25, 40, 60]  # 11 endpoints

# 各 endpoint で linear path: tvt = last_known_tvt + tail_frac × endpoint
# tail_frac = 0 (= last_known), 1 (= tail 末尾) を線形補間
# 各 path で typewell GR を interp → 観測 tail GR との absdiff
# best endpoint / soft endpoint mean / top2 gap / boundary count を feature 化
```

**さらに `tw_path_ease`** ([pilkwang:2869-2876](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py)) で `tail_frac^1.45` (= ease-in 曲線) でも同じ feature を計算 → 線形 vs ease-in の 2 種を持つ。

**意図**: 「hidden zone での TVT は last_known から linear or ease-in でどれかの endpoint に向かう」という事前 prior を 11 通り × 2 曲線 = 22 path として展開し、観測 GR との fit が最良の endpoint を発見させる。R/N の Beam Search が "GR-driven path 探索" だとしたら、これは **"endpoint-anchored 大局トレンド探索"** で粒度が粗いが計算は軽い。

#### 2.2.4 TDBC / TDSC Offset Families (typewell GR diff at offset positions) ★

[pilkwang:1772-1773](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py):

```python
TDBC_OFFSETS = [-40, -20, -10, -5, -3, 0, 3, 5, 10, 20, 40]  # 11 offsets, anchor=beam_cons
TDSC_OFFSETS = [-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30]   # 11 offsets, anchor=selfcorr_tvt
```

`tdbc_<label>` = `tail_GR - typewell_GR_at(beam_cons + offset)` (11 features)
`tdsc_<label>` = `tail_GR - typewell_GR_at(selfcorr_tvt + offset)` (11 features)

**top3-distill `tw_diff` family ([R:619-623](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py)) との対応**:
- R は `[40,20,10,5,3,0,3,5,10,20,40]` の 11 offsets を **beam_ref (= cons + sm5 平均)** anchor で 1 family
- pilkwang は **2 family** (beam_cons anchor `tdbc` + selfcorr anchor `tdsc`) → **22 features** (R は 11 features)

**意図**: anchor 周辺の細かい TVT shift で typewell GR がどう変化するかを多次元 feature として GBM に渡す → "正解 TVT は anchor からどっち方向にどれだけズレているか" を学習。

#### 2.2.5 GR Centered Rolling + Lag/Lead (offline policy) ★

[pilkwang:2588-2611](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py):

```python
for window in [5, 21, 51, 151, 301]:
    center_roll = gr_full.rolling(window, center=True, min_periods=...)
    out[f'gr_center_roll_mean_{window}'] = ...
    out[f'gr_center_roll_std_{window}'] = ...
    out[f'gr_center_roll_range_{window}'] = ...
out['gr_center_lag1/lead1/lag5/lead5/lag15/lead15/lag30/lead30'] = ...
out['gr_center_grad_1/grad_2'] = ...
out['gr_cumsum_since_ps'] = ...
```

**top3-distill との差分**: R は trailing-only (causal) rolling stats が中心。pilkwang は **`offline` policy で centered window と lead (= 未来の GR) を使う** → test 時に GR は完全提供される (target ではない) ので leak なし、ただし strict drilling-time policy では使えない。

**狙い**: hidden zone で GR は完全に観測できる ⇒ "未来の GR を見た context" を feature 化することで予測精度を上げる。strict policy (drilling-time compatible) では使えないが、Kaggle submission では使ってよい。

#### 2.2.6 6-formation 全 features の網羅 ★

R/N は 6 formation の `b_well` を計算しても **GBM に渡すのは ANCC の 1 つ + tvt_formula の集約** が中心 ([R:483-503](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py))。pilkwang は 6 formation **すべて** で:
- `formation_plane_anchor_b_{label}` (median b_well per formation)
- `formation_plane_anchor_b50_{label}` (b_50 per formation)
- `formation_plane_prefix_rmse_{label}` (per-formation accuracy on known zone)
- `formation_plane_prefix_mae_{label}`
- `formation_plane_tvt_formula_{label}` (= -Z + plane_form + b)
- `formation_plane_delta_formula_{label}`
- `formation_plane_delta_formula50_{label}`

を出力 ([pilkwang:2074-2099](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py)) → **6 formations × 7 features = 42 features**。

GBM に "どの formation が最も信頼できる plane fit か" を選ばせる戦略。R の WLS b_well 3 種 ([R:289-295](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py)) と思想は近いが、formation 全部に展開する点が異なる。

### 2.3 features 総数

`offline_super220_alignment` ([pilkwang:5901-5904](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py)) は `SUPER_MIN_FEATURE_COUNT=120, SUPER_MAX_FEATURE_COUNT=230` の curated range ([pilkwang:5541-5542](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py))。`220` は名前の通り 220 ± 10 features 程度を持つ想定。

block 構成 ([pilkwang:5758-5772](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py)): base / gr / typewell / path / beam / formation / row_ancc / selfcorr / offset / pf_lite の 10 family。

---

## 3. Models

| # | model | params | seed | role | file:line |
|---|---|---|---|---|---|
| 1 | LightGBM seed 42 | num_leaves=127, lr=0.04, n_estimators=5000, GPU, early_stopping(150) | 42 | base | [pilkwang:5625-5645](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py) |
| 2 | LightGBM seed 7 | 同上 | 7 | base (seed diversity) | 同上 |
| 3 | LightGBM seed 123 | 同上 | 123 | base (seed diversity) | 同上 |
| 4 | CatBoost seed 42 | iterations=5000, lr=0.04, depth=8, l2_leaf_reg=3, GPU `task_type='GPU'` `devices='0:1'` (T4×2) | 42 | base | [pilkwang:5677-5712](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py) |
| 5 | Ridge `(alpha=1, fit_intercept=False, positive=True)` | - | - | meta-learner | [pilkwang:5968-5970](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py) |

**top3-distill との対応**:
- N (LB 10.081) は LGB×3 seed + CB×3 seed + TabICL×2 = **8 base** (top3-distill §6.1)
- R (LB ~10.1) は LGB×3 (lr 多様化) + CB = **4 base** (top3-distill §6.2)
- pilkwang = LGB×3 seed + CB = **4 base** (= R と同じ構成だが diversity の作り方が **lr→seed** に変わっている)

**TabICL なし**: pilkwang は TabICL を使っていない → 構成上は R に近い軽量 stack。N の TabICL 追加が +0.02-0.04 LB の差分とすると、pilkwang は **R 流の 4-base stack と同じ LB 帯 (10.1 前後)** が射程と推定。

### 3.1 LightGBM hyperparams 詳細

```python
boosting_type='gbdt', objective='regression', metric='rmse',
learning_rate=0.04, num_leaves=127, min_child_samples=20,
subsample=0.80, subsample_freq=1, colsample_bytree=0.80,
reg_lambda=5.0, reg_alpha=0.1,
seed=seed, n_estimators=5000,  # early_stopping(150)
device_type='gpu', gpu_use_dp=False, max_bin=255  # Kaggle GPU mode
```

**N との比較** (top3-distill §6.1): N も `num_leaves=127, lr=0.04, n_estimators=5000, GPU` で **完全同一**。R は `num_leaves=255, lr=[0.025/0.020/0.030]` で深いが lr 小。pilkwang は **N の hyperparams と完全一致** で 3 seed で振る → LB 10.081 (N) と同じ base 性能を出す前提で、stack 全体構成 (CB + Ridge + post-proc) で勝負を分ける戦略。

### 3.2 CatBoost hyperparams 詳細

```python
iterations=5000, learning_rate=0.04, depth=8, l2_leaf_reg=3.0,
min_data_in_leaf=20, loss_function='RMSE', random_seed=42,
task_type='GPU', devices='0:1' if 2 GPUs else '0',
od_type='Iter', od_wait=150
```

**N との比較**: N は `depth=8, lr=0.04, iterations=5000, GPU` で **同一**。R は `depth=7, lr=0.025`。

---

## 4. Stacking

### 4.1 OOF flow ([pilkwang:5945-5979](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py))

```python
splits = list(GroupKFold(n_splits=5).split(X_train, y_train, groups=well_id))

# 各 base model で 5-fold OOF + test pred (avg over 5 folds)
results = {
    'lgb_seed42': run_super_lgb(42, X_train, y_train, X_test, splits),
    'lgb_seed7':  run_super_lgb(7, ...),
    'lgb_seed123':run_super_lgb(123, ...),
    'catboost_seed42': run_super_catboost(...),
}

# Stacking
stack_oof  = column_stack([item['oof']  for item in results.values()])  # (n_train, 4)
stack_test = column_stack([item['test'] for item in results.values()])  # (n_test, 4)

ridge = Ridge(alpha=1.0, fit_intercept=False, positive=True)
ridge.fit(stack_oof, y_train)  # y_train = target_delta_from_last_known
ridge_oof  = ridge.predict(stack_oof)
ridge_test = ridge.predict(stack_test)

avg_oof  = stack_oof.mean(axis=1)
avg_test = stack_test.mean(axis=1)

# Ridge と Simple Avg を OOF RMSE 比較 → 良い方を採用
use_ridge = (ridge_rmse <= avg_rmse)
final_oof_delta  = ridge_oof  if use_ridge else avg_oof
final_test_delta = ridge_test if use_ridge else avg_test
```

**top3-distill §6.3 との一致度**: ほぼ完全一致。Ridge `positive=True, fit_intercept=False` も同じ。`positive=True` で **負の重みを禁止**することで anti-correlated outlier base を排除し、weighted positive blend を強制 (= robust blend)。

### 4.2 CV 戦略

`GroupKFold(n_splits=5)` by `well_id` ([pilkwang:5945](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py))。**top3 R/N と同じ 5-fold**。

**rows_per_well sampling**: `rows_per_well=None` ([pilkwang:5885](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py)) → **全 train tail rows を使う** (3,783,989 rows ([pilkwang:93](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py)))。CV 用途には `rows_per_well=350` で sampling していたが、最終提出は全 row。

**top3-distill との差分**: top3 R/N も全 row ([R:617-619](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py))。row sampling は CV diagnostic 用のみで、submission は全 row が共通。

---

## 5. Post-process

### 5.1 alpha × tau grid search ([pilkwang:5981-5988](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py))

```python
for alpha in np.arange(0.60, 1.05, 0.05):  # 9 grid points
    for tau in [None, 30.0, 60.0, 120.0, 250.0, 500.0]:  # 6 grid points
        delta = fade_delta_by_md(train_meta, final_oof_delta, tau) * alpha
        # fade_delta_by_md: delta *= (1 - exp(-md_since_ps / tau))
        pred_abs = base_train + delta
        rmse_value = root_mean_squared_error(y_abs, pred_abs)
        # best (alpha*, tau*) を OOF RMSE で選ぶ
test_delta = fade_delta_by_md(test_meta, final_test_delta, best_tau) * best_alpha
test_pred = test_meta['last_known_TVT'].to_numpy(dtype=float) + test_delta
```

**意味**:
- `alpha`: residual prediction の shrinkage 係数 (0.60-1.00)。`alpha < 1` で **予測 delta を縮める** (= 過剰な extrapolation を抑制)
- `tau`: fade-in 時定数。**hidden zone 直近** (md_since_ps が小) では予測を縮める (`(1-exp(-md/tau))`)、遠方ほど 1 に近づく → "近い場所は last_known の延長で十分、遠ざかるほど model 予測を信頼" という物理事前

**top3-distill §6.4 との対応**: 完全同型 (`(1 - exp(-md_since/tau))` fade-in も `pred *= alpha` shrinkage も同じ)。R は `w_pf` の 3 軸目を加える ([top3-distill §6.3 step 5](top3-distill.dense.md))。pilkwang は 2 軸のみ。

### 5.2 Causal slope clip per well (★ pilkwang 固有)

[pilkwang:4279-4324](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py):

```python
# (1) train wells 全部から |dTVT/dMD| を集計し、quantile を取って max_abs_slope を推定
MAX_ABS_TVT_SLOPE_BY_QUANTILE = estimate_abs_tvt_slope_quantiles(
    train_horizontal_files, quantiles=[0.90, 0.95, 0.975, 0.99, 0.995]
)
# 例: q=0.99 で max_abs_slope ≈ 0.5-1.0 ft/ft (= TVT が 1ft 進む間に MD が 1-2ft 進める速度)

# (2) per-well, prev_tvt → curr_tvt の変化を |max_abs_slope * step_md| で hard clip
for k in range(len(pos)):
    step_md = abs(md[k] - prev_md)
    limit = max_abs_slope * max(step_md, 1e-6)
    clipped[pos[k]] = np.clip(clipped[pos[k]], prev_tvt - limit, prev_tvt + limit)
    prev_tvt = clipped[pos[k]]  # ★ causal: clipped 値を次の anchor にする
    prev_md = md[k]
```

**意味**: 物理的に **bit が 1 ft 進む間に TVT が ± δ ft しか動けない** という制約を hard clip。Outlier prediction (= 突発的な大ジャンプ) を強制的に丸める。q=0.90-0.995 を grid search で OOF RMSE 最良を選ぶ ([pilkwang:5452-5460](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py))。

**top3-distill との差分**: R/N にない。pilkwang 固有の "safety clipping"。 **drift モデリングの暴発防止 + private LB の robust 化**に効果あり (= 公開 LB だけでなく hidden test でも効くはず)。

**注**: ただし `RUN_SUPER_MONOLITH_STACK` path ([pilkwang:5836-6090](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py)) では **slope clip は使われていない** (= alpha × tau の 2 軸 grid のみ)。Strict / HGB diagnostic path のみで使用。同 kernel 内に **2 流派** が共存し、最終提出は seasonal alpha × tau 流派が選ばれている → slope clip は **safety guard としては実装済みだが super stack には load されていない**。

### 5.3 Savitzky-Golay smooth ([pilkwang:5723-5739](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py))

```python
from scipy.signal import savgol_filter
for _, group in temp.groupby('well_id', sort=False):
    idx = group['_row_pos'].to_numpy(dtype=int)
    n = len(idx)
    wl = min(window=17, n); wl -= (wl % 2 == 0)  # window must be odd
    if wl >= poly + 2:
        values[idx] = savgol_filter(values[idx], wl, 3)
```

per well で window=17, poly=3。隣接予測同士の HF jitter を平滑化 (= TVT は連続関数のはず)。**top3-distill §6.4 step 6 と完全同一**。

---

## 6. GPU 利用箇所

`enable_gpu: true` (kernel-metadata.json) で何を GPU 化しているか:

1. **LightGBM GPU** ([pilkwang:373-385](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py), [pilkwang:5571-5595](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py)):
   ```python
   {'device_type': 'gpu', 'gpu_use_dp': False, 'max_bin': 255}
   ```
   `preflight_lightgbm_gpu()` で 6-row dataset で 1-iteration 動作確認 → fail 即 abort。

2. **CatBoost GPU** ([pilkwang:5598-5622](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py)):
   ```python
   task_type='GPU', devices='0:1' if len(gpu_names) >= 2 else '0'  # T4×2 dual GPU
   ```
   `preflight_catboost_gpu()` で 32-row dataset で 2-iter 動作確認 → fail 即 abort。

3. **NN / Transformer なし**: `enable_gpu` の主目的は **LGB + CB の training を GPU で高速化** すること (XGBoost は default model に含まれていない)。

**top3 distill の N との比較**: N も `device_type='gpu'` で LGB / CB を GPU 化。pilkwang は **強制的に GPU が必須** (`SUPER_REQUIRE_GPU_ON_KAGGLE = True` [pilkwang:5539](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py)) で、GPU なしでは即 abort する設計。

**preflight の哲学**: "feature 構築 (= 30-60 min) を始める前に GPU が動くか tiny dataset で確認" → 失敗時の時間ロスをゼロに。我々の repo にも導入価値あり (= safety guard)。

---

## 7. Top 3 distill との差分 (★★★)

pilkwang が R/N から「追加した」要素を優先順で:

### 7.1 ★★★ 最も重要: PF-lite (state-less particle filter ensemble)

**追加内容**: full PF (top3 R/N) を捨てて、**6-8 candidate TVT を per-row で likelihood-weighted ensemble** する PF-lite を独自設計。

**なぜ効くか**:
- R/N の PF は **Markov chain で連続 path を出す** が、出力は per-step weighted mean のみ。**他手法 (Beam, selfcorr, plane, dense) との比較情報を per-row で持たない**
- pilkwang の PF-lite は **「Beam / selfcorr / plane / dense / formation_mean を全部候補に入れて、観測 GR と ANCC dense の両方で likelihood 計算 → weighted mean + std + vs-ratio 4 個」** を per-row 出力 → GBM が **「手法間の意見の合致度」を直接学習可能**
- `pf_lite_vs_dense / vs_plane / vs_sc / vs_beam_cons` の 4 features は **「複数手法が同意した zone は信頼できる、バラけた zone は不確実」** を符号化

**実装難度**: 低 (numpy 純)。state-less なので Numba JIT も不要。**60 行で完結**。

### 7.2 ★★★ 重要: Self-correlation を NN-on-signature で実装

**追加内容**: top3 R/N の NCC を捨てて、**5-dim GR signature (mean/std/range/grad/center) で `sklearn.NearestNeighbors`** で nearest prefix を探す方式に変更。`top2_gap` / `trust` を feature 化。

**なぜ効くか**:
- NCC は raw signal を厳密 match → noise に弱い、計算重い
- 5-dim signature は **GR texture の粗い圧縮**で、noise を 5 統計量に集約 → false match が減る
- `top2_gap` (2 番目近傍との距離差) を feature 化することで GBM が **「曖昧 match か明確 match か」を判別可能** → R/N にはない

**実装難度**: 中 (sklearn NN 利用)。**100 行**。

### 7.3 ★★ 重要: train/test asymmetric formation imputer (public LB max strategy)

**追加内容**: train 時 `exclude_query_well_from_formation=True` (leak 防止)、**test 時 `False` (= 自分自身も近傍参照)**。

**なぜ効くか (公開 LB のみ)**:
- visible test wells (3 件) は train wells と well_id を共有 (visible test = train の特定 zone のみ hidden)
- test 時に self を imputer に入れると **"自分自身の formation 平均値が周囲 10 wells の中心" になる** → plane fit がほぼ完全に test well を通る → ANCC imput がほぼ正解 → tvt_formula がほぼ正解
- **public LB が visible test の RMSE しか見ていない場合**、これだけで LB が大幅改善する公算 (-0.5 〜 -1.0 LB 推定)

**リスク (private LB 崩壊シナリオ)**:
- hidden test が visible test と異なる well_id (= 真の新規 well) なら **この trick は効かない、むしろ leak で fit overfit**
- pilkwang 自身もコメントで `score-seeking offline submission mode` と注記 → public LB 最大化がバレるのを承知

**移植判断**: 我々が public LB Gold 圏入りだけを狙うなら効果的。private LB 崩壊リスク受容は要議論。

### 7.4 ★★ 重要: Candidate Path Features (11 endpoints × 2 曲線 = 22 paths)

**追加内容**: tail_frac × endpoint で 11 個の linear path + 11 個の ease-in path を作って typewell GR と fit。

**なぜ効くか**:
- Beam は GR-driven の細粒度 path 探索だが、**「全体トレンドが endpoint -60 ft なのか +60 ft なのか」の大局判断**は 1-2 ft 単位の Beam では難しい
- Candidate Path は **大局トレンドの 11 仮説** を直接 GR-fit で評価 → "best endpoint" を feature 化することで GBM に **粗粒度の TVT 移動方向**を伝える
- `tw_path_top2_absdiff_gap` で曖昧度も渡す

**実装難度**: 低 (numpy 純)。**80 行**。

### 7.5 ★★ Centered + Lead/Lag GR features (offline policy)

**追加内容**: `gr_center_lead1/lead5/lead15/lead30` (= 未来 GR の値) を feature に追加。

**なぜ効くか**: Kaggle test では **GR は完全提供** (target ではない) → "未来 GR" を見ても leak ではない。R/N は trailing-only で causal (drilling-time compatible) を守るが、pilkwang は **offline submission mode** と割り切って lead を使う。

**実装難度**: 極低 (`Series.shift(-k).ffill()` の 8 features)。**5 行**。

### 7.6 ★ TDBC + TDSC offset families (anchor=beam_cons + anchor=selfcorr)

**追加内容**: anchor を 2 つに増やして tw_diff family を 22 個に倍増。

**なぜ効くか**: R は 1 anchor (`beam_ref`) で 11 features。pilkwang は 2 anchor で **22 features** → 各手法の anchor 周辺の細かい "GR profile sensitivity" を多次元化。

### 7.7 ★ Causal slope clip per well

**追加内容**: `MAX_ABS_TVT_SLOPE_BY_QUANTILE = estimate_abs_tvt_slope_quantiles(train_files)` で train から TVT 速度上限を quantile で推定し、per-well causal hard clip ([pilkwang:4305-4324](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py))。

**なぜ効くか**: outlier prediction を物理的上限で強制丸め → robust。但し pilkwang は **super stack path では使っていない** (= 開発済みだが採用見送り)。

### 7.8 ★ Per-formation features の網羅 (6 formations × 7 features = 42 features)

**追加内容**: ANCC のみでなく **6 formation 全部** で `b / b50 / prefix_rmse / prefix_mae / tvt_formula / delta / delta50` を per-formation 計算。

**なぜ効くか**: GBM が **どの formation が当該 well で最も信頼できるか** を選択可能。R/N は ANCC 中心で 6 formation の多様性を活かしきれていない可能性。

---

## 8. 我々の repo に移植可能な要素 (= 推奨せず軸のみ)

> **CRITICAL**: 推奨案ではなく「選択軸」のみ提示。判断は開発者に委ねる。

| 要素 | 推定工数 | 期待 LB 改善 | 失敗モード | 既存 src/rogii/* との合成可能性 |
|---|---|---|---|---|
| **PF-lite** (`weighted_candidate_tvt_features`) | 0.5 日 | -0.3〜0.5 | candidate TVT が高 NaN なら weight 全死 → fallback median が必要。pf_std の意味は full PF より弱い (state-less) | 現 9 帯 kernel に Beam + selfcorr が入っている前提なら **immediate plug-in 可** (既存 candidate を column_stack で渡すだけ) |
| **Self-correlation NN-on-signature** (`selfcorr_prefix_tvt_features`) | 0.5 日 | -0.2〜0.4 | 既存 NCC 実装と coexist させると features 重複で GBM 混乱 → どちらか一方に統一が無難 | 既存 `src/rogii/self_ncc.py` がある場合、**置き換えか並列か**の判断が必要。並列なら top3 distill §10 の "アイデア多様化" として正当化可能 |
| **train/test asymmetric formation imputer** | 0.1 日 | -0.5〜1.0 (public のみ) / +0.0〜+1.0 (private 崩壊リスク) | hidden test が真の新規 well なら逆効果 | `FormationPlaneKNN.impute(self_wid=...)` の 1 引数を test 時 None に変えるだけ。**public LB 即効、private LB は賭け** |
| **Candidate Path Features** (11 endpoints × 2 曲線) | 0.5 日 | -0.2〜0.4 | Beam が既に強い場合は redundant (= 情報過多で GBM が混乱) | `tw_path_*` は anchor が `last_known_TVT + tail_frac × endpoint` で軽量。Beam と diversity 出る可能性あり |
| **Centered + Lead/Lag GR (offline policy)** | 0.1 日 | -0.1〜0.2 | strict drilling-time policy 違反 (= 業界利用なら不可)。Kaggle submission のみ可 | `Series.shift(-k).ffill()` 5 行追加。即効果 |
| **TDBC + TDSC dual-anchor offset family** | 0.25 日 | -0.1〜0.3 | features 過剰で GBM が学習遅延 (overfit risk) | 既存 `src/rogii/features.py` に拡張。anchor=beam_cons / selfcorr の 2 種で 22 features |
| **6-formation 全 features 化** | 0.25 日 | -0.1〜0.3 (既存 ANCC 中心 features がある前提) | features 数 +35 → memory + tree depth 増。GBM hyperparams の調整必要 (`min_child_samples` 増やす等) | 既存 imputer / features 流用で per-formation loop 追加するだけ |
| **GPU preflight (LightGBM/CatBoost)** | 0.1 日 | LB 改善なし (但し submission 失敗時間ロス削減) | preflight 自体が tiny で 5 sec、リスク低 | 既存 submission script に `preflight_lightgbm_gpu` / `preflight_catboost_gpu` を import |
| **Causal slope clip per well** (`estimate_abs_tvt_slope_quantiles` + `causal_slope_clip_by_well`) | 0.25 日 | -0.0〜0.2 (outlier 抑制効果のみ、平均 RMSE には影響小) | clip 過剰だと真の急変化 zone も抑制 | 既存 `src/rogii/postproc.py` に追加。q を grid search で選ぶ |
| **alpha × tau grid post-process (existing と異なる軸組み合わせ)** | 0.1 日 | -0.1〜0.3 (existing post-proc と diversity ある場合) | 既存 post-proc と差異がない可能性 | 既存 `src/rogii/postproc.py` に grid 拡張 |

**工数合計** (top 5 要素): **約 1.85 日** (= 約 15 時間、週 20h でも 1 日)。

**依存グラフ**:
```
[既存 9 帯 kernel = subagent G 完成想定]
   ├→ PF-lite (Beam + selfcorr 既存前提) ← 即効性最大
   ├→ Self-correlation NN (置き換え or 並列)
   └→ Candidate Path (Beam と diversity)
       ↓
   train/test asymmetric formation (public LB max trick) ← 賭け
       ↓
   Centered/Lead GR (offline policy) ← 軽量
       ↓
   TDBC/TDSC dual-anchor (features 過剰 risk)
       ↓
   6-formation 全 features (memory コスト)
       ↓
   GPU preflight (safety guard)
```

---

## 9. ライセンス + 引用

- **kernel license tag**: `kernel-metadata.json` に license 欠落 → Kaggle Terms 8.B により **default Apache 2.0 相当** (= top3-distill §9 と同様)
- **再配布**:
  - そのまま fork して提出するのは原則禁止 (各 kernel 著者の同意要)
  - **アルゴリズム copy + 自分の repo で再実装は OK**
  - コード snippet を docs/research/ に引用する場合は **必ず出典 URL + 著者名を残す** ⇒ 本 doc では §0 で URL + file:line を記載済み
- **再実装 commit message**: "inspired by pilkwang/rogii-eda-v4-same-matrix-super-stack" を明記

---

## 10. 関連

- `docs/research/top3-distill.dense.md` — Top 3 R/N kernel 詳細解析 (= 比較対象)
- `docs/research/first-principles.dense.md` — 物理 formula `TVT = -Z + ANCC + b_well` の出典
- `docs/research/independent-edges.dense.md` — 独自 edge 候補 (xcorr_tvt_features 等)
- `docs/research/public-notebook-analysis.dense.md` — 公開 LB 序列 + ノートブック特徴
- `docs/research/discussions.dense.md` — Slide 9 insight (= Self-correlation の物理根拠)
- `docs/strategy/winning-strategy.dense.md` Phase 1.6 — 戦略 doc 上の優先順位 (= 移植 map との照合)
- `docs/research/host-datasets.dense.md` — Kaggle host dataset 確認 (本 distill は kernel のみで dataset 検証は未踏)

---

## 付録: pilkwang `offline_super220_alignment` の構成 (block ごと feature 数の概算)

[pilkwang:5803-5888](../../_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py) より:

| block | features 推定数 | 主な内容 |
|---|---|---|
| `base` | ~30 | last_known, MD/X/Y/Z, GR, tail_frac, dist_xyz_ps, prefix_tvt_slope etc |
| `gr` | ~35 | rolling means/stds, lag/lead 8 個, cumsum, gap_gr quantiles |
| `typewell` | ~22 | typewell GR stats, prefix_horizontal_vs_typewell stats, anchor_gr_diff 11 offsets, local_last200 features |
| `path` | ~12 | tw_path / tw_path_ease の min_absdiff / best_endpoint / top2_gap 6 stats × 2 曲線 |
| `beam` | ~25 (GPU enabled 時) | 6 configs × {delta, step} + gap + spread + mean/std + 3 cross-features |
| `formation` | ~30 | plane: anchor_b, b50, prefix_rmse/mae, tvt_formula, delta, delta50 × 6 formations + 4 stats |
| `row_ancc` | ~20 | dense_ancc, dense_std, dense_dist, dense_rmse, formation_row_*, spatial_vs_dense |
| `selfcorr` | ~9 | selfcorr_delta, selfcorr_score, trust, top2_gap, hyb_delta, beam_vs_sc, dense_vs_sc, plane_vs_sc |
| `offset` | ~22 | tdbc 11 offsets + tdsc 11 offsets |
| `pf_lite` | ~9 | pf_lite_delta, std, weight_sum, candidate_count, vs_dense/plane/sc/beam_cons, gr_abs_resid |
| **合計** | **~214** | (上下範囲 120-230 で curated、`SUPER_MIN_FEATURE_COUNT=120`, `SUPER_MAX_FEATURE_COUNT=230`) |

= "220" は **210-220 features の curated set** という命名。
