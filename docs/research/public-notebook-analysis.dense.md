# 公開ノートブック分析 (Code セクション、2026-05-10 取得)

> Kaggle CLI `kaggle kernels list --competition rogii-wellbore-geology-prediction --sort-by scoreAscending` で取得した best score 順の上位 8 ノートブックをダウンロード (`_research_kernels/`) し、コードを直接読解した結果。
>
> このコンペで **絶対に外してはいけない核心 insight** が判明した。Plan を全面的に修正する。

## 1. 注目ノートブック一覧

| ノートブック | 著者 | LB / CV | 主アプローチ | votes |
|---|---|---|---|---|
| `score-10-081-score-lb-32-rank` | needless090 | **LB 10.081** (32位) | Beam×5 + PF×2 + 6-form plane-fit + Self-NCC + LGB×3 + CB×3 + TabICL×2 + Ridge stack | 19 |
| `rogii-super-solution-lb-top-3` | romantamrazov | **LB ~10.1** (Top 3) | numba-PF + 7 beams + plane-fit + LGB×3 + CB | 133 |
| `physics-informed-baseline` | karnakbaev | **LB 10.784** (Top 2 老ver) | hybrid: PF + Beam + 6-form + Self-NCC + Affine GR cal + 5 base + Ridge | 86 |
| `rogii-plane-fit-formation-top-knn` | konbu17 | **LB 11.912** | 6-form plane-fit + row-level KNN ANCC + Beam + LGB×3 + XGB + Ridge | 51 |
| `triple-signal-beam-search-dual-pf-lightgbm` | shinya | LB ~12.x | Triple Signal + Dual PF + LGB | 51 |
| `12-388-a-better-baseline-particle-filter` | shinya | **LB 12.388** | simple PF (S = TVT + Z 平滑化) | 11 |
| `lb-11-068-rogii-wellbore-geology-prediction` | tasmim | **LB 11.068** | Beam + plane-fit + GBM | 15 |
| `rogii-eda-v4-same-matrix-super-stack` | pilkwang | (大型) | Same-Matrix Super Stack | 80 |

## 2. 決定的 insight ★★★★★

### 2.1 物理関係 `TVT = -Z + ANCC + b_well`

**ほぼ完全な線形関係 (Pearson = -1.0000, resid_std ~ 0.007 ft)** が成立する。
出典: konbu17 ノートブック冒頭の docstring (`_research_kernels/konbu17__rogii-plane-fit-formation-top-knn/rogii-plane-fit-formation-top-knn.py:7-13`)

```
TVT = -1.0 * (Z - ANCC) + b_well
    = -Z + ANCC + b_well
```

- `Z`: bit の TVD (= 観測済み、全 well 全 row)
- `ANCC`: 地層 ANCC top の depth (= **train でのみ提供、test では NaN**)
- `b_well`: well-specific bias (= visible 区間で TVT_input から推定)

**含意**:
- ANCC を test wells でも imput できれば、TVT は formula で**ほぼ決まる**
- 我々の exp001 が破綻した根本原因はこれを knowing しなかったこと
- 6 formations (ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA) すべてで同様の formula が成立 (各 formation でわずかに異なる b_well)

### 2.2 Target は **residual** で予測する

**全 top notebooks** がこれをやっている:
```python
target_train = TVT - last_known_TVT   # = TVT - TVT_input.iloc[last_visible]
final_test_pred = last_known_TVT + model.predict(X_test_features)
```
出典: needless090 line 547, konbu17 line 26-32, romantamrazov の同等コード

絶対値 TVT を直接予測すると、TVT 値域が 9000-13000 ft なので RMSE 30+ になる (= 我々の exp001)。
Residual (差分) を予測するなら、TVT 変動量が well 内 ~750 ft なので RMSE 5-15 帯になる。

### 2.3 ANCC imputation の 2 段戦略

**FormationPlaneKNN** (`konbu17` 流): `K=10` 最近傍 wells の centroid から weighted 2D plane fit
```python
# Per-formation: TVT_col_imputed(X, Y) = a*X + b*Y + c
# weight = 1 / (distance + ε)
# 6 formations (ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA) で同様
```
- 各 train well の `(X_median, Y_median, FORMATION_median)` を centroid 抽出 (1 well 1 行)
- test row の (X, Y) で K=10 nearest non-self centroid を取り、weighted plane fit で formation impute
- LB 11.912 で plane fit RMSE ≈ 17 ft (vs IDW 47 ft, 2.7x 改善)

**DenseANCCImputer** (`needless090` 追加): row-level IDW
- 全 train wells から 60 pts/well (1 well あたり 60 行 down-sample) → 全 train で ~46k points
- KDTree で K=20 nearest non-self → IDW で ANCC fine-resolution impute

両者の差: plane fit は extrapolation 強い、IDW は interpolation 細やか。**両方使うのが Top 解の流儀**。

## 3. 重要技術カタログ

### 3.1 Particle Filter (PF) — Bayesian state estimation

**目的**: hidden 区間で TVT (or `S = TVT + Z`) を逐次推定。GR likelihood が重要観測。

**2 流儀**:
1. **PF on TVT Z-velocity** (`needless090.run_pf_z`)
   - state: (TVT, dTVT/dMD)
   - rate model: `dTVT/dMD ≈ β * dZ/dMD + ε` (visible で β を fit)
   - measurement: GR(row) ≈ tw_GR(TVT) (likelihood Gaussian)
   - 500 particles, momentum 0.998, resample threshold ESS < 0.5N

2. **PF on `S = TVT + Z`** (`shinya` 流, `needless090.run_pf_ancc` の発展)
   - **insight**: `S = TVT + Z` は drilling 影響を消した **geology-only** signal
   - state: (S, dS/dMD); 平滑、roughening 小さい
   - measurement: GR(row) ≈ tw_GR(S - Z) で likelihood
   - alpha 0.998 (rate inertia), pos noise 0.005, GR sigma 10-60 ft
   - Numba JIT で高速化 (romantamrazov)

### 3.2 Beam Search — discrete typewell-tie

**目的**: typewell TVT 軸上の最適 path を find (≒ DTW alignment の前向き推定)

**仕組み** (`needless090.beam_search` line 138-163):
- state: typewell index (= TVT 軸上の位置)
- transition: ±1 / 0 step (= TVT が ±1 unit 変動 / 同じ)
- emit cost: `(GR_horizontal - GR_typewell[idx])² / emit_scale`
- move cost: `move_cost * |delta|`
- beam で top-K state を維持

**5-7 configs で diversity**:
```python
BEAMS = [
    (10, 20.0, 144.0, 2, "cons"),   # tight, conservative
    (10,  8.0,  64.0, 2, "loose"),  # exploratory
    ( 8, 35.0, 220.0, 1, "vcons"),  # very tight
    (10, 14.0,  90.0, 5, "sm5"),    # smoother
    (20,  4.0,  36.0, 3, "vloose"), # very loose
    # romantamrazov adds 2 more (mid, stiff)
]
```

各 well で **5-7 paths** → mean / std / median を feature 化 (`beam_mean_d`, `beam_std_d`, `beam_med_d`)。

### 3.3 Self-Correlation NCC (Self-NCC)

**目的**: visible prefix の GR pattern が hidden で repeat していないか探索

**実装** (`needless090.self_corr_tvt` line 120-136):
- prefix: visible GR (length n_known), 既知の TVT 値あり
- query: 各 hidden row 中心の小窓 (±15 ft → 31 samples)
- normalized cross-correlation で prefix のどの部分と最も似てるかを探す → 対応 TVT が候補
- ROGII では **going up/down strata で repetitive な GR pattern** が頻発するため有効

`sc_score` (NCC max value, 0-1) と `sc_d = sc_raw - last_known_TVT` (delta) を feature 化

### 3.4 Affine GR Calibration

各 well で `kgr ≈ a * tw_GR(known_TVT) + b` の線形変換を least-squares で fit。
これで well 間の GR スケール差を統一。`a_cal`, `b_cal` を feature 化。

### 3.5 tw_diff features (3 anchors × ~11 offsets)

各 hidden row で:
- anchor: `last_known_TVT`, `beam_ref`, `sc_raw` の 3 点
- offset: `[-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80]` (or smaller/larger sets)
- feature: `gr_horizontal - tw_GR(anchor + offset)` (差分)

これにより model は「typewell のどの TVT 候補位置が真値か」を統計学習できる。

### 3.6 fade-in & alpha post-process

予測 residual を以下で smooth:
```python
delta_smoothed = alpha * delta * (1 - exp(-md_since / tau))
final_TVT = last_known_TVT + delta_smoothed
```

- `alpha`: 0.6-1.0 (overfit 抑制 scaling)
- `tau`: 30 / 60 / 120 / 250 / 500 / None (fade-in time constant)
- grid search で OOF RMSE 最小化

### 3.7 Savitzky-Golay smoothing

各 well で final pred を `savgol_filter(window=17, polyorder=3)` で smooth。
高周波 noise 除去で +0.1-0.3 程度の改善。

### 3.8 Ridge stacking with non-negative coefficients

```python
ridge = Ridge(alpha=1.0, fit_intercept=False, positive=True)
ridge.fit(np.column_stack([oof_lgb1, oof_lgb2, oof_lgb3, oof_xgb, oof_cb, oof_tabicl]), y_train)
final_test = ridge.predict(np.column_stack([test_lgb1, ...]))
```
- positive コンストレイントで physically meaningful な weights
- intercept なしで residual prediction との整合
- Simple avg と比較して `final = stack if stack < avg else avg` の安全策

### 3.9 TabICL (Tabular In-Context Learning)

`needless090` が採用。事前学習済みの transformer-based tabular model `TabICLRegressor`:
- top-50 features を quick LGB で選定
- context size 4096 × 5 seeds + 8192 × 1 seed の 2 variants
- GPU 必須、tabicl wheel + ckpt を Kaggle dataset で持ち込む

LGB/XGB/CB と diversity が高く ensemble に強く効く。

## 4. 横断パターン分析

| 共通技術 | 必須? | LB 改善寄与 |
|---|---|---|
| **target = TVT - last_known_TVT (residual)** | ★★★★★ 全員 | 30 → 15 帯 |
| **6-formation plane fit imputation** | ★★★★★ Top 5 全員 | 17 → 13 帯 |
| **tvt_formula = -Z + ANCC + b_well** | ★★★★★ 全員 (formula が物理的に正しい) | core 機能 |
| **Beam search (5-7 configs)** | ★★★★ Top 4 採用 | 13 → 11 帯 |
| **Particle Filter** | ★★★★ Top 4 採用 | 11 → 10 帯 |
| **Self-correlation NCC** | ★★★ needless090, karnakbaev | 11 → 10 帯 |
| **Affine GR calibration** | ★★★ Top 3 採用 | 0.1-0.3 |
| **tw_diff (3 anchors × N offsets)** | ★★★ needless090, karnakbaev | 0.3-0.5 |
| **GroupKFold(5) by well** | ★★★★★ 全員 | leak 防止 |
| **LGB×3 seeds + CB or XGB** | ★★★★★ 全員 | 0.2-0.5 |
| **Ridge stacking (positive coef, no intercept)** | ★★★★★ 全員 | 0.1-0.3 |
| **post-proc: alpha × fade-in tau + Savitzky-Golay** | ★★★★ Top 3 | 0.1-0.3 |
| **TabICL** | ★★ needless090 | 0.05-0.1 |

## 5. Plan の全面改定 (= 案 A の中身を強化)

我々の Plan は方向は合っていたが、**核心 (residual target + tvt_formula + ANCC plane-fit imputation + Beam + PF)** を見落としていた。Plan の案 A を以下のように再定義する:

### 案 A v2 (revised) — Public LB 10.x 級ベースライン

**目標**: LB 10-12 帯 (= 公開 baseline 上位を再現)

1. **Build imputers** (一度だけ):
   - `FormationPlaneKNN` (centroid K=10 plane fit, 6 formations)
   - `DenseANCCImputer` (row-level KNN K=20)

2. **Per-well features (~80-100)**:
   - **Closed-form `tvt_formula = -Z + ANCC_imputed + b_well`** ← core
   - residual target = `TVT - last_known_TVT`
   - GR rolling (5/21/51/101) + lag/lead (1/5/15/30) + diff (1/2 階)
   - Beam Search × 5 configs → mean / std / median delta
   - Particle Filter (PF_z, PF_ancc) → pts, std
   - Self-correlation NCC → sc_d, sc_score, sc_trust
   - Affine GR calibration → a_cal, b_cal
   - tw_diff (3 anchors × 11 offsets)
   - prefix stats (rmse, slope, length)
   - position (md_since, frac, dx, dy, dz)

3. **Models**:
   - LGB × 3 seeds (gpu device)
   - CatBoost × 1-3 seeds (GPU)
   - XGBoost × 1 (GPU optional)

4. **CV**: GroupKFold(5) by well
5. **Stacking**: Ridge non-negative, no intercept
6. **Post-process**: alpha × fade-in tau grid search + Savitzky-Golay smoothing

期待 LB: **10.5 - 12** (公開 baseline を再現できれば十分)

### 案 B v2 (revised) — Sequence Transformer (case A の上に積む)

Phase 4 で実装。residual target の逐次予測を Transformer で行う。Beam の path を soft-attention で expand したような構造。

### 案 C v2 (revised) — TabICL + 物理 prior

Phase 5 で TabICL を ensemble に追加 (top-50 feature subset)。

### 最終 ensemble

A v2 の OOF (LGB×3 + CB×3 + XGB) + B v2 (Sequence Transformer OOF) + C v2 (TabICL OOF) を Ridge stacking。

期待 LB: **8-10 (1 位射程)**

## 6. 直近のアクション

1. ✅ 公開ノートブックを download / 解読 / docs 化
2. ⬇ Plan の `docs/strategy/winning-strategy.dense.md` に "Phase 1.5 知見" セクションを追記
3. ⬇ exp002 を実装 (案 A v2 の minimum subset):
   - residual target
   - FormationPlaneKNN (ANCC のみで OK)
   - tvt_formula = -Z + ANCC_imputed + b_well_prefix
   - simple GR rolling + position + tvt_formula → LGB GroupKFold(5)
4. exp002 を実行 → 第 1 提出 → LB 確認
5. exp003 で Beam Search + PF + Self-NCC を追加
6. exp004 で multi-seed LGB + CB + Ridge stack
7. exp005 で post-process alpha × fade-in 追加
8. Phase 3 以降で B/C を積む

## 7. 出典 (download 済み source)

- `_research_kernels/needless090__score-10-081-score-lb-32-rank/score-10-081-score-lb-32-rank.py` (799 行)
- `_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py` (774 行)
- `_research_kernels/karnakbaevarthur__physics-informed-baseline/physics-informed-baseline.py` (1978 行)
- `_research_kernels/konbu17__rogii-plane-fit-formation-top-knn/rogii-plane-fit-formation-top-knn.py` (675 行)
- `_research_kernels/shinyanagai123__triple-signal-beam-search-dual-pf-lightgbm/triple-signal-beam-search-dual-pf-lightgbm.py` (925 行)
- `_research_kernels/shinyanagai123__12-388-a-better-baseline-particle-filter/12-388-a-better-baseline-particle-filter.py` (412 行)
- `_research_kernels/tasmim__lb-11-068-rogii-wellbore-geology-prediction/lb-11-068-rogii-wellbore-geology-prediction.py` (663 行)
- `_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py` (6142 行)

これら notebook は **公開** されているのでライセンス的に再利用可。ただし fork ではなく、**自前で再実装** することで知見を内在化する (Plan ルール: 案 C を独自に積み上げて差別化)。
