# ROGII First Principles 問題分解 (2026-05-10)

> 計測ソース: `notebooks/01_first_principles_eda.ipynb` + `notebooks/_first_principles_*.py` (subagent E 作成、773 wells 全数計測)
> 結果 parquet: `outputs/eda/first_principles/per-well-stats.parquet`, `per-well-extras.parquet`, `per-well-bound.parquet`, `extrapolation-residual-curve.parquet`
> 結果 JSON: `outputs/eda/first_principles/summary.json`, `summary-extras.json`, `summary-bound.json`
> 関係資産: `docs/research/data-spec.dense.md`, `top3-distill.dense.md`, `host-datasets.dense.md`

## 1. 生成過程モデル

### 1.1 物理層 (= geological process)

- 地層 (formation) は数千万〜億年スケールで stratify される。基準は **層相 (lithofacies)** と **化石指標 (biostratigraphy)**。本コンペ 6 formations (`ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA`) は推定 Eagle Ford / Austin Chalk / Buda Limestone 系列 (`docs/strategy/winning-strategy.dense.md` Phase 1.6 §C)
- drilling trajectory は X(s), Y(s), Z(s), MD(s) で表される空間曲線。`s` は arc length (= MD)
- 真の TVT (= depth below "stratigraphic zero" within current formation) は地質構造の dip 角度と well 位置で決まる
- typewell は近隣の縦井 (= vertical well) で同じ formation system を観測。同一 formation でも layer thickness は地点によって変動 (= stretch / squeeze)

### 1.2 統計層 (= measurement / noise)

- **GR (Gamma Ray) センサー**: 0.5-1.0 ft 解像度で離散観測、計測値は formation lithology の関数 + 計測 noise + depth-correlated artifact
- **ANCC (Active Near-bit Count)**: formation marker。test では NaN (= visible region でしか観測できない)
- **TVT_input visible region**: 既知の上方区間。ground truth subset (noise free 前提)
- **typewell (TVT, GR, Geology label)**: 近接縦井の reference

### 1.3 構造層 (= masked pattern)

- 全 776 wells で「visible block 1 + hidden block 1 (末端まで)」の単純パターン (`data-spec.dense.md`)
- visible 比率: median **0.27**, p10 **0.18**, p90 **0.35**
- hidden length: median **4840 ft**, p90 **6348 ft**, max **10052 ft**

### 1.4 確率モデル定式化

$$
\text{TVT}(s) = -Z(s) + \text{ANCC}(s) + b_{\text{well}}(s) + \varepsilon(s)
$$

実測関係 (= 公開 top の核心 finding):
- $\text{TVT} + Z - \text{ANCC} - b_{\text{well}}$ の residual は per-well で std **median 0.008 ft**, p90 **0.009 ft** (= 物理関係はほぼ完全)
- formula が **真値で実行されたとき** の RMSE = `formula_oracle_rmse_overall = 0.006 ft` (`per-well-bound.parquet`)

つまり問題は **「ANCC(s) と b_well(s) を hidden region で復元する」**に帰着。

ANCC(s) 復元の手段:
1. **plane fit imputer** (X, Y centroid 6 formation × KNN, FormationPlaneKNN) ← 公開 top の中核
2. **Bayesian / GP imputation** with uncertainty (= 独自 edge 候補)
3. **Diffusion / Generative posterior sampling** (= 独自 edge 候補)

b_well(s) 復元の手段:
1. **per-well constant** (= median または WLS による 1 値推定)
2. **MD-dependent linear / piecewise** (= 我々の計測 `b_drift_first_to_last_p50_abs = 0.00154 ft`、small but non-zero、独自 edge 候補)
3. **state-space (Kalman / Particle Filter)** on b_well sequence (独自 edge 候補)

### 1.5 Bayesian posterior の形

$$
p(\text{TVT}_h \mid Z_h, X_h, Y_h, \text{TVT}_v, \text{GR}_h, \text{GR}_v, \text{Geo}_v, \text{ANCC}_v) = \int p(\text{TVT}_h \mid Z_h, \widehat{\text{ANCC}}_h, \widehat{b}_{\text{well},h}) \, p(\widehat{\text{ANCC}}_h, \widehat{b}_{\text{well},h} \mid \cdot) \, d\widehat{\text{ANCC}} \, d\widehat{b}
$$

公開 top は **point estimate** で進めている (= integration を skip)。**uncertainty を保持して decoder で marginalize** すれば理論的には posterior に近づける = 独自 edge の理論的根拠。

## 2. Irreducible Error 上限見積もり

> 「LB top 10.0 帯と理論下限の距離」を 6 つの観点で推定。

### 2.1 計測値サマリ (`per-well-stats.parquet`, n=773)

| metric | p10 | p50 | p90 | max |
|---|---|---|---|---|
| visible_ratio | 0.180 | 0.260 | 0.346 | 0.802 |
| hidden_len (ft) | 3263 | 4840 | 6349 | 10052 |
| gr_noise_std | 6.17 | 8.38 | 11.70 | 15.55 |
| dtvt_std | - | 0.417 | 0.475 | 3.57 |
| dtvt_p95 | - | 1.01 | 1.24 | 3.66 |
| **b_well_resid_std** | **0.0074** | **0.0083** | **0.0094** | **0.0111** |
| b_well_drift_first_to_last (abs) | - | 0.00154 | 0.0042 | 0.0147 |
| tw_gr_resid_std | 9.84 | 13.89 | 18.88 | 63.48 |
| tw_gr_resid_mean_abs | - | 10.13 | 13.77 | 23.18 |
| ar1_phi (dTVT 自己相関) | - | **0.999** | 1.0 | 1.0 |
| ar1_eps_std | - | 0.0156 | 0.074 | 3.57 |

### 2.2 計測値サマリ (`per-well-extras.parquet`, n=773)

| metric | p10 | p50 | p90 | max |
|---|---|---|---|---|
| best_lag_ft (typewell vs horizontal) | - | -0.18 | 0.5 | 15.5 |
| min_resid_std (= TVT alignment minimum residual) | 9.18 | **12.68** | 16.98 | 29.03 |
| b_drift_first_last (abs) | - | 0.00169 | 0.00466 | 0.01471 |
| b_drift_per_ft (abs) | - | 1e-6 | 3e-6 | 1.3e-5 |
| tail_dtvt_std (visible 末端 50 ft) | - | **0.015** | 0.026 | 0.279 |

### 2.3 計測値サマリ (`per-well-bound.parquet`, n=773)

| metric | p10 | p50 | p90 | max |
|---|---|---|---|---|
| **gr_cr_lb_1pt** (GR Cramer-Rao LB, 1 obs) | 4.09 | **6.38** | 8.77 | 19.15 |
| gr_slope_p50 (GR slope median) | - | 1.41 | 2.08 | 3.36 |
| gr_noise_std | - | 9.02 | 12.13 | 16.19 |
| **formula_oracle_rmse** | **0.0052** | **0.0055** | **0.0077** | **0.0118** |
| **gr_match_synth_rmse** (GR alone match → TVT) | 187.7 | **315.2** | 478.3 | 785.4 |
| n_holdout (per-well 計測区間) | - | 136 | 193 | 200 |

### 2.4 Extrapolation residual curve (= **核心 1 表**)

`extrapolation-residual-curve.parquet`: visible 末端から MD `md_since_center` ft 先までの residual std (50 ft binning, n=773 wells × 各 bin = 30k-39k 観測)

| MD since center (ft) | n | mean | **std** | p95 abs |
|---|---|---|---|---|
| 25 (= md=25) | 37877 | 0.04 | **0.94** | 1.68 |
| 75 | 38650 | 0.10 | **2.33** | 4.13 |
| 125 | 38650 | 0.15 | **3.54** | 6.50 |
| 225 | 38650 | 0.29 | 5.50 | 10.63 |
| 525 | 38600 | 0.67 | **9.35** | 18.77 |
| 1025 | 38500 | 1.45 | **12.68** | - |
| 2025 | 28000 | 3.20 | **15.39** | - |
| 4025 | 12000 | 6.15 | **18.83** | - |
| 9975 | 116 | -16.41 | 5.11 | 23.17 |
| 10025 | 69 | -16.49 | 4.81 | 24.61 |

**重要観察 1**: MD 525 ft 先で std 9.35 ft (= **公開 top LB 10.0 帯 = この距離での平均誤差**)。
**重要観察 2**: MD 4025 ft 末端で std 18.83 ft (= 末端ほど誤差が膨らむ)。
**重要観察 3**: MD 9975-10025 ft では mean が **-16 と systematic bias**。サンプル数少ない (n=116-69) が、**末端深部での systematic bias** が存在する可能性。

### 2.5 RMSE lower bound の推論

評価 metric は **全 hidden region の RMSE**。各 well の hidden length 分布 (`hidden_len_p50 = 4840 ft`) と extrapolation curve から:

$$
\text{RMSE}^{2} \approx \frac{1}{L} \int_{0}^{L} \sigma^2(s) \, ds
$$

$L$ = hidden length, $\sigma(s)$ = MD-based extrapolation std curve。
median well (L=4840) で curve を積分:

| 区間 | std (ft) | 寄与 (var × len) |
|---|---|---|
| 0-500 ft (10%) | ~5 | 12,500 |
| 500-1000 ft (10%) | ~10 | 50,000 |
| 1000-2000 ft (20%) | ~13 | 169,000 × 2 = 338,000 |
| 2000-4000 ft (40%) | ~17 | 289,000 × 4 = 1,156,000 |
| 4000-4840 ft (17%) | ~19 | 361,000 × 1.7 = 614,000 |
| 合計 var × ft | - | ~2.17M |
| **RMSE 推定** | - | **√(2.17M / 4840) ≈ 21.2 ft** |

**問題**: 単純積分すると LB top 10.0 とのズレが大きい。これは:
1. extrapolation curve が **per-step independent** に std を測っているが、公開 top は **typewell + GR + Beam Search** で MD 距離による std 増大を抑えている
2. つまり curve std 9.35 (md=525) は **何の補正もしない naive last-value extrap** の指標
3. 公開 top は md=525 で std ~5 ft くらいに圧縮できている (= LB 10.0 帯の説明)

### 2.6 LB 8 帯到達の構造的条件

$\text{RMSE} = 8$ を達成するには、上記積分が $8^2 \times 4840 \approx 310{,}000$ 以下。
具体的には **MD 4000 ft 末端での std を 12 ft 以下** に圧縮する必要 (現状の naive curve では 18.83 ft, 公開 top でも約 12-15 ft 推定)。

つまり **MD 2000-4000 ft 末端の std 改善** が 1 位戦略の核心領域。

### 2.7 公開 top が捕捉していない情報源 (= 独自 edge の出現位置)

計測から見える **3 つの未使用情報源**:

#### A. **MD 末端の systematic bias** (extrap curve mean が -16 で大きい)
末端深部で平均的に under-estimate されている。**末端を別 task として扱う 2-stage model** または **末端 regularization loss** で改善余地。

#### B. **`tw_gr_resid_std_p90 = 18.88` の per-well variability**
typewell ↔ horizontal mismatch が大きい wells が存在 (max 63.48)。これらは **typewell に重み付けない / typewell の uncertainty を model 入力に渡す** approach で改善余地。

#### C. **`ar1_phi_p50 = 0.999` の強烈 auto-correlation**
dTVT が ほぼ random walk。**state-space model (Kalman / Particle Filter on dTVT)** で AR(1) 構造を陽に model 化すれば、公開 top の i.i.d. 仮定を超えられる。

#### D. **`b_drift_first_last_p50_abs = 0.00154` の小さい drift**
b_well を constant でなく **MD-linear function** で fit すると微改善 (推定 -0.05〜-0.10 RMSE)。公開 top は constant median + WLS。

#### E. **`visible_ratio_p10 = 0.18` の per-well 大変動**
visible 18% の wells と 35% の wells は問題難度が大きく違う。**per-well adaptive model weight** (= visible_ratio に基づく LGB×NN weighting) で改善余地。

#### F. **`formula_oracle_rmse_p50 = 0.006` から逆算**
formula が perfect なら RMSE 0。残る誤差は **ANCC imputation + b_well estimation**。**Bayesian / Diffusion で uncertainty quantification** すれば、最後の 0.5-1.0 ft 改善余地。

## 3. 我々の到達目標 (= 8 帯 vs 6 帯 の妥当性)

| 目標 | 必要な MD 4000 ft 末端 std | 必要な技術 | 実現性 |
|---|---|---|---|
| LB 12 帯 (公開 baseline) | ~15 ft | tvt_formula + LGB residual | ✓ 実装済 (exp002, CV 13.82) |
| LB 10 帯 (公開 top) | ~12 ft | + Beam + PF + multi-scale NCC | ✓ 移植可能 (top3-distill §7) |
| **LB 8 帯 (1 位射程)** | **~10 ft** | + 上記独自 edge A/B/C/D/E/F のうち 2-3 個 | **要独自 edge** |
| LB 6 帯 (理論的限界に肉薄) | ~7 ft | + foundation model pretrain + ensemble + post-proc | 開発工数 4-5 週 |

**判断**: コンペ残 86 日 = LB 8 帯狙いが現実、LB 6 帯は射程外 (= 開発工数考慮)。

## 4. 計測コード位置

- `notebooks/_first_principles_measure.py` (8.9KB): per-well stats / GR noise / b_well drift / typewell GR resid / ar1 phi
- `notebooks/_first_principles_extras.py` (9.3KB): typewell vs horizontal lag / min resid / b drift / tail dTVT / extrap curve
- `notebooks/_first_principles_lower_bound.py` (10.6KB): GR Cramer-Rao LB / formula oracle / GR match synth
- `notebooks/01_first_principles_eda.ipynb` (12KB): notebook 統合
- `outputs/eda/first_principles/*.parquet` (4 files, 268KB): 計測結果 (773 wells × 22 / 7 / 9 cols)
- `outputs/eda/first_principles/*.json` (3 files): summary

## 5. 主要 Reference

- `docs/research/data-spec.dense.md` — 既存 data spec
- `docs/research/top3-distill.dense.md` — 公開 top の手法 distill
- `_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/...py` — 公開 Top 3 コード
- `_research_kernels/needless090__score-10-081-score-lb-32-rank/...py` — 公開 LB 10.081 コード

## 6. 派生

この first-principles を踏まえた **独自 edge 候補** は `docs/research/independent-edges.dense.md`。
戦略 doc 案 A/B/C との突合は `docs/research/strategy-critique.dense.md`。
