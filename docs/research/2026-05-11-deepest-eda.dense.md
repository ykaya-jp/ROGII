# Deepest EDA for ROGII (2026-05-11、参加者の誰よりも深く)

> 中央 deep EDA AG 担当。 ユーザー指示「軽量じゃなくてがっつりやれ、参加者の誰より EDA やって」 への応答。
> 公開 EDA (= pilkwang super stack 80 votes、 cdeotte starter 59 votes) + 既存 `first-principles.dense.md` (= subagent E、773 wells × 22 cols) を **構造的に超える** deep dive。
>
> 走行 environment: local CPU、 .venv、 メモリ 16 GB。 7 Phase で 776 wells × 全 CSV を全数走査。
> 生成 parquet 合計 11 件 (= `outputs/eda/deepest_eda/*.parquet`)、 計測 metric 130+ 種。
>
> ブランチ: `docs/deepest-eda-2026-05-11` (= `feat/phase-5-edge-r-online` から派生)。 commit 7 件、 全 `git add <file>` 限定。
>
> 関連:
> - `docs/research/first-principles.dense.md` (= 既存 22 col 計測、 本 doc の基盤)
> - `docs/research/data-spec.dense.md` (= Phase 1 EDA 確定)
> - `docs/research/top3-distill.dense.md` (= R/N kernel の 6 要素)
> - `docs/research/pilkwang-distill.dense.md` (= pilkwang SUPER STACK)
> - `docs/dev/2026-05-11-eda-findings.dense.md` (= 中央前回軽量 EDA、 22-col level)
> - `_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/...py`
> - `_research_kernels/cdeotte__eda-starter/...py`

---

## 0. 序 — 公開 EDA / first-principles との差分宣言

### 0.1 既存 EDA の scope

| source | scope | metric 数 | 視点 |
|---|---|---|---|
| pilkwang super stack | 776 well count + 統計 table + visualisation | ~30 | 全体俯瞰 + train/test 比較 |
| cdeotte starter | data loader + 視覚化 + baseline | ~20 | 教育的、 個別 well の signal |
| first-principles (= subagent E) | 773 wells × 22 cols + 6 観点 lower bound | 22 | 確率モデル定式化、 LB 上限推定 |
| 中央前回軽量 EDA (5/11 17:30) | 22-col の corr + 未活用識別 | 22 | redundancy 抽出 |

### 0.2 本 deepest EDA で追加した scope

| Phase | parquet | 新 metric | 計測単位 |
|---|---|---|---|
| 1 | `per-well-high-order.parquet` | 32 | GR の kurtosis/skewness/lag-1/lag-10/lag-50 + dTVT の高次 + trajectory (incl/azi/dogleg/z_dip) + formation boundary jump + visible-hidden boundary 近傍 |
| 2 | `per-well-outlier.parquet`, `md-dependent-dtvt.parquet` | 18 + 38k bin | 物理 formula residual の per-well detail + 末端 NCC self-similarity + MD offset 別 dTVT 分布 |
| 3 | `b-cluster-xy.parquet`, `cluster-xy-summary.parquet` | 14 + 40 cluster | per-well b_well と XY centroid の関係 + cluster 内 6 formation 一致度 |
| 4 | `tail-stats.parquet`, `gr-pca.parquet` | 33 + 12 | visible 末端 K=20/50/100/200 tail 統計 + cross-well GR PCA + KMeans cluster |
| 5 | `pseudo-test-errors.parquet` | 22 | 773 train wells を pseudo-test として 4 種 simple model error 計測 |
| 6 | `failure-correlations.csv` | 3×50 corr | 50 EDA feature × 3 error target の Spearman corr |
| 7 | `typewell-stats.parquet`, `formation-uniqueness.parquet` | 15 + 13 | typewell GR の structure + 6 formation top depth uniqueness |

**合計 130+ 新 metric**、 既存 22 metric を **6× 拡張**。 公開 EDA に **存在しない 4 件の核心発見** を抽出 (= §3 参照)。

---

## 1. 全 features の数理統計

### 1.1 per-well stats — 既存 22 col の上にさらに 32 metric

| metric | 計測 | p10 | p50 | p90 | min | max | 既存対応 |
|---|---|---|---|---|---|---|---|
| `gr_kurt` (Fisher) | n=776 | -0.54 | 0.27 | 2.52 | -1.48 | **18.80** | × 新 |
| `gr_skew` | n=776 | -0.19 | 0.30 | 0.88 | -1.11 | 2.75 | × 新 |
| `gr_ac1` | n=776 | 0.84 | **0.92** | 0.97 | 0.63 | 0.99 | × 新 (= GR 連続性) |
| `gr_ac10` | n=776 | 0.48 | 0.67 | 0.81 | 0.19 | 0.94 | × 新 (= 10-step lag) |
| `gr_ac50` | n=776 | 0.15 | 0.46 | 0.67 | -0.11 | 0.91 | × 新 (= 50-step lag, formation 識別?) |
| `dtvt_kurt` | n=773 | -1.71 | -1.37 | 1.78 | -1.85 | 1774 | × 新 (= max 1774 は数値計算 outlier) |
| `dtvt_skew` | n=773 | 0.06 | 0.38 | 0.97 | -41.8 | 12.5 | × 新 |
| `dtvt_ac1` | n=773 | 0.988 | **0.999** | 0.9998 | 0.009 | 0.9999 | △ `ar1_phi` と同等 |
| `d2tvt_std` | n=773 | 0.0081 | **0.0156** | 0.074 | 0.0067 | 5.0 | × 新 (= TVT 2 階差分 std) |
| `d2tvt_p95` | n=773 | 0.010 | **0.010** | 0.010 | 0.010 | 0.040 | × **新発見**: TVT が 0.01 ft discretization (= 量子化済) |

### 1.2 trajectory metric — 公開 EDA に皆無

| metric | p10 | p50 | p90 | 物理解釈 |
|---|---|---|---|---|
| `incl_mean` (deg from vertical) | 78.5 | **82.1** | 86.0 | well は ほぼ horizontal だが 4-8 deg の dip |
| `incl_std` | 15.9 | 21.8 | 25.5 | 各 well 内で incl が 大きく振れる |
| `azi_mean` (deg from north) | 137 | 244 | 333 | 主軸は SW (244°)、ばらつき大 |
| `azi_std` | 2.6 | **12.3** | 59.7 | well 内 azi はほぼ一定 |
| `dogleg_mean` (deg/100ft) | 48.8 | 50.2 | 51.4 | per-well 平均 50°/100ft (= **異常に大きい**、 単位確認要) |
| `dogleg_p95` | 81.8 | 97.0 | 98.8 | per-well 95% で 97° (= curve 急変) |
| `z_dip_mean` (dz/dmd) | -0.158 | **-0.107** | -0.052 | bit は 1 ft MD あたり 0.107 ft 下降 (= dip ~6°) |
| `z_dip_std` | 0.222 | 0.281 | 0.327 | dip 変動大 |

**ノート (dogleg)**: 50°/100ft の値は drilling engineering 上ありえない (= 通常 < 5°/100ft)、 実際は **連続点間の direction vector 内積角**を 100ft 正規化したもので、 当該物理的 dogleg severity とは異なる metric。 信号としては valid だが naming に注意。

### 1.3 typewell features — pilkwang EDA 部分カバー、本 doc で深掘り

`typewell-stats.parquet` (773 × 15):

| metric | p10 | p50 | p90 |
|---|---|---|---|
| `tw_n_rows` | 1250 | **1874** | 2986 |
| `tw_tvt_range` (ft) | 633 | 907 | 1104 |
| `tw_gr_mean` | 73.0 | 86.4 | 109.3 |
| `tw_gr_std` | 24.0 | 28.7 | 35.5 |
| `tw_geo_n_classes` | 6 | **6** | 10 |
| `tw_geo_top_count` | 269 | 337 | 625 |
| `tw_geo_class_gr_mean_spread` | 19.0 | 22.3 | 28.3 |
| `tw_tvt_step_p50` (ft) | 0.25 | 0.50 | 0.50 |

**geology label top-class 分布 (= 773 wells)**:
- `ANCC`: 631 (82%)
- `ASTNL`: 77 (10%)
- `EGFDL`: 36 (5%)
- **`OLMOS`: 28 (3.6%)** ← 公開 EDA で未言及、 5/12 plan 候補 (§4.7)
- `UPSN`: 1

### 1.4 物理 formula residual — formula は perfect、 問題は ANCC 復元

`pseudo-test-errors.parquet` で `tvt_formula = -Z + ANCC + b_well_median` を **hidden region で ANCC known** として走らせた:

| metric | p10 | p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| `formula_rmse` (ft) | **0.005** | **0.005** | 0.008 | 0.010 | 0.012 |
| `formula_mae` | 0.003 | 0.004 | 0.005 | 0.007 | 0.008 |
| `formula_max_err` | 0.020 | 0.030 | 0.040 | 0.040 | 0.040 |

つまり **ANCC が perfect に手に入れば RMSE 0.005 ft、 max 0.012 ft**。 LB 8 帯 vs LB 0.005 帯のギャップは **全部 ANCC plane-fit imputer + b_well 推定 + ε noise** で説明可能。

= **問題は ANCC 復元の正確性**。 first-principles §1.5 の Bayesian posterior で integral 取れば理論的に近づける。

### 1.5 物理 formula residual の per-well outlier

`b_resid_p99` (= per-well b_well residual の p99) は **全 wells で 0.04 ft** で完全一定。 これは:

- **TVT が 0.01 ft 単位で量子化されている** (= host による rounding)
- residual の 99 percentile は 4 ticks (= 0.04 ft) で頭打ち
- → 公開 EDA・first-principles で未指摘の発見

**含意**: predict も 0.01 ft 単位で **round-to-grid** すれば理論上 -ε ft 程度 LB 改善 (= Edge S と同型 strategy)。

---

## 2. failure pattern 分析

### 2.1 各 simple model の error 分布 (= 773 train wells を pseudo-test 化)

| model | p10 | p50 | p90 | p99 | max | 物理解釈 |
|---|---|---|---|---|---|---|
| `last-value extrapolation` | 5.0 | **10.7** | 23.0 | 47.8 | 70.6 | visible 末端の TVT を hidden 全行に放置 |
| `linear extrap from tail-100` | 8.8 | **36.6** | 135.3 | 252.2 | **1437** | 末端 slope で線形外挿、 急変で爆発 |
| `tvt_formula (ANCC known)` | 0.005 | **0.005** | 0.008 | 0.010 | 0.012 | ANCC 既知の理論下限 |
| `typewell GR-NCC naive match` | 218 | **313** | 439 | 607 | 762 | GR 値だけで 最近 TVT、 GR similarity の素 |

**重要観察**:
- last-value naive ですら p50 10.7 ft (= 公開 baseline 12.6 帯)
- linear extrap は **dangerous**: tail-100 slope を未来 4000 ft に投影で max 1437 ft
- typewell GR-NCC naive は worst (= GR 値が wells 間で normalize されてないので 313 ft の bias、 Beam Search や Self-NCC で形を見る必要)

### 2.2 MD-distance dependent error

last-value extrap で MD offset 別の abs mean error:

| MD offset (ft) | p50 | p90 | max |
|---|---|---|---|
| 100 | **1.32** | 4.16 | 40.5 |
| 500 | 4.95 | 15.44 | 63.4 |
| 1000 | 7.43 | 20.78 | 61.9 |
| 2000 | 8.48 | 25.60 | 75.6 |
| 4000 | **10.41** | 28.06 | 99.2 |

= MD 100 ft は p50 1.32 (= 直近は予測簡単)、 MD 4000 ft は p50 10.4 (= 公開 LB 10.0 帯と整合)、 **末端深部 p90 は 28 ft で大幅悪化**。

first-principles §2.4 の **`extrapolation-residual-curve` std (md=525: 9.35, md=4025: 18.83)** と整合 — naive last-value extrap が公開 LB の理論下限を提供している。

### 2.3 top-10 failure wells (= last_rmse 上位 10)

| well_id | last_rmse | n_hidden | n_visible | b_med | tail_50_dtvt_mean | tail_50_dtvt_std | visible_ratio | ar1_phi | dtvt_std |
|---|---|---|---|---|---|---|---|---|---|
| `1b1eba53` | **70.6** | 4655 | 2055 | NaN | 0.0245 | 0.0090 | 0.306 | 0.998 | 0.514 |
| `86454a6f` | **70.3** | 7964 | 1136 | 11855.20 | 0.0127 | 0.0044 | 0.125 | 0.996 | 0.216 |
| `a959858c` | **65.4** | 4404 | 1366 | 11756.39 | **0.569** | 0.0210 | 0.237 | 0.994 | 0.239 |
| `2fd68f7b` | 60.7 | 4730 | 1651 | 10370.34 | -0.0194 | 0.0031 | 0.259 | 0.998 | 0.422 |
| `5f4d2a52` | 56.9 | 5225 | 1887 | 10441.07 | -0.0224 | 0.0059 | 0.265 | 0.999 | 0.437 |
| `f88ddb26` | 51.4 | 4990 | 1595 | 11788.00 | -0.0196 | 0.0067 | 0.242 | 0.999 | 0.407 |
| `ba48188d` | 50.9 | 4166 | 1368 | 11855.20 | 0.0057 | 0.0049 | 0.247 | 0.998 | 0.448 |
| `389ae58f` | 49.8 | 6463 | 1738 | 11373.26 | **0.057** | 0.0045 | 0.212 | 0.978 | 0.512 |
| `f6d009f4` | 47.0 | 6715 | 1288 | 10559.18 | **0.321** | 0.027 | 0.161 | 0.9996 | 0.240 |
| `43e16325` | 45.8 | 3720 | 2125 | 10560.62 | -0.0086 | 0.0035 | 0.364 | 0.9996 | 0.482 |

**共通 signature**:
1. **`n_hidden > 4000`** (= 末端深部に行くほど extrapolation error 累積)
2. **`tail_50_dtvt_mean` の絶対値が p90 を超える wells** (= 0.057 〜 0.569)、 末端で急変
3. **`ar1_phi` 非常に高い** (= ほぼ random walk)
4. **`dtvt_std` 大きい** (= 全体的に dTVT 変動が大きい well)
5. **`b_round` cluster 集中**: `11855.20`, `10560.62`, `10934.02` 等の **大 cluster** (= **同 cluster 内で予測しやすい/しにくい が共起している = cluster 内 mixed**)

### 2.4 exp003 / exp005 v3 / exp007 失敗の data 由来推測

| 失敗 exp | 失敗 LB | data 由来仮説 (= 本 EDA からの裏付け) |
|---|---|---|
| exp003 (LB 17.51) | 自前 33 features 過剰 | `dtvt_kurt` が max 1774、 outlier wells の高次 features が外挿で爆発 (= `d2tvt_std max=5.0`) |
| exp005 v3 (LB 10.39) | Edge R online 悪化 +0.18 | tail K=100 の dTVT std p10=0.0041, p90=0.0176 = **per-well 4x spread**、 全 wells 一律の online step は cluster ベースで weight 必須 |
| exp007 (LB 10.68) | fold-misalign | typewell hash duplicate 13 group / b_well cluster 67 group = stratified fold 必要、 well_id GroupKFold だけは不十分 (`data-spec.dense.md` §10b で既に指摘) |

### 2.5 failure wells と correlation 強い feature (= Spearman top-5)

`last_rmse` (= last-value naive RMSE) の Spearman corr top 5:

| feature | spearman | n |
|---|---|---|
| `last_abs_at_500` | +0.348 | 810 (= 自己相関) |
| `n_hidden` | +0.187 | 811 |
| `visible_ratio` | -0.178 | 811 |
| `z_dip_mean` | +0.147 | 811 |
| `incl_mean` | +0.140 | 811 |

`lin_rmse` (= linear extrap RMSE) の top 5:

| feature | spearman |
|---|---|
| `tail_100_gr_std` | **+0.381** |
| `gr_pre_boundary_std` | +0.319 |
| `last_abs_at_500` | +0.314 |
| `n_hidden` | +0.280 |
| `visible_ratio` | -0.269 |

`formula_rmse` (= ANCC known) の top 5:

| feature | spearman |
|---|---|
| **`b_jump_max_abs`** | **+0.396** ★ |
| **`b_cross_formation_std_p50`** | **+0.395** ★ |
| `b_mad` | +0.302 |
| `y_med` | +0.275 |
| `b_quartile_diff` | +0.200 |

= **6 formation 間で b_well 推定が乖離する wells は formula も信頼できない**。 これは `independent-edges.dense.md` 案 G (MoE gating) の信号として直接利用可能。

---

## 3. 新 signal 発掘 — 公開 EDA を超える 4 件の核心発見

### 3.1 ★最重要発見: **b_well は 773 wells で たった 67 unique 値しか取らない**

`per-well-outlier.parquet` の `b_med` (= per-well median of `TVT + Z - ANCC` on visible region):

| 観点 | 値 |
|---|---|
| total train wells | 773 (= 766 ANCC-valid) |
| unique `b_med` (round to 0.01 ft) | **67** |
| 平均 cluster size | **11.6 wells / cluster** |
| 最大 cluster: `10762.50` | **71 wells** が共有 |
| 第 2 cluster: `10934.02` | 53 wells |
| 第 3 cluster: `11855.20` | 40 wells |
| 第 4 cluster: `11567.68` | 38 wells |

**cluster 内一致度** (= 同 cluster 内で 6 formation b 値の per-well std):
- p50 = **0.0000 ft** (= 完全一致)
- p90 = 0.0016 ft

= 同 cluster 内の 6 formation 値も完全に一致 → **cluster は geological zone identity** に対応 (= 同じ地下 geological structure を共有する wells 群)。

#### 3.1.1 cluster と XY centroid の関係

cluster 別の XY 分散:

| cluster `b_med` | n_wells | xy_diag_range (ft) | avg formation b std |
|---|---|---|---|
| 10762.50 | 71 | 25805 | 0.001 |
| 10934.02 | 53 | 26283 | 3e-12 (= 0) |
| 11855.20 | 40 | 25326 | 0.0004 |
| 11567.68 | 38 | 24807 | 0.0016 |

= 同 cluster の wells は **XY plane 上で 約 25000 ft 直径の zone** に分布 (= ~5 mi)。 これは Eagle Ford geological field の structure と整合。

= **`b_med` cluster は地理空間 (XY) の zone identifier** として機能。 KMeans XY centroid clustering を ANCC plane-fit の **per-cluster ridge alpha** に紐づければ predict 改善余地。

#### 3.1.2 cluster size 分布

| size | cluster 数 | 含 wells |
|---|---|---|
| 1 (singleton) | 23 | 23 |
| 2 (pair) | 3 | 6 |
| 3-5 | 2 | ~10 |
| 6-15 | 20 | ~200 |
| **16+** | **18** | **530** |

= 大 cluster 18 件が **wells の 69% (530/766)** をカバー → cluster 識別が決まれば multi-task learning で大幅改善余地。

#### 3.1.3 test 3 wells の cluster 帰属

| test well | nearest train (dist) | nearest b_med | cluster size |
|---|---|---|---|
| `000d7d20` (2983514, 1071407) | `ff0aea78` (481 ft) | **11373.26** | 8 wells |
| `00bbac68` (3008509, 1086906) | `155887bb` (1068 ft) | **11855.20** | 40 wells (= top-3) |
| `00e12e8b` (2969586, 1062517) | `42669188` (383 ft) | **11373.26** | 8 wells |

= 2/3 test wells は **同 cluster (11373.26)** に属し、 1 件は **超大 cluster (11855.20)** に属する。 5/12 plan で **cluster ID を直接 feature にする** だけで強力。

### 3.2 ★第 2 発見: train/test に同 well_id が完全一致で含まれている

**事実**: train 773 wells のうち test 3 wells (`000d7d20`, `00bbac68`, `00e12e8b`) は **train CSV にも存在し、 MD/X/Y/Z/GR/TVT_input/TVT/ANCC 全 columns が完全一致**。

- train `000d7d20`: 5278 行、 hidden 部 3836 行に **TVT 真値が完全に書かれている**
- sample_submission の `000d7d20_*` 3836 行 = train CSV の hidden TVT を読むだけで RMSE 0

しかし **cdeotte EDA line 147 が明示**: `"The visible test/ folder contains example data. In the official scoring run, Kaggle replaces it with the hidden test set."` — 公開 test/ は **example data のみ**、 official scoring 時には **hidden test set に置換される**。

**pilkwang EDA line 91 も同様**: `"public LB can be optimistic because test well ids overlap train wells"`、 line 291 `"Same-well diagnostics can be optimistic | Produce an all-train table and a public-clean table"` で **同 well_id の例文用 data**であり、 official LB は別 hidden set。

**含意 (= 戦略的に重要)**:
1. **直接 leakage は使えない** (= host が hidden set で置換)
2. ただし **train CSV 内の 3 wells を pseudo-test として model validation に活用** 可能 → submit せず CV で feature effect を測れる
3. CV-LB gap 監視に decisive (= 3 wells で local LB を完全再現可能、 ratio drift 検出に最適)

### 3.3 ★第 3 発見: TVT が **0.01 ft 単位で量子化** されている

`per-well-outlier.parquet` の `b_resid_p99` が **全 766 wells で完全に 0.04** で一定。 さらに:

- `d2tvt_p95` (= TVT 2 階差分の 95 percentile abs) が **p10/p50/p90/min 全部 0.010** (= 1 tick)
- max = 0.040 (= 4 ticks)

= TVT は **0.01 ft step quantization**、 dTVT は {0.00, 0.01, 0.02, ...} の整数倍。

**含意**:
- Edge S round-to-grid (= exp005 v2 で LB -0.114 改善) は **物理 inherent な quantization** に依存 → 強力 prior
- predict 0.01 ft 細かさで round すれば理論的 noise 削減 (= 既に effective)
- **TVT 単位が 0.01 ft 整数倍であることを model output layer に enforce** すれば +0.05〜0.10 改善余地

### 3.4 ★第 4 発見: GR cluster と b cluster は独立 (= purity 0.21)

`gr-pca.parquet` (= 776 wells × 10-dim GR PCA):

| metric | 値 |
|---|---|
| top-10 PC variance ratio | **0.484** (= GR profile の 48% を 10 dim で説明) |
| PC1 alone | 0.155 |
| PC2 | 0.080 |

**GR cluster (= KMeans n=40 on PCA top 10) と b cluster (= b_med round) の交叉**:
- median purity = **0.21** (= ランダム 1/n_clusters の倍程度)
- 0.5+ = strong link、 0.21 = **weak link / independent signal**

= **GR profile (= lithology fingerprint) と b_well cluster (= geological zone identity) は別 dimension の signal**。 両方を feature に投入する価値あり。

| test well | GR PCA cluster | b cluster |
|---|---|---|
| `000d7d20` | 37 (PC1=-5.47, PC2=-2.73) | 11373.26 |
| `00bbac68` | 15 (PC1=+6.87, PC2=+7.54) | 11855.20 |
| `00e12e8b` | 30 (PC1=+6.61, PC2=+14.53) | 11373.26 |

= test 3 wells で GR cluster は全部別 (= 37/15/30)、 b cluster は 2 種 (= 11373.26 / 11855.20) → **diverse coverage** がある。

### 3.5 (補助) 第 5 発見: typewell GR amplitude が horizontal の 1.25× (= scale 0.80)

`per-well-high-order.parquet` の `tw_scale_best` (= horizontal GR ≈ a × typewell_GR + b の OLS slope):

| metric | p10 | p50 | p90 |
|---|---|---|---|
| `tw_scale_best` | 0.66 | **0.80** | 0.95 |

= horizontal GR は典型的に typewell GR の **0.66-0.95×** (= horizontal は drilling fluid effect で attenuated)。 公開 R kernel の `affine_calibration` (= top3-distill §1.5) はこれを正規化している。

**含意**: GR-NCC matching で **scale_best を well 内で per-well 適用** (= affine cal) → typewell との signal alignment 改善。

### 3.6 (補助) 第 6 発見: tail_50_dir_sign が wells 間でほぼ完全 split

`tail-stats.parquet` の `tail_50_dir_sign` 分布:
- -1 (= 末端 50 行で TVT 下降): **376 wells (48.6%)**
- 0 (= flat): 2 wells
- +1 (= 末端 50 行で TVT 上昇): **398 wells (51.4%)**

= **末端 trajectory の方向は 50/50**。 hidden 部の dTVT 符号 prior は **per-well で別**。

**test wells**: 3 件全部 +1 (上昇方向)。 これは test set として **biased 標本** の可能性 → 全 wells 一律の last-value naive は **方向 prior を捨てている**、 per-well sign-aware modeling で改善余地 (= H1 visible 末端 dTVT sign を local prior にする提案、 `host-resources.dense.md`)。

### 3.7 (補助) 第 7 発見: 6 formation top per-well median は全 wells で unique

`formation-uniqueness.parquet`:

| formation | n | unique (round 0.01 ft) | range (ft) |
|---|---|---|---|
| `ANCC` | 766 | 765 | -10437 ~ -7503 |
| `ASTNU` | 773 | 771 | -10642 ~ -7710 |
| `ASTNL` | 773 | **773** | -10655 ~ -7815 |
| `EGFDU` | 773 | **773** | -10677 ~ -7905 |
| `EGFDL` | 772 | 771 | -10722 ~ -7932 |
| `BUDA` | 773 | **773** | -10894 ~ -8036 |

= 6 formation top depth は **per-well unique fingerprint** (= 同 cluster 内でも 0.01 ft 精度で全 wells 別)。
↔ **b_well は 67 cluster に discretize** されている (= 3.1)。

**逆算**: `b_well = TVT + Z - ANCC` という関係から、 `TVT - ANCC` の per-well 値が cluster 化されている (= Z の per-well 連続性が cluster discretization の起源を消す)。 これは **TVT 軸の原点が cluster 単位で choice されている** ことを示唆 (= ROGII discussion 698282 の "virtual reference line"、 cluster 内では同 reference line、 cluster 間で reference line が discrete shift)。

---

## 4. 5/12-13 plan 投入可能な new feature 候補 (= 上位 10 件)

> 中央軽量 EDA 5 件 (`docs/dev/2026-05-11-eda-findings.dense.md` §3) に加え、 さらに deep な 10 件を抽出。
> 推奨は出さない (= 主道原則 §11)、 トレードオフ表 + 選択軸 のみ提示。

| # | feature | 数理根拠 | LB 寄与推定 (ft) | 工数 (人日) | 依存 |
|---|---|---|---|---|---|
| **D1** | **b_well cluster ID (= 67 cluster)** | §3.1 で 67 cluster confirmed、 同 cluster 内 formation 値完全一致 = geological zone identity | -0.2 〜 -0.5 | 0.2 (= cluster ID round で生成、 plane-fit alpha を cluster 別に変える) | A (Imputer) |
| **D2** | **cluster size feature** (= 18 大 cluster vs singleton) | §3.1.2、 singleton 23 wells は学習 sample 不足、 cluster size 1 で loss weight × 0.5 検討 | -0.05 〜 -0.15 | 0.1 (= per-well loss weight 注入) | - |
| **D3** | **XY KMeans cluster (= geological zone proxy)** | §3.1.1、 XY 25000 ft 直径 zone、 cluster 別 plane-fit | -0.1 〜 -0.3 | 0.2 (= sklearn KMeans XY) | A |
| **D4** | **`b_jump_max_abs` (= 6 formation b 値の per-row std)** | §2.5 Spearman 0.396 vs formula_rmse、 MoE gating の最強信号 | -0.1 〜 -0.3 | 0.1 | - |
| **D5** | **`tail_50_dir_sign` (= 末端方向 prior)** | §3.6 で 50/50 split、 per-well sign-aware modeling | -0.05 〜 -0.15 | 0.1 | - |
| **D6** | **`gr_ac50` (= 50-step lag GR autocorr)** | §1.1、 p10=0.15 p90=0.67 = wells で大幅異、 formation 識別 signal | -0.05 〜 -0.15 | 0.1 | - |
| **D7** | **`tw_scale_best` (= horizontal/typewell GR scale)** | §3.5、 p50=0.80、 affine cal をモデル input 化 | -0.05 〜 -0.10 | 0.1 | - |
| **D8** | **`d2tvt_std` (= TVT 2 階差分 std)** | §1.1、 p50=0.0156、 curvature signal、 LGB に直接添加 | -0.05 〜 -0.15 | 0.1 | - |
| **D9** | **6 formation top per-well median (= 直接添加)** | §3.7、 per-well unique fingerprint、 train only だが pseudo-typewell 推定にも使える | -0.1 〜 -0.3 | 0.2 (= 6 列 × per-well median 計算) | - |
| **D10** | **`OLMOS` flag** (= typewell top class が OLMOS な 28 wells) | §1.3、 公開 EDA 未言及、 minority class 専用 model 検討余地 | -0.02 〜 -0.10 | 0.1 (= binary flag) | - |

### 4.1 中央軽量 EDA との重複/補完関係

| 中央軽量 (5/11 17:30) | 本 deepest | 関係 |
|---|---|---|
| `b_well_mean` (= 22-col stats) | D1 `b_well cluster ID` | **深化**: mean → 67 cluster の categorical |
| `b_well_drift_first_to_last` | D5 `tail_50_dir_sign` | **代替**: drift continuous → sign binary、 sign-aware modeling 用 |
| `tw_gr_resid_mean` | D7 `tw_scale_best` | **直交**: resid (= 加法) vs scale (= 乗法) |
| `gr_nan_frac` | (本 doc では同等指標なし) | 中央軽量で発見済、 並行 valid |
| `ar1_phi` 低 well | D6 `gr_ac50` | **直交**: dTVT AR(1) vs GR AR(50)、 別物 |

### 4.2 トレードオフ表

| 軸 | D1 (cluster ID) | D4 (b_jump_max_abs) | D9 (formation top med) |
|---|---|---|---|
| LB 寄与 (期待) | 中-大 | 中 | 中-大 |
| 工数 | 小 | 極小 | 小 |
| 既存 model への注入難度 | 小 (= 1 col) | 極小 (= 1 col) | 中 (= 6 col) |
| overfit risk | 中 (= 67 categorical、 fold で disjoint 保証要) | 小 | 小 |
| Edge との重複 | Edge Q stratified fold と相補 | Edge G MoE gating と同型 | tvt_formula と機能重複 |
| 公開上位の使用状況 | × 未使用 (= 公開上位は b_well を continuous で扱う) | × 未使用 | △ 部分使用 (= R kernel の `bw_{fn}` features と類似) |

### 4.3 選択軸 (= 5/12 plan で何を取るか)

- **新規 LB lift 重視** → D1 + D3 (= cluster + XY KMeans、 公開上位未使用 path)
- **既存 stack 内補強重視** → D4 + D8 (= GBM features に 1-2 col 追加するだけ)
- **MoE / multi-task 構造化重視** → D1 + D4 + D9 (= cluster ID で base 切替、 b_jump で gate、 formation top で aux supervision)
- **工数最小重視** → D4 + D5 + D8 + D10 (= 全部 0.1 人日、 4 col 追加で smoke 可能)

**判断は中央 (= 開発者) に委ねる**。

---

## 5. plot / 視覚化 — 軽量数値 summary

> context 制約のため plot は最小限。 主要 distribution は §1-3 の table で数値化済。
> 詳細 plot が必要なら `outputs/eda/deepest_eda/*.parquet` を pandas で読んで matplotlib に渡す。

### 5.1 cluster 数 vs 含 wells (= §3.1.2 を縦並べ可視化)

```
size 16+: ████████████████████████████ 530 wells / 766 = 69%
size  6-15: ██████████ 200 wells / 766 = 26%
size  3-5:  █ 10 wells / 766 = 1.3%
size    2:  █ 6 wells / 766 = 0.8%
size    1: █ 23 wells / 766 = 3.0% (singletons)
```

### 5.2 last_rmse 分布 (= §2.1)

```
RMSE ft:       0    5   10   15   20   25   30   40   50   60   70+
              |    |    |    |    |    |    |    |    |    |    |
percentile:   *    p10  p50  p70  p85  p90  p93  p97  p99  -    max
              0    5.0  10.7 15   20.5 23.0 27.7 38.4 47.8 -    70.6
```

### 5.3 MD-distance error curve (= §2.2)

```
ft offset:   100  500 1000 2000 4000
p50 abs err: 1.3  4.9  7.4  8.5 10.4
p90 abs err: 4.2 15.4 20.8 25.6 28.1
```

= **near-distance は強力に予測可能、 4000 ft 末端は ±10 ft が誤差幅**。

---

## 6. 残課題 — 時間切れで深掘りできなかった部分

1. **DTW (Dynamic Time Warping) cross-well GR similarity** — 簡易 GR PCA で代替したが、 真の DTW で wells 間の **lag 情報込みの similarity** を計測すれば精度向上 (= 工数 1 日)
2. **typewell-horizontal 完全 alignment** — `tw_scale_best` は OLS で粗計算、 NCC + Beam-style cross-correlation で **per-step lag** を計測すれば精度向上 (= 工数 0.5 日)
3. **6 formation top のクラスター構造** — `formation-uniqueness.parquet` で per-well unique と判明したが、 **大 cluster 内で 6 formation 間の関係**を線形/非線形構造で解析できていない (= 工数 1 日)
4. **test wells と train との XY distance < 1000 ft な neighbor 詳細** — §3.1.3 で確認したが、 **neighbor wells の visible/hidden pattern が test wells と一致するか** の検証はしていない (= 工数 0.5 日)
5. **NN-based dTVT prediction の lower bound** — 直接 NN を pseudo-test で走らせて per-well error を計測すれば、 LB 8 帯到達条件の bound を data 由来で確定可能 (= 工数 1-2 日)

これらは **5/12-13 plan の中で必要に応じて追加 EDA** として走らせる。

---

## 7. 完了報告 summary

### 7.1 「参加者の誰よりも深い」 EDA の **最も衝撃的な finding 1 件**

**`b_well` は 773 wells で たった 67 unique 値しか取らない** (= §3.1)。 公開 EDA / first-principles 全て `b_well` を **per-well 連続値**として扱っているが、 実際は **67 cluster の discrete identifier**。 同 cluster 内で 6 formation 値も完全一致 (= std 0) = **geological zone identity**。 test 3 wells は train cluster `11373.26` (2 件) と `11855.20` (1 件) に属する。

**LB 寄与**: cluster ID を categorical feature として 1 列追加で **-0.2 〜 -0.5 ft** 改善期待 (= 公開上位は continuous で扱っており、 categorical 化が新規 lift)。

### 7.2 5/12-13 plan に **immediately 投入すべき new feature 上位 3 件** (= 推奨せず軸のみ)

| # | feature | 軸 | LB 寄与 | 工数 |
|---|---|---|---|---|
| D1 | b_well cluster ID (= 67 categorical) | 新規 LB lift 重視 / overfit risk 中 / 工数 0.2 日 | -0.2 〜 -0.5 | 0.2 |
| D4 | b_jump_max_abs (= 6 formation b std) | 既存 GBM 1 列追加 / Spearman 0.40 vs formula_rmse / overfit risk 小 | -0.1 〜 -0.3 | 0.1 |
| D9 | 6 formation top per-well median | tvt_formula 機能重複 / per-well unique fingerprint / 工数 0.2 日 | -0.1 〜 -0.3 | 0.2 |

**選択軸**:
- 新規 LB lift 重視 → D1 + D3 (XY KMeans)
- 既存 stack 内補強 → D4 + D8
- MoE 構造化 → D1 + D4 + D9

判断は中央に委ねる。

### 7.3 残課題 (= §6 参照)

- DTW cross-well GR similarity
- typewell-horizontal 完全 alignment
- 6 formation cluster 内構造解析
- test wells neighbor 詳細
- NN-based lower bound 計測

---

## 8. 関連成果物 一覧 (= `outputs/eda/deepest_eda/`)

| file | rows | cols | 内容 |
|---|---|---|---|
| `per-well-high-order.parquet` | 776 | 32 | GR/dTVT/trajectory/formation/visible-boundary 高次 metric |
| `per-well-outlier.parquet` | 776 | 18 | 物理 formula residual + NCC self + b_well per-quartile |
| `md-dependent-dtvt.parquet` | 38218 | 8 | visible 末端 MD offset 別の dTVT 分布 |
| `b-cluster-xy.parquet` | 776 | 14 | per-well b_well + XY centroid |
| `cluster-xy-summary.parquet` | 40 | 10 | b cluster 別 XY range + formation 一致度 |
| `tail-stats.parquet` | 776 | 33 | visible 末端 K=20/50/100/200 tail 統計 |
| `gr-pca.parquet` | 776 | 13 | GR PCA top 10 + KMeans cluster 40 |
| `pseudo-test-errors.parquet` | 773 | 22 | 4 simple model の per-well RMSE/MAE/MD-distance |
| `typewell-stats.parquet` | 773 | 15 | typewell GR/geology label structure |
| `formation-uniqueness.parquet` | 773 | 13 | 6 formation top per-well median |
| `failure-correlations.csv` | 150 | 4 | 50 feature × 3 error target の Spearman |
| `summary-*.json` | - | - | 各 phase の per-metric summary |

合計 **11 parquet + 7 json + 1 csv**。 既存 first_principles 4 parquet と独立。

---

## 9. 関連 doc / 出典

- `docs/research/first-principles.dense.md` — 既存 22-col 計測、 本 doc が拡張
- `docs/research/data-spec.dense.md` — Phase 1 EDA 確定、 §10b typewell duplicate group 13 件
- `docs/research/top3-distill.dense.md` — R/N kernel の 6 要素 distill
- `docs/research/pilkwang-distill.dense.md` — pilkwang SUPER STACK 詳細
- `docs/dev/2026-05-11-eda-findings.dense.md` — 中央軽量 EDA (= 22-col 再観察)
- `_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/...py` — pilkwang EDA 公開 source
- `_research_kernels/cdeotte__eda-starter/...py` — cdeotte starter 公開 source (= line 147 で test set 置換明示)
- 出典 url:
  - https://www.kaggle.com/code/pilkwang/rogii-eda-v4-same-matrix-super-stack
  - https://www.kaggle.com/code/cdeotte/eda-starter
  - https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/rules
