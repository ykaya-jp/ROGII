# 2026-05-11 EDA Findings (= per-well-stats.parquet 軽量再観察)

> ユーザー指示「暇なら EDA」 への中央実行。`outputs/eda/first_principles/per-well-stats.parquet` (773 wells × 22 col) を未活用 column + 高 correlation の 2 視点で再観察。
> 設計原則: 軽量 EDA は local OK (= `docs/dev/remote-migration-2026-05-11.md` §1)、本 observation は 145 KB parquet で remote migration rule 抵触なし。

## 1. 未活用 column candidate (= 13 件、5/12 plan 候補)

| col | p10 | p50 | p90 | min | max | 既知活用 | 5/12 候補 |
|---|---|---|---|---|---|---|---|
| dtvt_std | 0.36 | 0.42 | 0.48 | 0.18 | - | × | 案 D Kalman の σ_ε 入力 (= ar1_eps_std と高 corr で redundant) |
| dtvt_p95 | 0.94 | 1.01 | 1.24 | 0.55 | - | × | fat-tail signal (= Huber δ tuning 根拠) |
| dtvt_mean_abs | 0.34 | 0.44 | 0.51 | 0.14 | - | × | per-well 平均変動 (= sample_weight 候補) |
| **b_well_mean** | 10441 | **10934** | 12118 | 9603 | - | × | **per-well b_well 値** = TVT 値域とほぼ同じ、強力 feature 候補 |
| **b_well_drift_first_to_last** | **-0.0022** | 0.0004 | **0.0042** | -0.0148 | - | × | **drift 小だが non-zero**、案 H (MD-linear b_well) の根拠 |
| tw_gr_resid_mean_abs | 3.30 | 10.13 | 13.77 | - | - | × | typewell-horizontal GR mismatch (= absolute) |
| **tw_gr_resid_mean** | **-17.75** | -0.62 | 2.76 | - | - | × | **signed mismatch、Edge M (visible-as-typewell) で改善余地** |
| ar1_eps_std | 0.0067 | 0.0156 | 0.074 | - | - | △ (Kalman) | dtvt_std と高 corr (= redundant) |
| gr_mean | 44.08 | 88.12 | 110.98 | - | - | × | per-well GR baseline (= normalize 候補) |
| gr_std | 13.16 | 22.99 | 30.04 | - | - | × | per-well GR amplitude |
| **gr_nan_frac** | **0.0036** | 0.24 | **0.46** | - | 0.73 | × | **NaN 比率 wells 間で大きく差、imputation strategy 影響** |
| tvt_range | 158.93 | 734.47 | 940.80 | - | - | × | per-well TVT 値域 |
| z_range | 197.84 | 787.84 | 1016.35 | - | - | × | per-well Z 値域 |

## 2. 高 correlation pair (= Pearson |r| > 0.7、redundancy 排除候補)

| col1 | col2 | r | 解釈 |
|---|---|---|---|
| n_rows | hidden_len | +0.986 | hidden_len = visible 後の残り、当然 |
| **dtvt_std** | **ar1_eps_std** | **+0.885** | **Kalman ε noise と dtvt 全体 std はほぼ等価** = 案 D Kalman の signature が dtvt_std で代替可 |
| **ar1_phi** | **ar1_eps_std** | **-0.904** | **AR(1) phi 高 = ε std 小** (= random walk closer)、これは AR(1) 構造の native property |
| n_rows | visible_ratio | -0.783 | well 長い → visible 比率 小 (= 物理的) |
| hidden_len | visible_ratio | -0.858 | 同上 |
| dtvt_mean_abs | tvt_range | +0.859 | 変動大 well = range 大 |
| tw_gr_resid_std | tw_gr_resid_mean_abs | +0.897 | std と mean abs はほぼ同等 metric |
| tvt_range | z_range | +0.756 | TVT-Z 物理関係から当然 |
| n_visible | tvt_range | +0.791 | visible 長 → range 大 |

## 3. 5/12 plan に組み込み可能な未活用 feature (= 上位 5 件)

| # | feature | 数理根拠 | LB 寄与推定 | 工数 |
|---|---|---|---|---|
| 1 | **b_well_mean** (= per-well b_well 値) | 物理 formula 直接、TVT 値域とほぼ同じ範囲 (= 9603-12118) で **強力 feature** | -0.1〜-0.3 ft | 0.1 日 (= LGB に 1 列追加) |
| 2 | **b_well_drift_first_to_last** | 案 H (= MD-linear b_well) の **データ駆動 root** (= p10 -0.0022 p90 0.0042 で非ゼロ) | -0.05〜-0.15 | 0.1 日 |
| 3 | **tw_gr_resid_mean** | signed mismatch、Edge M (visible-as-typewell) の **重要性 signal** | -0.05〜-0.15 | 0.1 日 |
| 4 | **gr_nan_frac** | wells 間で 0.004 〜 0.73 と大幅異、**per-well adaptive imputer** の signal (= 案 G MoE gating) | -0.1〜-0.3 | 0.5 日 (= 案 G feature) |
| 5 | **ar1_phi 低 well の検出** | p10 = 0.009 = **random walk でない well 存在**、案 D Kalman 効かない segmentation | -0.05〜-0.2 | 0.3 日 (= Kalman fallback 設計) |

## 4. 既存 features の redundancy = Ridge meta weight 配分の改善余地

旧 stack (= exp008 v2 で verified 9.957):
- karnakbaev 5 base (= LGB×3 + XGB + CB) で **Pearson > 0.7** 同士 (= 推定、kernel log の Ridge weights から推測):
  - LGB lr=0.05 と LGB lr=0.1 (= 同 base、lr 違いのみ) = redundant
  - karnakbaev LGB と 自前 LGB (= 同 features) = redundant (= exp007 で verify)

= **5/12 plan で base 選定**:
- karnakbaev XGB + CB (= 異 algorithm)、LGB lr=0.05 = 3 base のみで Ridge meta 効果最大化
- exp008 v3 (= subagent T 改修、Multi-seed MEDIAN) で seed diversity 強化 = redundancy 緩和

## 5. 直近の next action

1. **5/12 sub plan (= exp011-013)** で上記未活用 feature 5 件のうち 上位 2 件 (= b_well_mean + b_well_drift) を加算
2. **5/13 paradigm (= Mamba + PySR)** で gr_nan_frac segmentation を gating に使う (= 案 G MoE 統合候補)
3. **subagent AF re-dispatch** (= API rate limit 復活後、公開 24h sweep)

## 6. 関連 doc

- `outputs/eda/first_principles/per-well-stats.parquet` — EDA source
- `docs/research/first-principles.dense.md` §2 — 既計測値 + 5 component 分析
- `docs/research/mathematical-formulation.dense.md` §0.2 — 物理 formula
- `docs/dev/submission-postmortems.dense.md` — verified LB data
- `docs/dev/remote-migration-2026-05-11.md` — design rule (= 軽量 EDA は local OK)
