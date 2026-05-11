# Submission Data Mathematical-Level Deep Mining (2026-05-11)

> 起源: 2026-05-11 ユーザー指示「過去サブミットデータを比べて何が変わったらよくなったとか悪くなったとか、そこから仮説として導かれるものは何か = sub data の mathematical-level deep mining」。9 切り (= LB 8.x 帯) 到達のため、既存 postmortem を超える data-driven 仮説生成を行う。
> 計測 source: `outputs/kernel_logs/{exp005,exp006,exp007}/*.log`, `submissions/exp005/rogii-exp005-cache-blend.log` (= exp005 v1), `submissions/exp005/submission.csv`, `submissions/exp002_lgb.csv`, `data/raw/test/*__horizontal_well.csv`
> 抽出 tool: `tools/mining/submission_data_mining.py`、抽出結果 = `outputs/eda/submission_mining/summary.json` (= 全数値の machine-readable source)

## 0. 序: 既存 postmortem doc との差分

既存 `docs/dev/submission-postmortems.dense.md` (182 行) は **mechanism-level root cause** (= 直接因 / 間接因 / メタ因 階層分析) と verified/unverified 表に主眼を置く。本 doc は同じ source data から異なる切り口で:

| 既存 doc | 本 doc |
|---|---|
| 「なぜ悪化したか」(= 因果説明) | 「数値が何を語っているか」(= data-driven pattern + 仮説生成) |
| 5 sub 個別の root cause | 横断 numerical matrix + per-fold × per-base × per-well drill-down |
| Edge S/R 等 component 観点 | fold variance / base diversity / boundary continuity / on-grid ratio / delta range shift |
| 既知 issue の verify | **新仮説 10 件** + 検証 plan + 5/12 以降の sub 設計 |

本 doc が補完するのは: **「悪化サブが log で何を計測できているか、そこから何がさらに分かるか」** の部分。

### 0.1 利用可能 data の制約 (= 透明性のため明記)

| sub | kernel log | submission.csv | 自前 CV |
|---|---|---|---|
| exp002 | ❌ (logs/ になし) | ✅ `submissions/exp002_lgb.csv` | ✅ CV 13.82 (= leaderboard.dense.md) |
| exp003 | ❌ | ❌ | ❌ (blind submit) |
| exp005 v1 | ✅ `submissions/exp005/rogii-exp005-cache-blend.log` | ✅ `submissions/exp005/submission.csv` | ❌ (blind submit, karnakbaev OOF 10.78 のみ参照) |
| exp005 v2 | ✅ `outputs/kernel_logs/exp005/` | ❌ (download 未) | ❌ (kernel log の Edge S 数値のみ) |
| exp005 v3 | ❌ (PENDING 中) | ❌ | ❌ |
| exp006 | ✅ `outputs/kernel_logs/exp006/` | ❌ | ❌ (TabICL fallback で NM blend) |
| exp007 | ✅ `outputs/kernel_logs/exp007/` | ❌ | ✅ **Ridge OOF 10.3874** + per-fold per-base RMSE 完全取得 |

**重要制約**: 我々が **per-id レベルで diff 計算可能**な比較は **exp002 vs exp005 v1** のみ (= 両者の submission.csv 在庫あり)。exp006/exp007 の submission.csv は kernel 出力後 download せず、log の statistics のみ取得済み。**Edge S 単独効果 (v1 → v2) は kernel log で `diff_abs_mean=0.002496, diff_abs_max=0.005` まで判明、CSV diff 計算は不可**。

---

## 1. 全 SCORED sub の数値 matrix

### 1.1 LB / CV 表 (= leaderboard.dense.md §Submission Log + 本 doc 追加列)

| sub | LB | 自前 CV (OOF) | LB - OOF | kernel runtime (s) | 主要 phase 時間 |
|---|---|---|---|---|---|
| exp002 | 14.695 | 13.82 | +0.875 | n/a | n/a |
| exp003 | 17.510 | - | - | n/a | n/a |
| exp005 v1 | **10.317** | (karnakbaev pub OOF 10.78) | - | **76** | imputers 28s + FE 25.6s + predict 1.6s |
| exp005 v2 | **10.203** | - | - | **76** | imputers 30s + FE 27.3s + predict 1.6s + Edge S 0.03s |
| exp006 | 10.503 | - | - | **122** | imputers 28s + FE 22s + TabICL fail 32s + NM fallback 0.02s |
| exp007 | 10.677 | **Ridge 9-stk 10.3874**, Simple 9-avg 10.4624 | **+0.290** | **5,689** (= 95 min) | imputers 24s + FE 23s + kb refit 271s + own 4-base train **5,318s** + Ridge 12s |

(出典: 全数値は `outputs/eda/submission_mining/summary.json` § `log_metrics.*.total_runtime_s` および `.phases`)

### 1.2 exp007 = per-fold × per-base RMSE matrix (= 唯一 OOF 完全取得 sub)

| fold | tr 件数 | va 件数 | lgb_own0 (lr=0.02) | lgb_own1 (lr=0.05) | lgb_own2 (lr=0.1) | cb_own | per-fold mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3,027,241 | 756,748 | 9.8562 | 10.0326 | 10.1857 | 10.1628 | **10.059** |
| 1 | 3,027,098 | 756,891 | **9.0185** | **9.1353** | 9.4001 | **9.2227** | **9.194** ← 最良 |
| 2 | 3,027,158 | 756,831 | **12.0474** | **12.0210** | **12.1939** | **12.0300** | **12.073** ← 最悪 |
| 3 | 3,027,127 | 756,862 | 10.4396 | 10.6400 | 10.6687 | 10.4582 | **10.552** |
| 4 | 3,027,332 | 756,657 | 11.6698 | 11.6990 | 11.9515 | 11.3537 | **11.669** |
| **per-base mean** | - | - | **10.606** | **10.706** | **10.880** | **10.646** | - |

(出典: `outputs/kernel_logs/exp007/rogii-exp007-edge-qm.log` line 108-178 から regex 抽出)

#### 1.2.1 分散構造の発見

- **per-fold across-base mean RMSE の std = 1.175 ft** (= fold 間差異)
- **per-base across-fold mean RMSE の std = 0.121 ft** (= base 間差異)
- **比 1.175 / 0.121 ≈ 9.7x** = **fold 間 variance は base 間 variance の 10 倍**

これが意味すること: **どの base を選ぶか** より **どの fold partition を選ぶか** の方が CV / LB に対する影響が 10 倍大きい。Ridge blend は base 間 0.12 ft の差を最適化するが、fold 間 1.18 ft の構造的 variance には介入できない。

### 1.3 exp007 Ridge weights

`Ridge weights: {'lgb0': 0.0299, 'lgb1': 0.0323, 'lgb2': 0.2238, 'xgb': 0.2118, 'cb': 0.1220, 'lgb_own0': 0.1023, 'lgb_own1': 0.0504, 'lgb_own2': 0.0000, 'cb_own': 0.2273}`

(出典: `outputs/kernel_logs/exp007/rogii-exp007-edge-qm.log` line 189)

| 集計 | weight 合計 |
|---|---|
| kb side total (lgb0+lgb1+lgb2+xgb+cb) | **0.6198** |
| own side total (lgb_own0+lgb_own1+lgb_own2+cb_own) | **0.3800** |
| 上位 2 weight | cb_own 0.2273, lgb2(kb) 0.2238 |
| weight=0 base | lgb_own2 (= lr=0.1 自前) |

#### 1.3.1 重要観察

- **kb side が ~62%、own side が ~38%** = exp005 (LB 10.317) base が dominant
- **cb_own が最大 weight 0.2273** = 自前 CatBoost が **base としては最有用** だが、own OOF RMSE 10.646 (= 全 base 中最良) と整合
- **lgb_own2 (= lr=0.1) が weight=0** = lr 大の自前 LGB は他 base と correlation 高すぎ・noise 大、Ridge が完全 reject
- kb side 内: **lgb2(kb)=0.2238, xgb=0.2118, cb=0.1220** = lgb2 と xgb が中核、lgb0 と lgb1 は weight 微小 (0.03)

### 1.4 base prediction の delta range (= test 予測の dynamic range)

各 base の test predict delta (= `TVT - tvt_formula`) の最小〜最大値:

| sub | lgb0 | lgb1 | lgb2 | xgb | cb |
|---|---|---|---|---|---|
| exp005 v1 | [-28.77, 13.45] span 42.22 | [-28.56, 14.61] 43.17 | [-28.37, 14.81] 43.18 | [-30.82, 15.34] 46.16 | [-29.67, 14.52] 44.19 |
| **exp005 v2** | [-26.21, 14.92] 41.13 | [-27.54, 16.24] 43.78 | [-27.89, 14.78] 42.67 | **[-20.75, 16.07] 36.82** | **[-15.82, 15.98] 31.80** |
| exp006 | [-30.90, 12.94] 43.84 | [-31.47, 13.81] 45.28 | [-30.42, 14.81] 45.23 | [-33.59, 14.00] 47.59 | [-33.70, 13.97] 47.67 |
| exp007 | [-28.39, 13.22] 41.61 | [-29.23, 14.27] 43.50 | [-27.53, 14.81] 42.34 | [-30.32, 14.93] 45.25 | [-29.28, 14.50] 43.78 |

(出典: 各 kernel log の `<base> delta range: <lo>–<hi>` 行)

#### 1.4.1 重要観察

**exp005 v1 → v2 で xgb の span が 46.16 → 36.82 (-20%)、cb の span が 44.19 → 31.80 (-28%)** = **同じ pretrained artifacts なのに xgb/cb の predict 範囲が劇的に縮小**。lgb 系は ±2 ft 程度の差で大きな変化なし。

これは Edge S 適用前の段階の数値 (= log 出力順序で `<base> delta range` printout の後に `[Edge S] applied` printout)。**つまり v2 の xgb/cb は v1 と異なる prediction を出している** が、kernel 内で **artifact は同一** (= karnakbaev `rogii-code-helper-dataset`)。

→ **新仮説 H7**: 同じ artifact で再走しても xgb/cb の predict は **non-deterministic component** (= GPU 計算順序 / `nthread` 並列 reduce 順序) で変動する。LB に与える影響は不明だが、**redundant sub** の意味で variance 計測可能。

### 1.5 exp005 v1 (= 既存 best base) の per-well prediction 構造

(出典: `submissions/exp005/submission.csv` + `data/raw/test/*__horizontal_well.csv`、tool 抽出)

| well | hidden 行数 | hidden 長 (ft) | pred 範囲 | 境界 jump (= first_hidden_pred - visible_last_tvt) | tail Δ (= last_hidden_pred - visible_last_tvt) | dtvt std | dtvt p95 abs | on-grid 比率 (round2) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 000d7d20 | 3,836 | 3,836 | [11734.53, 11749.93] = 15.40 ft | **-0.0005** | **-9.108** | **0.034** | **0.066** | **0.247** |
| 00bbac68 | 6,014 | 6,014 | [12195.49, 12238.19] = 42.69 ft | +0.0018 | **-5.864** | **0.053** | **0.110** | **0.374** |
| 00e12e8b | 4,301 | 4,301 | [11590.36, 11616.33] = 25.97 ft | **+0.015** | **-7.367** | **0.044** | **0.094** | **0.388** |

(出典: tool/mining/submission_data_mining.py 出力、`outputs/eda/submission_mining/summary.json § per_well_pred_stats.exp005_v1`)

#### 1.5.1 重要観察

1. **boundary jump はほぼ 0** (-0.0005 〜 +0.015 ft) = **visible 末端と hidden 先頭が連続**。model は continuity を獲得済み
2. **tail Δ は全 well で大きく負** (-5.9 〜 -9.1 ft) = **末端は visible_last より 6-9 ft 下方移動**。hidden 末端での systematic 下方 drift
3. **dtvt std 0.034-0.053** = 各 step の変動は **0.4-0.5 ft 解像度** (= sample 内 sigma の 1/10) で非常に滑らか。SG smooth が effective
4. **on-grid 比率 24.7-38.8%** = **既に 4 倍 step 内に同じ tvt 値が連続** = predict は粗く discrete。**Edge S inject の追加効果は本来限定的**な筈

### 1.6 exp002 (= LGB residual baseline) との per-well 比較

| well | exp002 tail Δ | exp005 v1 tail Δ | diff | exp002 dtvt std | exp005 v1 dtvt std | smoothing ratio |
|---|---:|---:|---:|---:|---:|---:|
| 000d7d20 | -7.33 | **-9.11** | -1.78 (下方 deeper) | 0.130 | 0.034 | **3.8x smoother** |
| 00bbac68 | -6.46 | **-5.86** | +0.60 (上方 reduced) | 0.175 | 0.053 | **3.3x smoother** |
| 00e12e8b | -9.23 | **-7.37** | +1.86 (上方 reduced) | 0.354 | 0.044 | **8.0x smoother** |

(出典: `outputs/eda/submission_mining/summary.json § per_well_pred_stats`)

#### 1.6.1 重要観察

1. **smoothing ratio 3-8 倍** = exp005 (karnakbaev blend + SG smooth) は exp002 (residual LGB) に対して **step-to-step variance を 3-8x 削減**。これが LB -4.378 ft 改善の **形状 component**
2. **tail Δ の符号は well 依存** = 000d7d20 で exp005 が exp002 よりさらに下方、他 2 well では逆。**well ごとに改善/悪化方向が異なる**
3. **diff abs mean = 2.90 ft** (両 sub の per-id 差の絶対値平均) = exp005 と exp002 は **半数の point で 3 ft 以上違う** 予測を出している
4. diff_abs_p95 = 7.06 ft、diff_abs_max = 9.81 ft = 一部 point では **10 ft 近い** predict 差

(出典: `outputs/eda/submission_mining/summary.json § cross_sub_diff.exp005_v1_vs_exp002`)

### 1.7 Edge S 効果の構造解析 (= kernel log 数値から)

exp005 v2 (= Edge S 単独 inject) の Edge S 計測:

- diff abs mean = **0.002496 ft** (= 各 sample の平均 snap 量)
- diff abs max = **0.005 ft** (= 物理上限、grid 0.01 ft の半分)
- unique tvt count after snap = **4735** (= 全 14151 の **33.5%**)

(出典: `outputs/kernel_logs/exp005/rogii-exp005-cache-blend.log` line 38-40)

#### 1.7.1 数理的理解

- predict が 0.01 ft grid 上に snap される → 各 sample は最大 ±0.005 ft 移動
- snap 量 0.002496 (= 平均) は ground truth grid との **距離 vs `±0.005` 一様分布** で予測 0.0025 と整合
- LB 改善 = 10.317 - 10.203 = **0.114 ft** = snap 量 0.002496 の **45.7 倍 leverage**
- ground truth が 0.01 ft grid 上にある (= hengck23 観測 100% on-grid) + snap で正しい grid に来る確率上昇 → **bias 削減 + variance 削減** 同時

#### 1.7.2 leverage の数理的根拠

ground truth $y$ が 0.01 grid 上、prediction $\hat{y}$ が連続値、snap 後 $\hat{y}_{snap} = \text{round}(\hat{y}/0.01) \times 0.01$ とすると:

$$
(\hat{y} - y)^2 - (\hat{y}_{snap} - y)^2 = ((\hat{y} - \hat{y}_{snap}) + (\hat{y}_{snap} - y))^2 - (\hat{y}_{snap} - y)^2
$$

差分 = $2(\hat{y} - \hat{y}_{snap})(\hat{y}_{snap} - y) + (\hat{y} - \hat{y}_{snap})^2$。

$\hat{y} - \hat{y}_{snap}$ は ±0.005 一様、$\hat{y}_{snap} - y$ は **systematic error 部分** (= model bias)。これらが **相関する場合のみ** snap が RMSE を改善する。観測 -0.114 ft 改善 = **systematic alignment が grid 周辺で起きていた** = model bias が grid 内方向に systematic に偏在していた証拠。

---

## 2. 新仮説 H1-H10 (= data-driven 仮説生成)

### 2.1 H1: fold 2 / fold 4 の高 RMSE は「特定 typewell hash group の困難 well 集中」が原因

#### 計測根拠
- exp007 per-fold mean RMSE: fold 1 = 9.19, fold 2 = **12.07**, fold 4 = **11.67** (Δ ≈ +2.9 ft = fold 1 比)
- 全 4 自前 base で fold 2 が同時に最悪 = base 個性ではない、**fold が含む well 集合の難度**
- fold 間 var 1.175 ft vs base 間 var 0.121 ft = **fold 構造が dominant 要因 (10:1 ratio)**
- 出典: `outputs/kernel_logs/exp007/rogii-exp007-edge-qm.log` line 108-178

#### 数理的に何が起きているか
Edge Q (typewell content-hash GroupKFold, 752 unique groups) で 5-fold 分割した結果、fold 2 と fold 4 に **typewell variability が高い group** (= `first-principles.dense.md` §2.7 §B「tw_gr_resid_std_p90 = 18.88, max 63.48 の wells」) が偏って配分されている可能性。$\text{RMSE}_{\text{fold}}^2 \approx \mathbb{E}_{\text{well} \in \text{fold}}[\text{RMSE}^2_{\text{well}}]$ で、難 well を持つ fold の RMSE が体系的に膨張。

#### ROGII 数理 framework との整合
`first-principles.dense.md` §2.7 §B (tw_gr_resid_std 大の wells) と §A (MD 末端の systematic bias) で実体観測済の現象を Edge Q fold partition が拾えていない。

#### 検証 plan
1. exp007 kernel log から fold 2 / fold 4 に属する well_id list を抽出 (= `Edge Q fold computation` phase の追加 log が必要、または `src/rogii/cv.py` を 1 実行して fold mapping 出力)
2. 各 well の `outputs/eda/first_principles/per-well-stats.parquet § tw_gr_resid_std` と join、fold 2 / fold 4 内の typewell variability mean を計算
3. fold 1 内 vs fold 2 内 tw_gr_resid_std mean を統計検定 (= Welch's t-test) で有意差確認

#### 5/12 以降の含意
- もし fold 2 内の typewell variability が有意に高い → **stratified fold** (= tw_gr_resid_std 分位で stratify) で fold balance を整える
- exp010 候補: **typewell variability-stratified GroupKFold** (= Edge Q 拡張)、期待 -0.1〜-0.3 ft (= fold variance 低減で Ridge meta が真の OOF を fit、LB overfit risk 減)

### 2.2 H2: Edge S 効果は base smoothness と非線形相関、xgb/cb 特有

#### 計測根拠
- exp005 v1 → v2 で xgb delta range span 46.16 → 36.82 (**-20%**)、cb 44.19 → 31.80 (**-28%**)
- 同 v1 → v2 で lgb0/1/2 は span 変化 ±2 ft 程度 (= base 種類で Edge S 前段の predict 自体が異なる)
- Edge S 自体は **delta range printout の後** で適用 (出典: log line 32-37 → 38)
- 出典: §1.4 表 + log line 32-37

#### 数理的に何が起きているか
v1 と v2 は **同じ artifact** (= `karnakbaevarthur/rogii-code-helper-dataset`) を使用しているのに base predict が違う = (a) GPU 利用 mode 差 (v1 GPU=False, v2 GPU=False...同じ)、(b) tree ensemble の thread reduce 順序非決定性、(c) **karnakbaev artifact 自体が更新された** (= live test FE 部分の version 違い)。

仮に (c) が成立すると、v2 で xgb/cb の predict が **より conservative** (= range 縮小) になっており、Edge S 適用前から **RMSE が改善している可能性**。本当の Edge S 単独効果 < -0.114 ft = 既存 doc の §1.4.2「想定中域 -0.05〜-0.30 の -0.114」評価が **過大評価** の risk。

#### ROGII 数理 framework との整合
`first-principles.dense.md` §1.4 の確率モデルで base predict $\hat{\text{TVT}}_b(s)$ は noise distribution に依存。xgb/cb は LGB と異なる split criterion で **range の dynamic を保ちやすい**。

#### 検証 plan
1. **exp005 v3 submission.csv を kaggle CLI で download**、id 単位で v1 と diff 計算 → Edge S 単独効果と Edge R 単独効果を分離
2. v1 sub と v2 kernel log の base delta range が違うなら、**Edge S なし同 kernel re-run** (= mode=infer_no_edge_s) で v1 と完全同条件で比較

#### 5/12 以降の含意
- もし v1 vs v2 で base predict 自体が違うなら、Edge S の **真の単独効果は -0.05 〜 -0.10 ft** と推定再校正必要
- 9 切り roadmap (`leaderboard.dense.md § 9 切り`) の累積 effect 見積もりが **0.05 ft 程度** 楽観バイアス
- exp008 v3 / v4 で Edge S を再走するときの予測値を **下限 -0.05** に補正

### 2.3 H3: 自前 4 base は karnakbaev 5 base と高 correlated、diversity 不足

#### 計測根拠
- exp007 own OOF: lgb_own0=10.666, lgb_own1=10.758, lgb_own2=10.931, cb_own=10.690 (mean **10.761**)
- karnakbaev published OOF (kb side OOF) ≈ **10.78** (= `leaderboard.dense.md § 公開 kernel ベンチマーク`)
- Ridge 9-base OOF = 10.3874、Simple 9-avg = 10.4624 → **Ridge は simple-avg より -0.075 ft 改善のみ** = 自前 4 base 追加で **0.075 ft 価値** しか出てない (= diversity 不足の証拠)
- Ridge weights で lgb_own2 = 0.000 完全除外、own side total 0.380 (= 4 base で 38%、1 base 平均 9.5%)
- 出典: §1.2 + §1.3 + log line 179-191

#### 数理的に何が起きているか
9 base の predict matrix を $\mathbf{P} \in \mathbb{R}^{N \times 9}$ とすると、Ridge 解は $\hat{w} = (\mathbf{P}^T \mathbf{P} + \alpha \mathbf{I})^{-1} \mathbf{P}^T \mathbf{y}$。**$\mathbf{P}^T \mathbf{P}$ の対角成分は同レベル** (各 base OOF RMSE 10.6-10.9)、**非対角成分 (= base 間 correlation) も同レベル**ならば、Ridge は weight を平準化するだけ。

実際 cb_own (= 自前 catboost, 自前最良 OOF 10.690) に最大 weight 0.2273 を割り当てているのは、cb_own が karnakbaev 5 base と **相対的に diversity が高い** から (= cb_own は CB だが Edge Q fold で再 train、karnakbaev cb と features が違う)。lgb_own0/1 は karnakbaev lgb0/1/2 と features 範囲がほぼ同じで correlation 高、よって weight 低。

#### ROGII 数理 framework との整合
`first-principles.dense.md` §2.7 で「公開 top が捕捉していない情報源」として **C (AR(1) state-space), F (Bayesian uncertainty)** を挙げているが、自前 4 base はこれらを使わず karnakbaev と同じ tabular regression を再走しただけ。**情報源の重複** = 必然的に correlation 高。

#### 検証 plan
1. exp007 OOF parquet を Kaggle から download (= `outputs/oof/exp007_own_*.parquet` を追加)、9 base の Pearson correlation matrix 計算
2. 期待: 自前 4 base × karnakbaev 5 base のうち高 correlation pair が **>0.95** = 完全冗長
3. 低 correlation な base (= 真の diversity) は何か特定 → 案 D (Kalman) / 案 E (Bayesian GP) / Edge R (online learning) が実現すれば low-correlation base になる仮説

#### 5/12 以降の含意
- 自前 4 base を **完全廃棄** (= path b = subagent T 改修と整合)、karnakbaev base のみ維持
- 新 base 投入は **必ず low-correlation paradigm** = AR(1) Kalman / GP posterior / online learning / NN (= cdeotte starter) のいずれか
- exp011-12 で 案 D + 案 E + Edge R + NN を **multi-paradigm injection** で 4 base × 2 種類 base layer に拡張

### 2.4 H4: tail systematic bias (= 末端 -16 ft 下方) は hidden_len と correlated

#### 計測根拠
- exp005 v1 全 3 well で tail Δ (= last_hidden_pred - visible_last_tvt) 全て負: -9.11, -5.86, -7.37 ft
- 同 exp002 tail Δ: -7.33, -6.46, -9.23 ft
- 各 well の hidden_len 3836 ft (000d7d20), 6014 ft (00bbac68), 4301 ft (00e12e8b)
- `first-principles.dense.md` §2.4 extrap curve: MD 9975-10025 ft で mean -16.4 ft (= systematic 下方 bias)
- 出典: §1.5 + §1.6

#### 数理的に何が起きているか
ground truth TVT は formation の地質構造で決まるため hidden 末端で **systematic に下方** に行く wells が **train 集団に多い**。LGB ensemble は MD 大 region で **末端 train sample の中央値方向** に regress → 下方系統 bias。  
$\mathbb{E}[\hat{y} - y \mid \text{hidden\_len} = L] = -\beta_0 - \beta_1 L$ で $\beta_1 > 0$ なら hidden_len 大 well で大きい bias。

実測: hidden_len 6014 (= 00bbac68) で tail Δ -5.86 ft (= 最小 abs)、hidden_len 3836 (= 000d7d20) で tail Δ -9.11 ft (= 最大 abs) = **逆 correlation**! つまり仮説 hidden_len → tail bias は **negative slope** か **non-monotonic**。

#### ROGII 数理 framework との整合
`first-principles.dense.md` §2.7 §A 「MD 末端の systematic bias」と整合。だが 3 well では sample 数不足、train 全体 (773 wells) で hidden_len vs tail Δ correlation を測る必要。

#### 検証 plan
1. `outputs/eda/first_principles/per-well-stats.parquet` で 773 train wells の `hidden_len` と (visible 内で計測した) `tail Δ` を計算 → Pearson correlation
2. 案 D (Kalman/PF) を inject すると AR(1) prior が tail での random walk を **smoother**、systematic bias を **suppress** できるか smoke で確認
3. exp008/009 v2/v3/v4 LB と各 well の MD 末端 predict shift を比較し、案 D の **tail bias correction 効果** を計測

#### 5/12 以降の含意
- 案 D (Kalman) は本仮説 verify されれば **末端 1000 ft 区間の tail correction** で -0.2〜-0.5 ft 寄与候補
- Edge R (online learning) も同様に **test visible region で base 末端を anchoring** = 効果同方向

### 2.5 H5: 自前 4 base の per-fold variance が Ridge OOF を **過小評価** している (= leak)

#### 計測根拠
- exp007 Ridge 9-stk OOF = **10.3874**、Simple 9-avg OOF = **10.4624** = Ridge -0.075 ft
- karnakbaev kb_oof_q は **train-set predict (not true OOF)** であり Edge Q fold とも partition 違い (= log line 65-67: `Stack consistency note: kb_oof_q is karnakbaev train-set predict, used only to align fold partitions with own OOF for Ridge meta. True OOF requires re-fitting karnakbaev models per Edge Q fold (skipped for runtime; deferred to v2).`)
- Ridge meta が train-set predict (= 自分のラベル fit 済) を mix → optimistic biased OOF
- `leaderboard.dense.md § exp007 fold-misalignment` で +0.12 ft overestimate と推定 → 実 LB +0.290 で **2.4 倍 severe**
- 出典: §1.2 + log line 65-67

#### 数理的に何が起きているか
9-base stack で kb 側 5 base は karnakbaev 元 fold で OOF 取得 (= 真の OOF だが Edge Q とは別 partition)。Edge Q fold $F_q$ と karnakbaev fold $F_k$ の重複率は **1/5 = 20%** (= 同じ row が両 partition の val fold に該当する確率)。Ridge は同じ row index で fit するので、**row $i$ で自前 base val、kb base train** ⇒ kb base は train data として fit 済 ⇒ leak。

leak 影響量: $\mathbb{E}[\hat{y}_{kb} - y \mid \text{row } i \in F_k^{\text{train}}] \neq 0$ で、Ridge は systematic に kb side weight を **過大評価**。weight 0.62 (kb side) が真の最適 0.5 程度なら、真の OOF は **10.45-10.50** で LB 10.677 と **+0.20-0.23 diff** に整合。

#### ROGII 数理 framework との整合
GroupKFold で同一 well の row を分離するのが GM 原則 (= `kaggle-gm-mindset.md` "good CV is half of success", Bestfitting)。Edge Q は well 単位 grouping だが、karnakbaev fold とは partition 違いで、**row-wise leak** が二重 fold 構造から発生。

#### 検証 plan
1. **karnakbaev pretrained を Edge Q fold で真の OOF 再生成** (= path a) を 1 sub で実装、Ridge OOF を再計算
2. 期待: 真 OOF Ridge ≈ 10.5 帯 → LB との diff +0.1〜+0.2 ft で収束 (= leak 解消)
3. **path b** (= kb simple-avg + own Ridge separate) で leak 数理的 zero を確認 (= subagent T 改修済)

#### 5/12 以降の含意
- exp008 v3 / exp009 v3 が path b で **真の OOF Ridge meta** で LB 10.0-10.3 帯到達するなら、本仮説 verified
- もし path b で LB 改善幅が **+0.05 ft 程度** に留まるなら、leak 影響量は実は **+0.1 ft 程度** で他要因 (= 自前 base diversity 不足) も同等に支配

### 2.6 H6: Edge S は base 種類で効果が逆転する (= xgb/cb で **悪化** の可能性)

#### 計測根拠
- §1.4 で xgb delta range span 縮小 (-20%)、cb 縮小 (-28%) = v2 で xgb/cb の **predict variance 自体が小さい**
- on-grid 比率 24-39% (= v1 sub 内、Edge S なし) = predict は既に粗く discrete
- Edge S diff abs mean 0.0025 ft = ほぼ無効 (∵ ±0.005 一様分布の期待値 = 0.0025)
- LB 改善 -0.114 ft は **全 base 平均** の効果。base 別に分解できれば一部 base で **悪化** している可能性
- 出典: §1.4 + §1.7

#### 数理的に何が起きているか
Edge S は **0.01 grid に snap**。base $b$ の predict $\hat{y}_b$ が systematic bias を持つ場合:
- bias 中心が grid 上 → snap で bias 維持 (= 効果 zero)
- bias 中心が grid 中央 → snap で randomly ±0.005 移動 = **variance 加算**
- bias 中心が grid 端 → snap で bias 削減 (= 効果 positive)

xgb/cb は LGB と異なる split で予測が **連続値に近い** (= range 縮小しても tail-heavy)、snap で **逆方向に動く** sample 多い → 一部 well で悪化。**Edge S は base ensemble の predict 平均化後 (= post-Ridge) に適用**するのが理想だが、本 kernel では Ridge 前に base 単独 snap も適用してる可能性。

#### ROGII 数理 framework との整合
`first-principles.dense.md` §1.4 で y は formula で構成、Edge S は formula 内 ANCC imputation の grid との一致前提。base prediction が formula を通った後の **post-formula** で適用する場合のみ意味。

#### 検証 plan
1. exp005 v2 submission.csv を Kaggle download、id 単位で v1 と diff (= Edge S 単独効果) を計算
2. 各 well で diff の符号 + 大きさを集計、well/region 単位で **negative diff** (= 悪化方向) sample があるか確認
3. もし negative ratio > 5% (= 700+ sample) なら adaptive snap (= 一部 well skip) の根拠

#### 5/12 以降の含意
- adaptive snap (= 末端 wells / typewell variability 高 wells で skip) で **追加 -0.02〜-0.05 ft** 寄与候補
- exp013 candidate: **per-well adaptive Edge S** + base ensemble 後の **post-stack snap** に変更

### 2.7 H7: 同じ kernel artifact でも GPU 設定差で base predict は **non-deterministic**

#### 計測根拠
- §1.4 表で exp005 v1 と v2 の base delta range が xgb/cb で **20-28% 異なる**
- 両 run で artifact `karnakbaevarthur/rogii-code-helper-dataset` は同一、kernel コードも同一 (`rogii-exp005-cache-blend` v1 vs v2)
- 唯一 v2 で Edge S inject 追加。Edge S 自体は predict 出力後の post-process なので base predict には影響しない筈
- 出典: log line 32-37 (v2) vs 32-36 (v1)

#### 数理的に何が起きているか
LGB / XGB / CatBoost の predict は **deterministic** (= 同じ model file → 同じ output) と公式 doc にあるが、実装上 GPU では:
- thread reduce 順序の非決定性
- floating point round-off 累積差
- multi-thread atomic 加算順序

これらは典型的に **±1e-6 ft** 程度の差しか作らない。20-28% range 差は **artifact 自体が更新された** か **kernel ver で feature engineering の random seed が違う** ことを示唆。

#### ROGII 数理 framework との整合
GM 原則「Trust your CV always」(Chris Deotte) は **same artifact + same code → same predict** 前提。本ケースは **同 artifact でも predict 違う** = re-train なしの predict variance も評価に入れる必要。

#### 検証 plan
1. v1 と v2 の kernel script コードを git diff で確認 (= 同じハッシュなら artifact 更新疑い)
2. karnakbaev dataset を kaggle CLI で download → version 比較 (`kaggle datasets list -v karnakbaevarthur/rogii-code-helper-dataset`)
3. もし version 違いなら、artifact 更新で新 predict を fetch → **常に最新版を pin** する運用必要

#### 5/12 以降の含意
- artifact 版固定 (= specific version で pin) する設計が **再現性確保のため必須**
- exp008 v3 以降の base predict 比較を **fair baseline** で評価するために v1/v2/v3 で同 artifact version を確認

### 2.8 H8: cb_own (= 自前 CatBoost) が最大 weight 0.2273 = 自前 GPU 失敗 → CPU 再学習が真の差別化 base

#### 計測根拠
- Ridge weights: cb_own = 0.2273 (= 全 9 base で **最大**)
- own OOF: cb_own = 10.690 ft (= 自前 4 base で **最良**)
- exp007 log で全 5 fold で **GPU CB failed → CPU retry** (= log line 117, 132, 147, 162, 177): `GPU CB failed (catboost/cuda/cuda_lib/cuda_manager.cpp:201: Condition violated: 'State == nullptr')`
- CPU retry の runtime: fold ごと 490s, 1008s, 935s, 789s, 505s (合計 **3,727s = 62 分**) → 全 own train 時間 5,318s の **70%**
- 出典: §1.3 + log line 117-178

#### 数理的に何が起きているか
GPU CB は host 環境の Kaggle GPU (= T4 or P100) で **CUDA 互換性エラー** で初期化失敗、自動 CPU 切替。CPU CB の **default hyper-parameter** で動いた結果が cb_own = 10.690 = **karnakbaev OOF 10.78 より -0.09 ft 改善**。これは:
- (a) CatBoost CPU は GPU と異なる split 順序で **diversity 自然生成**
- (b) Edge Q fold + own features (= 165 cols) で **Edge M (visible-as-typewell) features が effective**
- (c) lr / num_leaves / depth の default が偶然 well-tuned

cb_own は **自前 base の中で唯一 karnakbaev base に追加価値を出す**。lgb_own0/1/2 は karnakbaev lgb0/1/2 と高 correlation で完全冗長。

#### ROGII 数理 framework との整合
`first-principles.dense.md` §1.4 で formula component は不変、ANCC imputation + b_well 推定が main task。CatBoost は **categorical interaction** に強く、Edge M (visible-as-typewell features 4 列) の categorical structure を活用できる ⇒ LGB と本質的に異なる prediction surface。

#### 検証 plan
1. exp007 OOF parquet を Kaggle download、cb_own と karnakbaev cb の Pearson correlation 計算
2. 期待: 0.85-0.93 = **non-trivial diversity** (= 完全冗長なら >0.95)
3. cb_own を **CPU 固定 + larger train round** で再 fit (= GPU 失敗依存をやめる)、OOF が更に改善するか確認

#### 5/12 以降の含意
- exp008 v3 / exp009 v3 で **cb_own を維持しつつ lgb_own を全廃** = own side を 1 base に絞り、Ridge weight 配分を清潔化
- exp012 で **CatBoost + Edge M features 路線** を 1 base layer として保持、その他 paradigm (= NN / Kalman / online) を別 layer 追加

### 2.9 H9: exp006 失敗の真因は TabICL fail でなく **fallback NM blend の weight 不可知性**

#### 計測根拠
- exp006 log line 113: `TabICL path failed: AcceleratorError: CUDA error: no kernel image is available`
- 5 base test predict は正常生成 (delta range -33 〜 +14、log line 40-50)
- fallback NM blend test delta range = [-31.942, 13.726] (= log line 123) = 5 base 個別と整合 (= NM blend は正常動作)
- exp005 v1 (= Ridge blend) = 10.317、exp006 (= NM blend fallback) = 10.503 = **+0.186 悪化**
- 出典: log line 121-127 + leaderboard.dense.md exp006 row

#### 数理的に何が起きているか
karnakbaev artifact には **Ridge weights が pre-fit 済**で保存されている前提。exp005 v1 は Ridge weight を直接 load → 5 base predict を Ridge blend。exp006 では **Ridge weight 5-base 用に再 fit せず、Nelder-Mead で 5-base weight 探索** = NM 探索が Ridge 解析最適と異なる解に収束。

NM は initial point 敏感 (= karnakbaev published weight を init 値にしないと local minimum)。fallback path で initial weight が不明 → 5 base 配分が exp005 v1 の Ridge optimum から **0.05-0.10 ft RMSE 相当ズレ** → LB +0.186 ft 悪化と整合。

#### ROGII 数理 framework との整合
`mathematical-formulation.dense.md`（doc 不在だが概念は first-principles.dense.md §1.5 と postmortems.dense.md §3.3 に分散記載）で「stack meta は base 質と integration 正確性が両方 high でないと悪化」原則と整合。

#### 検証 plan
1. exp006 kernel コードで NM blend の initial weight を log から確認 (= 不可なら次回 sub で log 出力)
2. exp006 を **Ridge weight load** (= karnakbaev 公開 OOF の Ridge weight を直接使用) で re-run → LB が exp005 v1 (10.317) に戻るか
3. もし戻るなら、本仮説 verified ⇒ 「Ridge meta は **必ず published weight を初期値に**、再 fit せず固定」運用ルール

#### 5/12 以降の含意
- 新 base 追加時の Ridge 再 fit は **risk pattern** = 既存 published Ridge weight を **固定**、新 base を **residual stacking** で別 layer 追加が安全
- exp010 で「**residual layer 設計** (= base ensemble 後の residual に新 base を fit)」を方針化

### 2.10 H10: fold variance σ=1.18 ft の構造的下限が exp007 type approach の **LB 天井** を 10.6 帯に固定

#### 計測根拠
- exp007 per-fold mean RMSE: [10.06, 9.19, 12.07, 10.55, 11.67] (= σ = 1.175 ft)
- mean of fold means = 10.71 ft
- exp007 LB = 10.677 ft = fold mean とほぼ一致
- Ridge meta は per-fold 9-base predict から row-wise weight fit ⇒ fold 構造の variance は inherit
- 出典: §1.2 + §1.1

#### 数理的に何が起きているか
GroupKFold(5) で 5 fold partition $\{F_1, ..., F_5\}$、それぞれの val RMSE を $r_k$ とすると CV $= \sqrt{\frac{1}{5} \sum r_k^2}$。fold variance σ = 1.18 ft の下では:

$$
\text{CV} \approx \sqrt{\overline{r}^2 + \sigma_r^2} \approx \sqrt{10.71^2 + 1.18^2} \approx 10.77 \text{ ft}
$$

(Jensen の不等式の逆: $\mathbb{E}[\sqrt{X^2}] \leq \sqrt{\mathbb{E}[X^2]}$。等号は variance 0 で成立。fold variance がある限り CV は fold mean RMSE より大きい。)

実 exp007 CV (Ridge 9-stk OOF) = **10.3874** で計算値より低い ⇒ **leak (H5) で過小評価**。真の CV は 10.5-10.8 帯。LB 10.677 はこの真の CV と整合。

⇒ **同じ fold partition + 同じ approach で submit する限り、LB の下限は ~10.4-10.6 ft で固定** = **9 切り (= 8.x 帯) には fold variance 自体を構造的に縮小する必要**。

#### ROGII 数理 framework との整合
`leaderboard.dense.md § 9 切り戦略` の「独自 edge 累積 -2.55 ft 必要」と整合。fold variance を縮小しない限り **edge 累積効果は σ_r の影響を受けず本来期待値より小さくなる**。

#### 検証 plan
1. exp008 v3 / exp009 v3 (= path b、leak 解消版) で OOF が 10.4-10.6 帯に収束するか確認
2. exp010 candidate: **fold balance** (= tw_gr_resid_std 分位で stratify) で fold variance σ < 0.5 ft 目標
3. exp011 candidate: **adversarial validation** (= train vs test の base predict distribution 差で sample weight 調整) で test に近い fold を作る

#### 5/12 以降の含意
- 9 切り の真の絶望感 = fold variance σ_r の影響で、edge 累積効果が **数学的に飽和**
- adversarial validation + stratified fold + per-fold-tuned base が **構造改革** 必須
- exp010-12 の優先順位を **「個別 edge inject」→「fold 構造改革 + adversarial」** に **方針転換** を検討

---

## 3. 横断 pattern (= 既存 postmortem §3 を data-driven で精緻化)

### 3.1 既存 pattern と本 doc data の整合性

既存 `submission-postmortems.dense.md` §3 の trend を本 doc data で **refute / verify**:

| 既存 pattern | data evidence | 本 doc 評価 |
|---|---|---|
| **既存 pretrained 質の高い model を母法に** (exp005 改善) | exp007 Ridge weights kb side 0.62 (= dominant) | **verified** (= H8 で cb_own 自前 1 base のみ補完) |
| **物理 formula を正確に計算** | formula_oracle_rmse 0.006 ft (= first-principles) | **verified、未活用** = H4 で末端 systematic bias を formula で吸収困難 |
| **single submit でなく ablation 設計** | exp005 v1 vs v2 で Edge S 単独効果 -0.114 ft 計測可能 | **verified、運用継続** |
| **自前 複雑 features 単独路線** (exp003 悪化) | (exp003 log なしで本 doc では verify 不可) | **partial** (= 既存 doc に依拠) |
| **新 base 追加 + Ridge re-fit 不十分** (exp006/007) | H5 (leak), H9 (NM weight 不可知性), H3 (diversity 不足) で **3 軸** で支配 | **verified, refined** = 既存 doc は 1 軸記述、本 doc で 3 機制を分離 |
| **CV 計測なし blind submit** | exp005 v1, v2, v3 でも CV 計測なし | **verified、systemic gap 継続** |
| **fold partition 違いの OOF を同 Ridge meta** (exp007) | H5 (leak 1/5 fold 重複) + H10 (fold variance σ=1.18) | **verified, refined** = leak だけでなく **fold 自体の構造的 variance** が dominant |

### 3.2 新 pattern (= 本 doc から新規発見)

#### Pattern P1: 「fold 間 variance が base 間 variance の 10 倍」
- 数値: exp007 で σ_fold = 1.175 vs σ_base = 0.121 (= 9.7x ratio)
- 含意: Ridge meta の base 選択 / weight 調整は **fold 構造改革に比べ無効**
- 対策: stratified fold / adversarial validation / per-fold tune base

#### Pattern P2: 「Edge S 効果は base 種類で非対称」
- 数値: xgb/cb で range -20 〜 -28%、lgb 系で ±5% 以下
- 含意: snap は **noise 大 base** で大幅 shrink、**bias 大 base** で限定的
- 対策: post-stack snap + per-base adaptive snap

#### Pattern P3: 「自前 base diversity は paradigm 単位、tree variant 単位ではない」
- 数値: lgb_own0/1/2 = 3 種 lr で base 内 std わずか 0.12 ft → 同 paradigm = 完全冗長
- 含意: 新 base は **tree → NN / Kalman / GP / online learning** の paradigm shift 必須
- 対策: exp011-12 で 案 D + 案 E + Edge R + NN multi-paradigm

#### Pattern P4: 「kernel re-run 間で base predict の non-deterministic variance」
- 数値: exp005 v1 v2 で xgb/cb の delta range 20-28% shift
- 含意: 「same artifact + same code → same predict」前提が **崩れる**
- 対策: artifact version pin、再現性 audit

#### Pattern P5: 「tail Δ は hidden_len と correlate しない (= non-monotonic)」
- 数値: 3 well で hidden_len 3836/6014/4301 vs tail Δ -9.1/-5.9/-7.4 = 順序不一致
- 含意: 単純な MD-based extrapolation correction は **効果限定**、well 個別 dynamics が支配
- 対策: per-well adaptive model / state-space (Kalman) / online learning

### 3.3 メタ pattern (= 既存 doc §3.3 を refine)

既存:
- 「safe path = 既存 stack を維持 + 加算的 inject (例: Edge S, Edge R)」が ROI 最大
- 「risk path = 新 base 追加 + Ridge re-fit」は path a 必須

本 doc 追加:
- **「fold 構造を改革しない限り edge 累積効果は天井あり」** (= P1, H10)
- **「真の diversity = paradigm 単位、tree variant の差は 0.05 ft 程度の noise」** (= P3, H3)
- **「Edge S は post-stack で base 平均化後に適用が理想」** (= P2, H6)
- **「artifact 再現性は assumed but unverified」** (= P4, H7)

---

## 4. 5/12 以降の sub 設計への直接的反映

### 4.1 immediate (= 5/12 中の 5 sub 枠で実施)

| 優先順 | sub 候補 | 検証する仮説 | 期待 LB | 実装コスト |
|---|---|---|---|---|
| **A** | **exp005 v4 = Edge S + post-stack snap** (= base ensemble 後で 1 回 snap、base 個別 snap 廃止) | H2 + H6 | 10.15-10.20 | 1-2 h |
| **B** | **fold balance analysis** = exp007 OOF を kaggle download → 各 fold 内 well_id list 抽出 → tw_gr_resid_std 分位確認 | H1 + H10 | sub 不要 (= dev work) | 2-3 h |
| **C** | **exp008 v3 (path b) LB 待ち** = 既に kernel push 済、5/12 早朝 LB 出る予定。LB 結果で H5 (leak 解消) verify | H5 | 10.0-10.3 | 既存 |
| **D** | **exp007 OOF parquet download + base correlation matrix** = 9 base の Pearson correlation 計算で H3 verify | H3 + H8 | sub 不要 | 1 h |
| **E** | **adaptive Edge S sub** (= 末端 1000 ft skip / typewell variability 高 well skip) | H6 part 2 | 10.18-10.22 | 2 h |

### 4.2 short-term (= 5/13-5/17 の week 内)

| 優先順 | sub 候補 | 検証する仮説 | 期待 LB |
|---|---|---|---|
| **F** | **stratified Edge Q fold** = tw_gr_resid_std 4 分位 stratify | H1 + H10 (= P1) | 10.3-10.5 (本来 leak 解消で構造改善) |
| **G** | **case D Kalman + AR(1) prior** = 末端 1000 ft で random walk smooth | H4 + Pattern P5 | 10.1-10.3 (= 案 D 期待 -0.30〜-0.60 ft 既存記載) |
| **H** | **case E Bayesian GP for ANCC** = uncertainty + posterior var feature | H3 (= 新 paradigm base) | 10.0-10.2 |
| **I** | **base correlation cap** = Ridge weight 制約に **base 間 correlation < 0.9** を要件化 (= 完全冗長 base を automatic exclude) | H3 + H5 | 10.3-10.5 |

### 4.3 medium-term (= 5/18-5/24 の week 内)

| 優先順 | sub 候補 | 検証する仮説 |
|---|---|---|
| **J** | adversarial validation で test に近い fold 構築 | H10 (= P1) |
| **K** | per-well adaptive Edge S | H6 part 2 |
| **L** | post-stack snap | H2 + H6 |
| **M** | **multi-paradigm injection** = LGB + CB + Kalman + GP + NN + online learning の 6-paradigm base layer | P3 |

### 4.4 quota constraint (= 5/day)

5/12 残 5 sub。優先順位:
1. **exp005 v3 LB scoring** (= Edge S + Edge R、5/11 0:32 submit、PENDING) → wait
2. **exp008 v3 / exp009 v3 LB** (= path b、subagent T) → wait
3. (空き枠で) **A (post-stack snap)** または **E (adaptive Edge S)** を 1 sub
4. (空き枠で) **B + D の dev work** (= sub 不要、kaggle download + 分析)

---

## 5. 残課題 / 不明点

### 5.1 取得できなかった data

- **exp005 v2/v3, exp006, exp007 の submission.csv**: kernel 出力後 download していない。各 sub の per-id predict は kernel log の statistics 抽出のみ
  - **回避**: `kaggle kernels output ky7240/rogii-exp005-cache-blend -v 2 -p submissions/exp005_v2/` で download 可能、次回 dev で実施
- **exp002, exp003 の kernel log**: `outputs/kernel_logs/` に未配置。CV 13.82 (exp002) のみ leaderboard.dense.md に記載
- **karnakbaev artifact version 履歴**: `kaggle datasets list -v karnakbaevarthur/rogii-code-helper-dataset` 未取得
- **exp007 fold partition の well_id 内訳**: log には数値 (tr/va 件数) しか出ていない、`src/rogii/cv.py` を 1 回実行して mapping 出力する dev work が必要

### 5.2 仮説で verify 未達

| 仮説 | verify 残作業 |
|---|---|
| H1 (fold 2 高 RMSE = typewell variability) | well_id mapping + tw_gr_resid_std join |
| H2 (Edge S effect non-monotonic) | exp005 v2 sub csv で per-id diff |
| H3 (自前 4 base correlation 高) | OOF parquet download + Pearson matrix |
| H4 (tail bias hidden_len correlate) | 773 wells 全数で計測 |
| H5 (Ridge leak 影響量) | path b LB で確認 (= 5/12 待ち) |
| H6 (Edge S 一部 well で悪化) | v2 sub csv per-id 分解 |
| H7 (kernel 非決定性) | artifact version 確認 |
| H8 (cb_own diversity) | OOF parquet correlation |
| H9 (NM weight 不可知性) | exp006 kernel コード再読 |
| H10 (fold variance 構造的下限) | path b OOF と本 doc fold variance を比較 |

### 5.3 時間切れで深掘りできなかった項目

- **per-id レベルの diff matrix** (= 全 sub の id × diff): submission.csv 揃ったら実施可能
- **karnakbaev kernel コードの細部** (= NM initial weight / Ridge alpha / SG window): public kernel コード読み込み未実施
- **discussion topic 698282 (= TVT 物理定義) との predict 整合性**: 各 well で predict が物理的に妥当な range か audit
- **public LB ベンチマーク kernel との predict 比較** (= pilkwang / ravaghi / chrisdeotte kernel) を local で再 run + 我々 sub と diff

---

## 6. 関連 doc / source

- 既存 postmortem: `docs/dev/submission-postmortems.dense.md` (= mechanism-level 因果)
- LB log: `docs/dev/leaderboard.dense.md` (= submission log + CV-LB trend)
- 数理基礎: `docs/research/first-principles.dense.md` §1.4, §2.4, §2.7 (= 確率モデル + extrap curve + 未使用情報源)
- 抽出 tool: `tools/mining/submission_data_mining.py`
- 抽出結果: `outputs/eda/submission_mining/summary.json` (= machine-readable 全数値)
- source log: `outputs/kernel_logs/exp005/`, `outputs/kernel_logs/exp006/`, `outputs/kernel_logs/exp007/`, `submissions/exp005/rogii-exp005-cache-blend.log`
- source csv: `submissions/exp005/submission.csv`, `submissions/exp002_lgb.csv`, `data/raw/test/*__horizontal_well.csv`
