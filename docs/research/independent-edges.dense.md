# 独自 Edge 候補 — 公開 Top 10.0 帯にない構造的に異なる 10 案 (2026-05-10)

> 母法: `docs/research/first-principles.dense.md` の計測値から **「公開 Top が捕捉していない情報源 A-F」** を出発点に、構造原理が異なる 10 案を生成。
> 推奨は出さない、**選択軸 + トレードオフ表 + 失敗モード** のみ提示 (CLAUDE.md 智者尽其慮 / 反証思考)。
> 判断は中央 (= 私 = ユーザー対応 Claude) が行う。

## 0. 設計原則

### 0.1 構造原理が "異なる" の定義

同一 paradigm 内のパラメータバリエーションは「異なる」と数えない。次の意味で異なる:

| 軸 | 区分 |
|---|---|
| 推論パラダイム | Frequentist point estimate / Bayesian posterior / Variational / Sampling-based |
| 時系列構造 | i.i.d. / AR / state-space / sequence-to-sequence |
| 学習信号 | Direct regression / Multi-task / Self-supervised / Pseudo-label |
| 推論時計算 | Static (1 forward) / Iterative (Kalman update) / MCMC / Diffusion sampling |
| 不確実性 | None / quantile / variance / posterior |

### 0.2 評価選択軸 (5 軸)

各案を以下で評価 (推奨せず軸のみ提示):
1. **構造原理の独自性** (= 公開 top にどれだけ "ない")
2. **期待 LB 改善** (first-principles の計測から推定)
3. **実装工数** (= 残 86 日のうちどれだけ消費)
4. **失敗リスク** (= overfit / numerical instability / runtime overflow)
5. **既存案 A/B/C / 移植 6 要素との合成可能性**

## 1. 案 D — State-Space Model on dTVT (Kalman / Particle Filter)

### 1.1 構造原理

公開 top は dTVT を **i.i.d.** 仮定で features に渡す。我々は **AR(1) 〜 random walk** として陽に model 化し、Kalman filter で hidden region を MAP estimate。

### 1.2 着想根拠

- **計測値**: `ar1_phi_p50 = 0.999` (`per-well-stats.parquet`)、`ar1_eps_std_p50 = 0.0156` ⇒ dTVT はほぼ random walk
- **学術出典**: Kalman 1960, Doucet et al. 2001 (PF), Vandewiele 2021 (Ventilator 1 位は LSTM だが PID 物理を post-proc に組込)
- **公開 top の Particle Filter** は ANCC + Z 上で動かしているが、**dTVT 上の AR(1) state-space は無い**

### 1.3 数学的定式

$$
\Delta\text{TVT}(s+1) = \phi \cdot \Delta\text{TVT}(s) + \varepsilon(s), \quad \varepsilon \sim \mathcal{N}(0, \sigma_\varepsilon^2)
$$

$$
\text{TVT}(s+1) = \text{TVT}(s) + \Delta\text{TVT}(s+1)
$$

$\phi$ は per-well で visible region から MLE 推定、$\sigma_\varepsilon$ も同様。Kalman filter の予測共分散は MD で grow:

$$
\text{Var}[\text{TVT}(s+t)] = t \cdot \sigma_\varepsilon^2 / (1 - \phi^2) \approx t \cdot \sigma_\varepsilon^2 \cdot \text{large}
$$

(= MD で std が増大する extrapolation curve を直接予測)

### 1.4 ROGII への適合性

- **適合度**: ★★★ (高、ar1_phi が 0.999 という強烈な auto-corr が直接効く)
- **公開 top との直交性**: ★★★ (= 公開 top の i.i.d. features に上乗せ可)

### 1.5 推定工数 + 期待 LB 改善

- 工数 **2 日** (= scipy で Kalman 1D 実装 + per-well MLE + features 化)
- 期待 LB 改善 **-0.3 〜 -0.6 ft** (= state-space MAP の優位性)

### 1.6 失敗モード 3 つ

1. **末端 drift**: $\phi < 1$ で長距離 extrap すると mean が 0 へ regress、bias 発生 → 対策: $\phi$ の per-well per-formation での fine fit、または piecewise linear b_well と組合わせ
2. **非ガウス noise**: `dtvt_p95 = 1.01` vs `dtvt_std = 0.42` → fat tail (kurt > 3 推定)、Kalman 最適性失う → 対策: Particle Filter で非ガウス likelihood
3. **Visible 末端の estimate noise**: $\sigma_\varepsilon$ MLE 推定が短い visible (p10=0.18) で不安定 → 対策: hierarchical Bayesian で global prior に shrinkage

### 1.7 既存案との合成

- 案 A (GBM Stack): Kalman MAP を **追加 feature** として入れる (= 1 列、orthogonal)
- 案 B (Sequence Transformer): state-space を decoder の prior として組み込む
- 案 C (Hybrid): plane-fit と Kalman の **2 stage** で fault detection と smoothing 両立

---

## 2. 案 E — Bayesian ANCC Imputation with Uncertainty (Gaussian Process)

### 2.1 構造原理

公開 top は ANCC を **point estimate** (FormationPlaneKNN, X/Y centroid から plane fit) で imput。我々は **GP で posterior** を出し、posterior variance を decoder に渡す = uncertainty を保持して decoding。

### 2.2 着想根拠

- **計測値**: `formula_oracle_rmse_p50 = 0.006 ft` ⇒ formula が真値で実行されれば RMSE ~0、不確実性は ANCC + b_well 推定誤差に集中
- **学術出典**: Rasmussen 2006 (GPML), Hensman 2013 (Sparse GP), Damianou 2013 (Deep GP)
- **公開 top の point estimate** vs **posterior** の差は 0.5-1.0 ft 推定 (= Bayesian quantification 文献の典型改善)

### 2.3 数学的定式

per formation $k$:
$$
\text{ANCC}_k(s) \mid \mathbf{X}_k, \mathbf{Y}_k, \mathbf{Z}_k, \text{formation} \sim \mathcal{GP}\left( \mu_k(\mathbf{X}, \mathbf{Y}), \, K_k\big((\mathbf{X}, \mathbf{Y}), (\mathbf{X}', \mathbf{Y}')\big) \right)
$$

$K_k$ = formation-specific kernel (例: Matérn 3/2 or RBF on $\mathbf{X}, \mathbf{Y}$ centroid)。
posterior:
$$
p\left(\widehat{\text{ANCC}}_k(s) \mid \mathcal{D}\right) = \mathcal{N}\left(m_k(s), v_k(s)\right)
$$

decoder feature: $m_k(s)$, $v_k(s)$, $m_k(s) \pm \sqrt{v_k(s)}$ を全部 LGB に渡す。

### 2.4 ROGII への適合性

- **適合度**: ★★★ (高、ANCC は formation × (X, Y) で連続性が強く、GP が hit する)
- **公開 top との直交性**: ★★★ (= plane fit imputer を GP imputer に置換 + posterior variance feature 追加)

### 2.5 推定工数 + 期待 LB 改善

- 工数 **3 日** (= scikit-learn GP or GPyTorch 実装、6 formation × 776 wells = 大規模なので Sparse GP 必須)
- 期待 LB 改善 **-0.4 〜 -0.8 ft** (= uncertainty quantification の効果)

### 2.6 失敗モード 3 つ

1. **Hyperparam tuning**: GP kernel の lengthscale / signal variance choice → 対策: marginal likelihood maximize で auto-tuning、または ARD kernel
2. **Scalability**: 6 formation × 776 wells × 数百 obs = 数十万点 → 対策: SVGP (Hensman 2013) で inducing points = 数百
3. **Non-stationarity**: formation boundary で kernel 不連続 → 対策: 各 formation で別 GP (= 6 GPs)、boundary smoothing は post-proc

### 2.7 既存案との合成

- 案 A: GP posterior を 6 formation × 2 (mean, var) = 12 features 追加
- 案 C (Geometry-Physics Hybrid): plane fit を GP に置換 = 直接的 augment

---

## 3. 案 F — Differentiable DTW + Cross-Attention (Soft-DTW + PatchTST)

### 3.1 構造原理

公開 top は typewell-horizontal alignment を **離散** Beam Search / DTW で決め、結果を features 化。我々は **soft-DTW** で alignment を differentiable 化し、Transformer cross-attention で end-to-end 学習。

### 3.2 着想根拠

- **計測値**: `tw_gr_resid_std_p50 = 13.89 ft` (= alignment 後の GR residual)、`min_resid_std_p50 = 12.68 ft` (= TVT alignment minimum residual) ⇒ alignment 改善余地あり
- **学術出典**: Cuturi-Blondel 2017 (Soft-DTW), Devlin 2019 (BERT cross-attention), Nie 2023 (PatchTST)
- **公開 top にない**: 離散 Beam Search vs differentiable alignment、後者は学習で alignment と prediction を joint optimize

### 3.3 数学的定式

soft-DTW loss (smoothed):
$$
\text{soft-DTW}_\gamma(x, y) = -\gamma \log \sum_{\pi \in \mathcal{A}} \exp\left( -\frac{1}{\gamma} \sum_{(i,j) \in \pi} d(x_i, y_j) \right)
$$

cross-attention layer:
$$
\text{out} = \text{Softmax}\left( \frac{Q_\text{horizontal} K_\text{typewell}^\top}{\sqrt{d}} \right) V_\text{typewell}
$$

multi-task head: TVT (主) + ΔTVT + Geology label (aux) + soft-DTW path (aux for alignment supervision)。

### 3.4 ROGII への適合性

- **適合度**: ★★ (中-高、200 wells で deep model overfit リスク、pretext pretrain 必須)
- **公開 top との直交性**: ★★★ (= 完全に異なる paradigm)

### 3.5 推定工数 + 期待 LB 改善

- 工数 **4 日** (= soft-dtw-cuda の install, PatchTST 実装, pretext pretrain on FORCE 2020/VOLVE)
- 期待 LB 改善 **-0.5 〜 -1.5 ft** (= deep model + cross-attention の典型成功幅)

### 3.6 失敗モード 3 つ

1. **Overfit**: 200 train wells で深層学習は不足 → 対策: FORCE 2020 (98 wells) / VOLVE (約 200 wells) で pretext pretrain、data aug (well drop / GR mixup / time shift)
2. **Memory**: hidden length 4840 ft × cross-attention = $O(L_h \cdot L_v) = 4840 \cdot 1700 \approx 8M$ → 対策: PatchTST (patch=64 or 128 ft) で sequence 短縮、FlashAttention で memory 削減
3. **Pretext task choice**: masked TVT reconstruction が effective でない → 対策: contrastive pretext (= GR alignment-based positive/negative pair)

### 3.7 既存案との合成

- 案 B (Sequence Transformer) を **置換** (= 案 B は cross-attention の前段が抽象だったが案 F は具体実装)
- 案 A の OOF と stack して final blend

---

## 4. 案 G — Per-Well Adaptive Model (Mixture of Experts)

### 4.1 構造原理

公開 top は **全 wells に同一 model**。我々は visible_ratio / hidden_len で wells を 3-5 cluster に分け、cluster 別に LGB / NN の重みを変える。

### 4.2 着想根拠

- **計測値**: `visible_ratio_p10 = 0.18 vs p90 = 0.35` の差、`hidden_len max = 10052 ft` 等の長尾 → wells 間の問題難度が大きく違う
- **学術出典**: Shazeer 2017 (Outrageously Large Neural Networks: MoE), Jacobs 1991 (Adaptive Mixtures of Local Experts)
- **公開 top にない**: well 別 adaptive な weighting

### 4.3 数学的定式

$$
\hat{y}(s) = \sum_{k=1}^{K} g_k\left(\text{visible\_ratio}, \text{hidden\_len}, \text{tw\_gr\_resid\_std}\right) \cdot f_k(\text{features}(s))
$$

$g_k$ = soft gating (softmax over cluster representatives), $f_k$ = cluster-specific LGB / small NN。

### 4.4 ROGII への適合性

- **適合度**: ★★ (中、cluster 数 K 選択 / gating overfit のリスク)
- **公開 top との直交性**: ★★ (= 中、orthogonal だが効果は moderate)

### 4.5 推定工数 + 期待 LB 改善

- 工数 **2 日**
- 期待 LB 改善 **-0.2 〜 -0.5 ft**

### 4.6 失敗モード 3 つ

1. **Cluster 数選択**: K=3 vs K=5 で効果が変わる → 対策: BIC / silhouette で K を選択、or hard clustering vs soft gating の比較
2. **小 cluster の under-train**: 一番難しい cluster (visible 18%) で train 数少ない → 対策: cluster 内 oversample、または全体 model + cluster-specific residual model
3. **Gating overfit**: gating function が train で過学習 → 対策: gating を visible_ratio only の hard rule に制限、または正則化強

### 4.7 既存案との合成

- 案 A の上に MoE wrapper として追加可
- 案 B/C と並列で OOF を blend

---

## 5. 案 H — MD-Dependent b_well Linear Function

### 5.1 構造原理

公開 top は b_well を **per-formation × per-well 1 値** (median + WLS) で推定。我々は **MD-linear** で fit:
$$
b_{\text{well},k}(s) = b_{0,k} + b_{1,k} \cdot s
$$

### 5.2 着想根拠

- **計測値**: `b_drift_first_to_last_p50_abs = 0.00154 ft`, `b_drift_per_ft_p50 ≈ 1e-6` ⇒ small but non-zero drift
- 物理的根拠: 地質構造の dip 方向に well が drilling すると formation thickness が連続的に変動

### 5.3 数学的定式

per-formation × per-well で WLS (recent-weighted, decay=0.02 が公開 top 値):
$$
\min_{b_{0,k}, b_{1,k}} \sum_s w(s) \left( \text{TVT}(s) + Z(s) - \text{ANCC}_k(s) - b_{0,k} - b_{1,k} \cdot s \right)^2
$$

### 5.4 ROGII への適合性

- **適合度**: ★ (低-中、drift が小なので effect も小)
- **公開 top との直交性**: ★ (= 既存 features の延長、minor improve)

### 5.5 推定工数 + 期待 LB 改善

- 工数 **0.5 日** (= 既存 wls_b_well 関数に slope 項追加)
- 期待 LB 改善 **-0.05 〜 -0.15 ft**

### 5.6 失敗モード 3 つ

1. **Noise overfit**: `b_drift_first_last p90 = 0.0042` のうち真の drift と noise の判別困難 → 対策: drift 推定 95% CI で 0 含めば slope=0 へ pull (regularization)
2. **片側 visible で fit 不安定**: visible が一方の端のみだと slope が global drift と違う方向 → 対策: bootstrap で安定 slope を選ぶ
3. **小 effect**: 全体 RMSE 改善が 0.05 程度 = 計測 noise に埋もれる可能性 → 対策: CV で必ず ablation 検証

### 5.7 既存案との合成

- 案 A の features 1-2 列追加のみ、orthogonal
- 案 D (Kalman) と組み合わせると state vector に b_well slope を入れる formulation 可

---

## 6. 案 I — Diffusion Model for Hidden TVT Distribution

### 6.1 構造原理

公開 top は **point prediction**。我々は **conditional diffusion model** で hidden TVT の posterior を sampling、mean を point prediction に、variance を uncertainty に使う。

### 6.2 着想根拠

- **学術出典**: Ho 2020 (DDPM), Tashiro 2021 (CSDI: Conditional Score-based Diffusion Imputer), Alcaraz 2022 (SSSD-S4)
- **計測値**: `extrap_resid std at MD 4000 ft = 18.83 ft` ⇒ 末端不確実性が大きい、posterior sampling で uncertainty を model 出力に持たせると ensemble 改善
- **公開 top にない**: generative + uncertainty quantification

### 6.3 数学的定式

Conditional score-based diffusion:
$$
\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t \mid \mathbf{c}), \quad \mathbf{c} = (Z, X, Y, \text{GR}_h, \text{GR}_v, \text{TVT}_v, \text{Geo}_v, \text{visible TVT})
$$

Sampling: reverse SDE で hidden TVT を K=20-50 samples、mean = ensemble。

### 6.4 ROGII への適合性

- **適合度**: ★ (低、200 wells で diffusion 学習 + Kaggle 9hr 制約 + 推論コスト高)
- **公開 top との直交性**: ★★★ (完全に異なる paradigm)

### 6.5 推定工数 + 期待 LB 改善

- 工数 **5 日**
- 期待 LB 改善 **-0.3 〜 -1.5 ft** (= diffusion + ensemble の典型)

### 6.6 失敗モード 3 つ

1. **Training 不安定**: small data + score matching loss → 対策: pretrain on FORCE 2020 / VOLVE, EMA model
2. **Sampling 遅い**: K=50 step × 200 wells = Kaggle 9hr 切れる → 対策: DDIM (= 20 step), DPM-Solver
3. **Mode collapse**: posterior が single mode に collapse → 対策: classifier-free guidance, multiple seeds

### 6.7 既存案との合成

- 案 A/B/C の OOF と stack
- 案 E (GP) と uncertainty source として **2 重 quantification**

---

## 7. 比較トレードオフ表

| 案 | 構造原理 | 着想根拠 | 期待 LB | 工数 | 独自性 | 主リスク | 既存案合成 |
|---|---|---|---|---|---|---|---|
| **D** | State-space Kalman/PF on dTVT | ar1_phi=0.999 | -0.3〜-0.6 | 2 日 | ★★★ | 末端 drift / fat tail | 案 A に feature 追加 |
| **E** | Bayesian GP for ANCC posterior | formula_oracle=0.006 | -0.4〜-0.8 | 3 日 | ★★★ | hyperparam / scale | 案 A/C に直接 augment |
| **F** | Soft-DTW + Cross-Attn | tw_gr_resid_std=13.9 | -0.5〜-1.5 | 4 日 | ★★★ | overfit / memory | 案 B 置換 |
| **G** | Per-well MoE | visible_ratio variance | -0.2〜-0.5 | 2 日 | ★★ | cluster K / gate overfit | 案 A wrapper |
| **H** | MD-linear b_well | b_drift=0.00154 | -0.05〜-0.15 | 0.5 日 | ★ | small effect | 案 A feature 追加 |
| **I** | Conditional Diffusion | extrap std 18.83 末端 | -0.3〜-1.5 | 5 日 | ★★★ | training 不安定 / sampling 遅 | 案 A/B/C OOF stack |

## 8. 案 A/B/C との合成 map (= 流路図)

```
                                              ┌── 案 D (Kalman MAP feature)
案 A (GBM Stack) ─────────────────────────────┼── 案 E (GP posterior + var feature)
                                              ├── 案 G (MoE per-well wrapper)
                                              ├── 案 H (MD-linear b_well feature)
                                              └── OOF ─┐
                                                       │
案 B (Sequence Transformer) ──── 案 F に置換 ──── OOF ─┤
                                                       │
案 C (Geometry-Physics Hybrid) ── 案 E で plane-fit 強化 ─ OOF ─┤
                                                       │
案 I (Diffusion) ──────────────────────────────── OOF ─┤
                                                       ↓
                                       Final Stack: Ridge meta + post-proc + median ensemble
                                                       ↓
                                                Sub 1 (CV best) + Sub 2 (LB best)
```

## 9. 補助案 (J/K/L/M, 簡略)

### 9.1 案 J — Two-Stage Stage-Aware Model (Far-Tail Specialist)

- 構造: hidden を MD bin (0-500 / 500-2000 / 2000+) で 3 stage に分け、各 stage で別 model
- 着想: extrap curve mean が MD 9000+ で **-16** の systematic bias、`bin50ft = 191-200` 領域 → 末端 specialist
- 工数 **1-3 日**, 期待 -0.2〜-0.5 ft, リスク stage 境界の discontinuity
- 失敗モード: stage boundary で値ジャンプ / stage-specific overfit / stage feature の definition (visible 末端からの距離が ground truth)

### 9.2 案 K — Graph Neural Network on Well-Pair Similarity

- 構造: 776 wells を node、X/Y 距離 + formation 類似度で edge → GAT で隣接 well の TVT pattern を retrieve
- 着想: Velickovic 2018 (GAT), Kaggle 1st Liverpool 2020 (GNN on station-pair)
- 工数 **4 日**, 期待 -0.2〜-0.5 ft, リスク over-smoothing / scalability
- 失敗モード: graph design (何で edge?) / over-smoothing / hidden region で feature aggregation できない

### 9.3 案 L — Causal Inference Frontdoor (Pearl)

- 構造: well_id を treatment, formation を mediator, TVT を outcome として causal effect を separate
- 着想: Pearl 2009 (Frontdoor adjustment), Bottou 2013 (Counterfactual ML)
- 工数 **3 日**, 期待 0〜-0.2 ft (要実証), リスク identifiability assumption 不成立
- 失敗モード: assumption 違反 / 効果測定難 / 通常 ML より弱い結果

### 9.4 案 M — Conformal Prediction for Sub Selection

- 構造: 各 prediction に conformal interval を付与、final sub 2 つ (CV best vs LB best) の選択時に LB 安全 sub を機械的に選定
- 着想: Vovk 2005 (Conformal Prediction), Romano 2019 (CQR)
- 工数 **1.5 日**, 期待 sub 選択の hedge (LB best vs private LB)
- 失敗モード: calibration が弱いと無意味、conformal interval が上の uncertainty source に依存

## 10. 中央が次に判断すべき選択軸

判断 1: **案 D/E のどちらを最初に組み込むか**
- 案 D: 工数低 + LB 確実改善 + 案 A に feature 追加で済む
- 案 E: 工数中 + LB 改善幅大 + 案 C を強化

判断 2: **案 F (deep model) を Phase 4 で投入するか**
- yes: independent paradigm 確保、LB 6-9 帯狙う
- no: 工数を案 D/E/G の磨き込みに振る、LB 8 帯狙い

判断 3: **案 I (Diffusion) を投入するか**
- yes: uncertainty quantification + 真に新規 paradigm
- no: 工数 5 日のリスク考慮、案 D/E でカバー

判断 4: **案 J (Stage-Aware) を Phase 5 post-proc で入れるか**
- 末端 systematic bias の事実 (`extrap mean = -16` at MD 9000+) を捕捉する最短 path
- 工数低 + 効果モデレート

判断 5: **case L (Causal) を試すか**
- 工数 3 日の博打 / 効果未知
- スキップが現実的

## 11. 関連ファイル

- `docs/research/first-principles.dense.md` (= 母法、計測値の出典)
- `docs/research/top3-distill.dense.md` (= 公開 top 6 要素の詳細)
- `docs/research/strategy-critique.dense.md` (= 案 A/B/C 批判的再評価)
- `outputs/eda/first_principles/per-well-stats.parquet` 等 (= 数値 evidence)
