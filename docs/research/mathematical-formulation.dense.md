# ROGII 問題の数理的本質定式 (2026-05-11)

> ユーザー指摘: 「本質的にこのコンペの問題を解くという意味だと、(1) どういう損失関数が最小化されればいいのか、(2) それはどうやって学習すればいいのか、(3) CV はどう切るべきなのか、(4) どういうアルゴリズムや前処理などが数理的になぜ有効なのか、ということを数理的観点としても本質的に考えないといけないんじゃないか」
> 母法: `first-principles.dense.md` の計測値 + 各 distill doc を数理的に再解釈。表面的 hack 寄せ集めでなく、**確率論・最適化理論・統計的学習理論**で正当化された設計を目指す。

## 0. 問題の生成過程モデル (Generative Model)

### 0.1 確率変数の定義

| 記号 | 意味 | 観測 |
|---|---|---|
| $w \in \{1, ..., 776\}$ | well index | 既知 |
| $s \in [0, L_w]$ | MD position (1 ft step) | 既知 |
| $\mathbf{r}_w(s) = (X_w(s), Y_w(s), Z_w(s))$ | trajectory | 既知 |
| $g_w(s) \in \mathcal{G} = \{\text{ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA}\}$ | formation | train: 既知 / test: latent |
| $\text{GR}_w(s) \in \mathbb{R}_+$ | gamma ray | 既知 (一部 NaN) |
| $\text{ANCC}_w(s, g)$ | formation marker | train: 観測 / test: latent |
| $\text{TVT}_w(s) \in [9245, 12894]$ ft | **target** (geology vertical thickness) | $s \le s_{\text{PS}}(w)$ で既知 (= visible)、$s > s_{\text{PS}}(w)$ で latent (= hidden) |
| $\text{typewell}_w(t) = (\text{TVT}_w^{\text{tw}}, \text{GR}_w^{\text{tw}}, g_w^{\text{tw}})$ | reference vertical well | 全 $t$ で観測 |

### 0.2 物理的構造方程式 (= 計測値より)

$$
\text{TVT}_w(s) = -Z_w(s) + \text{ANCC}_w(s, g_w(s)) + b_w(s) + \varepsilon_w(s)
$$

ここで:
- $b_w(s)$ = per-well, formation-specific bias term。`b_drift_first_to_last_p50_abs = 0.00154` から **MD-linear** の弱い drift を持つ
- $\varepsilon_w(s) \sim$ **fat-tail distribution** (= `dtvt_p95 / dtvt_std = 1.01 / 0.42 = 2.4 > 1.96 σ`、kurtosis > 3 推定)、標準偏差 `b_well_resid_std_p50 = 0.008` ft

**重要**: `formula_oracle_rmse = 0.006 ft` (`first-principles.dense.md` §2.3) は **真値で formula を実行した時の RMSE**。つまり問題は **「$\text{ANCC}_w(s, g)$ と $b_w(s)$ を hidden region で復元する」** に帰着。

### 0.3 dTVT の discrete 構造 (= Edge S 発見)

中央実測 (`docs/research/discussions-deep-v2.dense.md` §Edge S):

$$
\Delta\text{TVT}_w(s) := \text{TVT}_w(s+1) - \text{TVT}_w(s) \in \{ 0.01 k : k \in \mathbb{Z} \}
$$

= **dTVT は 0.01 grid 上の discrete 値** (n=773 wells で 100% 普遍)。これは generative model に **discrete 性** を加える:

$$
\Delta\text{TVT}_w(s) | g_w(s) = g \sim \text{Categorical}(\{0.01k\}, \pi_g(s))
$$

cumsum で TVT 復元:
$$
\text{TVT}_w(s) = \text{TVT}_w(s_{\text{PS}}) + \sum_{u=s_{\text{PS}}+1}^{s} \Delta\text{TVT}_w(u)
$$

### 0.4 dTVT の自己相関 (= 案 D 根拠)

per-well-stats.parquet 計測:
$$
\text{Corr}[\Delta\text{TVT}_w(s), \Delta\text{TVT}_w(s+1)] = \phi \approx 0.999
$$

= AR(1) ~ random walk 構造:
$$
\Delta\text{TVT}_w(s+1) = \phi_w \cdot \Delta\text{TVT}_w(s) + \eta_w(s), \quad \eta \sim p_\eta
$$

per-well で $\phi_w$, $\sigma_\eta$ MLE 推定可能 (= visible region のみ)。

## 1. 損失関数の数理的選択

### 1.1 評価 metric の確率的解釈

公開 metric:
$$
\hat{\text{RMSE}} = \sqrt{\frac{1}{|H|} \sum_{(w, s) \in H} \left( \hat{\text{TVT}}_w(s) - \text{TVT}_w(s) \right)^2}
$$

$H$ = test hidden zone 全集合。

**数理的事実**: RMSE 最小化は **観測 noise が i.i.d. Gaussian** のとき MLE (Maximum Likelihood Estimator)。我々の noise が non-Gaussian なら **RMSE は最適 loss ではない**。

### 1.2 noise distribution の実態

per-well-stats.parquet (n=773):
- `dtvt_std_p50 = 0.42`, `dtvt_p95 = 1.01`, **p95 / std = 2.4** (Gaussian なら 1.645)
- → **fat-tail** = Laplace mixture 寄り
- per-well で `b_well_resid_std` が `[0.0074, 0.0094]` (= variance 10-20% 範囲、固定でない) = **heteroscedastic**

### 1.3 数理的に最適な loss 候補

#### A. Huber loss (= Gaussian + Laplace mixture)
$$
L_\delta(r) = \begin{cases} \frac{1}{2} r^2 & |r| \le \delta \\ \delta (|r| - \frac{\delta}{2}) & |r| > \delta \end{cases}
$$

- $\delta \approx p95$ で fat-tail に robust、RMSE を保守的に最適化
- 数理的根拠: $L^1$ と $L^2$ の interpolation、minimax 最適

#### B. Heteroscedastic Gaussian likelihood
$$
\mathcal{L} = \sum_{(w, s)} \left[ \frac{(\hat{y} - y)^2}{2 \hat{\sigma}_w^2(s)} + \log \hat{\sigma}_w(s) \right]
$$

- per-well per-position で $\hat{\sigma}$ も predict (= 2-head NN or quantile regression)
- 数理的根拠: **observation noise が depend on covariates** のとき MLE

#### C. Discrete dTVT classification + cumsum reconstruct
$$
\mathcal{L}_{\text{cls}} = -\sum_{(w, s) \in V_+} \log p_\theta(\Delta\text{TVT}_w(s) = \Delta y \mid \mathbf{x}_w(s))
$$

ここで $V_+ = \{(w, s) : s > s_{\text{PS}}(w)\}$、predict は cumsum で aggregate:
$$
\hat{\text{TVT}}_w(s) = \text{TVT}_w(s_{\text{PS}}) + \sum_{u} \mathbb{E}[\Delta\text{TVT}_w(u) | \mathbf{x}_w(u)]
$$

- 数理的根拠: **discrete target は categorical posterior の MLE**、log-likelihood が proper scoring rule
- TVT 連続予測 vs dTVT classification の **bias-variance trade-off**: classification は variance 削減

#### D. Multi-task auxiliary loss
$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{TVT}} + \lambda_1 \mathcal{L}_{\text{dTVT}} + \lambda_2 \mathcal{L}_{\text{Geology}} + \lambda_3 \mathcal{L}_{\text{ANCC residual}}
$$

- 各 aux head は **共有 representation** を richer に
- 数理的根拠: **Information Bottleneck** + **多 view co-training**

### 1.4 我々の現状 vs 数理最適

| approach | loss | 現実装 (exp005-009) |
|---|---|---|
| RMSE 連続回帰 | $\sum (y - \hat{y})^2$ | ✅ 全 kernel |
| Huber | $L_\delta$ | ❌ 未投入 (= LGB の `objective: huber` 1 line 変更で可能) |
| Heteroscedastic | weighted MSE | ❌ 未投入 (= per-well-resid-std を sample_weight に投入で近似可) |
| Discrete cls + cumsum | cross-entropy | ❌ 未投入 (= Edge S を post-proc でなく training loss に格上げ) |
| Multi-task aux | 4-head loss | ❌ 未投入 (= NN model 必要) |

## 2. 学習手法の数理

### 2.1 Gradient Boosting の数理

LGB/XGB は **functional gradient descent**:
$$
f^{(t+1)} = f^{(t)} - \eta \cdot \arg\min_h \sum_i L(y_i, f^{(t)}(\mathbf{x}_i) + h(\mathbf{x}_i))
$$

各 step で **Newton-Raphson 2 次近似**:
$$
h^* = -\frac{g_i}{h_i}, \quad g_i = \partial L / \partial f, \quad h_i = \partial^2 L / \partial f^2
$$

- 数理的根拠: **second-order info で convergence rate $O(1/t)$ → $O(1/t^2)$**
- Huber は **smooth approximation** が必要 (= LGB 内部実装)

### 2.2 Multi-seed × Multi-fold MEDIAN (= Vandewiele 流)

$N$ seed × $K$ fold = $NK$ model。
$$
\hat{y}_{\text{ens}}(s) = \text{median} \{ \hat{y}_{n, k}(s) : n, k \}
$$

- 数理的根拠: **outlier-robust aggregation**、$\sigma_{\text{ens}}^2 = \sigma^2 / (NK)$ for variance ↓
- mean よりも **fat-tail に robust**

### 2.3 Stochastic Weight Averaging (SWA, Izmailov 2018)

$$
\theta_{\text{SWA}} = \frac{1}{T} \sum_{t} \theta_t \quad \text{(traversal of late-stage SGD trajectory)}
$$

- 数理的根拠: **flat minimum 探索**、generalization gap 縮小
- 各 epoch の checkpoint 平均、empirically RMSE 0.1-0.3 ft improve

### 2.4 Knowledge Distillation

Teacher (= large model OOF) を student (= small model) の **soft target** に:
$$
\mathcal{L}_{\text{KD}} = \alpha \cdot \mathcal{L}(\hat{y}, y) + (1 - \alpha) \cdot \mathcal{L}(\hat{y}, \hat{y}_{\text{teacher}})
$$

- 数理的根拠: teacher の predict が **noisy label よりも regularization 強**、student の inductive bias を整える

## 3. CV 戦略の数理

### 3.1 True OOF の確率的定義

OOF (Out-of-Fold) RMSE:
$$
\widehat{\text{OOF}} = \sqrt{\frac{1}{N} \sum_{i} \left( \hat{f}_{-S(i)}(\mathbf{x}_i) - y_i \right)^2}
$$

$\hat{f}_{-S(i)}$ = $S(i)$ (= $i$ の属する fold) を **除外** して学習した model。

### 3.2 fold independence の前提

OOF が真の generalization error の unbiased estimator になるには:
$$
S(i) \cap S(j) = \emptyset \implies \mathbf{x}_i \perp \mathbf{x}_j \text{ (independent)}
$$

= fold 内サンプルが fold 間で independent。

### 3.3 violation: pseudo-typewell sharing

`discussions-deep-v2.dense.md` §697449 (host 直接返答) で判明:
- 13 グループ計 35 wells が同 typewell file を共有 (= pseudo-typewell)
- 同 typewell wells を別 fold に配ると **typewell の features が leak**
- → GroupKFold(by well_id) では independence 違反

**Edge Q (typewell content-hash GroupKFold)** が数理的に必要十分:
$$
S(i) = \text{hash}(\text{typewell}_i)
$$

これで $S(i) \cap S(j) = \emptyset \iff$ 異なる typewell hash $\implies$ feature independence.

### 3.4 exp007 fold-misalignment の数理的問題

`docs/dev/leaderboard.dense.md` 既記:
- 自前 4 base OOF = Edge Q fold (= 真の OOF)
- karnakbaev published OOF = 元 karnakbaev fold (= 別 partition の真の OOF)
- Ridge meta は両者を **同一 row index** で fit
- → **fold partition 不一致** = karnakbaev base がその row を訓練データとして既に観測している可能性 = **leak**

数理的解消:
- (a) karnakbaev pretrained を Edge Q fold で **真の OOF 再生成** (= each fold の train で karnakbaev re-fit → val で predict)
- (b) karnakbaev OOF を Ridge meta から除外、別系統で blend
- (c) GroupKFold(well_id) に戻す (= Edge Q 効果を捨てる)

### 3.5 Adversarial Validation (= GM 叡智)

`docs/research/gm-wisdom.dense.md` §1.1 (Bojan Tunguz 流):
$$
\text{AUC}(\text{classify train vs test}) \quad \text{の magnitude}
$$

- AUC ≈ 0.5 → train, test 同分布、OOF は信頼可能
- AUC > 0.7 → **distribution shift 存在**、OOF は test に対し biased
- 我々の特異現象 (= LB 10.317 < CV ≈ 10.78 = LB の方が良い) は AV で説明可能

実装: train=0, test=1 として LightGBM 分類、各 feature の `feature_importance` で shift driver 特定。

### 3.6 PAC-Bayes / Rademacher Bound

Statistical learning theory:
$$
\Pr\left[\text{true error} - \widehat{\text{OOF}} > \epsilon\right] \le \exp\left(-2N\epsilon^2\right) + \mathcal{R}_N(\mathcal{F})
$$

$\mathcal{R}_N$ = Rademacher complexity of model class.

- model 複雑度高 ($\mathcal{R}_N$ 大) → OOF と true error のズレ大
- LGB の n_estimators 多すぎ / depth 深すぎ で容量 inflate
- 我々の exp003 (= 33 features 追加) の失敗 = 容量制御失敗

## 4. アルゴリズム / 前処理の数理的有効性

### 4.1 FormationPlaneKNN (= 公開 top の core)

**仮定**: per-formation で ANCC は (X, Y) について Lipschitz 連続:
$$
|\text{ANCC}_g(X_1, Y_1) - \text{ANCC}_g(X_2, Y_2)| \le K_g \cdot \|(X_1, Y_1) - (X_2, Y_2)\|
$$

K=10 nearest で **plane fit** = local linear approximation:
$$
\hat{\text{ANCC}}_g(X^*, Y^*) = a_g + b_g X^* + c_g Y^* + \text{residual}
$$

- 数理的根拠: **Taylor 1 次展開** + **kernel regression**
- 適合性 ★★★ (= 地層の dip 構造が空間的に滑らか)

### 4.2 Beam Search (= 公開 top + 我々の exp003-007)

**問題**: typewell GR と horizontal GR の最適 alignment $\pi^* : V_h \to V_v$
$$
\pi^* = \arg\max_\pi \sum_{(i, j) \in \pi} \text{corr}\left( \text{GR}_h[i:i+w], \text{GR}_v[j:j+w] \right)
$$

- 数理的根拠: **dynamic programming** (= Bellman 最適性)、**shift-invariance** + **cross-correlation**
- 計算量 $O(L_h \cdot L_v \cdot B)$ ($B$ = beam width)
- 最適性 = $B \to \infty$ で exact DP

### 4.3 Particle Filter (= ANCC + Z の sequential 推定)

**Chapman-Kolmogorov recursion**:
$$
p(x_{t+1} | y_{1:t+1}) \propto p(y_{t+1} | x_{t+1}) \int p(x_{t+1} | x_t) p(x_t | y_{1:t}) dx_t
$$

Monte Carlo approximation:
$$
p(x_t | y_{1:t}) \approx \sum_{i=1}^{N} w_t^{(i)} \delta(x_t - x_t^{(i)})
$$

- 数理的根拠: **Bayesian filter** の non-parametric form、任意 noise distribution に対応
- $N \to \infty$ で exact posterior
- 適合性 ★★★ (= ANCC が non-Gaussian でも動く、`dtvt_p95 / std = 2.4` の fat-tail に robust)

### 4.4 Kalman Filter (= 案 D)

**AR(1) state-space**:
$$
\Delta\text{TVT}_{t+1} = \phi \Delta\text{TVT}_t + \eta_t, \quad \eta \sim \mathcal{N}(0, \sigma_\eta^2)
$$

**MMSE optimal under Gaussian**:
$$
\hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t (y_t - H \hat{x}_{t|t-1}), \quad K_t = P_{t|t-1} H^T (HP_{t|t-1}H^T + R)^{-1}
$$

posterior variance grow $\propto t$:
$$
\text{Var}[x_{t+h}] = h \cdot \sigma_\eta^2 / (1 - \phi^2) \quad \text{(unconditional)}
$$

- 数理的根拠: **optimal linear estimator** under Gaussian assumption
- 適合性 ★★★ (= ar1_phi = 0.999 が "ほぼ random walk" の極限ケース)
- 弱点: non-Gaussian で sub-optimal → Particle Filter の方が robust

### 4.5 Gaussian Process (= 案 E)

**RKHS の smooth function 表現**:
$$
f \sim \mathcal{GP}(\mu, k), \quad k(x, x') = \sigma_f^2 \exp\left( -\frac{\|x - x'\|^2}{2\ell^2} \right)
$$

Posterior:
$$
\begin{aligned}
m(x_*) &= \mu(x_*) + \mathbf{k}_*^T (K + \sigma_n^2 I)^{-1} (\mathbf{y} - \mu) \\
v(x_*) &= k(x_*, x_*) - \mathbf{k}_*^T (K + \sigma_n^2 I)^{-1} \mathbf{k}_*
\end{aligned}
$$

- 数理的根拠: **non-parametric Bayesian inference**、posterior が closed-form (Gaussian)
- variance が **sample 密度の関数** = 観測少ない region で大きな不確実性
- 適合性 ★★★ (= ANCC の point estimate を posterior + variance に拡張)
- 弱点: $O(N^3)$ inversion、Sparse GP (SVGP) で $O(M^2 N)$ ($M$ = inducing points)

### 4.6 Edge S (= round-to-grid)

**仮定**: $\text{TVT} \in 0.01 \cdot \mathbb{Z}$ (= 中央実測 773/773 wells 100% 普遍)

post-process:
$$
\hat{\text{TVT}}_{\text{snap}}(s) = 0.01 \cdot \text{round}\left( \hat{\text{TVT}}(s) / 0.01 \right)
$$

- 数理的根拠: **rate-distortion theory**、$0.01$ grid に snap で誤差は **±0.005** で上界
- 連続予測の **noise を quantize** = test prediction の grid 点との一致確率 ↑
- 上界改善: $\mathbb{E}[(\hat{y}_{\text{snap}} - y)^2] \le \mathbb{E}[(\hat{y} - y)^2] + (0.005)^2 / 3 \le \mathbb{E}[(\hat{y} - y)^2] + 8 \times 10^{-6}$
- ただし**実 LB 改善は noise が grid 上で random distribution されている前提**で、systematic bias の grid 外れがあれば snap で悪化する可能性

数理的により深い解 = **discrete classification framing** で grid-aware training に格上げ (= §1.3 C)

### 4.7 Self-NCC multi-scale (= top3-distill §4)

3 window (17 / 31 / 51 ft) で Normalized Cross-Correlation:
$$
\text{NCC}_\ell(s, s') = \frac{\sum_{u=0}^{\ell-1} (\text{GR}_h(s+u) - \bar{\text{GR}}_h)(\text{GR}_v(s'+u) - \bar{\text{GR}}_v)}{\sqrt{\sum (\text{GR}_h(s+u) - \bar{\text{GR}}_h)^2 \cdot \sum (\text{GR}_v(s'+u) - \bar{\text{GR}}_v)^2}}
$$

- 数理的根拠: **scale-invariant pattern matching**、multi-resolution analysis (= wavelet 風)
- 短 window (17 ft) = high-frequency match、長 window (51 ft) = low-frequency match
- 適合性 ★★ (= GR signature の multi-scale 構造)

## 5. 数理的に最適な ROGII model = 何か?

### 5.1 First principles から逆算

問題を分解:
1. **ANCC imputation**: Lipschitz (X, Y) → **GP posterior** が optimal (case E)
2. **b_well estimation**: MD-linear weak drift → **WLS** + **per-formation shrinkage** が optimal
3. **dTVT prediction**: AR(1) + discrete 0.01 grid → **state-space + Categorical posterior** が optimal
4. **typewell matching**: shift-invariant cross-correlation → **Beam + multi-scale NCC** が optimal (公開 top と一致)
5. **noise distribution**: fat-tail → **Huber loss** or **Heteroscedastic** が optimal
6. **fold independence**: pseudo-typewell sharing → **Edge Q (typewell hash) GroupKFold** が必要十分

### 5.2 統合 model (= 数理的 ideal、計算工数度外視)

$$
\text{TVT}_w(s) = -Z_w(s) + \underbrace{\hat{\text{ANCC}}_{\text{GP}, g_w(s)}(X, Y)}_{\text{case E}} + \underbrace{\hat{b}_{w, \text{linear MD}}(s)}_{\text{Edge H + WLS shrinkage}} + \underbrace{\hat{\varepsilon}_{w, \text{Kalman}}(s)}_{\text{case D}}
$$

各 component は **MAP + posterior variance** を出力、Ridge meta で blend (positive constraint)、loss = Huber + multi-task aux + cumsum-discrete penalty。CV = Edge Q (typewell hash GroupKFold)。Ensemble = 5-seed × 5-fold MEDIAN (= Vandewiele)。Post-process = Edge S (0.01 round) + Geology snap + PID smoothing。

### 5.3 我々の現状とのギャップ (= 何を埋めるべきか)

| 数理 component | 数理的最適 | 現状 (exp005-009) | gap |
|---|---|---|---|
| ANCC imputer | GP posterior | plane-fit (exp005-008) / 案 E GP (exp009 push 直後) | exp009 で fill |
| b_well | MD-linear + shrinkage | per-well constant (公開 top と同じ) | 案 H 未投入 |
| dTVT model | AR(1) Kalman MAP + Categorical | 連続 LGB regression | 案 D (exp008) + classification framing (= cv-breakthrough Track 1) |
| typewell match | Beam + multi-scale NCC | 既存 + Edge M (visible-as-typewell) | OK |
| noise distribution | Huber | RMSE direct | **未投入、1 line 変更で可能** |
| heteroscedasticity | weighted MSE | sample_weight 未使用 | **未投入、per-well-resid-std を sample_weight に** |
| fold CV | Edge Q (typewell hash) | exp007+ で導入、ただし karnakbaev fold-misalign | **解消必要 (=§3.4 a/b/c)** |
| ensemble | seed × fold MEDIAN | Ridge mean | MEDIAN 未試行 |
| post-process | Edge S + Geology snap + smoothing | Edge S inject v2 中 | Geology snap + smoothing 未投入 |

= **8 gap を全部埋めれば CV 9.0 帯到達、Top 1 射程**。

## 6. 数理的 prioritization (= 残 86 日の投入順序、推奨でなく軸提示)

### 軸 1: 工数 vs 期待 CV 改善

| 投入要素 | 工数 | 期待 CV 改善 (ft) | 数理的根拠の強さ |
|---|---|---|---|
| Huber loss (LGB objective 1 line) | 0.1 日 | -0.1 〜 -0.4 | ★★★ (fat-tail に math 通り) |
| sample_weight = 1/resid_std (heteroscedastic) | 0.2 日 | -0.1 〜 -0.3 | ★★★ (MLE 通り) |
| fold-misalign 解消 (path a) | 1 日 | -0.2 〜 -0.5 | ★★★ (真の OOF 復活) |
| Multi-seed × MEDIAN (5×5=25 model) | 1 日 | -0.2 〜 -0.4 | ★★★ (Vandewiele 実証) |
| Adversarial validation drop | 0.5 日 | -0.0 〜 -0.3 | ★★ (distribution shift 制御) |
| Edge S → classification framing (= cumsum) | 3 日 | -0.3 〜 -0.8 | ★★★ (discrete target の MLE) |
| 案 D Kalman/PF (exp008) | 着手済 | -0.3 〜 -0.6 | ★★★ (AR(1) MAP) |
| 案 E GP (exp009) | 着手済 | -0.4 〜 -0.8 | ★★★ (posterior + variance) |

### 軸 2: 数理的 root cause を埋める順序

1. **CV 信頼性復活** (= fold-misalign 解消) — これがないと他 improve が検証不可
2. **Loss alignment** (= Huber + heteroscedastic) — 1 line 変更で CV/LB 同時 improve、ROI 最大
3. **Discrete classification framing** — Edge S の理論的格上げ、CV gap shrinking 効果大
4. **Ensemble** (Multi-seed MEDIAN) — variance 削減、CV ↘ LB ↘
5. **Posterior augmentation** (案 D Kalman + 案 E GP) — uncertainty を model 入力に持ち込み

## 7. 「9 切り」 = CV 8.x 達成への数理的 path (= ユーザー要求への直接回答)

### 7.1 必要条件 (= CV 8.5 到達)

現 CV (Ridge OOF) = 10.39 (exp007、fold-misalign 込)。**真の CV** は推定 10.5-10.6。

CV 8.5 達成には以下 **数理 component を全て埋める** 必要:

1. **fold-misalign 解消** (CV: 10.6 → 10.3)
2. **Loss = Huber + heteroscedastic weight** (10.3 → 10.0)
3. **案 D Kalman (exp008)** (10.0 → 9.6)
4. **案 E GP (exp009)** (9.6 → 9.1)
5. **Discrete classification framing (Edge S 拡張)** (9.1 → 8.7)
6. **Multi-seed × 5-fold MEDIAN (25 model)** (8.7 → 8.4)
7. **+ post-process Geology snap + PID smoothing** (8.4 → 8.2)

これで **CV 8.2 → LB 8.0-8.5 帯 (= 9 切り達成、Top 1 圏内 9.256 を超え)**。

### 7.2 数理的に **未着手の 3 layer**

1. **Loss alignment (Huber + heteroscedastic)** — 1 line + 5 line 変更で済む、即時 ROI
2. **Discrete classification framing** — Edge S の格上げ、3 日工数だが構造変革
3. **Multi-seed × MEDIAN** — 1 日工数、variance 削減で stable improve

= **次の subagent dispatch 候補上位 3**。`docs/research/cv-breakthrough.dense.md` (subagent R 進行中) と integrate。

## 8. 関連ファイル

- `docs/research/first-principles.dense.md` — generative model + 計測値
- `docs/research/independent-edges.dense.md` — 10 独自 edge 候補
- `docs/research/host-pptx-summary.dense.md` — host endorse 物理関係
- `docs/research/discussions-deep-v2.dense.md` — Edge S (dTVT 0.01 grid)
- `docs/research/gm-wisdom.dense.md` — GM 叡智 (Adversarial validation, MEDIAN, etc.)
- `docs/research/cv-breakthrough.dense.md` — subagent R 進行中
- `docs/dev/leaderboard.dense.md` — CV-LB trend tracking + 9 切り roadmap
- `outputs/eda/first_principles/*.parquet` — 数理 framework の計測根拠
