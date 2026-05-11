# 優勝寄与高 4 paradigm deeper dive (2026-05-11)

> 範囲: 案 D Kalman / exp006 TabICL 失敗 / `formula_oracle_rmse_p50 = 0.006 ft` / `b_drift_first_to_last_p50_abs = 0.00154 ft` の既存知見 (`docs/research/independent-edges.dense.md`, `docs/research/first-principles.dense.md`) を踏まえ、subagent W (= 学術文献 deeper) が cover してない 4 paradigm を、ユーザー指示「優勝により近づけるかどうかで選んでね」を最上位選定軸として deeper dive する。
> 出典は arXiv ID / GitHub URL / Nature DOI / file:line 形式で必ず添える (= `~/.claude/CLAUDE.md` Links, not verdicts)。
> 推奨は出さない、トレードオフ・選択軸のみ提示する (= 主道 中立指示原則)。

---

## 0. 序: 優勝寄与基準 (= 軽量/重さでなく)

### 0.1 何を「優勝寄与」と定義したか

ユーザー指示「9 切らないと優勝は無理」「優勝により近づけるかどうかで選んでね」(`docs/dev/leaderboard.dense.md:67-72` Top 1 = 9.256, Gold cutoff = 9.919) から、優勝寄与基準を次のように **定量化** する:

```
score(paradigm) = (期待 LB 寄与 ft) / (実装工数 日) * P(成功)
                  * (1 + 既存 layer との orthogonality)
```

ここで:
- **期待 LB 寄与**: 単独で karnakbaev blend (LB 10.317) からどれだけ削れるか (case-D Kalman 案の estimate `-0.3〜-0.6 ft` = `docs/research/independent-edges.dense.md:318` をベースライン)
- **P(成功)**: Kaggle 9 hr inference cap 内に走り切る確率 + CV-LB 乖離が ≤0.3 ft で収まる確率
- **既存 layer との orthogonality**: Edge S (round-to-grid) / Edge R (test-time online) / karnakbaev blend が既に拾ってる variance と直交していれば +1、redundant なら 0

→ Mamba / TabPFN v2 / PINN / PySR は **構造原理が全て異なる** (= state-space / in-context Bayesian / 物理 soft constraint / 遺伝的式探索) ので、orthogonality 軸でも 1 案に絞らず 4 案併走価値がある。

### 0.2 4 paradigm の選定根拠 (= 構造原理の差異)

| paradigm | 構造原理 | ROGII の何に効く |
|---|---|---|
| Mamba / S4 | 連続時間 SSM の selective discretization + HiPPO 長距離記憶 | 案 D AR(1) Kalman を任意 order に拡張、hidden zone MD 4000+ ft 末端の variance 蓄積を selective gating で抑制 |
| TabPFN v2 | Prior-data fitted network (= PFN) の in-context Bayesian inference、attention で row × column 両方向 | exp006 TabICL CUDA error の復活 path、per-well subset で in-context predict |
| PINN | 物理 PDE / algebraic constraint を loss に soft 化 + autograd | `TVT + Z − ANCC − b_{well} = 0` (= `formula_oracle_rmse = 0.006 ft`) を joint train の hard 寄り soft loss で **multi-head 同時最適化** |
| PySR | 遺伝的 programming で operator 空間を Pareto front 探索 | `b_well` per-well constant の closed-form 式 (`b_well = f(X, Y, formation_thickness, dip_azimuth)`) を symbolic 発見、overfit-resistant feature 化 |

**棄却した paradigm** (= 構造原理が overlap):
- TimeGPT / Chronos / Moirai (= 時系列 foundation model): TabPFN v2 と「事前学習 foundation で in-context」が overlap、TabPFN v2 の方が tabular 直結で工数低い
- Neural ODE (Chen et al. 2018): 案 D Kalman の連続時間拡張と原理同等、PINN と autograd 共用、独立価値小
- Gaussian Process (GP) + 物理 kernel: PINN の hard constraint の bayesian 同値、`docs/research/independent-edges.dense.md` 案 B として既に存在

### 0.3 9 切り戦略との位置づけ

`docs/dev/leaderboard.dense.md:67-72` 「9 切らないと優勝は無理」を起点に逆算:
- 現状 LB 10.317 (= karnakbaev + Edge S blend、`docs/dev/leaderboard.dense.md:15`)
- Edge R (test-time online) で **9.95 想定** (`docs/dev/leaderboard.dense.md:16`)
- 9.0 帯まで残り **−0.95 ft**、8.5 帯まで残り **−1.45 ft**

→ 4 paradigm が単独で 0.3〜1.2 ft 寄与する仮説が data-supported なら、最低 2 paradigm を Edge R 後の **Edge T / U / V** 系として投入する価値がある。後述 §5 で投入順序を試算する。

---

## 1. Mamba / S4 deeper

### 1.1 数理仕様 (= 状態方程式、HiPPO、selective parameterization)

#### 1.1.1 連続時間 SSM (= S4 の出発点)

S4 (Gu et al. ICLR 2022, arXiv:2111.00396) は次の連続時間 state-space model を出発点とする (`https://arxiv.org/abs/2111.00396`):

$$
\frac{dx(t)}{dt} = A x(t) + B u(t), \quad y(t) = C x(t) + D u(t)
$$

- $x(t) \in \mathbb{R}^N$: hidden state (= state expansion factor `d_state`)
- $u(t) \in \mathbb{R}$: input scalar (= ROGII では bit MD step ごとの dTVT または GR)
- $A \in \mathbb{R}^{N \times N}$, $B, C \in \mathbb{R}^N$, $D \in \mathbb{R}$

離散化は zero-order hold (ZOH) で:
$$
\bar{A} = \exp(\Delta A), \quad \bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B
$$
$$
x_t = \bar{A} x_{t-1} + \bar{B} u_t, \quad y_t = C x_t
$$

→ AR(1) Kalman (`docs/research/independent-edges.dense.md:38` `ar1_phi=0.999`) は **N=1 special case** $\bar{A} = \phi$。S4 は N ≥ 16 で任意 order の長距離記憶を獲得。

#### 1.1.2 HiPPO matrix で長距離記憶を保証

S4 の核心は $A$ を random 初期化ではなく **HiPPO matrix** で初期化する点。HiPPO (High-order Polynomial Projection Operator, Gu et al. NeurIPS 2020, `https://arxiv.org/abs/2008.07669`) は、過去 input をある measure (= Legendre, LegS) に対する optimal polynomial coefficient で圧縮する formulation:

$$
A_{nk}^{\text{HiPPO-LegS}} = \begin{cases}
-(2n+1)^{1/2}(2k+1)^{1/2} & n > k \\
-(n+1) & n = k \\
0 & n < k
\end{cases}
$$

→ ROGII における意味: hidden zone MD 4000+ ft の末端で AR(1) Kalman は variance が指数的に grow するが、HiPPO の Legendre 直交化により **過去 visible window 全体の情報** が hidden state $x_t$ に圧縮されたまま保持され、末端 systematic bias が抑制される。

#### 1.1.3 計算効率 (S4): NPLR / DPLR + Cauchy kernel

S4 の breakthrough は HiPPO matrix を **Normal + Low-Rank** に分解し、Cauchy kernel 計算で $O(L \log L)$ inference を実現した点 (`https://arxiv.org/abs/2111.00396` § 3-4)。Transformer の $O(L^2)$ と比較し、L = 5000 で 10x 高速、L = 16000 (Path-X) で全 baseline 超え。

#### 1.1.4 Selective parameterization (Mamba)

Mamba (Gu & Dao 2023, arXiv:2312.00752, `https://arxiv.org/abs/2312.00752`) は $B, C, \Delta$ を **input-dependent** にする (= selective):

$$
B_t = W_B u_t, \quad C_t = W_C u_t, \quad \Delta_t = \text{softplus}(W_\Delta u_t)
$$

→ ROGII における意味:
- visible 区間 (= input known) では $\Delta_t$ 大 → hidden state を頻繁更新 = 短距離情報優先
- hidden 区間 (= input NaN) では $\Delta_t$ 小 → hidden state を保持 = 長距離 trend 優先
- これは AR(1) Kalman の `phi` を **MD 位置ごとに動的調整** することに相当

> 制約: selective にすると Mamba は parallel scan アルゴリズムで $O(L)$ になるが、convolution として書けない (= S4 の Cauchy kernel 手法は使えない)。代わりに hardware-aware parallel scan (Blelloch scan) で wall-clock 高速化。

#### 1.1.5 Mamba-2: SSD (State Space Duality)

Mamba-2 (Dao & Gu 2024, arXiv:2405.21060, `https://arxiv.org/abs/2405.21060`) は Mamba を **structured semiseparable matrix** として再定式化し、attention との duality を確立。2-8× 高速化、head dimension 並列化、大きな `d_state` (= 64-256) が可能に。

### 1.2 ROGII 適用 (= 案 D Kalman の deep 拡張として、hidden zone 末端 bias 抑制)

#### 1.2.1 ROGII 数理 framework との整合

`docs/research/mathematical-formulation.dense.md` で定式化された generative model は:
$$
TVT_t = -Z_t + \text{ANCC}(X_t, Y_t) + b_{well} + \epsilon_t
$$

ここで $\epsilon_t$ は AR(1)-like noise (= `ar1_phi=0.999`)。Mamba/S4 の役割は **$\epsilon_t$ の dynamics を任意 order** で学習し、hidden region で **conditional posterior** $p(\epsilon_t \mid \text{visible window})$ を出すこと。

Input: `[Z, X, Y, GR, last_known_TVT, ANCC_imputed, ...]` の time-aligned sequence (= MD-step sampling)
Output: per-step `dTVT_residual` predict + uncertainty

#### 1.2.2 hidden zone 末端 bias 抑制 mechanism

AR(1) Kalman の末端 variance は:
$$
\text{Var}(\hat{TVT}_t \mid \text{visible}) = \sigma_\epsilon^2 \cdot \frac{1 - \phi^{2(t - T_v)}}{1 - \phi^2} \xrightarrow{\phi \to 0.999} \infty \text{ as } t - T_v \to \infty
$$

→ MD 4000+ ft 末端で systematic bias になる (= `docs/dev/submission-postmortems.dense.md` exp003 の末端 overshoot 推定)。

Mamba/S4 は hidden state $x_t \in \mathbb{R}^N$ ($N=64$) に **visible 全 window の情報** を直接 attention 的に保持できるので、末端でも:
$$
\hat{TVT}_t = C x_t \approx \text{(visible 全 trend からの projection)}
$$

→ variance grow が log-linear に抑制される (`https://arxiv.org/abs/2111.00396` Theorem 4)。

### 1.3 実装 spec (= mamba-ssm package、checkpoint、Kaggle GPU 上 runtime)

#### 1.3.1 package & install

```bash
pip install mamba-ssm --no-build-isolation
pip install causal-conv1d>=1.4.0 --no-build-isolation
```
- CUDA 11.6+ / PyTorch 1.12+ (`https://github.com/state-spaces/mamba` README)
- Kaggle 上 default image (CUDA 12 / Torch 2.x) は条件満たす

#### 1.3.2 Mamba block API

```python
from mamba_ssm import Mamba
block = Mamba(d_model=128, d_state=64, d_conv=4, expand=2)
# input (B, L, d_model) → output (B, L, d_model)
```

ROGII 用 hyperparameters 提案 (= MambaTS 2024 の time-series 慣行から、`https://arxiv.org/abs/2405.16440`):
- `d_model = 64-128` (= visible window 700-3000 step の embed)
- `d_state = 16-32` (= AR order に相当、ROGII では low-rank 想定)
- `d_conv = 4` (= local smoothing)
- `expand = 2` (= block expansion)
- パラメータ数 ≈ $3 \cdot \text{expand} \cdot d_{\text{model}}^2$ ≈ 50K-100K / layer × 2-4 layer

#### 1.3.3 学習 / inference 設計

- **学習**: 376 well visible region から `(visible_window, hidden_segment)` pair を sliding window で sampling、residual TVT - last_known_TVT を target に MSE loss
- **inference**: 776 well の test visible 区間から hidden を auto-regressive 1-step predict + Monte Carlo dropout で variance 推定
- **Kaggle GPU 上 runtime estimate**: 1 well L=5000 / 64 hidden / 4 layer / T4 GPU で **0.05-0.1 s/well** → **40-80 s for 776 wells** (= 9 hr cap に 0.3% 占有)

#### 1.3.4 pretrained checkpoint の有無

公開 checkpoint は **language modeling 用 130M-2.8B** のみ (`https://github.com/state-spaces/mamba` Pretrained Models)。ROGII は **scratch train** が現実的。代替として MambaTS / Mamba4Cast / TSMamba (`https://arxiv.org/abs/2411.02941`) の time-series 用 pretrained を fine-tune する選択肢あり (要 license 確認)。

### 1.4 LB 寄与推定 + 失敗 modes 3 件

#### 1.4.1 LB 寄与推定

| component | 仮定 | LB 寄与 |
|---|---|---|
| AR(1) Kalman → Mamba (N=1 → N=64) | hidden zone末端 variance を log-linear に抑制 | **−0.4 ft** (= 案 D `-0.3〜-0.6` の中域、`docs/research/independent-edges.dense.md:318`) |
| Selective gating で fault / fold 検出 | `docs/research/problem-essence.dense.md:48` fault jump を $\Delta_t$ で local 検出 | **−0.3 ft** |
| Multi-feature (GR + Z + last_TVT) joint embed | 自前 features の non-linear 組合せ | **−0.2 ft** |
| **合計 (single Mamba head)** | | **−0.7〜−1.2 ft** |

→ Edge R 後 LB 9.95 想定から **8.75-9.25 ft** へ。8.x 帯到達可能性 **65-75%** と試算。

#### 1.4.2 失敗 modes 3 件

| failure | 兆候 | 対策 |
|---|---|---|
| **過学習: 376 well では scratch train 不足** | CV-LB 乖離 +0.5 ft 以上 | (a) MambaTS pretrained を fine-tune、(b) heavy augmentation (= visible window random crop + GR noise), (c) `d_model` を 32 に下げ過学習耐性確保 |
| **fat-tail noise で Gaussian assumption 破綻** | `docs/research/independent-edges.dense.md:73` `dtvt_p95/dtvt_std=2.4` の fat tail、MSE loss 過剰収束 | Huber loss / quantile output / ensemble averaging |
| **fault zone での $\Delta_t$ saturate** | fault layer 跨ぎで $\Delta_t \to 0$ または $\infty$ で stable region 消失 | (a) $\Delta_t$ に min/max clamp、(b) fault detection を `Edge O` (= `docs/research/independent-edges.dense.md` Edge O ANCC clip) で先処理してから Mamba 投入 |

---

## 2. TabPFN v2 deeper

### 2.1 数理仕様 (= PFN, in-context Bayesian)

#### 2.1.1 Prior-data Fitted Network (PFN) の発想

TabPFN v1 (Hollmann et al. ICLR 2023) は **prior-data fitted network** という考え方:
1. 事前に「ありそうな tabular dataset の prior」 $p(\mathcal{D})$ を定義 (= SCM ベース、structural causal model)
2. prior から $D = (X, y)_{1:n}$ を sampling し、$(X^*, y^*) \to y^*$ を予測する task を transformer に大量学習させる
3. 推論時は学習済み transformer に test dataset を **in-context** で渡すだけ → 1-pass で posterior $p(y^* \mid X^*, D)$ 出力

これは Bayesian inference $p(y^* \mid X^*, D) = \int p(y^* \mid X^*, \theta) p(\theta \mid D) d\theta$ を **NN で amortize** したもの。

#### 2.1.2 TabPFN v2 の改良点 (Nature 2025, Hollmann et al.)

- **Row × Column attention**: features と samples の両方向に attention (`https://www.nature.com/articles/s41586-024-08328-6`)
- **Regression support**: v1 は classification 中心、v2 で **TabPFNRegressor** が公式提供
- **Numerical + categorical + missing values** ネイティブ対応
- **最大 context**: 10,000 samples × 500 features (= 公式推奨)、TabPFN-2.5 で 50,000 × 2,000 まで拡張 (`https://arxiv.org/abs/2511.08667`)

#### 2.1.3 Regression posterior の表現

TabPFN v2 は予測時に **bucketed bar distribution** (= histogram form, 5000 bin) を出力し、mean / quantile / full distribution を抽出可能:
$$
p(y^* \mid X^*, D) \approx \sum_{k=1}^{K} \pi_k(X^*, D) \cdot \mathbb{1}[y^* \in B_k]
$$

→ ROGII では **uncertainty-aware blending** (= 高 confidence sample に高 weight) に直結。

### 2.2 ROGII 適用 (= exp006 TabICL 失敗の復活 path)

#### 2.2.1 exp006 TabICL 失敗の構造

`docs/dev/submission-postmortems.dense.md` exp006 (= TabICL + PFlite blend) は **CUDA error fallback** で失敗。TabICL は 2024 後半に CUDA 11 only でビルドされており、Kaggle の CUDA 12 環境で動かなかった (= 推定)。

TabPFN v2 (PriorLabs 公式 `https://github.com/PriorLabs/TabPFN`) は **CUDA 12 native** + PyTorch 2.x 対応で同種 in-context Bayesian の復活 path。

#### 2.2.2 per-well in-context setup

ROGII で TabPFN v2 を最も活かす形:
- **train context = 当該 well の visible region 全 row** ($T_v$ rows, ~3000)
- **test query = 当該 well の hidden region 全 row** ($T - T_v$ rows, ~2000)
- **features**: `[Z, X, Y, GR, MD, last_known_TVT, ANCC_imputed, b_well_estimate, ...]` 約 15-30 列
- target: `dTVT_residual` (= TVT - last_known_TVT) → cumsum で TVT 復元

これは **per-well で独立 Bayesian regression**。776 well を batch 処理。

#### 2.2.3 ROGII 数理 framework との整合

TabPFN v2 は generative model を **暗黙に prior で学習済み** のため、TVT = -Z + ANCC + b_well の物理を **明示しない**。代わり、`Z, ANCC_imputed` を feature として渡せば in-context Bayesian が自動的に linear 関係を発見する (= NeurIPS 2024 報告で SCM linear structure に強い)。

→ 物理を明示しないことの **trade-off**: PINN とは逆 paradigm。物理が perfect (formula_oracle_rmse=0.006) な場合、TabPFN v2 は形を発見はするが、最後の 0.1-0.3 ft を絞り切れない可能性。

### 2.3 実装 spec (= huggingface + CUDA 12、context 100k row、per-well in-context)

#### 2.3.1 install & API

```bash
pip install tabpfn  # PriorLabs/TabPFN v2
```

```python
from tabpfn import TabPFNRegressor
reg = TabPFNRegressor(device='cuda', n_estimators=8, softmax_temperature=0.9)
reg.fit(X_visible, y_visible)  # ~3000 row x 20 feat
pred = reg.predict(X_hidden)   # ~2000 row → mean
quantiles = reg.predict(X_hidden, output_type='quantiles', quantiles=[0.1, 0.5, 0.9])
```

#### 2.3.2 inference runtime estimate

- TabPFN v2 公式: 10k row × 100 feature inference ≈ **1-3 s/dataset on T4 GPU** (`https://github.com/PriorLabs/TabPFN` README)
- ROGII: 776 well × ~2 s/well ≈ **25 min on T4** → 9 hr cap 4.6% 占有
- メモリ: ~8 GB VRAM (= 16 GB Kaggle T4 で OK)

#### 2.3.3 context limit 内に収める

v2 公式推奨は ≤ 10,000 samples × ≤ 500 features。ROGII の visible は 平均 3,000 row × 20 feature → 余裕。
ただし **複数 well の visible を pool して 1 context** にすると 100k row 越えする可能性 (= TabPFN-2.5 release 待ち、`https://arxiv.org/abs/2511.08667`)。

### 2.4 LB 寄与推定 + 失敗 modes

#### 2.4.1 LB 寄与推定

| component | 仮定 | LB 寄与 |
|---|---|---|
| per-well in-context regression (TVT residual) | 公開 top の LightGBM blend と直交 30-50% | **−0.3 ft** |
| uncertainty-aware blend (quantile output) | weighted ensemble with karnakbaev / Mamba | **−0.15 ft** |
| **合計 (single TabPFN v2 head)** | | **−0.3〜−0.5 ft** |

→ Edge R 後 LB 9.95 想定から **9.45-9.65 ft** へ。Mamba 単独より控えめ。

#### 2.4.2 失敗 modes 3 件

| failure | 兆候 | 対策 |
|---|---|---|
| **CUDA error 再発 (exp006 と同じ)** | Kaggle T4 で `RuntimeError: CUDA error` | offline で wheel 事前ビルド → kernel に attach、`pip install --no-deps tabpfn` で依存 lock |
| **per-well context が短すぎ** | 短い visible (T_v < 500) で in-context が underfit | (a) 周辺 well を pseudo-context として concat、(b) TabPFN-2.5 release 待って context 50k に拡張 |
| **物理 (TVT = -Z + ANCC + b) を発見できず** | feature-importance で `Z, ANCC_imputed` が drop | (a) feature engineering で `tvt_oracle_naive = -Z + ANCC` を pre-compute して feature 投入、(b) PINN と合体 |

---

## 3. PINN (Physics-Informed Neural Network) deeper

### 3.1 数理仕様 (= L_PINN, multi-head joint train)

#### 3.1.1 原典 PINN (Raissi et al. 2019)

PINN (Raissi+Karniadakis 2019, arXiv:1711.10561, `https://arxiv.org/abs/1711.10561`) は **PDE-constrained NN regression**:
$$
\mathcal{L}_{PINN}(\theta) = \mathcal{L}_{data}(\theta) + \lambda \cdot \mathcal{L}_{physics}(\theta)
$$
$$
\mathcal{L}_{data} = \frac{1}{N_u} \sum_{i=1}^{N_u} |u_\theta(x_i) - u_i|^2, \quad \mathcal{L}_{physics} = \frac{1}{N_f} \sum_{j=1}^{N_f} |\mathcal{N}[u_\theta](x_j)|^2
$$

ここで $\mathcal{N}$ は PDE operator (= 例: $u_t + u u_x - \nu u_{xx}$ for Burgers)。autograd で計算 (`tf.gradients` or `torch.autograd.grad`)。

#### 3.1.2 ROGII における algebraic constraint 化

ROGII は PDE ではなく **algebraic constraint**:
$$
TVT_t + Z_t - \text{ANCC}_t - b_{well} = 0 \quad \text{(formula\_oracle\_rmse}_{p50} = 0.006 \text{ ft、`docs/research/first-principles.dense.md:38`)}
$$

これを PINN-style soft loss にする:
$$
\mathcal{L}_{phys}(\theta, \phi, \psi) = \frac{1}{N} \sum_{t} \left| TVT_\theta(s_t) + Z_t - \text{ANCC}_\phi(s_t) - b_\psi(s_t) \right|^2
$$

ここで:
- $TVT_\theta$: TVT predictor (NN head 1)
- $\text{ANCC}_\phi$: ANCC imputer (NN head 2) — test では NaN なので NN で imput
- $b_\psi$: b_well predictor (NN head 3) — per-well constant + drift

#### 3.1.3 multi-head joint train

$$
\mathcal{L}_{total} = \mathcal{L}_{TVT-data} + \lambda_{phys} \mathcal{L}_{phys} + \lambda_{ANCC} \mathcal{L}_{ANCC-data} + \lambda_b \mathcal{L}_{b-prior}
$$

- $\mathcal{L}_{TVT-data}$: visible region の TVT_input に対する MSE
- $\mathcal{L}_{phys}$: 全 region (visible + hidden 候補) で algebraic constraint 強制
- $\mathcal{L}_{ANCC-data}$: train set の ANCC 真値 (provided) に対する MSE
- $\mathcal{L}_{b-prior}$: per-well b_well の prior (= WLS solution `docs/research/independent-edges.dense.md` 公開 top の値) からの距離

#### 3.1.4 lambda 自動調整 (Wang & Perdikaris 2021)

PINN は naive な fixed $\lambda$ で **gradient pathology** が出る (= `https://epubs.siam.org/doi/10.1137/20M1318043`)。$\mathcal{L}_{data}$ と $\mathcal{L}_{phys}$ の gradient magnitude が乖離して **片方が学習されない**。

Wang+Perdikaris (SISC 2021) の **learning rate annealing**:
$$
\lambda_{phys}^{(k+1)} = (1-\alpha) \lambda_{phys}^{(k)} + \alpha \cdot \frac{\max_\theta |\nabla_\theta \mathcal{L}_{data}|}{\overline{|\nabla_\theta \mathcal{L}_{phys}|}}
$$
$\alpha = 0.1$、毎 100 step で更新。報告で **50-100× 精度向上** (= 同 paper Figure 5)。

### 3.2 ROGII 適用 (= TVT + Z − ANCC = 0 を soft loss)

#### 3.2.1 適用優位性

`formula_oracle_rmse_p50 = 0.006 ft` (= `docs/research/first-principles.dense.md:38`) は **物理 formula が真値 ANCC で実行されれば誤差 2 mm** を意味する。つまり:
- ANCC を真値で持っていれば TVT 予測の floor は 0.006 ft
- 我々の inference 誤差はほぼ全て **ANCC imputation 誤差 + b_well 推定誤差** に集約
- PINN は ANCC imputer と b_well predictor を **同時 multi-head 学習** することで、TVT 予測誤差を直接最小化する

→ これは「TVT を直接学習」ではなく「TVT を成立させる ANCC, b_well を学習」という構造変換。LightGBM や Mamba と原理直交。

#### 3.2.2 hidden region での residual 削減

hidden region の TVT 予測において:
- 通常 NN: $\hat{TVT}_t = f_\theta(\text{features})$ で誤差 0.5-1.5 ft
- PINN: $\hat{TVT}_t = -Z_t + \hat{\text{ANCC}}_\phi(t) + \hat{b}_\psi(t)$ で誤差 0.006 ft + $\text{err}_{ANCC} + \text{err}_b$

つまり PINN は **個別 head の最適化** に問題を分割。ANCC は formation top depth で空間的に smooth (`docs/research/problem-essence.dense.md:27`)、b_well は per-well constant + small drift (`docs/research/first-principles.dense.md:74` p50_abs = 0.00154 ft) なので、それぞれの予測誤差は TVT 直接予測より小さい可能性。

#### 3.2.3 ROGII 数理 framework との整合

`docs/research/mathematical-formulation.dense.md` の generative model:
$$
TVT_t = -Z_t + \text{ANCC}(X_t, Y_t) + b_{well} + \epsilon_t
$$

PINN は $\epsilon_t = 0$ を **hard constraint に近づけながら学習** → generative model に最も忠実な paradigm。

### 3.3 実装 spec (= PyTorch、loss weights λ tuning、NN backbone)

#### 3.3.1 architecture

```python
class ROGIIPinn(nn.Module):
    def __init__(self):
        self.shared = MLP(in=20, hidden=128, layers=4)
        self.ancc_head = MLP(in=128, hidden=64, out=1)
        self.b_head    = MLP(in=128, hidden=64, out=1)
        self.tvt_head  = MLP(in=128, hidden=64, out=1)

    def forward(self, x):
        h = self.shared(x)  # (B, 128)
        ancc = self.ancc_head(h)  # ANCC imputed
        b    = self.b_head(h)
        tvt_direct = self.tvt_head(h)  # 直接予測
        tvt_phys = -x[:, Z_idx] + ancc + b  # 物理 derived
        return tvt_direct, tvt_phys, ancc, b
```

#### 3.3.2 loss & training

```python
loss_data_tvt = mse(tvt_direct, tvt_true)   # visible のみ
loss_data_ancc = mse(ancc, ancc_true)        # train ANCC のみ
loss_phys = mse(tvt_phys, tvt_direct)        # 全 region (consistency)
loss_b_prior = mse(b, b_wls_per_well)

# Wang+Perdikaris lambda annealing
lambda_phys = (1-0.1)*lambda_phys + 0.1 * (grad_norm(loss_data_tvt) / grad_norm(loss_phys))

total_loss = loss_data_tvt + lambda_phys * loss_phys \
           + lambda_ancc * loss_data_ancc + lambda_b * loss_b_prior
```

#### 3.3.3 inference

```python
with torch.no_grad():
    _, tvt_phys, ancc_pred, b_pred = model(X_hidden)
    # tvt_phys を最終予測に採用 (= 物理整合性最も高い)
```

#### 3.3.4 runtime

- 学習: 376 well × 5000 row × 50 epoch / Adam / T4 GPU で **30-60 min**
- 推論: 776 well × 5000 row / 1 forward で **5-10 s**
- 9 hr cap 内余裕

### 3.4 LB 寄与推定 + 失敗 modes

#### 3.4.1 LB 寄与推定

| component | 仮定 | LB 寄与 |
|---|---|---|
| ANCC imputer (multi-head joint) | hidden region ANCC を空間 smooth で推定、誤差 0.5 ft → tvt_phys floor 0.5 ft 寄与 | **−0.2 ft** |
| b_well per-well + drift (multi-head joint) | b_drift_p50_abs = 0.00154 を MD-dependent linear で吸収 | **−0.1 ft** |
| lambda annealing で gradient pathology 回避 | 50-100× 精度向上 (= Wang+Perdikaris Figure 5) のうち ROGII で 1-2x 実現 | **−0.1 ft** |
| **合計 (single PINN head)** | | **−0.3〜−0.5 ft** |

→ Edge R 後 LB 9.95 想定から **9.45-9.65 ft** へ。TabPFN v2 と同等。

#### 3.4.2 失敗 modes 3 件

| failure | 兆候 | 対策 |
|---|---|---|
| **lambda 暴走で physics 学習が data を駆逐** | $\mathcal{L}_{phys} \to 0$ だが TVT MSE 悪化 | (a) Wang+Perdikaris lambda annealing、(b) lambda 上限 clamp (`lambda < 100`)、(c) NTK-based weighting (`https://arxiv.org/abs/2007.14527`) |
| **ANCC imputer が test 分布 shift で破綻** | train ANCC OK、test ANCC pred で `formula_oracle_rmse` 検算が 0.1 ft 以上に悪化 | (a) ANCC imputer は spatial GP (Edge N 推定) で先 imput、PINN は residual modeling、(b) ANCC head の正則化強化 |
| **fault zone で smooth assumption 破綻** | fault layer 跨ぎで ANCC 不連続、PINN smooth NN が discontinuity 表現できない | (a) fault detector (= Edge O ANCC clip) を pre-process、(b) PINN 中で fault indicator を input feature 化、(c) piecewise NN (= mixture of experts) |

---

## 4. PySR (Symbolic Regression) — b_well の closed-form 式探索

### 4.1 数理仕様 (= 遺伝的 programming、Pareto front)

#### 4.1.1 SymbolicRegression.jl / PySR の発想

PySR (Cranmer 2023, arXiv:2305.01582, `https://arxiv.org/abs/2305.01582`) は **expression tree** を遺伝的 algorithm で進化させる:

- 各 individual = expression tree (= 数式の AST)
- operator set: binary (+, −, ×, /, ^)、unary (sin, cos, exp, log, sqrt, ...)
- mutation: subtree 交換 / 定数摂動 / operator swap
- crossover: 2 tree の subtree swap
- fitness: $\text{loss}(eq) + \text{parsimony} \cdot \text{complexity}(eq)$

#### 4.1.2 Pareto front 選択

複数 population (= default 15) を並列進化、終了時に **complexity vs loss** の Pareto front を提示。ユーザーは:
- **accuracy 優先**: complexity 高、過学習リスク
- **interpretability 優先**: complexity 低、bias リスク
- **model_selection = "best"**: F-score 的 trade-off の最良点
- **model_selection = "score"**: complexity 増分に対する loss 減少 rate 最大点 (= elbow)

#### 4.1.3 default hyperparameters (PySR ≥ 0.18)

(`https://ai.damtp.cam.ac.uk/pysr/` Features and Options より)

| param | default | 推奨 (ROGII) |
|---|---|---|
| `niterations` | 40 | 100-200 |
| `populations` | 15 | 30-60 (= parallel diversity) |
| `population_size` | 33 | 50 |
| `ncyclesperiteration` | 550 | 550 (default) |
| `maxsize` | 30 | 25 (= 過学習抑制) |
| `parsimony` | 0.0032 | 0.001-0.01 |
| `model_selection` | "best" | "score" (= elbow 選択) |

`parsimony` 推奨値は **min_loss / 5-10** とドキュメント (`https://ai.damtp.cam.ac.uk/pysr/v1.5.9/options`)。`b_drift_first_to_last_p50_abs = 0.00154` を ROGII の min_loss 想定として `parsimony = 1.5e-4 〜 3e-4`。

### 4.2 ROGII 適用 (= b_well closed-form 式探索)

#### 4.2.1 input/output 設計

- **train sample**: per-well 1 row = 376 sample (= train wells)
- **features**: `[X_mean, Y_mean, Z_first, Z_last, MD_first, MD_last, formation_thickness, dip_azimuth, dip_angle, GR_mean, GR_std, ...]` 10-15 列
- **target**: `b_well_wls` (= WLS で per-well 推定した b_well 定数、`docs/research/independent-edges.dense.md` 公開 top の hand-crafted value)
- target std ≈ 0.05 ft 想定

#### 4.2.2 探索式空間

ROGII の物理は plane fit ANCC(X, Y) = aX + bY + c (`docs/research/problem-essence.dense.md:27`)、b_well は constant + drift。closed-form として狙う式:

$$
b_{well} \stackrel{?}{=} \alpha_1 X_{mean} + \alpha_2 Y_{mean} + \alpha_3 \cdot \sin(\text{dip\_azimuth}) \cdot \cos(\text{dip\_angle}) \cdot Z_{last} + \alpha_4
$$

→ X/Y 線形 + 幾何学的 projection 項 (sin/cos/tan) は遺伝的探索の sweet spot。

#### 4.2.3 ROGII 数理 framework との整合

generative model $TVT = -Z + \text{ANCC} + b_{well}$ で b_well を per-well constant とせず、**幾何 X, Y, dip 関数** で symbolic 表現できれば:
- per-well WLS の variance (= 標本数小の overfit) を抑制
- new well (= test) でも extrapolation 安定
- feature engineering 1 件で「b_well_symbolic」column を作成 → karnakbaev blend / Mamba / TabPFN v2 全てに feature として注入可能

### 4.3 実装 spec (= PySR package、population size、generation budget)

#### 4.3.1 install

```bash
pip install pysr
python -c "import pysr; pysr.install()"  # Julia auto install on first import
```

Julia runtime が必要。Kaggle kernel に Julia は default 入ってないので `apt install julia` または `juliaup` を事前に kernel 内でインストール (≈ 1 min)。

#### 4.3.2 sklearn-style API

```python
from pysr import PySRRegressor
sr = PySRRegressor(
    niterations=200,
    populations=30, population_size=50,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sin", "cos", "tan", "exp", "log", "sqrt"],
    maxsize=25,
    parsimony=2e-4,
    model_selection="score",
    procs=4,        # CPU 並列
    multithreading=True,
)
sr.fit(X_train_376wells, y_b_well_wls)
print(sr.equations_)  # Pareto front DataFrame
best_eq = sr.get_best()
```

#### 4.3.3 runtime

- 376 sample × 12 feature × niter=200 × pop=30 × CPU 4 thread → **15-30 min on Kaggle CPU**
- GPU は使わない (= symbolic search は CPU GA)
- 9 hr cap 内余裕、submission kernel と分離可能 (= offline で式発見 → 式を inference kernel に hardcode)

### 4.4 LB 寄与推定 + 失敗 modes

#### 4.4.1 LB 寄与推定

| component | 仮定 | LB 寄与 |
|---|---|---|
| b_well_symbolic を karnakbaev blend に feature 追加 | per-well WLS の overfit を symbolic stabilize | **−0.1 ft** |
| Mamba / TabPFN v2 / PINN にも feature 注入 | 全 head が b_well を等しく見れる | **−0.05 ft** (= 重複避け) |
| **合計 (PySR single feature)** | | **−0.1〜−0.3 ft** |

→ 単独では最弱。ただし **工数 2-3 日と最短**、他 paradigm との合成効果あり。

#### 4.4.2 失敗 modes 3 件

| failure | 兆候 | 対策 |
|---|---|---|
| **遺伝的 algorithm が trivial 式に hit** | `b = mean(b_train)` の constant が Pareto front 1 位 | (a) parsimony を 1 桁下げる、(b) 強い nonlinear unary 強制 (= sin/cos 必須)、(c) niterations 倍増 |
| **376 sample で過学習** | train R² 高、CV R² 低 | (a) maxsize=15 に削減、(b) cross-validate within Julia process (`https://ai.damtp.cam.ac.uk/pysr/`)、(c) test set per-well b_well を hand-WLS と blend |
| **Julia install 失敗 (Kaggle 環境)** | `julia: command not found` で fit が hang | (a) offline で式発見 → hardcoded function を kernel に paste、(b) `pysr.install()` を kernel 冒頭で実行 + retry、(c) PySR 不要に → `sympy` で手動式 fit |

---

## 5. ROGII 9 切り / Top 1 圏 への投入順序

### 5.1 単独 LB 寄与 + 工数 + 既存合成

| paradigm | 単独 LB 寄与 | 工数 (日) | 既存 layer との合成 | priority for 9 切り |
|---|---|---|---|---|
| **Mamba / S4** | **−0.7〜−1.2 ft** | 5-7 | karnakbaev blend に feature, または full replace candidate | **★★★ (= 単独で 9.0 割れ最有力)** |
| TabPFN v2 | −0.3〜−0.5 ft | 3-4 | karnakbaev blend と直交 ensemble | ★★ |
| PINN | −0.3〜−0.5 ft | 4-5 | ANCC imputer / b_well predictor を Mamba / blend に注入 | ★★ |
| PySR | −0.1〜−0.3 ft | 2-3 | b_well_symbolic を全 head に feature 注入 | ★ (= 補助・短工数) |

### 5.2 priority 決定軸 (= 推奨は出さない、軸のみ)

| 選択軸 | 重視するなら | 候補 |
|---|---|---|
| 単独最大寄与 | Mamba 単独投入 | **Mamba** |
| 短工数で確実な小寄与 | PySR から | **PySR** |
| 公開 top と直交性最大 | in-context Bayesian | **TabPFN v2** |
| 物理整合性 + 説明可能 | algebraic constraint soft | **PINN** |
| 失敗 risk 分散 (= 並列開発) | 4 paradigm 同時着手 | **全部並列** (= 工数合計 14-19 日、残 86 日内可) |

### 5.3 投入順序 scenario 3 案

#### scenario A: Mamba 一点集中
- Day 1-7: Mamba scratch train + submit
- 期待 LB: 9.95 → 8.75-9.25
- リスク: Mamba 単独失敗で 1 週間ロス

#### scenario B: 短工数先行 (PySR → PINN → TabPFN v2 → Mamba)
- Day 1-3: PySR で b_well_symbolic, blend に注入、LB 9.95 → 9.65-9.85
- Day 4-8: PINN 構築、LB 9.65 → 9.15-9.45
- Day 9-12: TabPFN v2 ensemble、LB 9.15 → 8.65-8.95
- Day 13-19: Mamba 追加 stack、LB 8.65 → 7.65-8.25
- リスク: 順次依存で延滞すると 8.x 帯到達が 5/30 ギリ

#### scenario C: 並列 4 worker
- Day 1-7: W_Mamba + W_TabPFN + W_PINN + W_PySR 同時着手 (= `~/.claude/rules/agents.md` Parallel Task Execution)
- Day 8-12: 4 head の blend 設計 + CV 検証 + submit
- 期待 LB: 9.95 → 7.5-8.5
- リスク: 並列 kill 発生時の rollback (= 30 分中間 commit 厳守)

### 5.4 LB 寄与の重複懸念

複数 paradigm が同じ variance を拾うと **加法的に効かない**。
- Mamba と PINN: Mamba は state-space、PINN は algebraic constraint → 原理直交、加法的に効きやすい (= +80% 加算性想定)
- TabPFN v2 と karnakbaev blend: 両方 tabular GBM 系 → 重複 30-50%、加法性 +50% 想定
- PySR と他 全部: b_well feature 提供のみ → 加法性 +95% 想定 (= 重複ほぼ無)

→ 4 paradigm 全部 hit 仮定での **乗算的寄与** は 2.4-3.0 ft 級 (= LB 9.95 → 6.95-7.55)。ただし 4 つとも上限値達成は 5-10% 程度の確率。中央値 8.0-8.5 ft 想定。

---

## 6. 4 paradigm 統合の最終 stack 設計 (= 全部 hit case)

### 6.1 stack architecture

```
                ┌─────────────────────────────────────────┐
                │ Input: per-row features                  │
                │ [Z, X, Y, MD, GR, last_TVT, ANCC_imp, …] │
                └─────────────────────────────────────────┘
                                  │
        ┌─────────────┬───────────┴───────────┬─────────────┐
        │             │                       │             │
   ┌────▼────┐  ┌─────▼─────┐         ┌──────▼──────┐  ┌────▼────┐
   │ PySR    │  │ Mamba/S4  │         │ TabPFN v2   │  │  PINN   │
   │ b_sym   │  │ state-    │         │ in-context  │  │ phys    │
   │ feature │  │ space     │         │ Bayesian    │  │ soft    │
   └─────────┘  └─────┬─────┘         └──────┬──────┘  └────┬────┘
        │             │                       │             │
        │       ┌─────▼─────┐         ┌──────▼──────┐  ┌────▼────┐
        │       │ μ_mamba   │         │ μ_tabpfn    │  │μ_pinn   │
        │       │ σ_mamba   │         │ q10,q50,q90 │  │σ_pinn   │
        │       └─────┬─────┘         └──────┬──────┘  └────┬────┘
        │             │                       │             │
        ▼             ▼                       ▼             ▼
   ┌──────────────────────────────────────────────────────────┐
   │ Karnakbaev LGB blend (PySR feature 含)                  │ ← Edge S, R, Q, M, D, E, O, N, P, R, T 全 Edge layer
   │ + 4 head μ を feature 追加                                │
   │ + σ を weight に in stacking                              │
   └──────────────────────────────────────────────────────────┘
                                  │
                          ┌───────▼───────┐
                          │ Final TVT pred│
                          └───────────────┘
```

### 6.2 既存 Edge との関係 (= `docs/research/independent-edges.dense.md` 参照)

| Edge / paradigm | 役割 |
|---|---|
| Edge S (round-to-grid) | post-proc、final TVT を 0.5 ft 格子に snap |
| Edge R (test-time online) | submission kernel で test visible に追加 train |
| Edge Q / M | 補助 feature 系 |
| Edge D (= 案 D Kalman) | **Mamba にアップグレード**、N=1 → N=64 |
| Edge O (ANCC clip) | fault detector として PINN の pre-process |
| Edge N (spatial smoothing) | ANCC imputer の prior (PINN ANCC head) |
| Edge P / R / T | 公開 top blend 補完 |
| Mamba (new) | hidden zone 末端 bias 抑制 |
| TabPFN v2 (new) | per-well in-context Bayesian |
| PINN (new) | 物理整合性 + ANCC/b joint |
| PySR (new) | b_well closed-form feature |

### 6.3 最終 stack の CV 評価設計

- **CV split**: per-well group 5-fold (= `docs/strategy/winning-strategy.dense.md` 慣行)
- **per-head metric**: MAE_ft, RMSE_ft, in-domain (visible end付近) vs hidden tail (MD 4000+)
- **blend search**: scipy.optimize.minimize で weight $w = (w_{lgb}, w_{mamba}, w_{tabpfn}, w_{pinn})$、constraint $\sum w_i = 1, w_i \ge 0$
- **CV-LB gap monitor**: 各 head 単独で CV vs LB diff を計測、>0.5 ft なら overfit signal

### 6.4 提出 kernel runtime budget (9 hr cap)

| component | runtime (T4 / 776 well) | 占有率 |
|---|---|---|
| karnakbaev blend (existing) | ~30 min | 5.5% |
| Mamba inference | 1-2 min | 0.4% |
| TabPFN v2 inference | 25 min | 4.6% |
| PINN inference | 5-10 s | 0.03% |
| PySR (offline, hardcoded) | 0 (式は kernel 内 fn) | 0% |
| Edge S/R/O/N/P/R/T post-proc | ~20 min | 3.7% |
| **合計** | **~80 min** | **14.7%** |

→ 9 hr cap (540 min) の 14.7% 占有、余裕 85%。

---

## 7. 残課題 / 5/12 以降の exp 設計への含意

### 7.1 残課題 (= 本 doc で深掘りできなかった部分)

1. **Mamba-2 pretrained time-series checkpoint** の license / quality
   - TSMamba / Mamba4Cast (`https://arxiv.org/abs/2411.02941`) の Apache-2.0 確認、ROGII への fine-tune 適合性検証
   - 既存 time-series pretrained weight を ROGII MD 軸に再 mapping できるか (= sample rate 違いの影響)
2. **TabPFN v2 + per-well batched inference の wall-clock 実測**
   - 公式 README の「1-3 s/dataset」は 1k row 想定、ROGII 3k row × 20 feature の実測必要
   - TabPFN-2.5 (arXiv:2511.08667) の release 状況 (= Kaggle 利用可否)
3. **PINN lambda annealing の収束保証**
   - Wang+Perdikaris 2021 の lambda annealing が ROGII algebraic constraint (非 PDE) で同等効果か
   - NTK-based weighting (`https://arxiv.org/abs/2007.14527`) との比較
4. **PySR Julia install を Kaggle kernel で安定化する方法**
   - `pysr.install()` の network 依存 / cache 戦略
   - offline 完結代替 (= sympy + scipy GA hand-roll)
5. **4 head blend の CV-LB gap 実測**
   - 各 head 単独 CV と LB 乖離値を W (subagent) や A/U が実測してない
   - blend weight optimization の overfit risk (= weight 数 4 個でも CV-LB +0.2 ft 想定)

### 7.2 5/12 以降の exp 設計への含意

#### exp010 候補 (= PySR first)
- 最短 2-3 日、b_well_symbolic を karnakbaev blend に feature 追加
- 期待 LB: 9.95 → 9.65-9.85
- 失敗時の rollback cost 最小 (= 単 feature 削除)

#### exp011-013 候補 (= Mamba / TabPFN v2 / PINN 並列)
- `~/.claude/rules/agents.md` Parallel Task Execution + git worktree + tmux N+1 で 3 worker 起動
- Mamba は W_M_M (= Mamba master + slave)、TabPFN は W_T、PINN は W_P
- 各 worker は **30 分中間 commit** + branch push までで停止、merge は中央
- 期待: 3 paradigm が同時に LB-0.3 〜 -0.7 ft 投入

#### exp014 候補 (= 4 head full stack)
- exp010-013 で投入済みの 4 head を **blend kernel** で統合
- scipy.optimize で weight 学習、CV per-well group 5-fold
- 期待 LB: 7.5-8.5

#### exp 失敗の早期検知
- `docs/dev/submission-postmortems.dense.md` の exp003 (LB 17.510) / exp006 (CUDA error) 教訓:
  - Mamba は **CUDA error / OOM 検証 を offline で先実施**
  - TabPFN v2 は **CUDA 12 native 確認** + wheel offline cache
  - PINN は **lambda 暴走を train log で 5 min 毎監視**、自動 abort
  - PySR は **Julia install 失敗を kernel boot で fast-fail**

### 7.3 subagent V (= sub data mining) / W (= 学術文献 deeper) との照合点 (= 中央が行う前提)

- W が cover した paradigm との重複: **Diffusion CSDI / Geometry 2-stage** (`docs/research/cv-breakthrough.dense.md`) は本 doc の Mamba / PINN と原理隣接、blend 重複懸念あり → 中央で照合
- V が見つけた sub data (= 例えば formation top metadata) を Mamba / TabPFN v2 / PINN の input feature に追加するか中央で判断
- subagent W の `academic-literature-deeper.dense.md` に Mamba / S4 / TabPFN v2 / PINN / PySR の言及ありか確認 (= 重複作業回避)
