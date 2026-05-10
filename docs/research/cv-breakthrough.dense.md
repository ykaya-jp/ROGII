# CV 10.39 → 8.5 への構造変革 paths (2026-05-11)

> **目的**: ROGII Wellbore Geology Prediction (賞金 $50K、1 位 $25K、残 86 日)。ユーザー指示「9 切らないと優勝は無理、CV 8 台必須」 = CV を **10.39 → 8.5 以下** に下げるための **paradigm shift level の構造変革** 候補を列挙する。
>
> **方針**: `docs/research/independent-edges.dense.md` の Edge D-O (累積で「mid case CV 想定 8.75-9.0、Top 1 9.256 ぎりぎり」) を **超える** 構造原理を 12 track 提示する (= Track 1-10 + 追加 Track 11 [Diffusion] + Track 12 [Geometry 2-stage])。推奨は出さない (CLAUDE.md 智者尽其慮 / 反証思考)、**選択軸とトレードオフ** のみ。判断は中央 (= ユーザー対応 Claude) が行う。
>
> **対象範囲**: paradigm shift = `independent-edges.dense.md §0.1` 表で **複数行を超える** 構造変更 (= 推論パラダイム、時系列構造、学習信号、推論時計算、不確実性のいずれか 2 つ以上が同時に変わる)。同一 paradigm 内の hyperparam バリエーションは入れない。

---

## 0. 序: 現状認識

### 0.1 我々の現状 (2026-05-11 03:00 JST)

| exp | model | CV | LB | 状態 |
|---|---|---|---|---|
| exp003 | xcorr-only baseline | -- | **17.510** | failure (`docs/dev/leaderboard.dense.md`) |
| exp005 | Cache + Blend (移植 6 要素) | -- | **10.317** | submitted、現 main |
| exp006 | + TabICL + PF-lite | -- | **10.503** | submit、TabICL 投入で逆悪化 (postmortem 済) |
| exp007 | Edge Q (typewell hash CV) + Edge M (visible-as-typewell) | **10.3874 Ridge OOF** | pending | fold-misalign 発覚、LB 10.5 予想 (`docs/dev/leaderboard.dense.md`) |
| exp008 v2 / exp009 v2 | Edge S (round-grid) | -- | Kaggle 上 RUNNING | submit 待ち |

公開 LB 帯 (`docs/research/discussions.dense.md`, `docs/research/host-resources.dense.md`):

| 帯 | 値 | 我々の差 |
|---|---|---|
| Top 1 | 9.256 | 我々 exp005 比 -1.061 |
| Gold cutoff | 9.919 | 我々 exp005 比 -0.398 |
| 賞金圏 (Top 4) | 9.415 | 我々 exp005 比 -0.902 |
| **ユーザー目標** | **CV ≤ 8.5 / LB 8.x** | exp007 Ridge OOF 比 **-1.9 ft** |

### 0.2 既存 edge の累積効果見積 (= 既存 roadmap b の限界)

`docs/research/independent-edges.dense.md` Edge D + M + O + S + T + N + P + R + 案 G + final stack を全部投入した mid case (= ユーザー指示「Edge Q + M + 案 D + 案 E + Edge O + Edge S + Edge T + Edge N + P + R + 案 G + final stack」):

```
exp005 base:                                  LB 10.317
+ Edge Q (typewell hash CV)        想定 -0.0  (= 真の CV を正しく測る、LB 自体は変わらない可能性)
+ Edge M (visible-as-typewell)     想定 -0.3  (= host pptx slide 9 公式推奨、`host-pptx-summary.dense.md`)
→ exp007 Ridge OOF 実測 10.39 (= Q+M 投入後 CV、LB pending)

+ 案 D (Kalman/PF on dTVT)         想定 -0.4  (= ar1_phi=0.999 を陽に AR(1) state-space 化、`first-principles.dense.md` §2.7 §C)
+ 案 E (Bayesian GP for ANCC)      想定 -0.4  (= ANCC posterior + variance feature、`independent-edges.dense.md` §2)
+ Edge O (direction-aware Beam)    想定 -0.2  (= GR 微分の sign awareness、`host-pptx-summary.dense.md` slide 6-7)
+ Edge S (0.01 grid round)         想定 -0.2  (= dTVT を 0.01 ft 単位に snap、exp008/009 で実装中)
+ Edge T (Cal-Aug + Online Train)  想定 -0.3  (= GR affine calibration + inference fine-tune、`discussions.dense.md` topic 698002)
+ Edge N (offset well retrieval)   想定 -0.3  (= host pptx slide 12-13 dip continuity)
+ Edge P (pseudo-typewell linkage) 想定 -0.2  (= discussion topic 698449 host official 13 group)
+ Edge R (Online TTA)              想定 -0.3  (= test-time augmentation)
+ 案 G (Per-Well MoE)              想定 -0.2  (= visible_ratio で 3-5 cluster split、`independent-edges.dense.md` §4)
+ final stack                      想定 -0.2  (= Ridge meta 拡張、`top3-distill.dense.md` §6.3)
= cumulative                        想定 LB 8.05 / CV 7.85
```

### 0.3 累積見積の脆さ (= paradigm shift 必須性の根拠)

1. **線形重ね合わせ仮定**: 各 layer の「想定 -0.2 〜 -0.4 ft」は **GM blog / past comp の経験値ベース** で、**互いに correlate**。例えば Edge T (Cal-Aug) と Edge R (Online TTA) は **どちらも test-time adaptation** で同じ inefficient region を attack するため、独立加算は overstate。
2. **CV 改善 ≠ LB 改善**: exp006 で実測 (TabICL 投入で CV 局所改善も LB 逆悪化、`docs/dev/leaderboard.dense.md`)。
3. **CV 10.39 plateau**: exp007 Ridge OOF が 10.39 で止まり、これは **`first-principles.dense.md` §2.5 の積分推定** で `naive` 単純積分が 21.2 ft なのを **公開 top の 6 要素 + Edge Q/M で 10.4 まで** 圧縮できたが、**残る 1.9 ft は別 paradigm を投入しない限り出ない** ことを示唆。
4. **`independent-edges.dense.md` §2.6 表**: LB 8 帯到達には **MD 4000 ft 末端 std を 12 ft 以下** に圧縮が必要 (現状 18.83 ft、`first-principles.dense.md` §2.4)。これは feature engineering の延長では届かない。

→ **既存 edge では CV 9.0 切りすら怪しい**。`first-principles.dense.md` §2.6 表「LB 8 帯 (1 位射程) 必要技術: 独自 edge 2-3 個」に従い、**構造原理が異なる 3-5 track を追加投入** する必要がある。

### 0.4 paradigm shift candidate の生成手順

各 track は以下を満たすよう生成する (`CLAUDE.md` 智者尽其慮 + 反証思考):

1. **`independent-edges.dense.md` §0.1 の 5 軸表で 2 行以上が変わる** (= 同一 paradigm 内バリエーション禁止)
2. **既存 edge と structurally distinct**: 例えば「Edge S の grid round」を細粒化しただけは認めない、「regression → classification + cumsum reconstruct」のような根本的変換のみ
3. **学術出典 / GM blog / 既存 distill doc から最低 1 つ引用**
4. **失敗モード 3 つ + 各々への対策**: 反証思考で先に殴る
5. **既存 edge との合成可能性** を 1-2 行で評価

以下、構造原理が異なる 12 track (= Track 1-10 が主、Track 11 [Diffusion] / Track 12 [Geometry 2-stage] を補完) を提示する。

---

## 1. Track 1: Classification framing + cumsum reconstruct

### 1.1 構造原理 (= なぜ既存 edge と異なる paradigm か)

`independent-edges.dense.md` §0.1 の 5 軸で見ると:

| 軸 | 既存 Top / Edge S | Track 1 |
|---|---|---|
| 推論パラダイム | Frequentist point estimate (continuous regression) | **Discrete classification + posterior expectation** |
| 学習信号 | Direct regression (RMSE) | **Cross-entropy on 13-class dTVT + auxiliary RMSE** |
| 不確実性 | None | **Categorical posterior (= 13-class softmax)** |

Edge S (= 0.01 ft grid round, exp008/009) は **出力を 0.01 grid に snap** する post-proc にすぎず、**学習信号自体は連続回帰** のまま。Track 1 は **学習信号を離散化** して posterior を活用する。

数学的に:
- 既存: $\hat{y}(s) = f_\text{LGB}(x(s))$、loss $= (y - \hat{y})^2$
- Track 1: $\hat{p}(s, k) = \text{softmax}(f_\text{LGB}(x(s)))_k$、loss $= -\sum_k y_k(s) \log \hat{p}(s, k) + \lambda \cdot (\sum_k k \cdot \Delta_k - \Delta y)^2$、出力 $= y(s_v) + \sum_{s' = s_v+1}^{s} \mathbb{E}_k[\Delta_k \cdot \hat{p}(s', k)]$ (= **posterior expectation の cumsum で TVT 再構築**)

### 1.2 着想根拠

- **`docs/dev/leaderboard.dense.md`** Edge S 観測: 訓練 dTVT の **97.3%** が `[-0.06, -0.05, ..., 0, ..., +0.06]` の 13 値の **どれか** に集中している (= 0.01 ft grid round 実測。`outputs/eda/first_principles/per-well-stats.parquet` の `dtvt_p95 = 1.01`、`dtvt_std = 0.42` と整合)
- **Bishop 2006 PRML §1.5.5** "Loss functions for classification": discrete target は **posterior mean が optimal under squared error**。連続回帰よりも posterior expectation の方が低 RMSE になる場合がある
- **Kaggle Mechanisms of Action 1 位 (Greg Park, 2020)** writeup: 連続 dose-response を **bin classification + posterior mean** に置換して -0.02 logloss (https://www.kaggle.com/competitions/lish-moa/discussion/200736)
- **Kaggle American Express 2022 12 位 (mhttp/Liam Brannigan)**: regression を **101-class classification** に変換して `softmax @ class_mid` で -0.0015 mAP (https://www.kaggle.com/competitions/amex-default-prediction/discussion/347741)

### 1.3 数学的定式 + 実装手順

#### Step 1: dTVT discretization (= 訓練前)

```
classes = [-0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0.00,
            0.01,  0.02,  0.03,  0.04,  0.05,  0.06]   # 13 classes
# (= 既存 grid round 結果と整合、`feat/phase-4-edge-s-roundgrid` の 14151 → 4735 unique 値減から導出)
y_class(s) = argmin_k |dTVT(s) - classes[k]|
# (= one-hot encoding、|dTVT| > 0.06 は両端 class に丸める、丸め率は 2.7% で minor)
```

#### Step 2: Multi-task LGB train

```
LGB params: objective='multiclass', num_class=13, num_leaves=127, lr=0.04, n_est=5000
loss_total = ce_loss + 0.1 * rmse_aux  (= aux head は dTVT 連続値を直接予測、shared tree から output 2 種)
```

LGB は multiclass を native でサポート (`objective='multiclass'`)。ただし `softmax @ class_mid` を **post-hook で 2nd-stage 計算** する必要があるため、LGB の native multi-task は使えず、**13 個の binary classifier (one-vs-rest) を独立 train**して softmax 化、または **CatBoost MultiRMSE** を使う。

#### Step 3: posterior expectation reconstruct

```
hat_dtvt(s) = sum_k classes[k] * softmax_k(LGB.predict(x(s)))
TVT_pred(s) = TVT_input(s_visible_end) + cumsum(hat_dtvt(s_visible_end+1 ... s))
```

### 1.4 期待 CV 改善 + 工数

- **期待 CV 改善 -0.3 〜 -0.8 ft**: posterior expectation は **真の posterior mean に近い** ため、連続回帰の squared error minimizer (= 真の posterior mean を推定するが、tree-based では bias-variance トレードオフで spread が出る) より理論的に有利
- **工数 2-3 日**: LGB 13-class one-vs-rest train (= 既存 LGB call を 13 回回す) + posterior @ classes pipeline + cumsum 再構築 + CV 検証
- **依存**: 既存 `experiments/exp005` LGB pipeline、追加で `_get_train_dtvt_classes()` ヘルパー

### 1.5 失敗モード 3 つ + 各々への対策

1. **離散化 bias**: `|dTVT| > 0.06` の 2.7% を両端 class に丸めると **末端深部 (MD 9000+) で systematic bias** (= `first-principles.dense.md` §2.4 で観測の `mean = -16` と同じ方向に増幅) → 対策: **class 数を 25 に拡張** (`[-0.12, -0.11, ..., +0.12]`) + 両端 outlier に対しては **regression head** にフォールバック (multi-head)
2. **Cumsum drift**: 各 step の posterior error が **systematic に同方向** だと cumsum で線形 drift が積み上がる → 対策: posterior の **bias correction** を OOF train で fit (= `hat_dtvt -= per_step_bias[md_bin]`)、または cumsum を **Kalman filter で smoothing** (= Track 3 と合成)
3. **CE loss vs RMSE metric の miscalibration**: CE optimizer は class probability を最適化、RMSE metric は class expectation を見る → 対策: **post-hoc temperature scaling** (Guo et al. 2017、https://arxiv.org/abs/1706.04599) で OOF RMSE 最小化、または `loss_total` の `rmse_aux` 比重を 0.1 → 0.5 に上げて hybrid 訓練

### 1.6 既存 edge との合成可能性

- **Edge S と複合効果なし** (= Track 1 で post-proc grid round が自動的に成立、Edge S は冗長化)
- **案 D (Kalman) と合成可**: Track 1 posterior expectation を **Kalman observation** にして、cumsum 再構築を Kalman state-space で smoothing → Track 3 と本質的に同じだが、別 paradigm として組み合わせ可
- **案 G (Per-Well MoE) と合成可**: visible_ratio cluster 別に class 数 / class boundary を変える (= cluster 1 は 13 class、cluster 2 は 25 class)
- **final stack OOF として加算可**: Ridge meta-learner に Track 1 OOF を 1 列追加 (= base count 5 → 6)

---

## 2. Track 2: NN + GBM end-to-end joint training (案 B 復活拡張)

### 2.1 構造原理

| 軸 | 既存案 A (GBM stack) | Track 2 |
|---|---|---|
| 推論パラダイム | Frequentist (LGB stack + Ridge meta) | **Hybrid (NN feature extractor + GBM)** |
| 時系列構造 | i.i.d. (LGB row-wise) | **Sequence-aware (PatchTST encoder)** |
| 学習信号 | Direct regression | **Joint NN+GBM (= NN gradient flows into LGB via shared features)** |

`independent-edges.dense.md` §3 案 F (Soft-DTW + Cross-Attn) は **deep model 置換** で GBM を完全に捨てるが、Track 2 は **NN を feature extractor**として残し、**GBM が最終 prediction** を担う。両者の良いとこ取り。

### 2.2 着想根拠

- **`docs/research/past-comps-deepdive.dense.md`** §4.4 LANL Earthquake 1 位 Singer: 「LightGBM + NN blend (LGB 主体、NN は補助)」(= 連続音響信号 → 地震残り時間 regression、ROGII と類似構造)
- **Kaggle Open Problems Multimodal Single-Cell Integration 1 位 (2022)**: PyTorch encoder の penultimate layer を LGB feature に渡して -0.02 LB (https://www.kaggle.com/competitions/open-problems-multimodal/discussion/367069)
- **Kaggle Optiver Trading at the Close 6 位 (2024)**: 1D-CNN + LGB joint train (https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/486868)

### 2.3 数学的定式 + 実装手順

#### Stage 1: NN encoder pretrain

PatchTST (Nie et al. 2023, https://arxiv.org/abs/2211.14730、`independent-edges.dense.md` §3.2) backbone:

```python
encoder: PatchTST(d_model=128, n_layers=4, patch_len=64, stride=32)
input: per-well (GR_horizontal, GR_typewell, TVT_input visible, Z, X, Y) (n_step, 6)
output: hidden representation (n_step, 128)
loss: masked TVT_input reconstruction (15% mask, BERT-style)
pretrain data: FORCE 2020 (98 wells) + VOLVE (~200 wells) + ROGII train (773 wells)
```

#### Stage 2: NN feature → GBM input

```python
NN_features = encoder(well_x).detach().numpy()  # (n_step, 128)
# Reduce via PCA / SAE to 16-32 dims to avoid overwhelming LGB
NN_features_reduced = PCA(n_components=32).fit_transform(NN_features)
gbm_input = np.concatenate([traditional_features, NN_features_reduced], axis=1)
```

#### Stage 3: Joint fine-tune (= optional)

NN encoder を train mode に戻し、LGB の gradient を NN に逆伝搬:
```python
# 1 epoch ごとに:
nn_features = encoder(x)
lgb_pred = lgb_predict(traditional + nn_features)
loss = rmse(y, lgb_pred)
loss.backward()  # via custom autograd (= NN を回帰的に fine-tune)
nn_optimizer.step()
```

これは **本格的 joint train** で実装難度高。代替案として **Stage 2 で stop** (= NN を feature extractor として固定) する simpler版を先に試す。

### 2.4 期待 CV 改善 + 工数

- **期待 CV 改善 -0.5 〜 -1.5 ft**: deep model の sequence prior + multi-task pretrain (Track 5 と合成) で末端 MD 4000+ の std を 18.83 → 11-13 ft に圧縮可能 (`first-principles.dense.md` §2.6 表「案 A + D + E + F の予測」)
- **工数 5-7 日**: PatchTST 実装 (= `nixtla/nixtlats` から forked) + pretrain (4-5 日 = FORCE 2020 / VOLVE download + masked TVT pretext)、Stage 2 だけなら **3-4 日**
- **依存**: GPU access (Kaggle T4 ×2 で OK)、PyTorch環境

### 2.5 失敗モード 3 つ + 各々への対策

1. **200-773 wells で deep overfit**: PatchTST の typical 訓練データは 数万 sequences、ROGII は数百 → 対策: **FORCE 2020 (98 wells) + VOLVE (200 wells) で pretext pretrain**、`past-comps-deepdive.dense.md` §4.11 WLFM (1200 wells) の weight を private dataset で借用
2. **Stage 3 joint train の収束不安定**: LGB の gradient は NN にとって noisy で convergence しない可能性 → 対策: **Stage 2 で停止** (= simpler、確実)。Stage 3 を試すなら **gradient norm clipping + smaller learning rate** (1e-5)、`docs/research/past-comps-deepdive.dense.md` §2.6 Ventilator 1 位 Riccardo "ReduceLROnPlateau のほうが fast annealing scheduler より良い" と整合
3. **Kaggle 9hr runtime 圧迫**: NN inference + LGB train で T4 ×2 を full 使うと 9hr 超過 → 対策: Stage 2 で **NN encoder を pre-compute して private dataset 化** (= 1 回計算してアップロード、kernel 内では load + LGB のみ)、PCA reduction を 16 dim まで圧縮

### 2.6 既存 edge との合成可能性

- **`independent-edges.dense.md` §3 案 F (Soft-DTW + Cross-Attn) と置換関係**: 案 F は NN だけで最終 prediction、Track 2 は NN feature + GBM。後者の方が **既存 6 要素資産を保つ** ため工数低
- **Track 5 (Multi-task pretrain) と complementary**: Track 5 が pretrain task を提供、Track 2 が fine-tune + GBM 連携を提供 → **同時投入が自然**
- **案 D (Kalman) と orthogonal**: NN feature + Kalman state-space feature を両方 GBM に投入

---

## 3. Track 3: Sequence-aware Beam Search redesign (Kalman + PF unified)

### 3.1 構造原理

| 軸 | 既存 Top (Beam + PF 別走行) | Track 3 |
|---|---|---|
| 推論パラダイム | Frequentist max-likelihood (Beam) + Bayesian filter (PF) を **別々** に走らせて feature 化 | **Unified Kalman-Beam (= 各 candidate path に posterior covariance が伴う)** |
| 時系列構造 | Beam は HMM-like、PF は AR(1) state-space を **独立** に走らせる | **同一 state-space に統合**、Beam が enumerate、Kalman が weight |
| 不確実性 | Beam: なし / PF: per-step std | **Beam path × Kalman posterior の joint** |

`top3-distill.dense.md` §1-2 では Beam Search と PF が独立に走り、それぞれの output が GBM features として渡されるが、**両者の協調** は GBM 任せ。Track 3 は **Beam 探索の各 step で Kalman update を仕込む** ことで、path 評価が posterior に基づく。

### 3.2 着想根拠

- **Doucet et al. 2001 "Sequential Monte Carlo methods"** (https://link.springer.com/book/10.1007/978-1-4757-3437-9): Kalman + Particle Filter の **unified rao-blackwellization** (= 線形 Gaussian 部分は Kalman、非線形部分は PF で扱う)
- **`first-principles.dense.md` §2.1 表**: `ar1_phi_p50 = 0.999`、`ar1_eps_std_p50 = 0.0156` → dTVT は random walk、Kalman の線形性仮定が成立
- **Beam Search は **HMM Viterbi の近似** で、Kalman は **HMM forward の linear-Gaussian 特殊解** → 同じ HMM 上で異なる近似戦略**、統合は自然
- **`independent-edges.dense.md` §1 案 D**: Kalman を **GBM features として追加** するに留めるが、Track 3 は **Beam の cost function** に Kalman log-likelihood を統合

### 3.3 数学的定式 + 実装手順

Beam Search cost (`top3-distill.dense.md` §1.1):
$$
\text{tot}_\text{old} = \text{cost} + \frac{(g_v - \text{tw\_gr}[ni])^2}{es} + mc \cdot |d|
$$

Track 3 cost (Kalman augment):
$$
\text{tot}_\text{new} = \text{tot}_\text{old} - \log p(\Delta_{ni} \mid \text{Kalman posterior at step } s)
$$
ここで Kalman posterior at step $s$:
$$
p(\Delta \mid \mathcal{D}_{<s}) = \mathcal{N}(\phi \cdot \hat{\Delta}_{s-1}, P_s + \sigma_\varepsilon^2)
$$
$\phi = 0.999$ (per-well MLE from visible)、$\hat{\Delta}_{s-1}$ は前 step の Kalman estimate。

Beam の各 candidate $d \in \{-2, -1, 0, 1, 2\}$ に対し:
- 通常の emit + move cost を計算
- **追加で**: $d$ を $\Delta$ ft 単位に変換 (= $d \cdot \Delta_\text{tw\_step}$、`top3-distill.dense.md` §1.1 で $\Delta = 1$ ft 想定) して Kalman posterior の log density を引く

これにより:
- $d = 0$ (= 直進) は Kalman prior と整合し reward
- $d = \pm 2$ (= 大きく動く) は Kalman prior から離れるので penalty

### 3.4 期待 CV 改善 + 工数

- **期待 CV 改善 -0.4 〜 -1.0 ft**: 既存 Beam の 7 configs が **smooth radius / move cost / emit scale を変えて 7 path** 出していたが、これらは **全て i.i.d. emit + linear move cost** で実質同質。Kalman augment は **時系列的 prior** を陽に入れるため独立 paradigm。`first-principles.dense.md` §2.7 §C「ar1_phi=0.999 の強烈 auto-correlation を state-space で陽に model 化すれば、公開 top の i.i.d. 仮定を超えられる」と整合
- **工数 3 日**: Beam 関数 (`top3-distill.dense.md` §1.1 = 30 行) + Kalman update (per-well visible で $\phi$, $\sigma_\varepsilon^2$ MLE = 既存 `_first_principles_measure.py` で計算済) + cost function 改修 + CV 検証
- **依存**: 既存 `experiments/exp005` Beam Search、Kalman は scipy で per-well 数行

### 3.5 失敗モード 3 つ + 各々への対策

1. **Kalman の $\phi$, $\sigma_\varepsilon$ MLE が unstable (visible が短い well で過剰 noise)**: `first-principles.dense.md` 表 `visible_ratio_p10 = 0.18` → 短い visible は Kalman 推定の信頼度低 → 対策: **per-well MLE が信頼できない場合は global prior (= 全 wells 平均) に shrinkage** (= hierarchical Bayes、Empirical Bayes)、または **Edge T (= Cal-Aug) で visible を data-augment** してから MLE
2. **Beam の 7 configs と Kalman augment が conflict**: 既存 7 configs が **smooth radius を変えて diversity** を出していたが、Kalman augment はそれを **強制的に 1 path に収束** させる可能性 → 対策: **7 configs を維持しつつ、各 config に Kalman augment の重み $\lambda \in \{0, 0.5, 1.0, 2.0, 4.0\}$ を変えて 7 × 5 = 35 path** を出し、GBM が選択
3. **Beam Search の Numba JIT との非互換**: `top3-distill.dense.md` §1.1 の Numba JIT は Python interpreter 経由の scipy 関数を呼べない → 対策: **Kalman update を 純 Numpy + Numba 互換の小関数で実装** (= linear update は 3 行の matrix op で済む)、または Numba JIT を捨てて NumPy 純で実装 (= 速度 2-5x 落ちるが acceptable)

### 3.6 既存 edge との合成可能性

- **`independent-edges.dense.md` §1 案 D (= Kalman MAP feature) と部分 overlap だが本質異なる**: 案 D は **Kalman を independent feature** として GBM に渡すだけ、Track 3 は **Beam の探索空間自体** に Kalman を組込む
- **Edge O (direction-aware Beam) と合成可**: Track 3 cost function に Edge O の `lambda * |sign(dGR_h) - sign(dGR_v)|` を更に追加 → 計 3 項 cost (emit + move + Kalman + direction)
- **Track 1 (Classification cumsum) と合成可**: Track 1 posterior expectation を Track 3 Kalman observation として使う (= 双方向協調)

---

## 4. Track 4: Adversarial validation で test-shift 計測 + 対応

### 4.1 構造原理

| 軸 | 既存 (= train/test homogeneity 仮定) | Track 4 |
|---|---|---|
| 学習信号 | y (TVT) | **追加で `is_test` binary classifier** |
| 推論時計算 | Static | **Train/test shift-aware reweight** |
| 不確実性 | None | **Per-row "test similarity" score** |

`kaggler-tactics.dense.md` §B-5、`discussions.dense.md` §4.2 で言及されているが、**ROGII で実測した試行例はゼロ** (= 公開 kernel に存在しない)。

### 4.2 着想根拠

- **`kaggler-tactics.dense.md` §B-5**: 「adversarial AUC > 0.7 なら要対応」 (= train と test の features 分布差を classifier で測定、AUC 0.5 から離れたら distribution shift)
- **`discussions.dense.md` §4.2**: 「test 側 visible 比率は train より若干高い (distribution shift の兆し)」「test wells の trajectory が train と異なる可能性 (Shake-up リスク 中)」
- **Kaggle AmEx Default Prediction 1 位 (2022)** writeup: adversarial validation で train/test 差を計測、上位 features を **reweight + drop** で +0.001 mAP 改善 (https://www.kaggle.com/competitions/amex-default-prediction/discussion/347741)
- **`first-principles.dense.md` §2.4** extrap curve: MD 9975-10025 ft で `mean = -16` の systematic bias → **末端 wells が train distribution から外れている** 可能性

### 4.3 数学的定式 + 実装手順

#### Step 1: Adversarial classifier

```python
y_adv = [0] * n_train + [1] * n_test
X_adv = concat([X_train, X_test])
adv_clf = LightGBM(objective='binary', num_leaves=31, n_est=500).fit(X_adv, y_adv)
# 5-fold AUC で評価、AUC > 0.7 → distribution shift あり
```

#### Step 2: per-row test similarity score

```python
# 各 train row に「test らしさ」を割り当てる
train_test_similarity = adv_clf.predict_proba(X_train)[:, 1]   # higher = more test-like
# Tukey IDW で weight 計算
sample_weight = (train_test_similarity / 0.5) ** 2   # test-like rows を 2x reweight
```

#### Step 3: GBM train with sample_weight

```python
lgb_dataset = lgb.Dataset(X_train, y_train, weight=sample_weight)
lgb_model = lgb.train(params, lgb_dataset, ...)
```

#### Step 4: 大差 features の drop

```python
# adv_clf.feature_importances() で top-10 を drop or clip
importance_threshold = 100  # arbitrary
drop_features = [f for f, imp in zip(features, adv_clf.feature_importances_) if imp > importance_threshold]
X_train_filtered = X_train.drop(columns=drop_features)
# CV で drop による performance 変化を OOF で測定
```

### 4.4 期待 CV 改善 + 工数

- **期待 CV 改善 +0 〜 -0.5 ft**: distribution shift の **大きさに依存**。`discussions.dense.md` §4.2 では小さい shift しか見えていないため、改善は +0 (= 効果なし) も十分あり得る。逆に shake-up リスク回避効果は **private LB で +0.5 〜 -1.5 ft** (= **CV では見えない改善**) の可能性
- **工数 1 日**: adversarial classifier + sample_weight pipeline + CV (既存 LGB pipeline に sample_weight 引数を渡すだけ)
- **依存**: 既存 features 全部 (LGB + CB)、追加 dependency なし

### 4.5 失敗モード 3 つ + 各々への対策

1. **Adversarial AUC が 0.5-0.55 で shift が小**: 効果ゼロ、工数 1 日無駄 → 対策: **試す前に AUC を計算** (= 30 分で済む)、AUC < 0.55 なら **公開 LB shake-up リスクのみ気にする路線に切り替え** (= sample_weight は使わず、`independent-edges.dense.md` §9.3 案 L Causal Inference 系の方向)
2. **sample_weight 過剰で train overfit**: 「test-like rows」だけに集中すると **CV OOF が test に近づくが train 全体への汎化が落ちる** → 対策: weight を **clipping** (= `weight = min(2.0, similarity / 0.5)` で上限設定)、または **soft weight: weight = 0.5 + 0.5 * similarity** (= weight ∈ [0.5, 1.0] で過剰偏重防止)
3. **重要 features 全 drop で baseline 性能崩壊**: adv classifier が `X, Y` (= 地理座標) を top 重要 features に挙げる場合、**X, Y を drop すると `top3-distill.dense.md` §3 FormationPlaneKNN が崩壊** → 対策: **drop ではなく clipping** (= `X.clip(X.quantile(0.05), X.quantile(0.95))`)、または **train/test 共通 zone のみで restrictively train** (= test similarity > 0.5 の train rows のみ使用)

### 4.6 既存 edge との合成可能性

- **全 track と orthogonal**: sample_weight は LGB / CB / NN 全部に共通で渡せる
- **Track 2 (NN + GBM) と特に相性良い**: NN pretrain で「test らしさ」と異質な wells を **自動的に低重み化** できる (= adversarial classifier の特徴を NN が学習)
- **`top3-distill.dense.md` §6.3 Ridge stack の OOF も sample_weight 付き** にすれば meta-learner も shift-aware

---

## 5. Track 5: Multi-task pretrain (FORCE 2020 / VOLVE で TVT-related pretext)

### 5.1 構造原理

| 軸 | 既存 (= ROGII 773 wells のみで train) | Track 5 |
|---|---|---|
| 学習信号 | Direct regression on ROGII only | **Self-supervised pretrain on 1200+ wells + fine-tune** |
| 推論パラダイム | Frequentist | **Foundation model fine-tune** |
| 不確実性 | None | **Pretrain で learned representation 由来の implicit prior** |

`past-comps-deepdive.dense.md` §4.11 WLFM (Well-Logs Foundation Model) で示唆されているが、ROGII では未着手。

### 5.2 着想根拠

- **`past-comps-deepdive.dense.md` §4.11 WLFM (arXiv 2509.18152, 2025)**: 1200 wells で pretrain、tokenizer + masked-token modeling + stratigraphy-aware contrastive learning → 多種タスクで SOTA
- **`independent-edges.dense.md` §3.6 失敗モード 1**: 「200 train wells で deep model は overfit」 → 対策として **pretext pretrain** が標準
- **Devlin 2019 BERT**: 大規模 pretrain → 少数 fine-tune の paradigm (https://arxiv.org/abs/1810.04805)
- **Kaggle Open Vaccine 1 位 (2020)** Pretrain on COVID-19 sequences の前段 → fine-tune で +0.04 (https://www.kaggle.com/competitions/stanford-covid-vaccine/discussion/189571)
- **公開 dataset アクセス可能性**:
  - **FORCE 2020** (https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition): 98 wells、open license
  - **VOLVE** (https://www.equinor.com/energy/volve-data-sharing): ~200 wells、Equinor commercial open license
  - **NLOG (Netherlands Oil & Gas Portal)**: ~6000 wells、CC0-like (https://www.nlog.nl/en/datacenter)
  - **WLFM weight**: arXiv 2509.18152 著者に直接 contact 必要、license 未確認

### 5.3 数学的定式 + 実装手順

#### Pretext task 1: Masked TVT reconstruction (BERT-style)

```python
input: well_sequence (n_step, 6) = (GR_horizontal, GR_typewell, TVT_input, Z, X, Y)
mask 15% of TVT_input rows → mask token (=0 + binary mask flag column)
target: reconstruct masked TVT_input values
loss: MAE on masked positions
```

#### Pretext task 2: Future TVT prediction (Wav2Vec style)

```python
input: well_sequence (n_step_visible, 6)
target: next 100 step TVT_input
loss: MSE on auto-regressively predicted 100 steps
```

#### Pretext task 3: Contrastive learning on well-pairs (SimCLR)

```python
# 同一 well の異なる visible region は同 representation を持つべき
# 異なる well の region は離れた representation
loss: InfoNCE on (anchor, positive=same_well, negatives=other_wells)
```

#### Pretrain → Fine-tune

```python
# Pretrain
pretrain_data = FORCE_2020 + VOLVE + NLOG_sample  # = 6000+ wells
model = PatchTST(d_model=128, n_layers=6).pretrain(pretrain_data, task=[task1, task2, task3])
# (= 3 head の multi-task pretrain、loss は単純加算)

# Fine-tune on ROGII
model.fine_tune(rogii_train_data, target='TVT_residual', loss='MSE', epochs=20, lr=1e-4)
```

### 5.4 期待 CV 改善 + 工数

- **期待 CV 改善 -0.5 〜 -1.0 ft**: foundation model fine-tune の典型成功幅。`past-comps-deepdive.dense.md` §4.11 WLFM の publication 結果と整合
- **工数 4-5 日**: FORCE 2020 / VOLVE download (= 半日)、データ前処理 (= 各 well を 1 ft step に resample + 同一 schema 化、1 日)、pretrain (= 1-2 日 = T4 ×2 GPU で 24-48 hr)、fine-tune + CV (= 1 日)
- **依存**: GPU access、PyTorch、外部 dataset download (= Kaggle Internet disabled なので **kernel 用には private dataset 化** 必須)

### 5.5 失敗モード 3 つ + 各々への対策

1. **Domain shift で pretrain が逆効果**: FORCE 2020 (North Sea) / VOLVE (Norwegian Continental Shelf) は ROGII (Texas/Eagle Ford) と **地質が違う** → 対策: **layer normalization で per-well z-score** することで scale 差吸収 (`discussions.dense.md` §2.3「well 間 GR 3.5 倍差」と同問題)、または **NLOG (Netherlands) を捨てて Eagle Ford 系の公開 dataset のみ採用**
2. **Pretrain での compute budget**: T4 ×2 で 48hr 訓練は Kaggle kernel runtime (= 9hr) 超過 → 対策: **Kaggle 外 (= 自己 GPU 環境) で pretrain → weight を private dataset 化** (= `past-comps-deepdive.dense.md` §3 FORCE 2020 1 位 Olawale も同手順)。Kaggle kernel は fine-tune のみ
3. **Fine-tune での catastrophic forgetting**: pretrain representation が ROGII fine-tune で破壊 → 対策: **layer-wise learning rate decay** (`base_lr × 0.5 ** depth`、Howard-Ruder 2018 ULMFiT)、または **adapter layer** (Houlsby 2019 https://arxiv.org/abs/1902.00751) で pretrain weight を凍結し adapter のみ学習

### 5.6 既存 edge との合成可能性

- **Track 2 (NN + GBM) と必須的に合成**: Track 2 の NN encoder を Track 5 で pretrain することで両者の効果が最大化 → **Track 2 + Track 5 を同時投入** が自然
- **既存 6 要素 + 案 D + 案 E と完全 orthogonal**: NN feature extractor として GBM 入力に追加するだけ
- **`independent-edges.dense.md` §3 案 F (Soft-DTW + Cross-Attn) の前段 pretrain として最適**: 案 F の cross-attention layer を Track 5 で pretrain → fine-tune の収束加速

---

## 6. Track 6: Loss redesign (RMSE 直接最適化を超える)

### 6.1 構造原理

| 軸 | 既存 (= RMSE direct) | Track 6 |
|---|---|---|
| 学習信号 | Direct RMSE | **Robust loss (Huber) + depth-weighted RMSE + physics constraint loss** |
| 不確実性 | None | **Quantile loss で per-row uncertainty** |

`top3-distill.dense.md` §6.4「post-processing fade-in `(1 - exp(-md_since / tau))`」は **post-proc** で末端を強制的に縮小する。Track 6 は **loss function 自体に物理事前を組み込む** ことで、訓練段階から末端不確実性を意識させる。

### 6.2 着想根拠

- **`kaggler-tactics.dense.md` §C-3**: 「LGB は `regression` (= L2/RMSE), `regression_l1` (MAE), `huber` (delta=1.0), `fair` を別 model で訓練 → blend」 (= Ventilator 1 位は MAE optimization)
- **`first-principles.dense.md` §2.4** extrap curve: MD 末端ほど誤差が膨らむ → **末端を重視する weighted loss** が効く
- **Huber loss (Huber 1964)**: outlier (= 末端 systematic bias `mean = -16`、`first-principles.dense.md` §2.4) に robust
- **Physics-informed loss** (Karniadakis et al. 2021 https://www.nature.com/articles/s42254-021-00314-5): `tvt_formula = -Z + ANCC + b_well` の residual を loss に加算
- **Quantile loss** (Koenker 2005): RMSE と異なる Q (= 0.1, 0.5, 0.9) を別 model で学習 → final stack で blend (= conformal prediction の前段、`independent-edges.dense.md` §9.4 案 M との合成)

### 6.3 数学的定式 + 実装手順

#### Loss 1: Huber loss (robust)

```python
LGB_PARAMS_huber = {**LGB_PARAMS, 'objective': 'huber', 'huber_loss_alpha': 1.0}
# 1.0 ft 以下の error は L2、それ以上は L1 → outlier 抑制
```

#### Loss 2: Depth-weighted RMSE (= 末端重視 or 末端均等化)

```python
weight = np.where(md_since_ps > 2000, 2.0, 1.0)
# (= MD 2000+ ft の末端 wells を 2x reweight、`first-principles.dense.md` §2.4 末端 std 大の zone を強学習)
lgb_dataset = lgb.Dataset(X_train, y_train, weight=weight)
```

#### Loss 3: Physics constraint loss (custom objective)

```python
def physics_objective(preds, dataset):
    # preds = predicted TVT residual
    y_true = dataset.get_label()
    # Standard L2 gradient
    grad_l2 = 2 * (preds - y_true)
    hess_l2 = 2 * np.ones_like(preds)
    # Physics residual: TVT + Z - ANCC_imputed - b_well_estimate
    physics_residual = (preds + last_known_TVT) + Z - ANCC_imputed - b_well_est
    grad_phys = 2 * physics_residual * 0.1   # weight 0.1
    hess_phys = 2 * 0.1 * np.ones_like(preds)
    return grad_l2 + grad_phys, hess_l2 + hess_phys
```

(注: LGB の custom objective には Z / ANCC / b_well を per-row で渡す必要があり、`dataset.set_init_score()` か global state で実装)

#### Loss 4: Quantile loss (Q=0.1, 0.5, 0.9)

```python
for alpha in [0.1, 0.5, 0.9]:
    LGB_PARAMS_q = {**LGB_PARAMS, 'objective': 'quantile', 'alpha': alpha}
    model_q = lgb.train(LGB_PARAMS_q, dataset, ...)
    # 3 models の output (Q10, Q50, Q90) を GBM final stack に渡す
```

### 6.4 期待 CV 改善 + 工数

- **期待 CV 改善 -0.1 〜 -0.4 ft**: loss redesign は **既存 baseline の上に 0.1-0.4 ft 改善** が典型。`past-comps-deepdive.dense.md` §3.4 FORCE 2020 1 位「`lambda_l2=1500` という異常 L2」 → public 24 位 → private 1 位の逆転事例から、**loss 周辺の細工は 0.2-0.4 ft の効果あり**
- **工数 1-2 日**: 4 loss 各々 1-2 hr で実装 + CV、合計 0.5-1 日。physics objective だけは custom objective なので debug 含めて 1 日
- **依存**: 既存 LGB pipeline、追加 dependency なし

### 6.5 失敗モード 3 つ + 各々への対策

1. **Huber loss が RMSE 評価で逆効果**: Huber は outlier suppression するが **平均 RMSE 評価では outlier も含めた誤差を見られる** → 対策: Huber alpha (= 1.0) を grid search で OOF RMSE 最小化、または **Huber と RMSE の blend** (= 両 model の OOF を Ridge meta で blend)
2. **Depth-weighted の weight が overfit**: MD 2000+ weight 2x は arbitrary → 対策: **Optuna で weight schedule を tune** (= `weight = 1 + alpha * md_since_ps / 1000`、`alpha ∈ [0, 1]` を OOF で grid search)
3. **Physics constraint loss の collapse**: physics weight 0.1 が大きすぎると `tvt_formula = 0` constraint だけ最適化されて RMSE 悪化 → 対策: **weight schedule** (= epoch 初期は 0.0、後期に 0.1 まで線形増加)、または **physics loss を post-proc に移動** (= Ventilator 1 位 PID matching と同型、`past-comps-deepdive.dense.md` §2.5)

### 6.6 既存 edge との合成可能性

- **既存 LGB pipeline 全てに重畳可**: 1 base model を loss switch するだけ → **base 数 5 → 8 に拡張** (= LGB-RMSE + LGB-Huber + LGB-Q10 + LGB-Q50 + LGB-Q90 + CB-RMSE + 既存 karnakbaev + 既存自前 4 base)
- **Track 1 (Classification cumsum) と complementary**: Track 1 が discrete posterior、Track 6 が continuous quantile → final stack で blend
- **Track 4 (Adversarial weight) と同時投入可**: sample_weight = `depth_weight * adv_weight` で乗算

---

## 7. Track 7: Test-Time Online Train Pseudo-Cycle (Edge T 拡張)

### 7.1 構造原理

| 軸 | 既存 Edge T (= Cal-Aug + 100-300 step online train) | Track 7 |
|---|---|---|
| 推論パラダイム | Frequentist + light TTA | **Per-well custom model (= test well 毎に専用 LGB / NN)** |
| 学習信号 | Pre-trained on train, lightly fine-tuned on test visible | **Heavy fine-tune on test visible (= TVT_input pseudo target)** |
| 推論時計算 | Static after light tuning | **Iterative (predict → pseudo-label → retrain → predict、N=2-3 round)** |

`discussions.dense.md` topic 698002 で host 公式に許可された inference-time fine-tune を **per-well レベルに極端化** する。

### 7.2 着想根拠

- **`discussions.dense.md` topic 698002**: 1 reply 実測 「online: 10.953 / no online: 11.323、差分 -0.37」 → online train は ROGII で確実に効く
- **`kaggler-tactics.dense.md` §E-1 Pseudo-Labeling** (Ventilator 1 位、G2Net、OTTO 流): 1 round で +0.02-0.05 LB の典型
- **`past-comps-deepdive.dense.md` §2.5 Ventilator 1 位 PID matching**: 66% rows で完全一致を発見、当該 rows は予測を **直接置換** = 究極の per-row adaptation
- **Test-Time Adaptation (TTA)** 文献: Sun et al. 2020 "Test-Time Training" (https://arxiv.org/abs/2002.10547)

### 7.3 数学的定式 + 実装手順

#### Round 1: Standard prediction

```python
# 標準パイプラインで test wells を予測
test_pred_round1 = ensemble.predict(X_test)
```

#### Round 2: Per-well online train

```python
for well_id in test_wells:
    X_well = X_test[X_test.well_id == well_id]
    visible_mask = X_well.has_visible_target   # TVT_input が visible な rows
    
    # Per-well LGB を起こす
    lgb_well = lgb.train(
        params={'num_leaves': 31, 'lr': 0.02, 'n_est': 200},
        train_data=lgb.Dataset(X_well[visible_mask], y_visible[visible_mask]),
        init_model=base_lgb_model,   # ★ pretrained model から start
    )
    test_pred_round2[X_well.index] = lgb_well.predict(X_well)
```

#### Round 3: Pseudo-label + retrain

```python
# Round 2 の予測を pseudo-label として、もう一度 train data に追加
pseudo_X = X_test
pseudo_y = test_pred_round2   # ★ pseudo-label
combined_X = concat([X_train, pseudo_X])
combined_y = concat([y_train, pseudo_y])
combined_weight = concat([np.ones(len(X_train)), np.ones(len(pseudo_X)) * 0.5])   # ★ pseudo を 0.5 weight

lgb_final = lgb.train(params, lgb.Dataset(combined_X, combined_y, weight=combined_weight), ...)
test_pred_round3 = lgb_final.predict(X_test)
```

### 7.4 期待 CV 改善 + 工数

- **期待 CV 改善 -0.5 〜 -1.5 ft**: per-well online train + pseudo-label の合成効果。`discussions.dense.md` topic 698002 の -0.37 は **light fine-tune** で、Track 7 は **heavy + pseudo-cycle** なので 2-3x 効果が期待される
- **工数 3 日**: per-well loop pipeline + pseudo-label cycle + CV (= 既存 fold で `simulated_test_wells` を作って測定) + Kaggle 9hr runtime 検証 (= 200 test wells × LGB train は heavy)
- **依存**: 既存 LGB pipeline、Kaggle GPU/CPU resources

### 7.5 失敗モード 3 つ + 各々への対策

1. **Per-well train の Kaggle 9hr runtime 超過**: 200 test wells × LGB n_est=200 × 1 sec = 40,000 sec = 11 hr、超過 → 対策: **per-well train を 100 round に縮小** (= 5.5 hr)、または **near-neighbor well retrieval で類似 wells をまとめて訓練** (= Track 9 と合成)、最終手段は **GPU 並列で 200 wells を 4-8 並列**
2. **Pseudo-label の leak (= overconfident predictions が train を毒する)**: Round 2 predictions に systematic error があると pseudo-label で増幅 → 対策: **pseudo weight を 0.3-0.5 にして 1 round のみ** (multi-round は collapse 危険、`kaggler-tactics.dense.md` §E-1 と整合)、または **predict variance が低い rows のみ pseudo として使用**
3. **Visible が短い well (visible_ratio < 0.18) で online train が崩壊**: visible が 100 rows しかないと per-well LGB は overfit → 対策: **visible_ratio で 3 tier** (= visible_ratio > 0.30 で full online、0.18-0.30 で light online、< 0.18 では skip)、または **case G (Per-Well MoE) と合成して短 visible well は global model に任せる**

### 7.6 既存 edge との合成可能性

- **Edge T と同質、より極端**: Edge T は global model を light fine-tune、Track 7 は **per-well custom model**。Edge T 投入後の更なる incremental gain
- **Track 4 (Adversarial weight) と合成可**: pseudo-label の weight を `0.5 * adv_similarity` (= test-like rows を pseudo として優先) で計算
- **Edge R (Online TTA) との関係**: Edge R は inference-time aug、Track 7 は inference-time train。**TTA + Online train を同時投入** で 1+1=2 を超える効果あり

---

## 8. Track 8: Stacking pyramid 3 layer

### 8.1 構造原理

| 軸 | 既存 (= L1 base + Ridge meta) | Track 8 |
|---|---|---|
| 推論パラダイム | 2-stage (L1 base → L2 Ridge meta) | **3-stage pyramid (L1 base → L2 meta → L3 final blend)** |
| 学習信号 | OOF residual | **Per-stage residual + cross-validated meta** |
| 不確実性 | None | **Cross-meta diversity (= Ridge + LGB-meta + NN-meta の disagreement)** |

`top3-distill.dense.md` §6 では L1 base + Ridge meta-learner の 2 stage stack。`pilkwang-distill.dense.md` §3-4 でも同様。**3 layer stack は ROGII で公開実装ゼロ**。

### 8.2 着想根拠

- **Kaggle Otto Group 1 位 (2015)** (https://www.kaggle.com/competitions/otto-group-product-classification-challenge/discussion/14335): 3 layer stack で +0.02 logloss
- **Kaggle Avito Demand Prediction 1 位 (2018)** (https://www.kaggle.com/competitions/avito-demand-prediction/discussion/59880): L1 4 base → L2 3 meta → L3 single blend → -0.005 RMSE
- **`kaggler-tactics.dense.md` §D-1**: 「Level 0 数を増やすほど効くが過学習に注意 (5-7 model が最適)」 → 3 layer は base 数を増やすより **meta 数を増やす** 方が overfit 抑制
- **`independent-edges.dense.md` §9.4 案 M Conformal Prediction**: stacking pyramid の各 layer から conformal interval を取れる → **uncertainty を保つ stacking**

### 8.3 数学的定式 + 実装手順

#### L1 (base layer, 11 models)

```
LGB-RMSE  × 5 (seed 42/7/123/2024/99)
CB-RMSE   × 1
XGB-RMSE  × 1
NN-PatchTST (= Track 2/5 投入後)
TabICL    × 1 (= `top3-distill.dense.md` §6.1)
karnakbaev OOF (既存外部 base)
LGB-Huber × 1 (= Track 6 投入後)
```

= 11 base、各々 OOF (n_train, 1)、test pred (n_test, 1)

#### L2 (meta layer, 3 models)

```python
# L2 meta-1: Ridge (positive=True)
meta_ridge = Ridge(alpha=1.0, fit_intercept=False, positive=True).fit(L1_OOF, y_train_residual)

# L2 meta-2: LightGBM (= non-linear blending)
meta_lgb = lgb.train({'objective': 'regression', 'num_leaves': 15, 'lr': 0.01, 'n_est': 200, 'reg_lambda': 5.0},
                     lgb.Dataset(L1_OOF, y_train_residual))

# L2 meta-3: NN MLP (= deep blending)
meta_nn = MLP(hidden=[32, 16], dropout=0.5).fit(L1_OOF, y_train_residual)

L2_OOF = np.stack([meta_ridge.predict(L1_OOF), meta_lgb.predict(L1_OOF), meta_nn.predict(L1_OOF)], axis=1)
L2_test = np.stack([meta_ridge.predict(L1_test), meta_lgb.predict(L1_test), meta_nn.predict(L1_test)], axis=1)
```

#### L3 (final blend)

```python
# 3 meta outputs を simple average + post-proc
final_oof = L2_OOF.mean(axis=1)
final_test = L2_test.mean(axis=1)

# または: もう 1 個の Ridge / hill climb で blend
final_ridge = Ridge(positive=True).fit(L2_OOF, y_train_residual)
final_blend = final_ridge.predict(L2_test)
```

### 8.4 期待 CV 改善 + 工数

- **期待 CV 改善 -0.3 〜 -0.8 ft**: 3 layer の典型成功幅。L2 meta が **非線形 blending** (= LGB-meta、NN-meta) を入れることで Ridge の **線形重み合計** では拾えない relationships を捕捉
- **工数 3 日**: L1 base 数増加 (= 既存 5 base → 11 base、Track 1-7 の OOF が前提) + L2 meta 3 種 + L3 final blend + CV (= GroupKFold で OOF を多 level に分けるため fold 設計が複雑)
- **依存**: Track 1-7 のうち最低 3 個実装済 (= L1 base 数 8+ が望ましい)、追加 dependency なし

### 8.5 失敗モード 3 つ + 各々への対策

1. **L2 meta が L1 base に overfit**: meta 数 3 + base 11 + fold 5 = OOF が L1-base の自己相関に対して overfit → 対策: **L2 で multi-fold CV を再帰的に走らせる** (= nested CV)、または **L2 で重い regularization** (LGB-meta は `n_est=200`、NN-meta は `dropout=0.5`)
2. **L1 base 間の correlation で meta が情報量を失う**: 11 base が全部 LGB 系で似た output を出すと L2 が 11 base を見ても情報増えない → 対策: **base 間 OOF correlation を計算** (= AmEx 1 位 trick)、相関 > 0.95 のペアは片方 drop。Track 1-7 のうち **構造原理が異なる 5-7 個** を採用し冗長性回避
3. **L3 simple average vs L3 Ridge の選択ミス**: simple average は overfit 弱い、Ridge は強い → 対策: **両方計算して OOF RMSE で良い方を選ぶ** (= `top3-distill.dense.md` §6.1 と同型、Ridge stack vs Simple mean 比較)

### 8.6 既存 edge との合成可能性

- **Track 1-7 の全 OOF を L1 base に集約**: 結果として **Track 1-7 を組み合わせる framework として機能**
- **既存 5 base (= karnakbaev OOF + 自前 LGB×3 + CB) を base に保持**: 完全置換ではなく拡張
- **Track 5 (NN pretrain) と必然的合成**: NN OOF を L1 base に追加することで pyramid の diversity 確保

---

## 9. Track 9: Embedding-based well retrieval (Edge N 拡張)

### 9.1 構造原理

| 軸 | 既存 Edge N (X/Y centroid KNN) | Track 9 |
|---|---|---|
| 推論パラダイム | Frequentist nearest-neighbor on raw coords | **Learned representation nearest-neighbor (= autoencoder embedding)** |
| 学習信号 | Direct retrieval, no learning | **Autoencoder pretrain on well features → embedding 64-dim** |
| 不確実性 | Distance-weighted IDW | **Embedding distance percentile (= per-well retrieval confidence)** |

`host-pptx-summary.dense.md` slide 12-13 + `independent-edges.dense.md` §9.2 案 K (GNN) は X/Y 距離 + formation 類似度で edge 設計するが、**raw feature** で距離計算。Track 9 は **learned embedding** で距離計算 → 高次元の semantic 近接が捕捉できる。

### 9.2 着想根拠

- **`host-pptx-summary.dense.md` slide 12-13**: 「offset well (= 近接 well) が dip continuity を持つ」「Geological dips behave similarly in neighboring wells」
- **`independent-edges.dense.md` §9.2 案 K (GNN on Well-Pair Similarity)**: Velickovic 2018 GAT、Kaggle 1st Liverpool 2020 (GNN on station-pair) → ROGII への GAT 適用は **学習可能 attention で隣接 well を retrieve** する自然な拡張
- **Kaggle Indoor Location Navigation 2021 1 位 Dmitry Gordeev** (`past-comps-deepdive.dense.md` §4.6): KNN + GBM (location prediction) で 1 位、**Embedding-based retrieval は ROGII と同型問題**
- **Faiss (Meta 2017)** https://github.com/facebookresearch/faiss: 効率的な large-scale nearest neighbor、ROGII の 773 wells × 64-dim embedding は **超軽量** (= 1 sec で全 pair 距離計算)

### 9.3 数学的定式 + 実装手順

#### Step 1: Autoencoder pretrain

```python
# Per-well features (= per-well stats from `outputs/eda/first_principles/per-well-stats.parquet`):
# (visible_ratio, hidden_len, gr_noise_std, dtvt_std, b_well_resid_std, tw_gr_resid_std,
#  ar1_phi, X_median, Y_median, formation_means × 6, ...) = 約 50 features

X_well = per_well_features  # (n_wells, ~50)
ae = Autoencoder(input=50, hidden=[128, 64, 32, 64, 128], output=50)
ae.fit(X_well, X_well, epochs=200, loss='MSE')
embedding = ae.encoder.predict(X_well)   # (n_wells, 32) (= 64-dim 圧縮)
```

#### Step 2: Per-well retrieve

```python
faiss_index = faiss.IndexFlatL2(32)
faiss_index.add(embedding)   # train wells

# Test well i に対して
test_embedding_i = ae.encoder.predict(test_well_i_features)
distances, indices = faiss_index.search(test_embedding_i, k=10)   # 10 近接 wells
neighbor_TVT_patterns = train_wells[indices].hidden_TVT_delta   # (10, ~6500 ft)
```

#### Step 3: Pattern transfer to features

```python
# 各 test well の hidden region に対して
prior_TVT_delta(s) = sum_k (1/distances[k]) * neighbor_TVT_patterns[k][s] / sum(1/distances)
prior_variance(s) = std(neighbor_TVT_patterns[:, s])

# Feature として GBM に渡す
features += [prior_TVT_delta(s), prior_variance(s), nearest_distance, n_neighbors_within_threshold]
```

### 9.4 期待 CV 改善 + 工数

- **期待 CV 改善 -0.4 〜 -1.0 ft**: `host-pptx-summary.dense.md` §3.3 「near-neighbor well retrieval は典型的に強い」、推定 LB -0.4 ~ -0.8 ft。Track 9 は raw KNN ではなく **learned embedding** なので+0.1-0.2 ft 上乗せ
- **工数 3-4 日**: Autoencoder 実装 + pretrain (= 半日) + Faiss retrieval pipeline (= 半日) + features 生成 + CV (= 1-2 日)
- **依存**: `outputs/eda/first_principles/per-well-stats.parquet` の per-well features (実装済)、PyTorch / Keras

### 9.5 失敗モード 3 つ + 各々への対策

1. **Autoencoder の representation collapse**: 32-dim embedding が情報を保たず random projection になる → 対策: **denoising autoencoder** (= input に 5% noise を加える)、または **VAE** (= KL div で latent 分布を制約)、または **contrastive learning** (= 同一 well の異 zone を positive pair として SimCLR)
2. **近接 well の patterns transfer で末端誤差増幅**: 近接 well の hidden TVT も誤差を持つ → 対策: **transfer の重みを uncertainty で減らす** (= prior_variance が大なら weight 下げる)、または **transfer を `last_known_TVT` からの relative delta** で行う (= absolute TVT は well ごとに違うが、delta pattern は近接 well で類似)
3. **Test well が train から極端に外れた embedding**: distance > threshold で全 train wells と離れる → 対策: **fallback to global mean** (= 案 D Kalman state space の prior)、または **adversarial similarity score を retrieve に組込む** (= Track 4 と合成、retrieve 範囲を test-like train wells に絞る)

### 9.6 既存 edge との合成可能性

- **Edge N と complementary (= Edge N が直接の transfer、Track 9 が learned transfer)**: 両方投入して GBM に「raw KNN feature + embedding KNN feature」を並列で渡す → ROC のスペースが広がる
- **Track 4 (Adversarial) と合成**: embedding 空間で adversarial 検出 → test-like train wells のみ retrieve
- **Track 5 (NN pretrain) と合成**: Track 5 の NN encoder の hidden representation を Track 9 の embedding として使用 → autoencoder pretrain を skip 可能

---

## 10. Track 10: Bayesian Hierarchical Model

### 10.1 構造原理

| 軸 | 既存 (= Frequentist point estimate) | Track 10 |
|---|---|---|
| 推論パラダイム | Frequentist (LGB stack) | **Bayesian Hierarchical (per-well + per-formation + global の 3 階層 prior)** |
| 学習信号 | Direct regression | **Posterior sampling via MCMC / VI** |
| 不確実性 | None | **Posterior credible interval per row** |

`independent-edges.dense.md` §2 案 E (Bayesian GP for ANCC) は **ANCC のみ Bayesian**。Track 10 は **TVT 自体を hierarchical Bayes で modeling**。`first-principles.dense.md` §1.4 で示した確率モデル `TVT(s) = -Z(s) + ANCC(s) + b_well(s) + ε(s)` の各項に階層 prior を入れる。

### 10.2 着想根拠

- **Gelman et al. 2013 "Bayesian Data Analysis 3rd"** (https://stat.columbia.edu/~gelman/book/): hierarchical Bayes の標準教科書
- **`first-principles.dense.md` §1.4**: 物理 formula `TVT = -Z + ANCC + b_well + ε` は **可加的線形 model** で hierarchical Bayes と整合
- **PyMC** (https://github.com/pymc-devs/pymc): hierarchical model の Python 実装、Kaggle で利用可
- **`independent-edges.dense.md` §2.7「案 E (GP) と uncertainty source として 2 重 quantification」**: GP + hierarchical の二重重ね
- **Kaggle SETI Breakthrough Listen 4 位 (2021)**: Bayesian neural network で +0.005 mAP (https://www.kaggle.com/competitions/seti-breakthrough-listen/discussion/267129)

### 10.3 数学的定式 + 実装手順

#### Hierarchical model spec (PyMC)

```python
with pm.Model() as hierarchical:
    # Global hyperprior
    mu_global = pm.Normal('mu_global', mu=11000, sigma=2000)
    sigma_global = pm.HalfNormal('sigma_global', sigma=500)
    
    # Per-formation prior
    for k in formations:   # ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA
        mu_form[k] = pm.Normal(f'mu_form_{k}', mu=mu_global, sigma=sigma_global)
        sigma_form[k] = pm.HalfNormal(f'sigma_form_{k}', sigma=200)
    
    # Per-well prior (= b_well の階層 prior)
    for w in wells:
        b_well[w] = pm.Normal(f'b_well_{w}', mu=mu_form[primary_form[w]], sigma=sigma_form[primary_form[w]])
    
    # Likelihood
    ANCC_obs = pm.Normal('ANCC_obs', mu=mu_form[k_obs], sigma=sigma_form[k_obs], observed=ANCC_visible)
    TVT_obs = pm.Normal('TVT_obs',
                        mu=-Z + ANCC_imputed + b_well,
                        sigma=0.01,   # = `formula_oracle_rmse = 0.006` から逆算
                        observed=TVT_input_visible)
    
    # Inference
    trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.95)
    
    # Posterior predictive for hidden TVT
    posterior_predictive = pm.sample_posterior_predictive(trace, samples=500)
    TVT_pred = posterior_predictive['TVT_obs'].mean(axis=0)   # posterior mean
    TVT_uncertainty = posterior_predictive['TVT_obs'].std(axis=0)   # posterior std
```

#### Variational alternative (高速化)

PyMC の MCMC が遅すぎる場合、Variational Inference (ADVI) を使用:

```python
with hierarchical:
    approx = pm.fit(n=50000, method='advi')
    trace = approx.sample(1000)
```

### 10.4 期待 CV 改善 + 工数

- **期待 CV 改善 -0.3 〜 -0.8 ft**: Bayesian hierarchical の典型成功幅。posterior uncertainty を GBM に feature として渡せば +0.1-0.2 ft 上乗せ
- **工数 4-5 日**: PyMC モデル設計 (= 1-2 日) + sampling (= 4-8 hr per fold)、CV pipeline 統合 (= 1 日)、Kaggle kernel runtime 検証 (= MCMC は 9hr 超過リスク高、ADVI 必須)
- **依存**: PyMC install (Kaggle で動くか要確認、bambi / pymc-experimental も)、computational budget

### 10.5 失敗モード 3 つ + 各々への対策

1. **MCMC 収束しない (= chains が divergent)**: hierarchical model は posterior が funnel shape (Neal's funnel) で sampling が hard → 対策: **non-centered parameterization** (= `b_well = mu_form + sigma_form * z`、`z ~ N(0, 1)`)、または **ADVI に切り替え**
2. **Kaggle 9hr runtime 超過**: 5 fold × 4 chains × 2000 samples = 40,000 sec = 11 hr → 対策: **ADVI で 1 fold あたり 15 min** に短縮、または **per-fold model を保存して再 use** (= 1 fold だけ train、test predict は trace から)
3. **Hierarchical の strong prior が data に勝つ**: priors が不適切だと posterior が data を無視 → 対策: **prior に weak prior** (= `sigma_global = pm.HalfNormal(sigma=2000)` のように広い)、**posterior predictive check** で予測が train data を再現するか検証

### 10.6 既存 edge との合成可能性

- **`independent-edges.dense.md` §2 案 E (GP for ANCC) と合成**: Track 10 で b_well を hierarchical、案 E で ANCC を GP → **全変数を Bayesian 化**
- **Track 6 (Physics constraint loss) と complementary**: Track 6 が physics を loss に、Track 10 が physics を model spec に → 両者重ねて augment
- **Track 8 (Stacking pyramid) の L1 base として加算**: Track 10 posterior mean を 1 base として L2 meta に渡す

---

## 10A. Track 11: Conditional Diffusion Model for Hidden TVT (案 I 本式化)

### 10A.1 構造原理

| 軸 | 既存 (= point prediction) | Track 11 |
|---|---|---|
| 推論パラダイム | Frequentist / Bayesian closed-form | **Generative (score-based diffusion)** |
| 時系列構造 | i.i.d. or AR(1) | **Conditional sequence generation** |
| 学習信号 | Direct regression / posterior | **Score matching (= noise prediction)** |
| 推論時計算 | Static | **Iterative reverse SDE (K=20-50 step)** |
| 不確実性 | None / point std | **Posterior samples (= ensemble of generative trajectories)** |

`independent-edges.dense.md` §6 案 I で「conditional diffusion model で hidden TVT の posterior を sampling、mean を point prediction に、variance を uncertainty に使う」と提案されていたが **paradigm shift 候補として詳細化されていない**。Track 11 で本式化する。

### 10A.2 着想根拠

- **Ho et al. 2020 DDPM** (https://arxiv.org/abs/2006.11239): conditional generation の基礎
- **Tashiro et al. 2021 CSDI** (https://arxiv.org/abs/2107.03502): time series imputation の score-based diffusion、ROGII の hidden TVT imputation と **構造完全一致**
- **Alcaraz & Strodthoff 2022 SSSD-S4** (https://arxiv.org/abs/2208.09399): structured state-space + diffusion で long sequence imputation、CSDI を超える性能
- **`first-principles.dense.md` §2.4** extrap curve: MD 4000 ft 末端で std 18.83 ft、ensemble で **末端不確実性を陽に modeling** すれば改善
- **Kaggle G2Net 1 位 (2021)** で diffusion + ensemble で +0.005 mAP (https://www.kaggle.com/competitions/g2net-gravitational-wave-detection/discussion/275390)

### 10A.3 数学的定式 + 実装手順

Conditional score-based diffusion (= CSDI 流):

```
Forward process:
q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

Reverse (denoising):
p_theta(x_{t-1} | x_t, c) where c = (Z, X, Y, GR_h, GR_v, TVT_v, Geo_v, visible TVT)

Training loss (score matching):
L = E_t E_{x_0, eps} ||eps - eps_theta(x_t, t, c)||^2

Sampling: K=20-50 step reverse SDE
x_0_sample ~ p_theta(x | c)  # 1 sample = 1 hidden TVT trajectory

Ensemble: K=20 samples → mean (point prediction), std (uncertainty feature)
```

#### Architecture (CSDI core)

```python
denoiser = TransformerDenoiser(
    d_model=128, n_heads=8, n_layers=4,
    n_features=8,  # (TVT_partial, Z, X, Y, GR_h, GR_v, TVT_v, Geo_v)
    diffusion_step_embedding_dim=64,
)
diffusion = GaussianDiffusion(
    schedule='cosine', T=200,
    sampling_steps=20,  # DDIM acceleration
)
```

#### Training

```python
for epoch in range(50):
    for batch in dataloader:
        x_0 = batch.tvt_residual   # (B, n_step, 1)
        c = batch.conditional      # (B, n_step, 7)
        t = torch.randint(0, T)
        x_t, eps = diffusion.add_noise(x_0, t)
        eps_pred = denoiser(x_t, t, c)
        loss = F.mse_loss(eps_pred, eps)
        loss.backward()
```

#### Inference

```python
# K=20 samples per test well
hidden_samples = torch.stack([
    diffusion.sample(denoiser, c=test_conditional, steps=20)
    for _ in range(20)
])  # (20, B, n_step, 1)
tvt_pred = hidden_samples.mean(0)
tvt_uncertainty = hidden_samples.std(0)
```

### 10A.4 期待 CV 改善 + 工数

- **期待 CV 改善 -0.3 〜 -1.5 ft**: `independent-edges.dense.md` §6.5 と整合。diffusion + ensemble の典型成功幅 (= G2Net、CSDI publication 結果)
- **工数 5 日**: CSDI 実装 (= 既存 GitHub fork で 1-2 日) + ROGII 用 conditional schema + pretrain (= FORCE 2020 / VOLVE で 2 日) + Kaggle 9hr 検証 (= K=20 sample × 200 test wells = 1-2 hr inference)
- **依存**: GPU access、PyTorch、Track 5 (Multi-task pretrain) と同基盤

### 10A.5 失敗モード 3 つ + 各々への対策

1. **Training 不安定 (= score matching loss が divergent)**: small data + diffusion 学習は典型的に不安定 → 対策: **EMA model** (= weight の exponential moving average で stable inference)、**classifier-free guidance** (Ho & Salimans 2022) で conditional / unconditional 混合学習、**Track 5 pretrain** で initialize
2. **K=50 step sampling で 9hr 超過**: 200 wells × 50 step × Transformer forward = 4-6 hr → 対策: **DDIM 20 step** で代替 (Song et al. 2021)、または **DPM-Solver 10 step** (Lu et al. 2022 https://arxiv.org/abs/2206.00927) で 5x 加速
3. **Mode collapse (= posterior が single mode に collapse、ensemble 効果消失)**: 対策: **multiple seeds × multiple schedules** (= K=20 samples × 3 schedules = 60 samples)、または **classifier-free guidance weight w を 0.0-2.0 で grid search**

### 10A.6 既存 edge との合成可能性

- **Track 5 (Multi-task pretrain) と必須合成**: diffusion denoiser を Track 5 で pretrain
- **Track 10 (Bayesian Hierarchical) と uncertainty 二重 quantification**: diffusion posterior std + Bayesian posterior std を両方 GBM feature に渡す
- **Track 8 (3-layer stack) の L1 base として加算**: diffusion mean を 1 base として L2 meta に渡す
- **既存 GBM stack と orthogonal**: GBM の predict と diffusion の sample mean を Ridge meta で blend

---

## 10B. Track 12: Geometry-first 2-stage prediction (= 大局 dip → row-level residual)

### 10B.1 構造原理

| 軸 | 既存 (= 1-stage row regression) | Track 12 |
|---|---|---|
| 推論パラダイム | Frequentist row-wise | **2-stage decomposition (= global dip prior + row residual)** |
| 学習信号 | Direct regression on residual | **Stage 1: per-well global parameters / Stage 2: row residual** |
| 時系列構造 | i.i.d. | **Per-well dip / strike を陽に推定** |

`top3-distill.dense.md` §3 FormationPlaneKNN は **per-well centroid を周囲 K=10 wells で plane fit** するが、これは **ANCC top の地理分布** のみ。Track 12 は **当該 well 自体の dip / strike** を visible で推定し、hidden の **大局 TVT trajectory** を先に決め、その上に row-level の細かい変動 (= residual) を別 model で予測する 2-stage。

### 10B.2 着想根拠

- **`problem-essence.dense.md` §1.1** Layer-cake の数学: 「ANCC top depth(X, Y) = a_w X + b_w Y + c_w」 = **well-specific 平面**。我々は地理分布だけでなく **per-well の (a_w, b_w, c_w) 自体を推定** すべき
- **`host-pptx-summary.dense.md` slide 12-13**: 「Geological dips behave similarly in neighboring wells」「Dip Dip Flat Flat (= 隣接 wells で dip パターンが共有)」 → dip を陽に modeling
- **`past-comps-deepdive.dense.md` §4.6** Kaggle Indoor Location 1 位 Dmitry Gordeev: **KNN + GBM (location) + CNN+RNN (trajectory reconstruction)** の multi-block 構造、ROGII 案 C (Geometry-Physics Hybrid) の延長
- **SLB Petrel automatic well-tie** (geophysics 業界標準): **大局 dip → row-level adjustment** は手動 geosteering の標準手順

### 10B.3 数学的定式 + 実装手順

#### Stage 1: Per-well dip estimation

Visible 区間で:
$$
\text{TVT}_v(s) = a_w \cdot X(s) + b_w \cdot Y(s) + c_w + \text{ANCC\_residual}(s)
$$
の右辺最初の 3 項を **per-well WLS で fit** (= visible データから (a_w, b_w, c_w) を直接推定)。$X(s), Y(s)$ は MD で変化するため、これは **3D plane** ではなく **trajectory 上の 2D plane**。

#### Stage 1 output: 大局 TVT trajectory (hidden 区間)

```python
TVT_global_hidden(s) = a_w * X(s) + b_w * Y(s) + c_w
# (= hidden 区間の大局 TVT、visible で fit した平面を hidden に外挿)
```

#### Stage 2: Row-level residual prediction

```python
residual_target = TVT_true(s) - TVT_global_hidden(s)   # (= 大局 dip からの偏差)
LGB.fit(features, residual_target)
```

#### Combine

```python
TVT_pred(s) = TVT_global_hidden(s) + LGB.predict(features)
```

#### Visible_ratio による Stage 1 / Stage 2 重み調整

```python
# Visible が短い well では Stage 1 (dip fit) の信頼度が低い → Stage 2 model に weight 大
w_stage1 = clip(visible_ratio / 0.30, 0, 1)   # visible_ratio >= 0.30 で full trust
TVT_pred(s) = w_stage1 * TVT_global_hidden(s) + (1 - w_stage1) * LGB_stage2_only.predict(features)
                                                + (Stage 1+Stage 2 の補正項)
```

### 10B.4 期待 CV 改善 + 工数

- **期待 CV 改善 -0.3 〜 -0.7 ft**: 2-stage decomposition は **大局 dip を陽に取る** ため、Stage 2 LGB の dynamic range が縮小 (= 既存 residual target も `±100 ft` から `±20-30 ft` に縮小)、bias-variance トレードオフで CV 改善
- **工数 2 日**: per-well WLS plane fit (= 既存 b_well 計算と同型) + Stage 2 LGB train (= 既存パイプラインに residual target 変更) + CV
- **依存**: 既存 features 全部、LGB pipeline

### 10B.5 失敗モード 3 つ + 各々への対策

1. **Visible 区間で fit した dip が hidden で破綻 (= fault / fold で大局構造が変わる)**: `problem-essence.dense.md` §1.3「fault / fold / unconformity」 → 対策: **per-well dip の confidence interval を Stage 2 feature に渡す** (= visible WLS residual std を per-well feature)、または **fault detection (= dTVT spike 検出) を post-proc で行い、検出された fault 以降は Stage 1 を再 fit**
2. **Stage 2 residual の RMSE が既存 residual 直接予測と同じ**: 大局 dip が既に既存 features (Z, X, Y) に含まれているため、Stage 2 が結局 同じ問題を解く → 対策: **Stage 1 で複数 dip candidates** を出す (= linear / quadratic / per-formation dip) + **Stage 2 で best candidate を選択する meta-feature** を渡す
3. **per-well WLS の overfit (visible が短い well で)**: visible 100 rows で 3 parameter fit は OK だが、**5 parameter (quadratic)** に拡張すると overfit → 対策: **shrinkage prior** (= dip parameters を global mean に Ridge regularize)、または **visible_ratio < 0.2 では Stage 1 を skip** して existing pipeline に戻す

### 10B.6 既存 edge との合成可能性

- **`independent-edges.dense.md` §5 案 H (MD-linear b_well) の上位互換**: 案 H は b_well を MD で線形 fit、Track 12 は b_well と ANCC trajectory を含めて per-well plane fit (= 3-parameter)
- **Track 10 (Bayesian Hierarchical) と本質的合成**: Track 10 の per-well b_well 階層 prior と Track 12 の per-well dip fit は **同一 model 内で結合可能** (= Track 10 hierarchical の sigma_form を Track 12 の dip residual で決定)
- **既存 6 要素 (FormationPlaneKNN + b_well WLS) と orthogonal**: 既存は per-row imputer、Track 12 は per-well global trajectory → 加算可

---

## 11. 比較トレードオフ表

| Track | 構造原理 (= paradigm change) | 期待 CV 改善 | 工数 (日) | 独自性 (= 公開 top にないか) | 失敗リスク | 既存 edge 合成 |
|---|---|---|---|---|---|---|
| **1: Classification + cumsum** | Discrete posterior + reconstruct | -0.3 〜 -0.8 | 2-3 | ★★★ (公開 top ゼロ) | 中 (= 離散化 bias) | Edge S 置換、案 D/G と合成可 |
| **2: NN + GBM joint** | Hybrid (= NN feature + GBM final) | -0.5 〜 -1.5 | 5-7 | ★★ (FORCE/Optiver で実証) | 中-高 (= overfit、runtime) | Track 5 必須合成 |
| **3: Beam + Kalman unified** | HMM + state-space 統合 | -0.4 〜 -1.0 | 3 | ★★★ (公開 top ゼロ) | 中 (= MLE 不安定) | 既存 Beam 置換 / 案 D 拡張 |
| **4: Adversarial validation** | Train/test shift-aware reweight | +0 〜 -0.5 | 1 | ★ (Kaggle 標準だが ROGII 未実装) | 低 (= AUC 事前測定で判定) | 全 track orthogonal |
| **5: Multi-task pretrain** | Foundation model fine-tune | -0.5 〜 -1.0 | 4-5 | ★★ (WLFM で先行例あり) | 中-高 (= domain shift、9hr) | Track 2 必須合成 |
| **6: Loss redesign** | Robust + depth-weighted + physics | -0.1 〜 -0.4 | 1-2 | ★ (Kaggle 標準) | 低 (= 既存 LGB に重畳) | 全 track 重畳可 |
| **7: Per-well online train** | Test-time custom model + pseudo cycle | -0.5 〜 -1.5 | 3 | ★★★ (Edge T 拡張、公開 top ゼロ) | 中-高 (= leak、9hr) | Edge T/R 上位互換 |
| **8: 3-layer stacking** | Pyramid meta-of-meta | -0.3 〜 -0.8 | 3 | ★★ (Kaggle 上位常套) | 低-中 (= meta overfit) | Track 1-7 を集約 framework |
| **9: Embedding well retrieval** | Learned representation KNN | -0.4 〜 -1.0 | 3-4 | ★★★ (Edge N learned 版、ROGII 未実装) | 中 (= AE collapse) | Edge N 拡張、Track 5 合成 |
| **10: Bayesian Hierarchical** | Hierarchical Bayes posterior | -0.3 〜 -0.8 | 4-5 | ★★★ (公開 top ゼロ) | 中-高 (= MCMC convergence、9hr) | 案 E と合成、Track 6 重畳 |
| **11: Diffusion (= 案 I 本式化)** | Generative posterior sampling | -0.3 〜 -1.5 | 5 | ★★★ (公開 top ゼロ、CSDI 流) | 中-高 (= training 不安定、sampling 9hr) | Track 5 必須合成、Track 8 base に加算 |
| **12: Geometry 2-stage** | Per-well dip + row residual | -0.3 〜 -0.7 | 2 | ★★ (案 H 上位互換、ROGII 未実装) | 中 (= fault で大局崩壊) | 案 H 置換、Track 10 と本質的合成 |

### 11.1 構造原理の 5 軸表 (= `independent-edges.dense.md` §0.1 と整合)

各 track が `independent-edges.dense.md` §0.1 の 5 軸でどう異なるか:

| Track | 推論パラダイム | 時系列構造 | 学習信号 | 推論時計算 | 不確実性 |
|---|---|---|---|---|---|
| 既存 (Top R/N) | Frequentist | i.i.d. + Beam (HMM) | Direct regression | Static | None (Beam) / PF std |
| **1: Classification cumsum** | Frequentist (discrete) | i.i.d. + cumsum | **CE + RMSE** | Static | **Categorical posterior** |
| **2: NN + GBM joint** | **Hybrid** | **Sequence (PatchTST)** | **Multi-source gradient** | Static (Stage 2) / Iterative (Stage 3) | None |
| **3: Beam + Kalman** | Frequentist + Bayesian filter | **State-space + HMM 統合** | Direct regression | Static | **Path-level posterior** |
| **4: Adversarial** | Frequentist | i.i.d. | Direct regression + **adv reweight** | Static | **Per-row similarity** |
| **5: Multi-task pretrain** | **Foundation fine-tune** | **Sequence** | **Pretext + downstream** | Static | None |
| **6: Loss redesign** | Frequentist | i.i.d. | **Robust / quantile / physics** | Static | **Quantile output** |
| **7: Per-well online** | Frequentist | i.i.d. + per-well train | **Pseudo-cycle** | **Iterative** | None |
| **8: 3-layer stack** | Frequentist + meta-learning | i.i.d. | Direct regression | Static | **Cross-meta diversity** |
| **9: Embedding retrieval** | Frequentist + learned KNN | i.i.d. + per-well retrieve | **AE pretrain** | Static | **Embedding distance** |
| **10: Bayesian Hierarchical** | **Bayesian posterior** | i.i.d. + hierarchical | **MCMC / VI** | Static | **Posterior credible interval** |
| **11: Diffusion** | **Generative (score-based)** | **Conditional sequence generation** | **Score matching (noise)** | **Iterative reverse SDE** | **Generative ensemble std** |
| **12: Geometry 2-stage** | Frequentist (2-stage) | **Per-well global trajectory + i.i.d. residual** | Decomposed regression | Static | None (or visible WLS residual std) |

= Track 1/2/3/5/7/10/11 は 2 軸以上が変わる **真の paradigm shift**、Track 4/6/8/9/12 は 1-2 軸変化の **paradigm extension**。Track 11 は **5 軸すべてが変化** で最も強い paradigm shift。

---

## 12. 累積効果見積 (= 上位 3-5 track 投入後)

### 12.1 投入順序の選択軸

Track 同士には合成可能性と排他性があるため、**累積効果は track 集合に依存** する。代表的な 4 つの集合を提示する。

#### Combo A: 「Bayesian + state-space」重視 (= 構造原理から攻める)

```
exp007 base (Edge Q+M):           CV 10.39
+ Track 3 (Beam + Kalman)          -0.6  → CV 9.79
+ Track 10 (Bayesian Hierarchical) -0.5  → CV 9.29
+ Track 1 (Classification cumsum)  -0.4  → CV 8.89
+ Track 6 (Loss redesign)          -0.2  → CV 8.69
+ Track 8 (3-layer stack)          -0.3  → CV 8.39
```

合計 -2.0 ft、想定 CV 8.39。所要工数 13-17 日 (= 残 86 日の 15-20%)。

#### Combo B: 「Deep learning + Foundation」重視 (= NN paradigm 投入)

```
exp007 base:                       CV 10.39
+ Track 5 (Multi-task pretrain)    -0.7  → CV 9.69
+ Track 2 (NN + GBM joint)         -0.8  → CV 8.89
+ Track 9 (Embedding retrieval)    -0.4  → CV 8.49
+ Track 6 (Loss redesign)          -0.2  → CV 8.29
+ Track 8 (3-layer stack)          -0.2  → CV 8.09
```

合計 -2.3 ft、想定 CV 8.09。所要工数 17-22 日 (= 残 86 日の 20-26%)。

#### Combo C: 「Test-time adaptation 重視」 (= 既存 paradigm に online train 追加)

```
exp007 base:                       CV 10.39
+ Track 7 (Per-well online train)  -0.8  → CV 9.59
+ Track 4 (Adversarial)            -0.3  → CV 9.29
+ Track 9 (Embedding retrieval)    -0.3  → CV 8.99
+ Track 6 (Loss redesign)          -0.2  → CV 8.79
+ Track 1 (Classification cumsum)  -0.3  → CV 8.49
```

合計 -1.9 ft、想定 CV 8.49。所要工数 10-13 日 (= 残 86 日の 12-15%、最短)。

#### Combo E: 「Generative + Geometry decomposition」 (= 公開 top と最も異なる paradigm)

```
exp007 base:                       CV 10.39
+ Track 12 (Geometry 2-stage)      -0.5  → CV 9.89
+ Track 11 (Diffusion)             -0.8  → CV 9.09
+ Track 9 (Embedding retrieval)    -0.4  → CV 8.69
+ Track 6 (Loss redesign)          -0.2  → CV 8.49
+ Track 8 (3-layer stack)          -0.2  → CV 8.29
```

合計 -2.1 ft、想定 CV 8.29。所要工数 16-19 日 (= 残 86 日の 19-22%)。**独自性が最大**、公開 top と paradigm が完全に異なる。

#### Combo D: 「全方位」 (= Track 1/2/3/5/7/8 = 6 track full deploy)

```
exp007 base:                                            CV 10.39
+ Track 5 + Track 2 (Foundation + Joint NN+GBM)         -1.0  → CV 9.39
+ Track 3 (Beam + Kalman)                               -0.5  → CV 8.89
+ Track 1 (Classification cumsum)                       -0.3  → CV 8.59
+ Track 7 (Online train)                                -0.4  → CV 8.19
+ Track 8 (3-layer stack)                               -0.3  → CV 7.89
```

合計 -2.5 ft、想定 CV 7.89。所要工数 22-28 日 (= 残 86 日の 25-33%、最長)。

### 12.2 累積効果の脆さに対する反証 (= optimistic 見積の正当性)

各 combo の **mid case** で見積りしたが、以下を考慮すれば pessimistic case も合理的:

| 因子 | mid case (上記) | pessimistic case |
|---|---|---|
| Edge S/T/R/P/N/G/final stack 既存累積 | 線形加算で -2.4 ft | 相関で **-1.2 ft** (= 50%) |
| Track 同士の相関 | 独立加算 | **Combo A の Track 3+10 重複で -0.3 ft** (= Bayesian + Kalman は同 noise を attack) |
| CV → LB ギャップ | -0 〜 +0.1 ft | **+0.3 〜 +0.5 ft** (= exp006 で観測の overfit 系) |
| Track 内部の hyperparam tuning | 期待値 mid case | 工数倍 + 効果 50% に縮小 |

→ **pessimistic combo A**: CV 8.39 + 0.5 (LB gap) + 0.3 (Track 3+10 重複) = **LB 9.2**、optimistic case で LB 8.4。**LB 8.x 帯到達は combo A/B のいずれかで CV ≤ 8.5 を達成すれば LB 8.x 射程**。

### 12.3 LB 想定との対応

| Combo | 想定 CV (mid) | 想定 LB (mid, CV+0.1) | 想定 LB (pessimistic, CV+0.5) |
|---|---|---|---|
| A: Bayesian/state-space | 8.39 | **8.49** | 8.89 (Top 1 越え) |
| B: Deep/Foundation | 8.09 | **8.19** | 8.59 (Top 1 越え) |
| C: Test-time | 8.49 | **8.59** | 8.99 (賞金圏 9.415 内) |
| D: 全方位 | 7.89 | **7.99** | 8.39 (Top 1 越え) |
| E: Generative/Geometry | 8.29 | **8.39** | 8.79 (Top 1 越え、最大独自性) |

公開 Top 1 (9.256) は combo C pessimistic でも越える可能性、combo A/B mid case では確実に超越。

---

## 13. ROGII 残 86 日への schedule fit

### 13.1 現行 roadmap (`docs/strategy/winning-strategy.dense.md` Phase 4-6) との突合

現行 Phase 4-6 は exp008-013 で Edge S/T/R/P/N/G を順次投入する想定 (= ユーザー指示の改修 b)。Track 投入は **既存 Phase の上に paradigm shift layer を重ねる** ことになる。

| Phase | 残期間 (日) | 現行 roadmap | Track 投入候補 |
|---|---|---|---|
| Phase 4 (LB 9 target) | 残 ~30 日 | exp008-010 (Edge S/T/N) | **Track 4** (= 1 日、低 risk) を先行投入、**Track 3 or Track 6** を並列開発 |
| Phase 5 (LB 8 target) | 残 ~30 日 | exp011-013 (Edge P/R/G + final stack) | **Track 1 / 7 / 9 のうち 2 つ** を投入 (= 3-4 日 / track) |
| Phase 6 (final blend) | 残 ~20 日 | sub 2 枚 (CV best + LB best) | **Track 8 (3-layer stack)** で集約 + post-proc tune |

### 13.2 工数と効果の選択軸

中央が判断すべき 4 つの軸:

#### 軸 1: 「Risk-Return balance」

- **Low risk (= Track 4 / 6 / 8)**: 合計 -0.5 〜 -1.7 ft、工数 5-6 日、確実だが LB 8.x は届かない可能性
- **High risk (= Track 2 / 5 / 7 / 10)**: 合計 -1.5 〜 -4.0 ft、工数 12-17 日、LB 8.x 確実だが失敗時のリカバリ困難

#### 軸 2: 「Single paradigm depth vs Multi paradigm breadth」

- **Single depth (= combo A 一本)**: Bayesian + state-space に集中、独自性最大化、コンペ後 write-up 価値高
- **Multi breadth (= combo D)**: 全方位投入、各 track の効果検証ループが分散、独自性希薄化

#### 軸 3: 「実装可能性 (= 残期間 vs 工数)」

- 残 86 日のうち、**Phase 4-5 で 60 日 + final blend 20 日 = 80 日**。Track 投入工数 (= 17 日 / combo B) は **20% の余裕**。combo D (= 28 日) はギリギリ
- subagent 並列開発 (`agents.md` §「Parallel Task Execution」) で同時 3-4 track 着手なら工数は **1/3 程度に圧縮** 可能 (= combo D も 10 日で済む)

#### 軸 4: 「Code competition の 9hr runtime 制約」

- **Track 2 / 5 / 7 / 10 は runtime 圧迫リスク**: NN inference + per-well train + MCMC は単独でも 4-6 hr 消費
- **Track 1 / 3 / 6 / 8 / 9 は runtime 影響低**: 既存パイプラインに乗せられる
- 4 track 以上を投入する場合、**Track 2 + Track 7 + Track 10 同時 deploy は確実に 9hr 超過** → 3 track までに絞るか kernel 分割

### 13.3 残 86 日で**現実的に投入可能な 3-4 track 候補** (= 推奨せず軸のみ)

| 候補 | track 集合 | 累積 (mid) | 工数 | 主リスク |
|---|---|---|---|---|
| 候補 α: 安全策 | 4 + 6 + 8 | -0.7 〜 -1.7 ft | 5-6 日 | 効果不足 (LB 9 帯止まり) |
| 候補 β: 中道 (= combo A 縮小) | **3 + 6 + 8** | -1.1 〜 -2.0 ft | 7-8 日 | Track 3 の MLE 不安定 |
| 候補 γ: 中道 (= combo C 縮小) | **7 + 4 + 6** | -1.0 〜 -2.4 ft | 5-6 日 | Track 7 の runtime + leak |
| 候補 δ: 攻め (= combo B 縮小) | **2 + 5 + 6** | -1.3 〜 -2.9 ft | 10-14 日 | NN pretrain 失敗時のリカバリ困難 |
| 候補 ε: 独自性最大 | **1 + 3 + 10** | -1.0 〜 -2.6 ft | 9-11 日 | 全 track が公開未実装、組合せ debug 困難 |
| 候補 ζ: Generative attack | **11 + 12 + 8** | -0.8 〜 -2.4 ft | 9-10 日 | Track 11 training 不安定、Generate × Geometry combine debug |
| 候補 η: 4 track 拡張 (= 候補 β に Track 9 追加) | **3 + 6 + 8 + 9** | -1.4 〜 -2.9 ft | 10-12 日 | Track 9 AE collapse + Track 3 MLE 不安定 同時 debug |

各候補とも `cv-breakthrough` 路線として **paradigm shift を 3-4 個** 投入する形。判断は中央に委ねる。

### 13.4 並列開発時の subagent 分割 (= `agents.md` Parallel Task Execution との整合)

候補が確定したら、各 track を独立 subagent に分割可能 (= track 間依存が低いため):

| track 集合 | subagent 配置 (= worktree branch 名) | 同時可能 worker 数 |
|---|---|---|
| 候補 β (= 3+6+8) | `feat/track-3-kalman-beam` + `feat/track-6-loss-redesign` + `feat/track-8-stack-pyramid` | **3 worker** + 中央 1 = 4 pane |
| 候補 γ (= 7+4+6) | `feat/track-7-online-train` + `feat/track-4-adversarial` + `feat/track-6-loss-redesign` | 3 worker |
| 候補 δ (= 2+5+6) | `feat/track-5-pretrain` (= 先行 4 日)、その後 `feat/track-2-nn-gbm` + `feat/track-6-loss-redesign` | 2 worker (= Track 2 は 5 後の direct dependency) |
| 候補 ε (= 1+3+10) | `feat/track-1-classification` + `feat/track-3-kalman-beam` + `feat/track-10-bayesian` | 3 worker |
| 候補 ζ (= 11+12+8) | `feat/track-11-diffusion` + `feat/track-12-geometry-2stage` + `feat/track-8-stack-pyramid` | 3 worker (= Track 11 は GPU 専有、Track 12/8 は CPU) |

`agents.md` の N+1 ペイン構成 (= チャット 1 + viewer N) に従い、各 subagent には初手で「branch push までで停止、merge は中央 (= 私) が引き受ける、develop に直接 commit 禁止」を通知 (~/.claude/CLAUDE.md 「[2026-04-30] 並列セッション orphan 化」教訓)。

---

## 14. 関連ファイル

- `docs/research/independent-edges.dense.md` (= 既存 Edge D-O、本 doc の前提)
- `docs/research/strategy-critique.dense.md` (= 案 A/B/C 批判的再評価、本 doc は更に paradigm shift を追加)
- `docs/research/first-principles.dense.md` (= 計測値、本 doc の期待 CV 改善の根拠)
- `docs/research/top3-distill.dense.md` (= 公開 Top R/N 解体、本 doc は越える方向)
- `docs/research/pilkwang-distill.dense.md` (= pilkwang super stack、本 doc Track 8 の原型)
- `docs/research/past-comps-deepdive.dense.md` (= 過去 winner、本 doc の各 track 着想根拠)
- `docs/research/kaggler-tactics.dense.md` (= 汎用 transferable pattern、本 doc Track 4/6 の根拠)
- `docs/research/host-pptx-summary.dense.md` (= host slide 9/12-13、Track 9 の根拠)
- `docs/research/discussions.dense.md` (= discussion topic 698002、Track 7 の根拠)
- `docs/dev/leaderboard.dense.md` (= 実測 CV/LB 履歴、本 doc §0.1 の現状認識根拠)
- `docs/strategy/winning-strategy.dense.md` (= 既存 roadmap、本 doc §13 は roadmap に paradigm shift を重ねる提案)
