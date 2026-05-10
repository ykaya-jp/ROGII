# ROGII の本質 — なぜこのコンペでこれらの手法が効いているのか

> 公開ノートブック (LB 10-12 帯) がやっていることを **なぜそれが効くのか** という観点で物理 / 統計 / 数学的に整理。
> 表面的な真似ではなく、独自 improvement の起点にする。
> 出典は併記。書き手の推論は明示的に "推測:" でラベル化。

## 0. このコンペが本質的に解いている問題

水平井 (horizontal well) を掘削中、bit から得られる **限られた観測** (3D 座標 + Gamma Ray) と **縦穴 typewell の reference signal** から、bit が今 **地層内 (formation 内) のどこにいるか** = TVT (True Vertical Thickness) を当てる。

これは業界用語で **Geosteering** (= 地質ステアリング) の中核問題。手動だと expert geologist が毎時間レベルで判断するのを、自動化したい。

### 数学的定式化

- 観測: $\{(MD_t, X_t, Y_t, Z_t, GR_t)\}_{t=1}^{T}$, $t = 1 \ldots T$ は ft 毎の 1D index
- visible 部分: $t \in [1, T_v]$ で `TVT_input` ≠ NaN (= 真の TVT 既知)
- hidden 部分: $t \in [T_v + 1, T]$ で `TVT_input` = NaN (= 予測対象)
- 補助: typewell の $\{(TVT_k^{tw}, GR_k^{tw}, Geology_k^{tw})\}$ は **TVT 軸上に並んだ垂直 reference**

求めたいもの: $\hat{TVT}_t$ for $t \in [T_v + 1, T]$ で hidden 全部。

## 1. なぜ `TVT = -Z + ANCC + b_{well}` が物理的に正しいのか

### 1.1 Layer-cake の数学

地層は **層序学的 stratigraphic** には平面層 (layer cake) として理想化できる。各 formation (ANCC, ASTNU, ...) は 3D 空間で:
$$ \text{ANCC top depth}(X, Y) = a_w X + b_w Y + c_w $$
の形 (well-specific 平面)。dip と strike が局所的に一定なら厳密、緩く変動するなら近似。

bit の TVD (Z, observed) と ANCC top の depth との差 = bit が ANCC layer に対してどれだけ深く入っているか:
$$ Z_t - \text{ANCC}(X_t, Y_t) = (\text{depth of bit below ANCC top}) $$

TVT は **「地層内での bit 位置」** = layer top からの downward 距離。Z は海面基準 (negative downward) なので符号反転:
$$ \text{TVT}_t = -(Z_t - \text{ANCC}(X_t, Y_t)) + b_{well} = -Z_t + \text{ANCC}(X_t, Y_t) + b_{well} $$

### 1.2 なぜ Pearson = -1.0 / resid_std = 0.007 ft なのか

konbu17 が報告 (`konbu17/...py:7-13`): visible 区間で `TVT vs (Z - ANCC_provided_in_train)` が **完全線形** で、residual std は **0.007 ft** (= 2 mm)。

物理的に: train ノートブックでは ANCC は per-row で正確に提供されている (= geologist が interpret した formation top)。**TVT_input 自体が ANCC 引き算で逆算されている** 可能性が高い。

つまり $TVT = f(\text{ANCC measured})$ という決定関係。test では ANCC が NaN だから自分で imput が必要。

> **推測**: 主催の ROGII は実際の StarSteer software 内部で、各 formation top depth から TVT を機械的に算出している。コンペの "TVT" の定義はその出力。

### 1.3 いつ formula が破綻するか

- **fault** (断層): ANCC layer が垂直方向に jump → TVT も jump
- **fold** (褶曲): ANCC が局所的に大きく曲がる → plane fit 誤差大
- **unconformity** (不整合): 一部 layer が消失 → 隣接 wells から impute するときに不正確
- **lateral facies change** (側方相変化): 同じ層位でも岩相が変わって GR 関係が崩れる

これらが **公開 notebook の Ridge stack による補正** で吸収されている (= LGB が plane fit residual を学ぶ)。

## 2. なぜ residual target (TVT - last_known_TVT) が効くのか

### 2.1 統計的理由

直接予測 vs residual:
- 直接 TVT: target $\sim \mathcal{N}(\mu = 11000, \sigma \sim 750)$ (Phase 1 EDA で TVT range p50 = 758 ft, std 大)
- Residual TVT - last_known: target $\sim \mathcal{N}(\mu = 0, \sigma \sim 100)$ (visible end からの相対変動)

**変動量が約 7-10 倍小さい** ので model の有効 dynamic range が増える。同じ relative error でも RMSE は 7-10 倍小さい。

### 2.2 「visible 区間の終端は既知」を活かす

最後の visible TVT は完全に既知 (TVT_input.iloc[-1])。これを **anchor** に取れば:
- hidden の最初の row → 真の TVT は anchor から 1 ft 程度しか変動しない (Phase 1 EDA: dTVT/dMD は ±0.5 以内)
- 深部に行くほど residual が大きくなる

⇒ fade-in tau による smoothing が物理的に意味を持つ (= "anchor から離れるほど不確実性増大"を表現)。

### 2.3 コードでの実装

```python
target_train = TVT - last_known_TVT
# ...predict residual...
final_test = last_known_TVT + alpha * predicted_residual * (1 - exp(-md_since / tau))
```

## 3. なぜ Beam Search が効くのか — Viterbi vs Beam の違い

### 3.1 問題構造

bit の GR が typewell の どの TVT に当たるか? これは **隠れマルコフモデル (HMM)** の問題:
- 状態 $s_t$: typewell index (TVT 軸上の位置)
- 遷移確率 $P(s_{t+1}|s_t)$: 隣接 ($\pm 1$) は高、跳躍は低
- 観測 $P(GR_t^h | s_t)$: $\propto \exp(-(GR_t^h - GR_{s_t}^{tw})^2 / 2\sigma^2)$

理論的には Viterbi で最尤 path 求まる。**でも、なぜ Viterbi だけだとダメか**:
- bit が **layer を上下動** する (going up/down strata) → GR pattern が repeat → posterior が **multimodal**
- Viterbi は単一 path を返す → 局所最適に陥る (= 1 個の peak しか追えない)
- **Beam K state 維持** → 上位 K modes を maintain → 終端で最良 path 選択

### 3.2 5-7 configs の意義

各 config は (move_cost, emit_scale) のトレードオフを変える:
- **conservative** (高 move_cost, 低 emit_scale): 慎重、stay at same TVT 多い
- **loose** (低 move_cost, 高 emit_scale): 探索的、TVT 変動が大きい
- **smooth** (radius 5): GR を window 5 で smoothing → noise robust

5 config で **mean / std / median** を feature 化することで、model は **「Beam の不確実性」** を学習する。conservative と loose の差が大きい区間 = 多 modes = 不確実性高い → 他特徴 (PF, NCC) で補完。

### 3.3 emit_scale の意味

$$ P(GR | s) \propto \exp\!\left( -\frac{(GR - GR_s^{tw})^2}{\text{emit\_scale}} \right) $$

emit_scale が小さい (= 低 64) と GR mismatch を厳しく penalty → "厳格" matching。
emit_scale が大きい (= 220) と soft → "緩やか" matching。

GR sensor noise は約 10-30 API → emit_scale = $2 \sigma^2 \approx 200-1800$ が物理。Top notebooks の 64-220 は **やや厳しめ** ⇒ noisier likelihood で smoothness 優先。

## 4. なぜ Particle Filter が効くのか — Beam では足りない subgrid 精度

### 4.1 連続 state の必要性

Beam Search の状態空間は typewell の TVT grid (0.5-1.0 ft 間隔)。bit は連続的に動くので **subgrid (sub-foot)** 精度が要る。

Particle Filter:
- state: $(\text{TVT}_t, v_t)$ where $v_t = dTVT_t/dMD_t$ (TVT velocity)
- transition: $v_{t+1} = \alpha v_t + \mathcal{N}(0, \sigma_v)$, $\text{TVT}_{t+1} = \text{TVT}_t + v_{t+1} \cdot dMD$
- 連続値で表現 ⇒ Beam の grid より細かい

500 particles で multimodal posterior を近似 → resampling で degeneracy 防止 → roughening で diversity。

### 4.2 Z velocity prior の効力

`needless090.run_pf_z` が使う:
$$ v_t \approx \beta \cdot dZ/dMD + \text{intercept} $$
$\beta$ は visible 区間で fit した well-specific 係数。これが **TVT の velocity を Z velocity (= bit の geometry) で予測** する constraint。

物理的: bit の geometry (Z 動き) は完全観測。地層 dip がこの well の visible 部分で fix していると仮定すると、$\frac{dTVT}{dMD} = -\frac{dZ}{dMD} + \frac{d(\text{ANCC})}{dMD}$。後者 (formation top の MD 方向 gradient) は visible で fit。

### 4.3 `S = TVT + Z` PF の優雅さ

shinya の発想 (`shinyanagai123/12-388-...py:142-184`): 
$$ S_t \equiv TVT_t + Z_t $$

なぜ?:
$$ S = TVT + Z = (-Z + \text{ANCC} + b) + Z = \text{ANCC} + b $$

つまり $S$ は **bit の Z (drilling 経路の影響) を消した、純粋な geology signal**。$S$ は ANCC layer の depth に等しいので、空間的に滑らか:
- $S$ の rate $dS/dMD$ は地層 dip だけで決まる (drilling path に依存しない)
- $S$ の noise は drilling vibration や Z measurement error から来る

この simpler signal を PF で track。Z が動いても $S$ は smooth 保つ。

## 5. なぜ FormationPlaneKNN が効くのか — 地理的滑らかさ

### 5.1 ANCC layer の地理分布

Permian Basin / Eagle Ford 等の堆積盆地では:
- ANCC top depth は (X, Y) に対し **滑らかに変動** (regional dip + minor folding)
- 隣接 wells で ANCC depth は ~ 数 ft 以内の差
- 遠 wells では数百 ft の差

### 5.2 K=10 centroid の選択理由

各 well を 1 centroid (X, Y median, formation depth median) に縮約することで:
- データ規模を 5M rows → 773 points に圧縮 (KDTree 構築 / query 高速)
- well 内の noise を除去 (well 内 ANCC の median を取る)
- K=10 で安定 plane fit (4 点で plane が決まるが、ノイズ込みで K=10 が経験的に良い)

### 5.3 plane fit vs IDW

konbu17 報告 (`konbu17/...py:6`): plane fit RMSE 17 ft vs IDW 47 ft (2.7x 改善)。

物理: ANCC が **平面状に変動** (regional dip) するなら plane fit が低 bias。IDW (= 距離反比例平均) は局所平均で over-smooth ⇒ regional dip を捉えない。

## 6. なぜ Self-NCC が効くのか — visible pattern の repeat 検出

### 6.1 layered 地質での repeat

bit が geosteering で layer を **上下動** すると、GR pattern は visible 区間で見た形が hidden 区間で **再現** する (= same layer, same depth → same GR)。

例: bit が visible で TVT 80 → 100 → 80 ft と変動 → typewell の TVT 80-100 区間の GR pattern が visible で観察 → hidden で同じ pattern が出たら、対応 TVT は同じ 80-100 区間内のどこか。

### 6.2 NCC (Normalized Cross-Correlation) による検出

$$ \text{NCC}(i, j) = \frac{(G^{vis}_i - \bar{G}^{vis}_i) \cdot (G^{hid}_j - \bar{G}^{hid}_j)}{\sigma^{vis}_i \cdot \sigma^{hid}_j \cdot W} $$

window $W=31$ で sliding。各 hidden row $j$ について、visible のどの window $i$ と最もマッチするか探す → そのマッチ位置の visible TVT を hidden の TVT 候補とする。

### 6.3 sc_score の重要性

NCC の最大値 (0-1) が **マッチの信頼度**。score 高 → repeat 確定 → sc_d を強く信頼。score 低 → repeat 確認できない → 他 feature 優先。

`sc_trust = clip(n_visible / 200, 0, 0.6)` ⇒ visible が 200 ft 以下では信頼度低 (= sample 不足)。`hyb = (1 - sc_trust) * beam + sc_trust * sc_raw` で Beam と weighted blend。

## 7. なぜ Affine GR Calibration が効くのか — well 間 GR スケール統一

### 7.1 GR sensor の well 間誤差

Phase 1 EDA: GR_mean range は 37-130 (3.5x 差)。原因:
- センサーの calibration 違い
- 掘削条件 (mud weight, hole size) による attenuation
- formation 違い (Permian で shale rich vs sand rich)

### 7.2 affine cal の式

$$ kgr = a \cdot tw\_GR(known\_TVT) + b $$

visible 区間で `kgr` (horizontal GR) を `tw_GR(known_TVT)` (typewell GR at known TVT) に対して回帰 → $(a, b)$ を fit。

これで model は "well 内 GR 相対変動" を学習 (= absolute GR 値ではない)。tw_diff features の $a, b$ で transformation を入力に与えれば、`gr - (a * tw_GR + b)` が **calibrated residual** になる。

## 8. なぜ tw_diff (3 anchors × 11 offsets) が効くのか — multimodal posterior の encode

### 8.1 idea

各 hidden row の bit は typewell のどの TVT 位置にいるか? — 3 つの hypothesis (last_known / beam_ref / sc_raw) を anchor とし、各 anchor から ±11 offset で **typewell GR の "候補値"** を生成 → bit GR との差を 33 列の feature 化。

### 8.2 model がここから何を学ぶか

LightGBM は決定木 ensemble で:
- "anchor X + offset Y で gr_diff = 0 になる組み合わせ" → 真の TVT
- "全 33 列が大きい (mismatch)" → どの anchor も間違い
- "anchor X だけ small で anchor Y は large" → X が真値、Y は誤った hypothesis

これは LGB に **「anchor 候補の事後分布」を教える** feature engineering。

### 8.3 なぜ 11 offsets か

offsets `[-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80]` は **対数的に間隔を広げる**。物理的には:
- 微小変動 (±5 ft) は GR の高周波で識別
- 中規模 (±20-40 ft) は layer 内変動
- 大規模 (±80 ft) は隣接 layer

これで **multi-scale matching** を一気に encode。

## 9. なぜ Ridge stacking (positive coef, no intercept) が効くのか

### 9.1 positive coef

stacking で `coef >= 0` を強制 → 各 base model の予測を **正の重み** で足し合わせる。
- negative weight は overfit signal (model が他 model の bias を打ち消す方向に効く) → 削除
- positive weight = "all models agree to push up if they all predict up" の物理的合理

### 9.2 no intercept

`intercept = 0` を強制 → residual 0 は base = last_known で完結 (anchor 整合)。

### 9.3 alpha 1.0 (mild regularization)

`Ridge(alpha=1.0)` の L2 が control → fold 毎の noise を除去しながら、main signal を保持。

## 10. なぜ post-process (alpha × fade-in tau + Savitzky-Golay) が効くのか

### 10.1 alpha (0.6-1.0)

predicted residual に linear scaling。

物理: model が **過剰に extrapolate** している場合、true residual の magnitude が予測より小さい → alpha < 1.0 で縮小 → fold-wise OOF RMSE で grid search → optimal alpha 0.85-0.95 が経験的。

### 10.2 fade-in tau

$$ \delta_{smoothed} = \delta_{predicted} \cdot (1 - e^{-md\_since/\tau}) $$

- $md\_since = 0$ (= visible end の真上) → factor = 0 → predicted = 0 → final = anchor (= last_known)
- $md\_since \to \infty$ → factor → 1 → predicted そのまま

これは **物理的事前知識** (= visible boundary 直後では真の TVT は anchor に近い) を直接 encode。tau は smoothing length scale。

### 10.3 Savitzky-Golay window=17, poly=3

各 well の predicted TVT を 17 ft window × 3 次多項式 fit。
- 高周波 noise (model variance) を除去
- 地層は物理的に滑らか (mm/ft オーダーの変動はノイズ)
- window=17, poly=3 は **約 1 layer 分の長さ × 二次曲率を保持** という選択

## 11. **なぜ我々が exp001 で失敗したか** (= 学び)

実装した: 絶対 TVT 予測, plane fit (visible 25% で fit), GR rolling features。
結果: val RMSE 30+ (LB baseline 12.6 に遠く及ばず)。

**根本原因**:

1. **Target を residual 化していなかった** ⇒ TVT の dynamic range 7000 ft で model が overwhelmed
2. **`tvt_formula = -Z + ANCC + b_well`** を使わなかった ⇒ 地層 dip 構造を全く model に教えていない
3. **typewell GR との関係** を model に教えていない (Beam / PF / NCC / tw_diff なし) ⇒ 「bit が typewell のどこにいるか」を学習する手がかりが無い

**学び**: 「データが何を表しているか」「目的変数が何によって決まるか」を **物理的に理解** してから ML を当てる。データ駆動のみだと方向違いを引く。

## 12. 我々が独自にできる improvement (= 公開 LB 10 を超える edge)

公開 top notebooks (LB 10.0-12.0) は以下を全部やっている:
- residual target ✓
- tvt_formula via FormationPlaneKNN ✓
- Beam Search (5-7 configs) ✓
- Particle Filter (PF_z + PF_ancc) ✓
- Self-NCC ✓
- Affine GR cal + tw_diff ✓
- LGB×3 + CB or XGB + Ridge ✓
- post-proc fade-in + SG ✓
- TabICL (1-2 ノートブック)

### 12.1 ノートブック群がやっていない・改善余地ある領域

| アイデア | 物理的根拠 | 実装難度 | 期待 LB 改善 |
|---|---|---|---|
| **Sequence Transformer + cross-attention to typewell** | bit 全 sequence を 1 forward で予測、Beam の path を soft-attention で表現 | High | 0.3-1.0 |
| **Multi-task DL** (TVT main + ΔTVT + Geology aux + 6 formation aux) | representation を richer に、Geology label が train 限定でも有効 | Mid-High | 0.2-0.5 |
| **PINN soft constraint** ($TVT + Z - \text{ANCC}_{predicted}$ を loss に組み込み) | model が formula を必ず満たすよう学習 | Mid | 0.1-0.3 |
| **FORCE 2020 / VOLVE で pretext pretrain** | 200-773 wells では DL に少ない | Mid | 0.2-0.5 |
| **Beam search with learnable transition/emission** (= NN-based HMM) | hand-tuned hyper を model に学ばせる | High | 0.3-0.7 |
| **Multi-resolution PatchTST** (32/64/128 patch) | layer scale 違いを階層で表現 | High | 0.2-0.4 |
| **Gaussian Process on residual** (uncertainty 推定込み) | PF/Beam の uncertainty を GP で fold | Mid-High | 0.1-0.3 |
| **Layer-aware attention** (typewell Geology label を pre-encoded vocabulary として attention key に乗せる) | 6 layer の categorical structure を直接利用 | Mid | 0.2-0.4 |
| **k-NN well retrieval を test 時動的に** (BM25 風 / well embedding) | 近傍 wells の feature を test 推論時に補完 | High | 0.1-0.3 |

### 12.2 重要な仮説 — まだ誰もやっていない idea

**"Beam path + PF + NCC を soft-attention で blend する Sequence Transformer"**:

公開 notebook は Beam / PF / NCC を **decoupled な feature** として LGB に与えるだけ。これらを **共通 hidden state を持つ DL model** で **end-to-end** に学習すれば、:
- 各時点で最適な weighting (= どの signal を信じるか) を model が学習
- Beam の不確実性 (5 configs std) を soft probability として decoder に渡す
- typewell との cross-attention を multi-head で多重に試す

これは Plan 案 B に直接該当。Phase 4 で実装する。

## 13. ROGII の **絶対に外してはいけない 3 原則** (実装の前に再確認)

1. **target は必ず residual** (TVT - last_known_TVT)。絶対値予測は禁止。
2. **`tvt_formula = -Z + ANCC_imputed + b_well_prefix`** を **必ず features に入れる**。これだけで RMSE 14 → 12 帯。
3. **typewell との対応を model に教える** (Beam / PF / NCC / tw_diff のいずれか必須)。

これら 3 つを満たすだけで LB 12 帯。+ 各種 ensemble + post-proc で LB 10 帯。+ DL hybrid で LB 8-9 (1 位射程)。

## 出典

- 公開ノートブック (`_research_kernels/` 内 8 件)
- konbu17 docstring (`_research_kernels/konbu17__rogii-plane-fit-formation-top-knn/rogii-plane-fit-formation-top-knn.py:1-37`)
- needless090 全コード
- Phase 1 EDA (`outputs/eda/aggregate-summary.json`)

## 付録: コンペが要求している数値感

- TVT 値域: 9245-12894 ft (well 平均 ~ 11000)
- TVT range per well: p50 758 ft, p95 997 ft
- visible 比率: median 26%
- LB benchmark:
  - all-zero submission (= residual 0): 推定 RMSE 200+ (TVT range の 1/4 程度)
  - last-known broadcast (= residual 0 だが anchor 使う): 推定 RMSE 50-100
  - simple LGB on absolute TVT: 30 (= 我々の exp001 fold 1)
  - simple LGB on residual + tvt_formula: ~14-15 (推定)
  - 公開 baseline 12.602 (LightGBM)
  - 公開 LB top 10.081 (Beam+PF+stacking)
  - 1 位狙い: LB 8-9 帯
