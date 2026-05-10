# 過去類似コンペ Winner Solution 完読 (2026-05-10)

> **目的**: 戦略 doc `winning-strategy.dense.md` で「構造一致度 最高 / 高 / 中」とされた 3 件を行単位で完読し、ROGII (masked TVT regression) に直接転用すべき要素と転用すべきでない要素を切り分ける。Track D-2 で追加類似コンペを横断探索し、共通する勝ち筋を抽出する。
>
> **方針**: コードを丸写ししない。具体的 file:line / arXiv ID / GitHub URL を必ず添え、ROGII の現 baseline (= R/N kernel 完読済み、`top3-distill.dense.md` 参照) に対する **構造的差分** を評価する。
>
> **CRITICAL**: 推奨案ではなく **トレードオフと選択軸** のみ提示。判断は開発者に委ねる。

---

## 0. 出典 1 次資料 (この doc で参照する全 winner solution の URL)

| # | コンペ | 順位 | チーム / 著者 | 出典 |
|---|---|---|---|---|
| 1 | SPWLA 2023 PDDA Depth Shift | 1 位 | Dreamstar (Fan Meng, SiYuan Chen, YingYing Ye, HaiLong Jiang) | https://github.com/arrayofstar/dreamstar (notebook `SPWLA_2023_dreamstar_lite.ipynb`) / 公式 https://github.com/pddasig/Machine-Learning-Competition-2023 (`Solutions/All for a dream.ipynb`) |
| 1b | SPWLA 2023 PDDA Depth Shift | 2 位 | WELL-LOGGED | `Solutions/WELL-LOGGED_SPWLA_2023_ML.ipynb` |
| 1c | SPWLA 2023 PDDA Depth Shift | 3 位 | ARC_CNLC | `Solutions/ARC_CNLC/` |
| 2 | Kaggle Ventilator Pressure Prediction | 1 位 | Vandewiele / He / Zidmie / Khahuras / Riccardo | https://github.com/GillesVandewiele/google-brain-ventilator (PID post-proc) + https://github.com/Shujun-He/Google-Brain-Ventilator (LSTM+CNN+Transformer) + https://medium.com/data-science/winning-the-kaggle-google-brain-ventilator-pressure-prediction-2d4c90d831ec (write-up) + https://www.kaggle.com/c/ventilator-pressure-prediction/discussion/285256 |
| 3 | FORCE 2020 Lithology Prediction | 1 位 | Olawale Ibrahim | https://github.com/olawaleibrahim/2020_FORCE_Lithology_Prediction + https://ibrahim-olawale13.medium.com/force-2020-machine-learning-lithology-predictionwinning-solution-8cbf78290b41 + 公式結果 https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition/blob/master/readme.md |
| 4 | SPWLA 2020 PDDA Sonic Synthesis | top5 | MLogging / Oilers / UTFE | https://github.com/pddasig/Machine-Learning-Competition-2020/tree/master/Solutions |
| 5 | Kaggle IceCube Neutrinos in Deep Ice | 1-3 位 | (Transformer 系 3 件) | https://arxiv.org/abs/2310.15674 |
| 6 | Kaggle LANL Earthquake | 1 位 | Philipp Singer | https://medium.com/@ph_singer/1st-place-in-kaggle-lanl-earthquake-prediction-competition-15a1137c2457 |
| 7 | Kaggle OpenVaccine COVID-19 mRNA | top10 (4 位 = Vandewiele) | 複数 | https://github.com/GillesVandewiele/covid19-mrna-degradation-prediction |
| 8 | Kaggle Indoor Location Navigation 2021 | 1 位 | Dmitry Gordeev チーム | https://github.com/kuto5046/kaggle-indoor (補助参考) |
| 9 | Kaggle Jane Street Market Prediction 2021 | top | (Autoencoder + MLP) | https://github.com/scaomath/kaggle-jane-street |

---

## 1. SPWLA 2023 Depth Shift — 1 位 Dreamstar (= ROGII 構造一致度 **最高**)

### 1.1 問題設定 (= ROGII との対応関係)

- **入力**: 9 wells (train) の整列済み GR (基準) + RHOB / NPHI / RD (シフト前)
- **出力**: 3 wells (test) の **シフト補正後の値** + **depth shift 量**
- **評価**: NMSE (補正後値) + MAD (depth shift) の rank 平均
- **ROGII との対応**:
  - 基準 GR ⇄ ROGII の **typewell GR** (両者とも reference signal)
  - 整列対象 RHOB/NPHI/RD ⇄ ROGII の **horizontal GR と TVT_input**
  - depth shift 量 ⇄ ROGII の **TVT** (= bit が typewell axis 上のどこにいるか)
- **構造一致度が "最高" な理由**: 「複数 channel time series を reference signal で整列させる」点が完全一致。違いは ROGII では shift 量が直接の target で、最終的に shifted-curve 値ではなく **alignment index** が問われる点

### 1.2 アーキテクチャ詳細

**mWDN-GRU** (Multi-level Wavelet Decomposition Network + GRU) を core として採用 (`SPWLA_2023_dreamstar_lite.ipynb` cells 65-67):

- **WaveBlock** (cell 65): seq_len の linear layer 2 本 (`mWDN_H` / `mWDN_L`) を学習可能 wavelet filter として初期化。filter 初期値はデフォルトで Daubechies-4 風の係数 `h_filter = [-0.2304, 0.7148, -0.6309, ...]`、`pywt.Wavelet(name)` でも可。Sigmoid → AvgPool1d(2) で 1/2 downsample。`L_H_loss = α||W_L - W_L^orig||₂ + β||W_H - W_H^orig||₂` (filter が wavelet coefficient から離れすぎないよう regularization、α=β=0.5)
- **mWDN class** (cell 66): `levels=3` で 3 段 wavelet 分解。各段の `x_h` (high freq) と最終段の `x_l` (low freq) を独立 GRU encoder (`hidden=256, num_layers=1`) に通す → `F.interpolate(size=seq_len, mode='linear')` で全段を input 解像度に戻す → cat → Linear(256*4 → 256) → Linear(256 → 3)
- **入力 5 ch**: `[DEPT, GR, RHOB, NPHI, log_RD]` のうち forward 内で DEPT を除き 4 ch を使用 (`x_l = x[:, :, 1:]` cell 66 forward)。**GR がここで reference として常に input に居る** のが ROGII でいう "typewell injection" に相当
- **出力 3 ch**: RHOB / NPHI / log_RD の補正後値
- **seq_len = 224** (cell 47 注記)

### 1.3 loss 設計

**NMSE** (cell 77):
$$\mathrm{NMSE}(\hat{y}, y) = \frac{\mathbb{E}[(\hat{y}-y)^2]}{\mathbb{E}[y^2]}$$

総 loss (`cell 83 line "loss = metrics_nmse(...) + L_H_loss * 0.01"`):
$$\mathcal{L} = \mathrm{NMSE}(\hat{y}, y) + 0.01 \cdot (\alpha L_L + \beta L_H)$$

= 主 loss (NMSE on values) + 0.01 倍の wavelet filter regularization。**MAD は metric のみで loss には入れていない** (DTW post-proc が depth shift を修正する設計)

### 1.4 data augmentation — `random_stretch` (cell 30)

これが Dreamstar の **ROGII 直接転用候補 ★★★**:

```python
def random_stretch(curve, depth0, ank_p=0.6, seed=10):
    n = int(length * ank_p)
    anchor_points0 = sorted(np.random.choice(arange(length), n, replace=False))
    anchor_points1 = sorted(np.random.choice(arange(length), n, replace=False))
    depth1 = np.interp(depth, depth[anchor_points0], depth[anchor_points1], ...)
    f = interpolate.interp1d(depth1, curve, kind=1)
    new_curve = f(depth0) + np.random.normal(0, 0.0001, curve.shape)
    new_depth = np.interp(depth0, depth1, depth0, ...)
    return new_curve, new_depth
```

- **anchor 60% を 2 セットランダム抽出**。1 セット目を「現在の」depth、2 セット目を「歪めた」depth として interp で対応付け → curve 本体を 1D 線形補間で新 depth grid に再サンプル
- `cell 32`: 1 well あたり `num` 回繰り返して `9 * num` wells に拡張。num は cell 21 namespace に居る (典型 5-10)
- ROGII 適用: visible 区間の TVT_input/GR を non-uniform に伸縮 → augmentation。dip 角度違いの well を疑似生成できる ★★★

### 1.5 post-process — DTW (cell 88-100)

予測した補正後値 `*_pred` (= mWDN 出力) は GR ↔ 各 channel の対応をまだ完全には合わせきっていない。最終 alignment は古典 DTW で行う:

- **library**: `dtwalign` (`cell 93 import`、Sakoe-Chiba window=180, step_pattern=`symmetricP05`)
- **smooth**: 入力 GR / pred curve は `tsaug.Convolve(window='triang', size=7)` で三角窓 7 ft 平滑化してから DTW 適用
- **DDTW** option: `dtwalign_ddtw` で `__derivation` (3-point central diff) を取ってから DTW (cell 92)
- **path 抽出 2 方向**: `path_query2ref` / `path_ref2query` を `get_warping_path` (cell 91) で float index → int 化
- **depth shift 復元**: `data['DEPT'].values[path_query2ref]` で各 query 位置に対応する reference depth を取得 → 重複 index は drop & `pchip` で補間
- ROGII 適用: hidden TVT 予測後、**typewell GR との DTW で TVT を再 anchor** する post-proc に転用可 ★★

### 1.6 訓練の細部

- **EarlyStopping** patience=5 (cell 81)
- **optimizer**: Adam (cell 75)
- **train/val split**: 9 wells を 70%/30% でシンプルに分割 (cell 35)、CV はしていない (= 9 wells しかなく fold 分割が不安定なため)
- **sliding window 推論 + median 集約** (cell 123): 重複 idx を `groupby(idx).median()` で集約。ROGII の typewell side-by-side prediction に直接転用可

### 1.7 ROGII に転用すべき要素 ★ / 転用しない要素

| 要素 | 評価 | 理由 |
|---|---|---|
| **mWDN (multi-scale wavelet decomposition layer)** | ★★ | ROGII の GR は high-freq / mid-freq / low-freq の 3 層構造をもつ (`problem-essence.dense.md §8.3`)。wavelet で明示的に分けるのは Self-NCC multi-scale (R: 17/31/51 ft window) と同じ着想だが学習可能版 |
| **GRU encoder per scale** | ★ | PatchTST と比べて表現力低だが parameter 軽量。Phase 4 の Transformer よりまず先に試す価値あり |
| **wavelet filter regularization (`L_H_loss`)** | ★ | 学習可能 wavelet filter が「単なる任意 linear layer」になるのを防ぐ。ROGII で wavelet decomposition を試すなら α=β=0.5 で coefficient 0.01 をそのまま流用可 |
| **`random_stretch` augmentation** | ★★★ | **ROGII で確実に効く**。bit dip 角・lateral 軌道の variation を擬似的に作れる。anchor 60% 設定もそのまま流用可 |
| **DTW post-proc (sakoechiba w=180, symmetricP05)** | ★★ | typewell GR との最終 alignment。`tsaug.Convolve` 平滑化 + `dtwalign` の組合せがすぐ使える |
| **DDTW (derivative DTW)** | ★ | GR の 1 次微分で alignment。HF noise に robust |
| **NMSE loss + filter reg 0.01** | × | ROGII は RMSE 評価。NMSE は variance で割るので well 間スケール差を吸収するが、ROGII では既に well 内 z-score normalize で同等の効果あり |
| **EarlyStopping patience=5** | ★ | そのまま流用可 |
| **9 wells 70/30 split (no fold)** | × | ROGII は 773 wells で GroupKFold (R/N kernel 通り) が前提 |
| **sliding window + groupby median** | ★ | ROGII で patch-based 推論する場合に必須 |

---

## 2. Kaggle Ventilator Pressure Prediction — 1 位 (= ROGII 構造一致度 **高**)

### 2.1 問題設定 (= ROGII との対応関係)

- **入力**: 80 step の `u_in` (input pressure) sequence + R, C (one-hot)
- **出力**: 80 step の `pressure` (target sequence)
- **評価**: MAE
- **ROGII との対応**:
  - input sequence (80 step) → ROGII の visible+hidden 全 sequence (1ft × ~6500 ft)
  - target sequence regression → ROGII の hidden TVT prediction
  - PID 物理 post-proc → ROGII の `tvt_formula = -Z + ANCC + b_well` を loss / post-proc に組み込む発想に対応
- **構造一致度が "高" な理由**: pure sequence regression であり target と input の関係が物理 (PID controller / `TVT = -Z + ANCC + b_well`) で記述できる点が共通

### 2.2 アーキテクチャ詳細 (Shujun He 担当部分)

GitHub: https://github.com/Shujun-He/Google-Brain-Ventilator

- **LSTM + 1D CNN + Transformer** の hybrid
  - LSTM: 過去依存の強い target (pressure) を直接モデル化
  - 1D CNN: 局所パターン抽出
  - Transformer: 長距離依存
- **`ResidualLSTM`**: 標準 LSTM に residual connection を加えた custom module。grad flow 改善
- **input features**: `u_in`, lag features (`u_in[t-k]`), diff features (`u_in[t]-u_in[t-1]`), `area_true = cumsum(u_in * dt)`, R/C one-hot + R×C cross
- **学習**: Ranger optimizer, ReduceLROnPlateau scheduler

### 2.3 loss 設計と multi-task

write-up (Medium / Kaggle discussion 285256):

- **主 loss**: MAE on pressure (= 評価指標と一致)
- **multi-task の証拠**: `target+diff+cumsum` の 3 head 同時予測 (Vandewiele write-up)
  - target = pressure
  - diff = pressure[t] - pressure[t-1]
  - cumsum = Σ pressure
- **3 head の loss は単純加算 with 同重み** (write-up 内記述)、または fold-OOF で重み grid search

### 2.4 ensemble = **weighted MEDIAN** (1 位の決定打)

Vandewiele write-up より:

> "We typically see weighted MEDIAN performs better than mean when MAE is the metric"

- **15 seed × 5 fold = 75 model** の予測を seed/fold 毎に保存
- 各 seed の OOF MAE で **重み = 1 / OOF_MAE^β** (β は経験値)
- 各 t 毎に 75 個の予測値を **重み付き median** で集約 (= np.median ではなく `weighted_quantile(preds, weights, q=0.5)`)
- mean ではなく median の理由: outlier model 1 個が ensemble を引きずる効果を排除

### 2.5 PID matching post-process — Yves / Vandewiele の発見

これが **2650 チーム中 2 チームしか出来なかった** 決定打:

- 物理: ventilator は P/I/D parameter (`kp, ki, kt`) に従って `u_in` を決める
- 観察: target `pressure` は `u_in[t] = kp*(kt - P[t]) + ki * I[t]` を満たすはず
- 整数分解: 全可能な (kp, ki, kt) 950×950 組合せから OOF 内で最も整合する組を探索
- **66% の test rows で完全に一致する PID parameter set を発見** → 当該 rows は予測値を PID 解で **直接置換** (= 0 error)
- **線形外挿 trick**: 950^2 を 950 倍高速化、`x_intersect = (u_in[t] - u_in_hat2) / slope` で integer 解検出

### 2.6 ROGII に転用すべき要素 ★ / 転用しない要素

| 要素 | 評価 | 理由 |
|---|---|---|
| **LSTM+CNN+Transformer hybrid** | ★★ | Phase 4 の Sequence Transformer 案 B に直接該当。ResidualLSTM だけ採用しても OK |
| **multi-task head (target + diff + cumsum)** | ★★★ | ROGII で **TVT + ΔTVT + cumΔTVT** の 3 head は直接転用可。`problem-essence.dense.md §12.1` の "Multi-task DL" と整合 |
| **weighted MEDIAN ensemble** | ★★★ | ROGII (RMSE 評価) でも seed 多様性を活かせる。R/N の Ridge stack に **追加で** ensemble 候補。注意: RMSE では median は厳密に最適ではない (mean が最適) ので **MEDIAN を取る対象は base model 出力** で、最終 stacking は Ridge mean のままでも OK |
| **Ranger optimizer + ReduceLROnPlateau** | ★ | DL 訓練時のデフォルトとして採用可 |
| **PID 物理 post-proc** | ★★★ | **ROGII で直接対応する物理は `TVT = -Z + ANCC + b_well`** (`top3-distill.dense.md §3-5`)。R/N kernel は既に b_well を 3 種計算しているが、**predicted TVT に対して formula 残差を最小化する dual-optimization** をかけるアイデアは未着手。Phase 5 post-proc 候補 |
| **15 seed × 5 fold = 75 model** | △ | ROGII 推論 9hr 制約下で 75 model は厳しい。15 seed → **5 seed × 5 fold = 25 model** が現実的妥協点 |
| **R, C one-hot** | × | ROGII は連続値特徴のみで categorical 入力なし |
| **`area_true = cumsum(u_in * dt)`** | ★ | ROGII で `cumsum(GR_horizontal * dMD)` / `cumsum(dZ)` の cumulative feature が同等。既存 R kernel の rolling sum とは別アングル |

### 2.7 ChatGPT-3 / 4 系で前評価不能だった隠し技

write-up を直接読んで初めて分かる細部:

- **only 2 teams out of 2650 found PID exploit**: Vandewiele は 2026-10-26 という競技終了 4 日前にこの match を発見。ROGII でも `tvt_formula` の **完全分解** に近い物理 trick が眠っていないか、Phase 5 で deep dive する価値あり
- **Riccardo の発見**: **ReduceLROnPlateau のほうが fast annealing scheduler より良い** (= overfit 抑制)。ROGII Transformer で OneCycleLR / cosine annealing の代わりに plateau-based を試す価値

---

## 3. FORCE 2020 Lithology Prediction — 1 位 Olawale Ibrahim (= ROGII 構造一致度 **中**)

### 3.1 問題設定 (= ROGII との対応関係)

- **入力**: well log channels (CALI, RDEP, RHOB, DHRO, SGR, GR, RMED, RMIC, NPHI, SP, GROUP/FORMATION/WELL label-encoded)
- **出力**: 12-class lithology label per row
- **評価**: penalty matrix を使った custom score (一部誤り pair に重い penalty)
- **ROGII との対応**:
  - lithology classification → ROGII の **Geology aux head** (train 限定 supervision)
  - GR / RHOB / NPHI features → ROGII で input にあるが TVT 軸ではなく MD 軸
  - 78 train wells → ROGII の typewell ~700 well 群 (オーダー一致)
- **構造一致度が "中" な理由**: ROGII の主タスクは regression、FORCE は classification。だが **aux head の設計** には直接参考になる

### 3.2 アーキテクチャ詳細

**single 10-fold stratified XGBoost** (= ensemble なし、neural net なし) で 1 位 (final test -0.4690)。

- `n_estimators` ≈ 数千 (lr/n_est は明示せず tuned by hand)
- `max_depth = 10`
- `lambda (L2 reg) = 1500` (= **異常に強い L2 regularization**)
- `colsample_bytree = 0.9`
- `subsample = 0.9`
- `StratifiedKFold(n=10, shuffle=True)`
- single algorithm (= LGB / CB / NN ensemble せず)
- DTS / ROPA / RXO / SGR は欠損率高すぎて drop

### 3.3 loss 設計

**custom penalty matrix を loss に組み込みたかったが時間切れ** で標準 multiclass log-loss を使用 (= 評価 metric と訓練 loss が一致しない)。著者本人が "Final Challenges" と記述。

- 後発の参加者 (top10) は **scipy.optimize.minimize で penalty matrix-aware threshold** を探索 → 標準 softmax 出力に対して threshold 移動で penalty を間接的に最小化

### 3.4 Bestagini polynomial features

Paolo Bestagini, Vincenzo Lipari, Stefano Tubaro (2017) "A machine learning approach to facies classification using well logs" SEG Technical Program Expanded Abstracts 2017 (https://library.seg.org/doi/10.1190/segam2017-17729805.1) で提案された augmentation:

- 各 row の同一 channel の **隣接 row の値** (上 N row + 下 N row) を feature 追加 → 局所文脈を window 化
- **隣接行の差分・比** も追加
- **degree 2 多項式** 展開 (`feature_i * feature_j` 全ペア)
- 結果として元 9 channel が ~50-100 channel に膨張、勾配ブースティングがその中から重要なものを選ぶ

Olawale も "Bestagini augmentation を適用" と明記。

### 3.5 data augmentation

- 各 well 内で **GR / RHOB / NPHI に小さい noise** (σ=0.001 オーダー) を加えて well 数を拡張
- **mixed-well augmentation** (= 異なる 2 well の接合) は **やっていない**

### 3.6 ROGII に転用すべき要素 ★ / 転用しない要素

| 要素 | 評価 | 理由 |
|---|---|---|
| **single XGBoost 10-fold stratified** | × | ROGII では既に R/N kernel が LGB×3+CB+TabICL の 8 base stack で 10 帯。single base への戻しは LB 後退 |
| **`max_depth=10, lambda=1500` の超強 L2** | ★ | ROGII の LGB hyperparameter として一度試す価値。`lambda_l2=1500` は異常値だが、772 wells の小データでは regularization 効果あり |
| **Bestagini polynomial features** | ★★ | ROGII でも有効。R/N kernel に **明示的にはなく**、これを feature engineering で追加すれば LB 改善余地。`top3-distill.dense.md §7 移植 map` の現行 14 要素に **15 番目** として追加候補 |
| **`StratifiedKFold` (= label 分布で stratify)** | △ | ROGII は GroupKFold by well_id が前提 (well 内 leak 防止)。stratify は適用不可。ただし well 群を **layer median 値で stratify** する変種は可 |
| **penalty matrix-aware loss** | × | ROGII は単純 RMSE で penalty 不要 |
| **GROUP/FORMATION/WELL label-encoded** | △ | ROGII で test の Geology が NaN なので test 推論時に使えない。train 時のみ aux feature |
| **DTS/ROPA/RXO drop (高欠損)** | ★ | ROGII でも ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA は test で NaN → drop 必須 (既に `data-spec.dense.md §1` に記述あり) |

### 3.7 補足: open test 24th が final test 1st に逆転した理由

Olawale の Medium 記事より:

- public LB は test の一部 (open test) のみで評価 → public LB 24 位
- final LB は full test で評価 → 1 位
- **理由**: 強 L2 regularization で **public LB に overfit せず汎化** していた
- ROGII への教訓: CV-LB diff > 0.5 で overfit 警戒 (`winning-strategy.dense.md` リスク 4) と整合。**強 regularization が public LB 後退と引き換えに private LB 上位を取る** パターンを意識する

---

## 4. 追加類似コンペ (Track D-2 結果)

### 4.1 SPWLA 2023 — 2 位 WELL-LOGGED / 3 位 ARC_CNLC

公式 README より (https://github.com/pddasig/Machine-Learning-Competition-2023):

| 順位 | チーム | スコア | 手法 |
|---|---|---|---|
| 1 | Dreamstar | 550 | mWDN+GRU+DTW (= §1) |
| 2 | WELL-LOGGED | 551 | Random Forest + DTW |
| 3 | ARC_CNLC | 552 | Pick-Pair-Based DTW |

注: ファイル容量の制約で full ipynb 解析できず。**順位差 1-2 ポイント** という超接戦から見て、3 件全てで **DTW post-process** が共通要素。Random Forest 単体でも 2 位という事実は、Dreamstar の DL モデルの優位性は **わずか 1 ポイント** にすぎないことを示唆 → ROGII でも DL モデル (案 B) に過度に投資せず、案 A (GBM) + DTW post-proc で底固めする戦略の妥当性を裏付け。

### 4.2 SPWLA 2020 — Sonic Log Synthesis

問題: Well #1 の従来 log → Well #2 の合成 DTC/DTS log を予測 (= cross-well log generation)

- benchmark Random Forest RMSE = 17.93
- top5 平均で benchmark 比 -27%
- top5 手法: NN (LSTM 含む) と GBM の hybrid stack
- 公開 solutions: `Solutions/MLogging Team submission.ipynb`, `Solutions/Oilers_solution.ipynb`, `Solutions/UTFE_Code .ipynb`

ROGII 転用: **cross-well prediction** という構造は ROGII の「typewell → horizontal の TVT を当てる」と直接対応。ただし ROGII の typewell は **同じ well の縦穴** (= 自己整合) なのに対し、SPWLA 2020 は **別 well** で非自明性が高い。SPWLA 2020 の手法ライブラリは ROGII で **lower bound** として参考になる程度。

### 4.3 IceCube Neutrinos in Deep Ice (Kaggle 2023, 賞金 $50K)

問題: 神経 detector の hit timing/position sequence → 入射 neutrino の direction vector を回帰
- 上位 3 解全てが **Transformer** (top-3 解析論文 https://arxiv.org/abs/2310.15674)
- 1 位: GraphNeT GNN + Transformer
- 2 位: Transformer + DynEdge (graph)
- 3 位: pure Transformer with masking

ROGII 転用: **graph 構造** (= 検出器配置) を持つ Transformer が勝ったという事実は、ROGII の **6 formation 平面構造を graph 化** する案 (Track D-3 GNN 系) の正当性を補強。

### 4.4 LANL Earthquake Prediction (Kaggle 2019)

問題: 連続音響信号 → 地震までの残り時間を回帰 (sequence regression)
- 1 位 Philipp Singer: **LightGBM + NN blend** (LGB 主体、NN は補助)
- 1 位の決め手: **特徴量設計** (= 1D CNN 特徴 + 統計特徴の組合せ)
- LSTM/GRU は試したが効かなかった

ROGII 転用: 「**複雑な DL より統計特徴 + GBM** が勝つ」事例。ROGII でも案 B Transformer が苦戦したら案 A 全力を選択する根拠の 1 つ。

### 4.5 OpenVaccine COVID-19 mRNA Degradation (Kaggle 2020)

問題: 107 nucleotide sequence + structure graph → 各位置での degradation rate を回帰 (= masked sequence regression そのもの)
- 上位解: **GRU+LSTM stack** または **1D CNN + Transformer**
- Vandewiele 4 位 (https://github.com/GillesVandewiele/covid19-mrna-degradation-prediction)
- multi-task: 5 reactivity labels を同時予測 (= ROGII の TVT + ΔTVT + Geology 構造に相当)

ROGII 転用: **multi-task head 設計** の参考に最適 ★★。

### 4.6 Indoor Location Navigation (Kaggle 2021)

問題: WiFi RSSI sequence + IMU sensor → 屋内 (x, y, floor) 位置を回帰
- 1 位 Dmitry Gordeev: **KNN + GBM (location prediction)** + **CNN+RNN (trajectory reconstruction)** + **optimization (combine)**
- multi-block 構造 (= 複数手法を物理的事前で結合)

ROGII 転用: ROGII 案 C (Geometry-Physics Hybrid) の構造原理と完全一致。**Plane-fit (= geometry) + DL residual + DTW (= optimization combine)** のパターン。

### 4.7 Jane Street Market Prediction (Kaggle 2021)

問題: 130-feature 行列 → trade or skip を予測
- 上位解: **Autoencoder + MLP** (= dimensionality reduction → predictor)
- 連続値 features の **MLP regression** で十分という事例

ROGII 転用: Phase 4 で simple MLP baseline を 1 つ持っておく動機 (= 過剰な architecture engineering を避ける safety net)。

### 4.8 Geosteering Robot (SPE 2025, Alyaev et al.)

論文: https://onepetro.org/SJ/article/30/03/995/631101/Geosteering-Robot-Powered-by-Multiple

問題: ROGII Geosteering World Cup simulator で human expert と benchmark
- 手法: **Particle Filter (= state assimilation) + Reinforcement Learning (= look-ahead decision)**
- 結果: 80% reservoir contact (薄層・断層付き synthetic) で expert level に到達
- **コンペとの直接関係**: ROGII (= Texas-based) の **同じ simulator family** のデータ。Alyaev チームの PF approach は R/N kernel の PF と同じ着想だが、**RL look-ahead** は未着手領域

ROGII 転用 ★★: PF はすでに kernel に居るが、**RL-based look-ahead post-proc** (= 各 step で next 数 step の予測整合性を最大化する action を選ぶ) は新規領域。Phase 5 post-proc 候補。

### 4.9 (ROGII 直接関連 文献) "Deep Hierarchical Graph Correlation" (Wei et al. 2025, MDPI Mar Sci Eng 14:66)

論文: https://www.mdpi.com/2077-1312/14/1/66

- **2 段階 well-log alignment**: Stage 1 = CNN による粗 alignment, Stage 2 = DTW による fine alignment + graph で multi-well 整合性
- ROGII への教訓: **CNN + DTW の 2 段階** は SPWLA 2023 1 位 Dreamstar と同型
- 新規性: **graph correlation** で複数 well の整合を同時最適化 → ROGII の typewell 群を graph node 化する案

### 4.10 ASHFormer (IEEE Xplore 2023, https://ieeexplore.ieee.org/document/10187164)

論文タイトル: "ASHFormer: Axial and Sliding Window-Based Attention With High-Resolution Transformer for Automatic Stratigraphic Correlation"

- 3 種 attention: sliding-window / horizontal-axis / vertical-axis
- HRNet (= multi-scale feature fusion) backbone
- 用途: 複数 well の stratigraphic correlation

ROGII 転用: PatchTST より **明示的に multi-scale** で、ASHFormer の axial attention は (TVT 軸 / MD 軸) の 2 次元 attention に対応 → ROGII の 2D 化 (TVT × MD plane で attention) の根拠。

### 4.11 WLFM (arXiv 2509.18152, 2025)

論文: "WLFM: A Well-Logs Foundation Model for Multi-Task and Cross-Well Geological Interpretation"

- **1200 wells で pretrain** (= ROGII の 773 well より 1.5 倍規模)
- Tokenizer (= log patch を geological token 化) + masked-token modeling + stratigraphy-aware contrastive learning
- ROGII 転用 ★★: **このモデル weight を private dataset としてアップロード → fine-tune** で 1 位射程の 1 つの edge。ただし license / weights public availability を要確認

---

## 5. 共通する勝ち筋 (= ROGII でも効く高確度パターン)

過去 9 件の winner solutions / 2 件の最新文献から抽出:

### 5.1 [パターン 1] **DTW 系 post-process は外すと負ける** (出典: SPWLA 2023 全 top3, 文献 4.9, 4.10)

- 1 位 (Dreamstar) の DTW = `dtwalign` sakoechiba w=180, symmetricP05
- 2 位 (WELL-LOGGED) も DTW (Random Forest にすら追加で勝てた)
- 3 位 (ARC_CNLC) も Pick-Pair-Based DTW
- **ROGII 含意**: typewell GR ↔ horizontal GR の最終 alignment は **古典 DTW 必須**。R/N kernel は Beam Search/PF で代替しているが、これに **追加で** DTW post-proc を載せると LB 0.1-0.3 ft 改善余地

### 5.2 [パターン 2] **multi-task head が representation を richer にする** (出典: Ventilator, OpenVaccine, FORCE 2020 後発組)

- Ventilator: target + diff + cumsum
- OpenVaccine: 5 reactivity targets 同時
- ROGII で対応: **TVT (主) + ΔTVT (lag-1) + ΔTVT² (lag-1 squared) + Geology (aux) + 6 formation top (aux)** の 5 head
- 実装難度 中 (= shared encoder + 5 prediction head)

### 5.3 [パターン 3] **weighted MEDIAN ensemble (mean ではない)** (出典: Ventilator 1 位)

- seed 多様性を 15 seed ×5 fold = 75 model
- mean だと 1 model の bias が引きずる
- median だと robust
- ROGII での補強: R/N の Ridge stack は mean (= positive coef Ridge は本質的に重み付き平均) → ここに **median branch** を並列で計算し、両者の OOF を比較して良い方を採用

### 5.4 [パターン 4] **物理事前を post-proc で組み込む** (出典: Ventilator PID matching, Geosteering Robot RL+PF, ROGII tvt_formula)

- Ventilator: PID equation 整合 → 66% rows で予測直接置換
- Geosteering Robot: PF likelihood で multi hypothesis 維持 → RL で最適 action
- ROGII: `tvt_formula = -Z + ANCC + b_well` の **double-check** に使える。predicted TVT に対して `predicted_TVT + Z - imputed_ANCC - b_well = 0` の constraint を **post-proc で再 fit** する変種が未着手

### 5.5 [パターン 5] **強い L2 regularization で private LB 逆転** (出典: FORCE 2020 1 位)

- public 24 位 → private 1 位
- `lambda_l2=1500` という異常値
- ROGII 含意: **public LB に過度に最適化せず**、CV best と LB best の両方を提出する戦略 (`winning-strategy.dense.md` Phase 6) と整合

### 5.6 [パターン 6] **augmentation で疑似 well を増やす** (出典: SPWLA 2023 1 位 random_stretch, FORCE 2020 noise injection)

- ROGII の visible 区間 TVT_input/GR を anchor 60% で非線形伸縮 → bit dip 角違いの疑似 well 生成
- 1 well あたり 5-10 倍に拡張可能
- 案 B Transformer の 200-773 wells 不足問題を補完

### 5.7 [パターン 7] **multi-scale (wavelet / patch / window) で時間スケール依存を分離** (出典: Dreamstar mWDN, R kernel multi-scale Self-NCC, TimeMixer, ASHFormer)

- ROGII の R kernel は Self-NCC を 17/31/51 ft の 3 window で実装済み
- これに加えて **学習可能 wavelet decomposition** (mWDN) または **patch-based PatchTST** で multi-scale 化を model 内部に焼き込む方向

### 5.8 [パターン 8] **single algorithm 深掘り vs ensemble 多様化のトレードオフ**

| 戦略 | 例 | LB 改善 | リスク |
|---|---|---|---|
| single 深掘り | FORCE 2020 1 位 (XGBoost only) | -0.1 -- -0.3 | overfit 警戒 |
| ensemble 多様化 | Ventilator 1 位 (15×5=75), R/N kernel 8 base stack | -0.3 -- -0.6 | runtime 圧迫 |
| **両立** | LANL Earthquake 1 位 (LGB+NN blend) | 中庸 | 工数倍 |

ROGII 残 86 日で全方位は不可。**Phase 5 で必ずどちらかを選ぶ** 必要 (case study 参照)。

---

## 6. 戦略 doc 案 A/B/C との突合

### 6.1 案 A (GBM Stack) との突合

| 観点 | 戦略 doc 案 A 現行 | 過去 winner からの示唆 |
|---|---|---|
| base model 数 | LGB / XGB / CB の 3 model | Ventilator は 75 model、R/N は 8 base。**3 base は不足**、せめて LGB×3 (lr 多様化) + CB ×1 + XGB ×1 = 5 base 必要 |
| feature engineering | rolling stats / derivatives / Plane-fit residual / DTW dist / Bestagini | **Bestagini は明示されているが** R/N kernel に居なかった ★。**xcorr_tvt (tasmim) も明示** |
| CV | GroupKFold by well_id 10-fold | FORCE 2020 1 位は 10-fold stratified、Ventilator は 5 fold + 15 seed。**fold 数より seed 数を増やす** ほうが ROGII では効率高 |
| pseudo-label | 1 round | Ventilator では使われていない、FORCE 2020 1 位も使わず。**ROGII では使うべきか open question** |

### 6.2 案 B (Sequence Transformer) との突合

| 観点 | 戦略 doc 案 B 現行 | 過去 winner からの示唆 |
|---|---|---|
| backbone | PatchTST | Dreamstar = mWDN+GRU、Ventilator 1 位 = LSTM+CNN+Transformer hybrid、IceCube 1 位 = GraphNeT+Transformer。**PatchTST 単独は未検証** ★ Dreamstar の mWDN-GRU は **PatchTST より軽量** で baseline として先に試す価値あり |
| cross-attention | typewell key/val + horizontal query | Dreamstar は GR を input に直接混ぜる (cross-attention 不使用)。**ROGII では cross-attention 採用が独自性 ★** |
| multi-task | TVT (主) + ΔTVT + Geology | Ventilator = target+diff+cumsum、OpenVaccine = 5 head。**ROGII で 5 head に拡張 ★** |
| ensemble | 15 seed × 5 fold = 75 model | Ventilator と同型。runtime 9 hr 制約で **15→5 seed に縮小必要**、後で実測で再判断 |
| pretrain | FORCE 2020 / VOLVE で masked TVT pretext | WLFM (1200 wells pretrain) や Wav2Vec 2.0 自己回帰 pretrain と整合。**masked TVT 15% pretext は標準的選択** |

### 6.3 案 C (Geometry-Physics Hybrid) との突合

| 観点 | 戦略 doc 案 C 現行 | 過去 winner からの示唆 |
|---|---|---|
| Step 1 plane-fit | visible TVT_input から最小二乗 | R/N kernel の `FormationPlaneKNN` が既にこれを上位互換で実装 ★ (top3-distill §3) |
| Step 2 DTW alignment | typewell-horizontal GR | Dreamstar の `dtwalign sakoechiba w=180 symmetricP05` を直接借用 ★★ |
| Step 3 CUSUM fault | 統計的 fault detection | 過去 winner で CUSUM 採用例なし、**ROGII 独自要素** |
| Step 4 1D U-Net residual | residual を NN で学習 | Indoor Location 1 位 (CNN+RNN trajectory reconstruction) に類似 |
| pretrain | FORCE 2020 で random window mask | 案 B と共有可 |

### 6.4 矛盾点・補完点

- **矛盾点 1**: 案 A は `kagglib.baselines.run_all()` で LGB/XGB/CB 3 base に固定するが、**過去 winner では 5+ base が標準**。`top3-distill.dense.md §6.1` の R/N の 8 base 構成と整合させるべき
- **矛盾点 2**: 案 B の "PatchTST + cross-attention" は **動作確認なし**。Dreamstar の mWDN-GRU の方が **学習可能 wavelet** で multi-scale が組み込まれており、開発コストも低い (= GRU のみ)。先に mWDN-GRU を試して baseline、その上で cross-attention 化を後付けする dev path も有力
- **矛盾点 3**: 戦略 doc は **DTW post-proc を案 C 内部にしか入れていない** が、Dreamstar / 文献 4.9 / 4.10 から見て DTW は **全案共通の post-proc** にすべき
- **補完点 1**: `random_stretch` augmentation (Dreamstar) が戦略 doc に未記載。案 B/C の data aug に追加 ★★★
- **補完点 2**: weighted MEDIAN ensemble (Ventilator) が戦略 doc に書かれているが、**具体的重み式** (= `1/OOF_MAE^β`) と適用範囲 (= base output level) が未明記
- **補完点 3**: PID-style physics post-proc (Ventilator) は ROGII の `tvt_formula` 直接適用に対応 → **Phase 5 post-proc に独立項目として追加**

---

## 付録 A. ROGII 構造に最も近い winner (= 1 件選ぶなら)

**SPWLA 2023 Dreamstar** (`SPWLA_2023_dreamstar_lite.ipynb`)

理由 1 行:
- GR を reference signal として「他 channel の depth shift / 値補正」を予測する pure sequence-to-sequence + DTW post-proc という構造が、ROGII の typewell GR を reference に horizontal の TVT を当てる構造と **入出力の役割まで含めて 1 対 1 対応**。さらに `random_stretch` augmentation と DTW alignment は **コード snippet レベルでそのまま borrow 可能**。
