# TimeXer Implementation Spec for ROGII (2026-05-11)

> 担当 AE: subagent (Auto-mode), branch `docs/timexer-spec-2026-05-11` (派生元 `feat/phase-5-edge-r-online`)
> 残コンペ: **86 日** / 賞金 $50K / **5/13 着手** prerequisites をここで凍結する
> 文脈: subagent W (= `docs/research/academic-literature-deeper.dense.md` §2.4, branch `docs/literature-deeper-2026-05-11`) + subagent Z (= `docs/research/3-hack-implementation-spec.dense.md` §2, branch `docs/3hack-spec-2026-05-11`) で確証された **TimeXer = 3 hack 中最大 contribution (-0.5 ft、 12 benchmark SOTA)** を offline pretrain + inference-only kernel の二段構成に refine する
> 中立指示原則: 推奨は出さない (= `~/.claude/CLAUDE.md` 主道フレームワーク § 中立指示原則)。 各章ごとに **選択軸 + トレードオフ + 失敗モード** を提示し、 go/no-go は開発者に委ねる
> 出典 URL は §末尾 + 本文中で明示、 `file:line` 参照は `feat/phase-5-edge-r-online` HEAD = 1bb1523 基準

---

## 0. 序: subagent W §2.4 + subagent Z §2 推奨と 9 hr 制約への対応

### 0.1 採択論理 (= TimeXer を 5/13 prerequisite に置く根拠)

subagent W §2.4 (= `academic-literature-deeper.dense.md`) で確認された TimeXer の構造的優位:
- endogenous = horizontal well の (TVT_input, GR_h, MD) sequence
- exogenous variates = (Z, ANCC_imputed, X, Y) + typewell の (TVT_tw, GR_tw, Geology_tw)
- **物理事前** `TVT = -Z + ANCC + b_well` を **cross-attention で陽に encode** (= subagent W §2.4 「ROGII の構造に最も自然な fit の 1 つ」)
- 12 real-world benchmark で SOTA (= NeurIPS 2024 Wang et al., arXiv:2402.19072)
- **公式 GitHub**: `thuml/TimeXer` (= 478 stars, 251 行/`models/TimeXer.py`, push 2024-11-27)

subagent Z §0.2 (= `3-hack-implementation-spec.dense.md`) の選択軸表より、 TimeXer は **「9 切り upper bound」 軸で最大** (= -0.2〜-0.5 ft、 12 benchmark SOTA 実績)。 ただし subagent Z §2.5 #1 で「**train 5.3 hr + Edge R 1.7 hr + 既存 pipeline 2 hr = 約 9 hr ぎりぎり**」 と判明、 単純 kernel 内 train は **9 hr 制約に対し fail-prone**。

→ **offline pretrain + inference-only kernel** の二段構成に切替が必須:
1. local / Colab Pro で **5-10 hr 学習** (= GPU 必須、 Kaggle 外で実施)
2. checkpoint (= ~100-300 MB) を Kaggle private dataset として upload
3. inference-only kernel で checkpoint load + predict (= 推論 ~15 sec、 9 hr cap 余裕大)

### 0.2 5/13 着手 prerequisite として固定する不変要素

以下 4 件を本 spec で凍結し、 中央 (= main agent) が 5/13 朝 dispatch する直前まで動かさない:

| 不変要素 | 凍結内容 | 動かさない理由 |
|---|---|---|
| GitHub source | `thuml/TimeXer` @ commit hash main (= 2024-11-27 push) | 上流 update で API 変更すると adapter 再書込み発生 |
| 適用 base kernel | `kaggle_kernels/exp005_cache_blend/exp005_cache_blend.py` (= 2320 行、 5-base + Ridge stacking + path b blend) | exp009 系より tested、 70 sec runtime baseline で TimeXer 追加余地大 |
| inference path 名 | **path c** (= path a karnakbaev + path b 自前 base と並列、 Ridge meta の external blend layer) | exp008 v3 / exp009 v2 の path b blend と直交、 衝突なし |
| Kaggle dataset id | `ky7240/rogii-timexer-ckpt` (= 20 char 以内、 衝突回避 short slug) | mamba runtime upload (= `ky7240/rogii-mamba-runtime`) と命名規律一致 |

### 0.3 9 切り (= LB <9.5) への定量分解 (= `kaggle/CLAUDE.md` §11.2 数理本質 criterion)

現状: exp005 v2 LB 10.203 ft (= `docs/dev/submission-postmortems.dense.md` 経由、 Edge S +0.114 ft 改善後)。 TimeXer path c blend 単独寄与は以下の幅で推定:

| 期待 component | LB 寄与 (ft) | 累積 LB |
|---|---|---|
| 現在 (exp005 v2) | baseline | 10.203 |
| **+ TimeXer path c (本 spec)** | **-0.2 〜 -0.5** | **9.70 - 10.00** |
| + exp008 case D Kalman + Huber + hetero MEDIAN (= subagent T v3) | -0.4 〜 -0.6 | 9.10 - 9.60 |
| + exp010 stratified Edge Q fold + adversarial drop | -0.05 〜 -0.10 | 9.05 - 9.55 |

**9 切り確率**: TimeXer 単独で累積 LB 9.10-9.60 帯 → **9 切り (= LB <9.5) 確率は中央値で 50-60%** (= subagent X §1.4.1 と整合)。 ただし TimeXer 自体の寄与下限 (= -0.2 ft) が真値だった場合は 9 切りに届かない、 これは subagent Z §2 で「12 benchmark SOTA」 が ROGII の hidden region imputation に必ずしも transfer しない数理 risk (= benchmark は forecasting、 ROGII は **bidirectional imputation**) を反映している (= §1.2 で詳述)。

---

## 1. TimeXer architecture (= endogenous + exogenous cross-attention)

### 1.1 model spec (= 公式実装の数式 + ROGII 適用)

**Wang 2024 NeurIPS arXiv:2402.19072 (= `https://arxiv.org/abs/2402.19072`) 構造**:

1. **endogenous patch embedding**: target $\mathbf{x}_{\text{en}} \in \mathbb{R}^{T_{\text{en}}}$ を **非重複 patch** $\{\mathbf{x}_i\}_{i=1}^P$ ($\mathbf{x}_i \in \mathbb{R}^{L_p}$, $P = T_{\text{en}} / L_p$) に分割、 各 patch を線形射影 + 位置 embedding で $\mathbf{e}_i = W_{\text{patch}} \mathbf{x}_i + \mathbf{p}_i \in \mathbb{R}^D$
2. **global endogenous token**: 学習可能 token $\mathbf{g}_{\text{en}} \in \mathbb{R}^D$ を patch tokens の先頭に concat (= $\mathbf{H} = [\mathbf{g}_{\text{en}}; \mathbf{e}_1; \ldots; \mathbf{e}_P]$, $\mathbf{H} \in \mathbb{R}^{(P+1) \times D}$)
3. **exogenous variate embedding (= `DataEmbedding_inverted`)**: 各 exogenous variate $\mathbf{y}^{(k)} \in \mathbb{R}^{T_{\text{ex}}}$ を全長 1 token に圧縮 ($\mathbf{v}_k = W_{\text{var}} \mathbf{y}^{(k)} \in \mathbb{R}^D$)、 $K$ 個 exogenous variates で $\mathbf{V} \in \mathbb{R}^{K \times D}$ (= patch せず、 1 variate = 1 token、 iTransformer 流)
4. **encoder block** (= $L$ 層、 各層で):
   - self-attention (= endogenous patches 間): $\mathbf{H}' = \text{SelfAttn}(\mathbf{H})$
   - cross-attention (= global token query → exogenous): $\mathbf{g}'_{\text{en}} = \text{CrossAttn}(Q = \mathbf{g}_{\text{en}}, K = \mathbf{V}, V = \mathbf{V})$
   - exogenous information flow back: $\mathbf{H}[0] \leftarrow \mathbf{g}'_{\text{en}}$、 次層 self-attention で endogenous patches へ伝播
5. **prediction head** (= `FlattenHead`): 最終層 endogenous patch tokens を flatten → 線形射影で $\hat{\mathbf{x}}_{\text{future}} \in \mathbb{R}^{T_{\text{pred}}}$

**公式 PyTorch 実装の構成**:
- `models/TimeXer.py` (= 251 行、 class `Model`、 forward / forecast / forecast_multi)
- `layers/SelfAttention_Family.py` (= 12 KB、 `FullAttention`, `AttentionLayer`)
- `layers/Embed.py` (= 7 KB、 `DataEmbedding_inverted`, `PositionalEmbedding`)
- 出典: `https://github.com/thuml/TimeXer/blob/main/models/TimeXer.py`

**公式 ETTh1 script の hyperparameter** (= 出典 `scripts/forecast_exogenous/ETTh1/TimeXer.sh`, pred_len=96 case):
- `seq_len=96, label_len=48, pred_len=96, e_layers=2, factor=3, enc_in=7, dec_in=7, c_out=7, d_model=512, d_ff=512, features=MS`
- `patch_len, learning_rate, train_epochs, batch_size, dropout, use_norm, loss` は `run.py` default を継承: **`patch_len=16, lr=1e-4, train_epochs=10, batch_size=32, dropout=0.1, use_norm=1, loss=MSE`**

### 1.2 ROGII 適用 (= endogenous TVT/GR/MD vs exogenous Z/ANCC/X/Y/typewell)

ユーザー指示で endogenous = `(TVT_input, GR, MD)`、 exogenous = `(Z, ANCC, X, Y, formation_one_hot, typewell features)`、 patch_size=32 ft、 seq_len=256 と凍結。 これを公式 TimeXer の `(features=MS, enc_in, c_out)` API に mapping する:

| TimeXer 概念 | ROGII data | 備考 |
|---|---|---|
| endogenous (= target side) sequence | (TVT_input, GR_h, MD) を MD 順 256 row | TVT_input は visible 部分のみ観測、 hidden で NaN (= imputation target) |
| `features` mode | `MS` (multivariate input, single-target output) | 公式 ETTh1 と一致、 TVT 1 値予測 |
| `enc_in` | 3 (= TVT_input + GR_h + MD) | endogenous channel 数。 ただし TimeXer `EnEmbedding` は **target univariate** を想定 (= MS mode で `enc_in=7` のうち 6 ch は exogenous side を兼ねている)、 ROGII で TVT 単 ch endogenous + GR/MD は exogenous 側に回す代替案あり (= §1.3 選択軸 A) |
| `c_out` | 1 (= TVT 予測のみ) | 公式 MS mode と一致 |
| exogenous variates | (Z, ANCC_imp, X, Y, formation_one_hot 6 col, typewell TVT, typewell GR, typewell Geology one_hot) = 約 13-15 variates | 各 variate は seq_len=256 で長さ揃え、 typewell は anchor-based alignment で sample |
| `patch_len` | **32** (ユーザー固定) | 公式 default 16 から拡大、 ar1_phi=0.999 → correlation length ~1000 ft / 1 ft 解像度 → 32 ft 内に local 構造 + smoothing 効果 (= subagent W §2.1 P=32 候補と一致) |
| `seq_len` | **256** (= ユーザー固定) | per-well visible region 中央値 580 ft の半分 + hidden 一部、 256 / 32 = 8 patch で transformer 計算 cheap |
| `pred_len` | **64-128** (= hidden region 中央値 250 を超えない、 §2 でさらに detail) | rolling window 推論で長 hidden 対応 (= §4.2) |
| `d_model, d_ff, e_layers, n_heads` | 256 / 512 / 2 / 8 (= ETTh1 pred_len=192 case の `d_model=128, d_ff=128` と公式 base case `d_model=512, d_ff=512` の中央、 9 hr 制約と representation のバランス) | §2.4 GPU memory 試算で調整 |

**選択軸 A: endogenous channel の構成 (= 3 ch か 1 ch か)**

ユーザー指示は endogenous = (TVT_input, GR, MD) の 3 ch だが、 公式 TimeXer `EnEmbedding` は univariate target を patch する設計 (= `models/TimeXer.py` の `forecast` method)。 3 ch endogenous は公式 `forecast_multi` (= channel-independent) ルートを通る必要があり、 cross-attention で exogenous → endogenous の transfer が **per-channel 別** に走る。

| 軸 | 案 A1: endogenous = TVT 1 ch のみ | 案 A2: endogenous = TVT + GR + MD の 3 ch (channel-independent) |
|---|---|---|
| 公式 implementation との一致 | ◎ (= `forecast` method 直接利用) | △ (= `forecast_multi` 必要、 cross-attention 多重化) |
| TVT 予測精度 | GR/MD は exogenous 側に回ってもよい (= variate token として `DataEmbedding_inverted` で encode、 表現損失なし) | TVT と GR/MD が同 self-attention を共有、 short-range coupling は厳密だが over-fit risk |
| GPU memory | 低 (= patch token 数 8 + variate 13 = 21) | 中 (= patch token 数 8 × 3 ch = 24 + variate 11 = 35) |
| 工数 (= adapter 行数) | 小 (= 公式 API そのまま) | 中 (= channel loop + cross-attention 多重化を adapter で wrap) |

→ 推奨せず、 開発者判断。 ただし **9 hr cap + offline pretrain 工数最小化** の選択軸では案 A1 が有利。

**物理事前 `TVT = -Z + ANCC + b_well` の attention 符号化**:
- 案 A1 で TVT 1 ch endogenous、 Z + ANCC を exogenous variate に入れた場合、 cross-attention の重み行列 $W^Q, W^K \in \mathbb{R}^{D \times D}$ が「Z token と ANCC token に高 weight、 X/Y token に低 weight」 を学習することで物理線形関係を **暗黙に encode**
- subagent W §2.4 「endogenous/exogenous 分離が ROGII の cross-attention 化と完全整合」 は **公式 implementation そのまま** で達成可、 ROGII 固有 adapter は不要

---

## 2. offline pretrain pipeline

### 2.1 training data (= 776 wells visible-region pretext)

**全 well 利用方針**:
- train wells 773 件 + test wells 3 件の **visible region (= TVT_input 観測済み部分)** を全 pool
- test wells も pretrain に含める根拠: **TVT_input の visible 部分は leakage ではない** (= ROGII 公式 data spec で「test wells の visible region は input として提供」、 hidden region のみ predict 対象)
- → 776 wells × per-well 約 580 ft (visible 中央値) = 約 450k row、 1 ft 解像度

**well-level sample 構成**:
- 各 well の MD ordered sequence から **sliding window** で seq_len=256 + pred_len=64 のペアを切り出す
- 1 well = 約 580 ft visible / stride 128 = **約 4 sample/well**、 計 **約 3100 sample**
- pretext task: 各 sample で seq_len=256 region の末端 64 ft を **artificial mask** → predict (= §2.2)

**選択軸 B: pretext task の設計 (= forecasting flavor か bidirectional imputation か)**

| 軸 | 案 B1: forecasting (= 末端 64 を future として predict) | 案 B2: bidirectional imputation (= 中央 64 を random mask、 両端 visible) |
|---|---|---|
| 公式 TimeXer との一致 | ◎ (= `forecast` method そのまま) | △ (= mask 位置の embed 変更、 自前 adapter 必要) |
| ROGII 実 hidden 構造との一致 | △ (= ROGII hidden は visible 末端から末端 70%、 forecasting に近いが「visible 末端」 が train で常に末端とは限らない) | ◎ (= subagent W §2.5 SAITS MIT 設計と同型、 真の imputation) |
| 工数 | 小 (= 公式 dataset loader そのまま) | 中 (= mask position random sampling + position embedding 修正) |
| 期待精度 | 標準 | やや高 (= ROGII test の hidden 位置 mode と一致) |

→ 案 B1 = offline pretrain 工数最小、 9 hr cap 内で確実完走。 案 B2 = 数理本質的に ROGII test 構造と完全整合だが adapter 工数 +1 day。 推奨せず。

### 2.2 pretext task (= masked TVT reconstruction)

**案 B1 を採用した場合の loss**:
$$
\mathcal{L}_{\text{pretrain}} = \frac{1}{N \cdot T_{\text{pred}}} \sum_{n=1}^{N} \sum_{t=1}^{T_{\text{pred}}} \left( \hat{x}_{n,t}^{\text{TVT}} - x_{n,t}^{\text{TVT}} \right)^2
$$
= 標準 MSE (= 公式 `loss=MSE` default)。 Huber 化 (= subagent T 整合) は **fine-tune phase で別途**、 pretrain は MSE で stability 優先。

**hidden 化方針**:
- seq_len=256 sample の末端 64 row の `TVT_input` を **0 で zero-fill** (= 公式 `forecast` method の future input は dec_in 0 で渡す慣行)
- exogenous side (= Z, ANCC, X, Y, typewell features) の末端 64 row は **観測値を保持** (= visible なので zero-fill しない)
- target = 末端 64 row の真の `TVT` 値

### 2.3 hyperparameters (= epoch, lr, batch_size)

| param | value | 根拠 |
|---|---|---|
| `seq_len` | **256** (= ユーザー固定) | per-well visible 半分、 §1.2 |
| `pred_len` | **64** | 1 sample あたり 64 ft hidden、 rolling window で長 hidden 対応 (= §4.2) |
| `patch_len` | **32** (= ユーザー固定) | ar1_phi=0.999 correlation 1000 ft の 1/30、 §1.2 |
| `d_model, d_ff` | **256 / 512** | 公式 ETTh1 base 512 と pred_len=192 case 128 の中央、 GPU memory T4 16 GB 内 |
| `e_layers, n_heads` | **2 / 8** | 公式 default |
| `dropout` | **0.1** | 公式 default |
| `enc_in, c_out` | **1 / 1** (= 案 A1 採用時) | TVT 1 ch endogenous、 GR/MD は exogenous variate へ |
| `features` | **MS** | 公式 ETTh1 mode |
| `learning_rate` | **1e-4** | 公式 default |
| `batch_size` | **32** | 公式 default、 約 3100 sample / 32 = 97 step/epoch |
| `train_epochs` | **50-100** (= ユーザー指示の pretrain epoch 範囲) | early stop with patience=10 |
| optimizer | Adam | 公式 default |
| LR scheduler | cosine annealing | 公式 default |

**選択軸 C: pretrain epoch (= 50 vs 100)**

| 軸 | 案 C1: epoch=50 | 案 C2: epoch=100 |
|---|---|---|
| 工数 | 約 2.5-5 hr | 約 5-10 hr (= ユーザー指示の上限) |
| 収束安定性 | early stop で patience 不足 risk | 余裕、 後半 epoch でも val loss 安定 |
| over-fit risk | 中 (= 3100 sample / 256k 推定 param で sample 不足気味) | 大 (= 50 で十分なら 100 は memorize 進行) |

→ ユーザー指示の「epoch 50-100」 範囲内で early stop に委ねる。 推奨せず。

### 2.4 工数 + ハードウェア要件 (= local CPU/GPU or Colab Pro)

**GPU 要件試算** (= T4 16 GB or 同等):
- forward + backward 1 batch (= bs=32, seq_len=256, patch=32, d_model=256, e_layers=2): 約 30-80 ms (= 公式 ETTh1 base case の推定)
- 1 epoch (= 97 step) = 3-8 sec の forward + backward
- **50 epoch = 2.5-7 min** (= 公式 ETT benchmark に近い、 ETT は約 17k sample で 100 epoch / 20-30 min)
- ROGII の 3100 sample は ETT の 1/5、 epoch 時間も 1/5 で済む

**ユーザー指示「5-10 hr local 想定」 との乖離**:
- ユーザー想定 (= 5-10 hr) は **CPU** での pretrain と思われる、 GPU なら 1/10 以下
- GPU local (= RTX 3060 / 3090) なら **0.5-1 hr で完走**、 Kaggle Notebook GPU (= P100 / T4) で実行可
- CPU local (= M1 Mac / 8 core x86) なら 5-10 hr、 Colab Pro free tier で運用可能

**実行 path 選択軸 D**:

| 軸 | 案 D1: GPU local (RTX 3060+) | 案 D2: Kaggle Notebook GPU | 案 D3: Colab Pro (T4) | 案 D4: CPU local |
|---|---|---|---|---|
| 工数 | 0.5-1 hr | 0.5-1 hr (= 別 kernel として実行) | 0.5-1 hr | 5-10 hr |
| Kaggle 出力との接続 | upload 必要 | **不要** (= kernel output → dataset 直接 publish) | upload 必要 | upload 必要 |
| 並列性 | 開発者 PC 占有 | 他 kernel と共存可 (= GPU pool 別) | Colab session 9 hr 制限 | PC 占有、 long |
| cost | 0 (= sunk cost) | 0 (= 月 30 hr GPU free) | ~$10/月 | 0 |

→ **案 D2 (Kaggle Notebook GPU で pretrain) が工数 + dataset publish path で最良**、 ただし pretrain notebook は **本番 inference kernel と別**、 2 kernel 構成になる。 推奨せず、 中央判断。

### 2.5 checkpoint size (= ~100-300 MB)

**TimeXer model size 試算** (= d_model=256, e_layers=2, n_heads=8, d_ff=512, patch_len=32, seq_len=256):

- `EnEmbedding`: $W_{\text{patch}} \in \mathbb{R}^{D \times L_p} = 256 \times 32 = 8\,k$ param + $\mathbf{g}_{\text{en}} = 256$ param
- `DataEmbedding_inverted`: $W_{\text{var}} \in \mathbb{R}^{D \times T_{\text{en}}} = 256 \times 256 = 66\,k$ param × $K=13$ variate (= weight tied across variates) → 66 k param
- `Encoder × 2`:
  - self-attention: $W^{Q,K,V,O} \in \mathbb{R}^{D \times D} = 4 \times 65\,k = 260\,k$ param per layer
  - cross-attention: 同 260 k param per layer
  - FFN: $2 \times D \times D_{ff} = 2 \times 256 \times 512 = 260\,k$ param per layer
  - LayerNorm × 2 + 残差: 約 1 k param per layer
  - → 1 layer ≈ **780 k param**、 2 layers ≈ **1.56 M param**
- `FlattenHead`: $W_{\text{head}} \in \mathbb{R}^{(P \cdot D) \times T_{\text{pred}}} = (8 \times 256) \times 64 = 131\,k$ param
- **合計 ≈ 1.76 M param**、 float32 で **約 7 MB**、 float16 で 3.5 MB

ユーザー指示「~100-300 MB」 と乖離する理由:
- 公式 ETT 設定の d_model=512, d_ff=512 なら param 数 4x、 約 28 MB
- ROGII で d_model=256 採用なら **7 MB 程度で十分**、 100-300 MB は 10-40x の過剰

**checkpoint 内容**:
- model state_dict (= 7 MB float32)
- optimizer state_dict (= Adam で param 数 × 2、 14 MB float32) ← 推論時は不要
- val loss history (= 数 KB)
- config dict (= JSON, < 1 KB)

→ **inference-only kernel に上げるべきは model state_dict のみ、 約 7 MB**。 Kaggle dataset で 7 MB は overhead 最小、 upload 数秒。

**選択軸 E: precision (= float32 / float16 / int8)**

| 軸 | float32 | float16 | int8 (= dynamic quant) |
|---|---|---|---|
| size | 7 MB | 3.5 MB | 1.8 MB |
| inference 精度 | baseline | 約 0 影響 (= 数値範囲内) | -0.01〜-0.05 ft 劣化 risk |
| Kaggle env 互換 | ◎ | ◎ (= PyTorch 標準) | △ (= `torch.quantization` API、 推論 path 変更) |

→ float16 で十分、 ただし開発者判断。

---

## 3. Kaggle dataset upload spec

### 3.1 id, title, license

**Kaggle dataset metadata** (= `dataset-metadata.json` template):
```json
{
  "title": "ROGII TimeXer ckpt",
  "id": "ky7240/rogii-timexer-ckpt",
  "licenses": [{"name": "Apache-2.0"}],
  "is_private": true,
  "convertToCsv": false,
  "subtitle": "TimeXer pretrained checkpoint for ROGII Wellbore Geology Prediction (Wang 2024 NeurIPS, exogenous-aware transformer)"
}
```

**license 議論 (= ユーザー指示の Apache-2.0 整合性)**:

ユーザー指示は「Apache-2.0 (= TimeXer is MIT、 互換)」 だが、 **公式 `thuml/TimeXer` repo は LICENSE file なし** (= `gh api repos/thuml/TimeXer/contents/LICENSE` で 404)、 README にも license 言及なし (= WebFetch 確認済)。 **MIT ではない**。

| 解釈 | 帰結 |
|---|---|
| **MIT 想定が誤り** | TimeXer 公式 repo は厳密には **license 不明 (= デフォルトで著者 copyright)**、 GitHub の terms of service で「fork / clone は許諾」 程度しか保証されない (= `https://docs.github.com/en/site-policy/github-terms/github-terms-of-service#5-license-grant-to-other-users`) |
| Kaggle dataset upload | TimeXer source 自体を upload するなら license clarification が必須。 ただし本 spec の dataset は **学習済 checkpoint** (= model weight、 author code ではない) のみで、 source code は kernel 内で別途 clone or paste、 dataset には含めない構成にすれば license 問題は低減 |
| ckpt の license | **学習済 weight の copyright 帰属** は法的に grey zone (= 米国 著作権局 2023 guidance: 「機械生成出力は著作権なし」 寄り、 ただし train data や code の copyright が transfer する可能性あり)。 安全 path は dataset license を **Apache-2.0 で publish**、 ただし notebook description に「checkpoint は TimeXer architecture (Wang 2024, https://github.com/thuml/TimeXer) を local pretrain した weight です」 と明記 |

**recovery if license clarification 必要**:
- step 1: 著者 (= `wangyuxu22@mails.tsinghua.edu.cn`、 README 記載) に短文 mail で「Kaggle competition で MIT/Apache-2.0 等の OSI 互換 license で公開してもよいか」 問合せ
- step 2: 返答待ち中は **private dataset として upload + 我々の kernel のみ参照、 public publish 待機**
- step 3: 著者返答が「license 不可」 だった場合、 **TimeXer architecture を再実装** (= `models/TimeXer.py` 251 行を ROGII 内に paraphrased re-write、 同一 algorithm だが文言 100% 書換え) → 法的に最 conservative

### 3.2 upload 手順 + status check

**手順** (= 既存 `rogii_mamba_runtime` upload と同パターン、 5/13 着手前に中央が判断):

```bash
# Step 1: staging dir 作成
mkdir -p data/external/upload_staging/rogii_timexer_ckpt/
# Step 2: pretrain 出力 を staging に copy
cp outputs/timexer_pretrain/timexer_rogii_v1.pt data/external/upload_staging/rogii_timexer_ckpt/
cp outputs/timexer_pretrain/config.json data/external/upload_staging/rogii_timexer_ckpt/
# Step 3: dataset-metadata.json 作成 (= §3.1 内容)
cat > data/external/upload_staging/rogii_timexer_ckpt/dataset-metadata.json <<EOF
{ ... §3.1 内容 ... }
EOF
# Step 4: create (= 初回) or version (= 2 回目以降)
kaggle datasets create -p data/external/upload_staging/rogii_timexer_ckpt/ --dir-mode zip
# or
kaggle datasets version -p data/external/upload_staging/rogii_timexer_ckpt/ -m "v2: pretrain 100 epoch" --dir-mode zip
# Step 5: status check (= upload は数秒〜数分、 size 7 MB なら即完了)
kaggle datasets status ky7240/rogii-timexer-ckpt
```

**注意**: **本 subagent は dataset 実 upload を実行しない** (= ユーザー指示「dataset upload は subagent 完了報告まで実行しない (= 中央が判断、 subagent は spec doc + script のみ)」)。 上記は中央が 5/13 朝に手動実行する手順 freeze。

**Kaggle env compatibility risk**:
- TimeXer model state_dict は **PyTorch 標準 `.pt` format**、 Kaggle env (= torch 2.x preinstalled、 出典 `https://github.com/Kaggle/docker-python/blob/main/kaggle_requirements.txt`) で `torch.load` 互換
- ただし pretrain GPU が新 torch (= 2.10+) で保存、 Kaggle が旧 torch (= 2.5+) の場合、 `_pickle` protocol mismatch risk。 **pretrain 時に `torch.save(model.state_dict(), path, _use_new_zipfile_serialization=True)` で旧 protocol fallback**

---

## 4. inference-only kernel design

### 4.1 kernel_metadata (= dataset_sources に rogii-timexer-ckpt 追加)

**kernel-metadata.json** (= `kaggle_kernels/exp_timexer/kernel-metadata.json` template):
```json
{
  "id": "ky7240/rogii-exp-timexer",
  "title": "ROGII exp TimeXer path c",
  "code_file": "exp_timexer.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_tpu": "false",
  "enable_internet": "false",
  "dataset_sources": [
    "karnakbaevarthur/rogii-code-helper-dataset",
    "thermostatic/rogii-tabicl-v2-public-assets",
    "thbdh5765/rogii-v1-train-cache",
    "ky7240/rogii-per-well-stats",
    "ky7240/rogii-timexer-ckpt"
  ],
  "competition_sources": ["rogii-wellbore-geology-prediction"],
  "kernel_sources": []
}
```

**exp 番号の placeholder**:
- ユーザー指示の `kaggle_kernels/exp_timexer/exp_timexer.py` は **5/13 朝 dispatch 時に確定**、 現在の最新 exp は exp010 (= `kaggle_kernels/exp010_fold_reform/`)、 衝突回避で `exp011_timexer` か `exp012_timexer` を予想

**TimeXer source code の kernel 内取込み**:
- Kaggle internet off (= `enable_internet=false`) なので `pip install` 不可、 `git clone` 不可
- 選択軸 F:

| 軸 | 案 F1: kernel script 内に TimeXer source 直接 paste (= `models/TimeXer.py` + `layers/*.py` 約 30 KB を kernel script 末尾に paste) | 案 F2: 別 dataset `ky7240/rogii-timexer-source` を作成、 source code を upload | 案 F3: ckpt dataset 内に source も同梱 |
|---|---|---|---|
| script 行数 | exp005 2320 行 + TimeXer 600 行 ≈ 2900 行 | exp005 + 30 行 import | exp005 + 30 行 import |
| license 整理 | kernel description で明示が必要 | dataset description で明示 | dataset description で明示 |
| 更新性 | source 変更 = kernel 全 push | source 変更 = dataset version up のみ | 同 F2 |
| 工数 | 小 (= copy-paste) | 中 (= 2 dataset 管理) | 小 (= ckpt と同梱) |

→ **案 F3 が運用 + 工数で最良候補**、 ckpt dataset 内に `timexer_source/` subdir として `models/TimeXer.py` と `layers/SelfAttention_Family.py`、 `layers/Embed.py` を同梱。 推奨せず、 中央判断。

### 4.2 inference workflow (= load → predict → cumsum → blend)

**kernel script 構成案** (= subagent Z §2.2 inject 位置と整合):

```python
# === Section 0: imports ===
import sys
sys.path.insert(0, "/kaggle/input/rogii-timexer-ckpt/timexer_source/")  # 案 F3
from models.TimeXer import Model as TimeXerModel

# === Section 1: knobs ===
TIMEXER_ENABLE = True
TIMEXER_CKPT = "/kaggle/input/rogii-timexer-ckpt/timexer_rogii_v1.pt"
TIMEXER_CONFIG = "/kaggle/input/rogii-timexer-ckpt/config.json"
TIMEXER_WEIGHT = 0.3  # path c blend weight、 grid search by CV
TIMEXER_PRED_LEN = 64  # rolling window 単位
TIMEXER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Section 2: existing exp005 base pipeline (= 5-base + Ridge stacking + path b) ===
# ... 既存 2320 行そのまま、 test_delta_a / test_delta_b 出力 ...

# === Section 3: TimeXer path c ===
if TIMEXER_ENABLE:
    config = json.load(open(TIMEXER_CONFIG))
    model = TimeXerModel(SimpleNamespace(**config)).to(TIMEXER_DEVICE)
    state = torch.load(TIMEXER_CKPT, map_location=TIMEXER_DEVICE)
    model.load_state_dict(state)
    model.eval()

    test_delta_c = predict_timexer(model, test_df, well_features)
else:
    test_delta_c = test_delta_b  # fallback

# === Section 4: 3-way blend ===
W_PATH_B = 0.5
W_PATH_C = TIMEXER_WEIGHT
test_delta_final = (
    (1.0 - W_PATH_B - W_PATH_C) * test_delta_a +
    W_PATH_B * test_delta_b +
    W_PATH_C * test_delta_c
)

# === Section 5: cumsum to TVT + submission ===
# 既存 path b と同じ post-processing
```

**rolling window predict** (= hidden region 長 > pred_len=64 の well 対応):

```python
def predict_timexer(model, test_df, well_features, pred_len=64, seq_len=256):
    """rolling window で hidden region を 64 row ずつ predict。
    更新済み hidden を次の input visible として再 forward。"""
    predictions = []
    for well_id, well_df in test_df.groupby("well_id"):
        visible_end_idx = well_df["TVT_input"].notna().sum()
        hidden_len = len(well_df) - visible_end_idx
        n_windows = math.ceil(hidden_len / pred_len)

        current_tvt = well_df["TVT_input"].copy()
        for w in range(n_windows):
            # endogenous + exogenous batch を組み立て、 末端 64 を predict
            inputs = build_timexer_input(well_df, current_tvt, visible_end_idx + w * pred_len, seq_len)
            with torch.no_grad():
                pred = model(**inputs)  # (1, pred_len, 1)
            current_tvt[visible_end_idx + w * pred_len : visible_end_idx + (w + 1) * pred_len] = pred.cpu().numpy().squeeze()
        predictions.append(current_tvt.iloc[visible_end_idx:].values)
    return np.concatenate(predictions)
```

### 4.3 既存 stack との合成 (= subagent T path b 整合)

**3-way blend (= karnakbaev path a + 自前 path b + TimeXer path c)**:

| layer | output | 寄与 |
|---|---|---|
| path a | karnakbaev pretrained LGB3+XGB+CB blend → Ridge meta | LB 10.317 baseline (= exp005 v1) |
| path b | 自前 Multi-seed Huber MEDIAN + hetero weight (= subagent T v3) | -0.4 〜 -0.6 ft (= 期待) |
| path c | **TimeXer endogenous + exogenous cross-attention** | -0.2 〜 -0.5 ft (= 本案) |

**blend weight 決定**:
- 5 fold CV で `(W_PATH_B, W_PATH_C)` grid search (= 各 0.1 刻みで 11×11 = 121 combination、 CV time 60 sec/combination で約 2 hr)
- 推奨 grid: `W_PATH_B ∈ {0.3, 0.4, 0.5, 0.6, 0.7}`, `W_PATH_C ∈ {0.1, 0.2, 0.3, 0.4, 0.5}` (= 25 combination、 30 min)

**diversity 確認**:
- TimeXer OOF vs path b OOF の Pearson correlation を 5 fold で計算
- $\rho < 0.85$ なら blend に寄与あり (= subagent T v3 path b 採用基準と整合)、 $\rho > 0.95$ なら redundant

### 4.4 9 hr cap 内 runtime budget

**inference-only kernel の runtime 試算**:

| component | 時間 | 根拠 |
|---|---|---|
| exp005 既存 pipeline | 70 sec (= 既存実測、 `docs/dev/leaderboard.dense.md`) | 5-base + Ridge stacking + path b blend |
| TimeXer ckpt load | < 5 sec | model state_dict 7 MB、 PyTorch `torch.load` |
| TimeXer inference (= 776 wells × rolling window 4 回平均) | **約 15-60 sec** | 1 forward 30 ms × 776 × 4 = 93 sec で 1 well 4 window 想定、 GPU T4 |
| 3-way blend grid search (= optional) | 30 min - 2 hr | §4.3、 ablation 時のみ |
| 合計 (= grid search なし) | **約 90-140 sec** | 9 hr cap (= 32400 sec) の 0.5% |

**結論**: inference-only kernel は **9 hr cap に対し圧倒的余裕**。 さらに Edge R (= 1.7 hr)、 Edge D Kalman、 MEMO (= 1.7-3.5 hr) を追加しても余裕。 ただし grid search で W 決定する場合は kernel 内ではなく **別 CV kernel で先に決め、 inference kernel には best W を hard-code** が安全。

---

## 5. failure modes 3 件 + recovery

### 5.1 pretrain 不安定 → epoch 削減 + warmup

**兆候**:
- pretrain 中の `train_loss` が epoch 5-10 で nan / inf
- val_loss が train_loss を逆転 (= 異常 over-fit)
- gradient norm が 1e+5 以上で explode

**原因 (= 数理本質)**:
- ROGII の TVT 値は absolute depth scale (= 数千 ft オーダー)、 `use_norm=1` で z-score 正規化されるが、 patch 内変動が極小 (= 1 ft 解像度で 1 patch=32 ft なら $\sigma \approx 0.1$ ft) で **数値不安定** (= gradient under-flow)
- exogenous variates (= Z, ANCC, X, Y) も scale 不揃いで cross-attention の dot-product が saturate

**recovery (= 3 段)**:
- **a)** `train_epochs=30` + `warmup_steps=500` (= 公式 default の linear warmup) + `lr=5e-5` (= 半減) で再 train
- **b)** input normalization を `RevIN` (= Reversible Instance Normalization, Kim 2022 ICLR、 公式 TimeXer の `use_norm=1` 実装) → **per-sample mean/std で正規化**、 patch 間 scale drift 抑制
- **c)** TVT を **dTVT (= 1 ft 差分)** に変換して endogenous に入れる、 train 後 cumsum で TVT 復元 (= ROGII 既存 path b と同手法、 数値範囲 0.01 ft scale で stable)

### 5.2 checkpoint size 過大 → smaller model variant (= patch_size 64)

**兆候**:
- pretrain 出力 `.pt` file が 300 MB 超 (= 想定 7 MB の 40x)
- Kaggle dataset upload で size limit (= public dataset 100 GB / private 20 GB cumulative) 圧迫

**原因**:
- d_model=512 (= 公式 ETT base case) を pretrain で誤採用、 param 数 8x
- `FlattenHead` の出力次元 = $P \cdot D \cdot T_{\text{pred}} = 8 \times 512 \times 64 = 262\,k$ param が肥大
- optimizer state_dict (= Adam で param 数 × 2) を同梱して 3x 膨張

**recovery (= 3 段)**:
- **a)** `torch.save(model.state_dict(), path)` で **state_dict のみ save** (= optimizer 除外)、 7 MB に圧縮
- **b)** `d_model=128, d_ff=128` (= 公式 ETTh1 pred_len=192 case の値) に縮小、 param 数 1/4 で 2 MB
- **c)** `patch_size=64` (= ユーザー固定の 32 から倍増) で patch 数を 256/64=4 に半減、 `FlattenHead` 出力次元半減、 ただし local 構造 capture が荒くなる risk

### 5.3 cross-attention OOM → seq_len 削減

**兆候**:
- Kaggle T4 16 GB で `RuntimeError: CUDA out of memory`
- batch_size=32 で memory 12+ GB 占有 (= 25% 上限 4 GB を 3x 超過)

**原因 (= 数理本質)**:
- self-attention の memory complexity = $O(P^2 \cdot D)$ where $P = T_{\text{en}} / L_p + 1 = 256/32 + 1 = 9$
- cross-attention の memory complexity = $O(P \cdot K \cdot D)$ where $K=13$ exogenous variates
- 1 sample: $9 \times 9 \times 256 + 9 \times 13 \times 256 = 50\,k$ float = 200 KB、 bs=32 で 6.4 MB → **OOM ではなく安全**
- ただし `seq_len=2048+` (= 長 well の visible 全長を 1 sample に詰める変更) で $P=64$、 memory 50x 増 で OOM 化

**recovery (= 3 段)**:
- **a)** `seq_len=128` に半減 (= 1 well の sliding window 数を倍に増やして compensate)
- **b)** `batch_size=8` に縮小、 gradient accumulation で effective bs=32 維持
- **c)** **flash-attention 化** (= `torch.nn.functional.scaled_dot_product_attention` の `is_causal=False` mode、 PyTorch 2.0+ で memory $O(P)$)、 ただし公式 `models/TimeXer.py` の `FullAttention` を書換える adapter 工数 +0.5 day

---

## 6. 5/13 着手 prerequisite check list

### 6.1 GitHub repo clone

**状態**: 未完了 (= 本 subagent は code を書かない、 local clone も実行しない)

**5/13 朝 中央 dispatch 直前のチェック**:
- [ ] `git clone https://github.com/thuml/TimeXer.git external/TimeXer/` (= ROGII repo 内 untracked)
- [ ] commit hash freeze (= `cd external/TimeXer && git rev-parse HEAD > ../TimeXer_commit.txt`)
- [ ] `models/TimeXer.py` + `layers/SelfAttention_Family.py` + `layers/Embed.py` の 3 file 存在確認 (= 推論最小限)
- [ ] license file 確認 → 不在 confirmed (= §3.1)、 著者 mail 必要かどうか判断
- [ ] requirements.txt 確認 (= einops==0.4.0, torch==2.0.0, numpy==1.23.5)、 Kaggle env 互換確認

### 6.2 pretrain 開始

**状態**: spec 凍結のみ、 実 train は 5/13 着手

**5/13 朝 中央 dispatch 直前のチェック**:
- [ ] ROGII data loader 整備 (= visible region sliding window → seq_len=256, pred_len=64 sample 生成、 約 3100 sample)
- [ ] exogenous variate 13 ch 準備 (= Z, ANCC_imp, X, Y, formation_one_hot 6 col, typewell TVT/GR/Geology)
- [ ] GPU 環境準備 (= 案 D1 local RTX or 案 D2 Kaggle Notebook、 §2.4)
- [ ] pretrain script (= 公式 `run.py` を ROGII data loader 用に minimal modify) 作成、 dry run 1 epoch で sanity 確認
- [ ] full pretrain 開始 (= 50 epoch with early stop)

### 6.3 checkpoint upload

**状態**: spec のみ、 実 upload は中央判断 (= ユーザー指示「dataset upload は subagent 完了報告まで実行しない」)

**5/13 朝 中央 dispatch 直前のチェック**:
- [ ] pretrain 完了 → `outputs/timexer_pretrain/timexer_rogii_v1.pt` 出力確認 (= 7 MB 想定)
- [ ] `config.json` 同梱 (= `seq_len, pred_len, patch_len, d_model, ...` の全 hyperparameter snapshot)
- [ ] staging dir 作成 (= `data/external/upload_staging/rogii_timexer_ckpt/`)
- [ ] `dataset-metadata.json` 作成 (= §3.1)
- [ ] `kaggle datasets create` で初回 upload → status `READY` 確認
- [ ] CV smoke test (= local 1 fold で TimeXer predict の OOF RMSE 計測、 既存 path b 比 ±5% 内なら sanity OK)

### 6.4 inference kernel script

**状態**: spec のみ、 実 script は 5/13 中央が作成

**5/13 朝 中央 dispatch 直前のチェック**:
- [ ] `kaggle_kernels/exp_timexer/` dir 作成 (= exp 番号は中央が確定)
- [ ] `kernel-metadata.json` 作成 (= §4.1、 `dataset_sources` に `ky7240/rogii-timexer-ckpt` 追加)
- [ ] `exp_timexer.py` script 作成 (= §4.2 構成、 exp005 base に TimeXer Section 3 + 4 inject)
- [ ] local smoke test (= 3 test wells で predict 出力 → submission.csv shape 確認、 NaN 不在確認)
- [ ] `kaggle kernels push` で submission、 LB 確認待ち

### 6.5 prerequisite 通過率予測

| check item | 通過確度 | risk |
|---|---|---|
| GitHub clone + license clarification | 90% (= clone 自体は確実、 license clarification は著者返答待ちで遅延 risk) | license 不明で公開 dataset 化に法的 grey、 private のみ運用なら回避可 |
| Kaggle env 互換 (= einops, torch) | 95% (= einops は Kaggle base image 含む、 torch 2.x 互換) | reformer-pytorch (= TimeXer requirements にあるが TimeXer model 自体は import しない) は不要 |
| pretrain 完走 | 80% (= GPU 案 D1 or D2 で 0.5-1 hr、 失敗時は §5.1 recovery 3 段) | 数値不安定 risk、 RevIN / dTVT 変換で逃げる |
| dataset upload | 95% (= 7 MB なら数秒で完了) | license confirmed なら private 公開即時 |
| inference kernel push + SCORED | 80% (= exp005 baseline + TimeXer inject の minimal change、 ただし path c blend で test_delta が wrong scale なら post-cumsum で TVT 異常 risk) | smoke test 必須、 NaN guard |

**総合通過率 (= 全 6.1-6.4 連続 success)**: $0.9 \times 0.95 \times 0.8 \times 0.95 \times 0.8 = 52\%$

→ 5/13 着手から **最短 5/15-5/16 で TimeXer 単独 LB 確認可能**、 9 切り到達は path c blend grid search + 他 hack (= Kriging / MEMO / Kalman) との合成で **5/18-5/20 目処**。

---

## 7. 残課題 (= 5/14 以降の statefull pretrain 等)

### 7.1 statefull pretrain (= test wells で fine-tune)

ユーザー指示の Step 5「v2 (= 5/14 以降): test wells で fine-tune (= TTA との combination)」 の実装方針:

**transductive fine-tune の数理**:
- 各 test well の **visible region のみ** で 1-5 epoch fine-tune (= MEMO / TTT 流、 subagent Z §3 と数理的に直交)
- loss: visible region の TVT 観測値を artificial mask して predict → 観測値との MSE
- 推論時間: 3 test wells × 5 epoch × 数 sec = **< 1 min** (= 9 hr cap への影響 negligible)

**残課題**:
- statefull pretrain で v1 (= static) よりさらに -0.05〜-0.15 ft 期待、 ただし test wells 3 件しかない → over-fit risk 大
- v2 で per-well fine-tune した model checkpoint は **submission 毎に再生成**、 dataset upload は v1 のみで OK

### 7.2 path c blend weight の CV-based 最適化

§4.3 で grid search 提案、 ただし CV time が 2 hr 必要。 5/15-5/16 の inference kernel 第 1 submission では `W_PATH_C=0.3` を初期値 hard-code、 LB 反映後に grid search で確定 (= 5/17-5/18)。

### 7.3 案 A1 (= TVT 1 ch endogenous) vs 案 A2 (= 3 ch endogenous) の ablation

§1.2 で「推奨せず」 としたが、 LB 改善幅が -0.2 ft (= 下限) で停滞した場合、 案 A2 への切替で +0.1 ft 上振れ余地。 ablation は 5/19-5/20 帯。

### 7.4 patch_len 32 vs 64 の ablation

§5.2 recovery c で patch_size=64 案を出したが、 ar1_phi=0.999 → correlation 1000 ft の現象を **patch 1 個に 64 ft 詰める** ことで local 構造が荒くなる risk。 ablation 必要、 5/19-5/20 帯。

### 7.5 dTVT 変換 (= §5.1 recovery c) vs absolute TVT

ROGII の path b は既存で dTVT (= 1 ft 差分) を予測、 TimeXer も dTVT 化が数値安定性で有利の可能性大。 pretrain 開始時に **dTVT 採用が default 安全 path**、 absolute TVT は失敗 fallback。 中央判断。

### 7.6 TimeXer license clarification (= §3.1)

著者 mail で `Apache-2.0 / MIT 互換 license 要請` → 返答待ち。 不可なら architecture re-implementation (= 同 algorithm を ROGII repo 内で 100% paraphrase) で legal-safe 化。 5/13 朝に判断、 私的 dataset 運用で先行可。

---

## 8. 参考文献

- TimeXer paper: Wang et al. "TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables", NeurIPS 2024, arXiv:2402.19072, https://arxiv.org/abs/2402.19072
- TimeXer GitHub (= 公式実装): `thuml/TimeXer`, https://github.com/thuml/TimeXer, push 2024-11-27, 478 stars, license **不明** (= LICENSE file 不在 confirmed)
- RevIN (= §5.1 recovery b): Kim et al. "Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift", ICLR 2022, https://openreview.net/forum?id=cGDAkQo1C0p
- SAITS (= §2.1 案 B2 数理的根拠): Du et al. "SAITS: Self-Attention-based Imputation for Time Series", Expert Systems with Applications 219:119619, arXiv:2202.08516, https://arxiv.org/abs/2202.08516
- iTransformer (= §1.1 exogenous variate embedding 起源): Liu et al. "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting", ICLR 2024, arXiv:2310.06625, https://arxiv.org/abs/2310.06625
- subagent W: `docs/research/academic-literature-deeper.dense.md` @ branch `docs/literature-deeper-2026-05-11` (= commit 53307f7), §2.4 TimeXer
- subagent Z: `docs/research/3-hack-implementation-spec.dense.md` @ branch `docs/3hack-spec-2026-05-11` (= commit dcf71ce), §2 Hack 2 TimeXer × ROGII
- ROGII first principles: `docs/research/first-principles.dense.md` §2.1 (= ar1_phi=0.999、 patch_size 32 の根拠)
- ROGII top3 distill: `docs/research/top3-distill.dense.md` (= typewell cross-attention の発想起源、 R/N kernel 数理)
- Kaggle docker requirements: https://github.com/Kaggle/docker-python/blob/main/kaggle_requirements.txt (= sktime / reformer-pytorch 不在 confirmed)
- ROGII LB baseline: `docs/dev/leaderboard.dense.md` (= exp005 v1 LB 10.317, exp005 v2 LB 10.203)
- ROGII submission postmortems: `docs/dev/submission-postmortems.dense.md`
- Mamba runtime upload pattern (= 本 spec の template): `docs/research/mamba-implementation-spec.dense.md` @ branch `docs/mamba-spec-2026-05-11` (= commit dda4142)

---

## 9. 中間 commit log (= 30 分粒度、 kill 復旧用)

| 時刻 | commit hash | 内容 |
|---|---|---|
| 2026-05-11 (本 spec 作成中) | (= 本 commit) | §0-9 一括 first draft |

(以降、 spec が更新される度に追記、 30 分以内の中間 commit を厳守する)
