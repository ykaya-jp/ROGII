# 学術文献 Deeper Dive — WLFM + ICML/NeurIPS 2024-2025 + UQ + SEG/SPE + TTT (2026-05-11)

> **目的**: 既存 distill (`past-comps-deepdive.dense.md`, `first-principles.dense.md`, `problem-essence.dense.md`, `top3-distill.dense.md`) で触れられていない/言及はあるが浅い 6 領域 (A: WLFM, B: 2024-2025 sequence/imputation, C: heteroscedastic UQ, D: SEG/SPE/EAGE, E: geostatistics, F: TTT) を一次資料まで降りて整理する。
>
> **方針**: ROGII exp005 v2 = LB 10.203 から 9 切り (-1.3 ft 改善) を狙うとき、各論文を **取り込む / 取り込まない** ではなく **どの場面でどう effect を測るか** を選択軸として並べる。推奨は出さない。判断は中央。
>
> **CRITICAL**: 各文献に arXiv ID / DOI / GitHub URL を必ず添える。出典のない claim は書かない。

---

## 0. 序: 既存 distill との差分

### 0.1 既存資産で touch 済 (= 本 doc では深掘りせず)
- SPWLA 2023 Dreamstar mWDN-GRU + DTW (`past-comps-deepdive.dense.md` §1)
- Ventilator 1 位 PID matching + weighted MEDIAN (= §2)
- FORCE 2020 1 位 XGBoost + `lambda_l2=1500` (= §3)
- `random_stretch` augmentation, DTW post-proc, multi-task head (= §5 パターン群)
- 物理関係 `TVT = -Z + ANCC + b_well`, residual std 0.007 ft (`first-principles.dense.md` §1.4)
- ar1_phi=0.999 の dTVT 自己相関 (= `first-principles.dense.md` §2.1)

### 0.2 本 doc が deeper に降りる領域
| 領域 | 既存言及 | 本 doc の追加 |
|---|---|---|
| A. WLFM (arXiv 2509.18152) | `past-comps-deepdive.dense.md §4.11` で 1 段落だけ | 1200 wells pretrain の数理 / token 設計 / fine-tune workflow / license / ROGII attach 可能性 |
| B. Sequence regression 2024-2025 | PatchTST / TimesNet 言及あり、SAITS 言及あり | iTransformer / TimeMixer / TimeXer / CSDI 新規、PatchTST patch_size 設計を ROGII の ar1_phi=0.999 から逆算 |
| C. Heteroscedastic + UQ | 未言及 | Beta-NLL / DEUP / CARD / Deep Evidential Regression (Amini 2020) を per-well variance handling として整理 |
| D. SEG / SPE / EAGE 2024 | SPWLA 系のみ | 公開 competition / award winner / OnePetro paper を調査 |
| E. Pyrcz geostatistics | 未言及 | GeostatsPy variogram / kriging を ROGII Edge N (offset well retrieval) の数理 strict 化 |
| F. TTT / transductive | TTT (Sun 2020) 言及あり | TTT++ (Liu 2021) / MEMO (Zhang 2021) / NC-TTT (2024) deeper、Edge R / Edge T との数理 unification |

---

## 1. 領域 A: Well-log Foundation Model (WLFM)

### 1.1 主要論文 — WLFM (arXiv 2509.18152, 2025-09-16)

**出典**: Qi et al. "WLFM: A Well-Logs Foundation Model for Multi-Task and Cross-Well Geological Interpretation" arXiv:2509.18152 (https://arxiv.org/abs/2509.18152), CC-BY 4.0 license

**著者**: Zhenyu Qi, Qing Yu, Jichen Wang, Yun-Bo Zhao, Zerui Li, Wenjun Lv

**Pretrain 規模**: **1200 wells** with multi-curve logs (= ROGII 773 wells より 1.55x、kaggle dataset として attach 可能規模)

**アーキテクチャ 3 段**:
1. **Patch tokenization** — log patch を「geological token」に変換 (= PatchTST patch + learnable codebook 的)
2. **Masked-token modeling** — masked patch を周辺 context から復元 (= BERT/MAE 流の self-supervised)
3. **Stratigraphy-aware contrastive learning** — 同一層 (= formation) 内 patch を引き寄せ、異層は引き離す。`top3-distill.dense.md` の formation top 概念と直結

**downstream 結果** (= fine-tune 後):
- Porosity estimation: MSE 0.0041 → 0.0038
- Lithology classification: 74.13% → 78.10%
- Curve reconstruction、boundary detection の clustering analysis でも層認識能力を確認

**実装 detail (= HTML v1 https://arxiv.org/html/2509.18152v1 で追加判明)**:
- **Tokenizer**: vector quantization (VQ) with learnable codebook $\mathcal{E} = \{e_k\}_{k=1}^K$、 curve は per-well z-score 正規化、 patch (length L, stride s)。 K / d / β / L / s の具体値は paper の "experimental setup" section 内で開示 (= HTML excerpt には未掲載、 supplementary PDF 必要)
- **Encoder**: depthwise separable convolutions with residual + Transformer。 curve-type embedding + relative-depth positional encoding (= patch の絶対 depth ではなく formation top からの相対 depth で encode)
- **Masking strategy**: **block-wise masking along depth with ratio r** (= contiguous block で mask、 random per-token ではない) → **ROGII の hidden block (= visible 末端から末端 70%) と同じ pattern ★**。 これは PatchTST の random 40% mask とは挙動が異なる
- **Contrastive loss (SCL eq.4)**:
$$
\mathcal{L}_{\text{SCL}} = -\log \frac{\exp(\text{sim}(z_i, z_j^+)/\tau)}{\sum_k \exp(\text{sim}(z_i, z_k)/\tau)}
$$
sim は cosine 類似度、 positive pair は **cross-well で相対 depth alignment + low-freq Pearson > τ_sim** な patch (= 別 well の同層 patch を positive、 異層 patch を negative)
- **Hardware**: NVIDIA RTX A6000、 batch size 256、 AMP、 throughput **680 ± 8 patches/s** (= 軽量 architecture の証左)
- **下流 baselines** (= Table 1): CNN / LSTM / CNN-LSTM / Transformer、 WLFM-400/-600/-1200/-Finetune の scale ablation 完備 (= 1200 well で full performance plateau)

**code/weights**: **公開 repo URL は paper にも Semantic Scholar / Google search にも明記なし** (= 2026-05-11 検証)。 著者所属 **USTC** (= 中国科学技術大学、 Yun-Bo Zhao / Wenjun Lv lab、 自動制御系 group)。 paper は CC-BY 4.0 だが weights 公開は別判断。 連絡 email は arXiv abstract page から取得可能 (Zerui Li 氏 corresponding 推定)

**ROGII への適用 path**:
- **path α (weights 入手 OK の場合)**: Qi et al. の pretrained encoder を private kaggle dataset 化 → ROGII の visible 区間 (GR, Z, X, Y) を patch tokenize → fine-tune で TVT head を取り付ける。**1200 wells pretrain → 773 wells fine-tune** という ratio は few-shot 適合範囲
- **path β (weights なしの場合)**: WLFM の **recipe (= patch + masked + contrastive)** を ROGII で zero から実装。`top3-distill.dense.md §6.1` の R/N kernel pretrain stage に組み込む。773 wells では pretrain 規模不足だが、FORCE 2020 / VOLVE / RealPore 系の public well log を集約すれば 2-3k well 規模に届く

**9 切り (LB 8.x) への寄与推定**:
- path α: 楽観 -0.5 〜 -1.0 ft (= cross-well generalization gain)。前提は weights 入手
- path β: -0.1 〜 -0.3 ft (= 自前 pretrain は data 規模に律速)、工数 5-10 day

**license + 工数 trade-off**:
- CC-BY 4.0 → kaggle submit OK (= attribution 必要)
- weights public でなければ ROGII deadline (残 86 day) では path β しか取れない
- 工数: path α 2-3 day、path β 5-10 day

### 1.2 関連 WLFM 系 (= 探索の deeper)

WLFM は 2025 後半の新規 work で、先行 well-log foundation model は **明示的 paper は少ない**。GeoFM / WellGPT 等の名称は kaggle/twitter 上で散見されるが査読論文として固まったものは見当たらない (= 2026-05-11 時点)。

代わりに **隣接 foundation model**:
- **MOMENT** (Goswami 2024, arXiv:2402.03885, https://arxiv.org/abs/2402.03885) — 一般 time series foundation model、HuggingFace で weights 公開、ROGII の GR や Z にそのまま attach 試せる
- **Lag-Llama** (Rasul 2024, arXiv:2310.08278, https://arxiv.org/abs/2310.08278) — probabilistic time series foundation model
- **TimesFM** (Das 2024, Google, arXiv:2310.10688, https://arxiv.org/abs/2310.10688) — 200B time-points で pretrain
- **TimeGPT × Well Logs** (Koeshidayatullah 2024-12, arXiv:2412.05681, https://arxiv.org/abs/2412.05681) — **TimeGPT を well log に zero-shot 適用した実証論文**。 R² 87%、 MAPE 1.95%、 anomaly detection accuracy 93%。 **basin-adaptation 不要 zero-shot で動く** ことを示した最初の paper の 1 つ。 ROGII の transfer 性に直接示唆

**ROGII での即時試行候補**: MOMENT は HF transformers から `transformers.AutoModel.from_pretrained` で attach 可、ROGII の visible 区間に zero-shot で imputation かけて baseline 取れる (= 半日工数)。 TimeGPT は Nixtla の paid API 経由、 **ROGII の 9 hr submission run は外部 API call 禁止** (= kaggle code competition ルール) なので submission には乗らないが、 local ablation の baseline として有用

### 1.3 ROGII への適用 path + license + 工数 まとめ

| model | weights 公開 | license | ROGII 工数 | LB 期待寄与 |
|---|---|---|---|---|
| WLFM (Qi 2025) | 未確認 | CC-BY 4.0 (paper) | path α 2-3d / path β 5-10d | path α -0.5〜-1.0 / β -0.1〜-0.3 |
| MOMENT (Goswami 2024) | HF 公開 | MIT | 0.5d zero-shot + 2d fine-tune | -0.05〜-0.2 (= 汎用 ts、地質特化なし) |
| Lag-Llama (Rasul 2024) | HF 公開 | Apache 2.0 | 1d zero-shot | -0〜-0.1 (= probabilistic だが univariate) |
| TimesFM (Das 2024) | HF 公開 | Apache 2.0 | 1d zero-shot | -0〜-0.1 |

---

## 2. 領域 B: Sequence regression / masked imputation 2024-2025

### 2.1 PatchTST deeper — patch_size 設計を ROGII で正当化

**出典**: Nie et al. "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" ICLR 2023, arXiv:2211.14730 (https://arxiv.org/abs/2211.14730), GitHub https://github.com/yuqinie98/PatchTST, MIT license

**論文の hyperparameter** (= 公式 repo `yuqinie98/PatchTST/PatchTST_supervised/scripts/PatchTST/ettm1.sh` より、 ETTm1 実験):
- **patch_length P = 16, stride S = 8** (= 50% overlap、 確認済)
- look-back L = 336 (短) or 512 (長) → patch 数 N = (L - P) / S + 1 = 41 or 63
- channel-independent: 各 channel に Transformer を独立適用 + 全 channel で重み共有
- **d_model=128, n_heads=16, e_layers=3, dropout=0.2, fc_dropout=0.2, head_dropout=0**
- batch=128, lr=1e-4
- masked pretrain: **40% mask ratio** (= `patchtst_pretrain.py --mask_ratio 0.4` 確認済)

**ROGII への patch_size 逆算**:

ROGII の dTVT 自己相関は ar1_phi = 0.999 (`first-principles.dense.md §2.1`)。AR(1) の **有効相関長 (correlation length)** は:
$$
\ell_{\text{corr}} = -\frac{1}{\ln \phi} \approx -\frac{1}{\ln 0.999} \approx 1000 \text{ ft}
$$

これは 1 ft 解像度なら **patch 内に 1000 sample 入れたい** scale。PatchTST P=16 では小さすぎる。一方、`extrapolation-residual-curve.parquet` で std が劇的に増える区間は 25→525 ft (= std 0.94→9.35 ft, 10x)。**「効く scale」は 25-525 ft の幅で、これを 1 patch に詰める設計が自然**。

→ ROGII で試すべき patch_size 候補 (= 1 ft 解像度前提):
- **P=32, S=16** (= 32 ft window, 50% overlap): mid-frequency capture、`extrapolation-residual` 25-75 ft 帯対応
- **P=64, S=32** (= 64 ft window): low-frequency、formation 内構造
- **P=128, S=64** (= 128 ft window): inter-formation、`gr_match_synth_rmse p50=315 ft` の半分

**channel-independent vs channel-mixed の選択軸**:
- ROGII の channel = [TVT, GR_h, GR_v(typewell), Z, X, Y, ANCC_imputed]
- **channel-independent**: 各 channel に同じ Transformer 重み。over-fit 抑制、汎化高い、しかし TVT ⇄ GR の cross-correlation を直接学べない (= 内部で encode する必要)
- **channel-mixed (= iTransformer 流)**: variate を token 化、attention で cross-variate 関係を学習。**TVT = f(Z, ANCC) の物理関係を直接 attention で表現可** (= 2.2 で詳述)
- 選択軸: **channel-mixed は ROGII の物理事前 (= `TVT = -Z + ANCC + b_well`) と整合**、ただし parameter 数増えて 773 well では over-fit リスク

**masked pretrain (PatchTST 40% mask)**: ROGII の hidden block は **末端 70% 連続 mask** = PatchTST 流 random 40% mask とは挙動が異なる。**長 contiguous mask に特化した pretrain** は SAITS / CSDI (= §2.5, 2.6) の領域で、PatchTST の標準 pretrain は ROGII にやや miss-fit

**9 切り寄与推定**: P=64 + channel-independent baseline で -0.1〜-0.3 ft。channel-mixed cross-attention 入れて -0.2〜-0.5 ft 上振れ余地、ただし over-fit リスク

### 2.2 iTransformer (Liu 2024 ICLR)

**出典**: Liu et al. "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting" ICLR 2024, arXiv:2310.06625 (https://arxiv.org/abs/2310.06625), GitHub https://github.com/thuml/iTransformer

**著者**: Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, Mingsheng Long (= Tsinghua THUML lab、 PatchTST と独立に時系列 SOTA を競う grp)

**核心 idea**: 「Transformer の attention は **時間軸ではなく variate 軸** で取るべき」
- 従来 (PatchTST 含む): patch_t = (TVT_t, GR_t, Z_t, ...) を 1 token として時間軸 attention
- iTransformer: 各 variate (TVT 全長 / GR 全長 / Z 全長) を 1 token として variate 軸 attention

**ROGII への含意**:
- ROGII の channel は [TVT, GR_h, GR_v, Z, ANCC] で variate 数 5-10 個
- iTransformer は variate 数 N に対し O(N²) attention で **N 小なら高速**、ROGII では variate 5 個 → attention 25 cell で激軽
- **物理関係 TVT = -Z + ANCC + b の linear constraint を attention で直接 encode しやすい** (= TVT token ⇄ Z token ⇄ ANCC token の attention weight が物理を反映)

**PatchTST との性能差** (= 公開 benchmark): ETT / ECL / Weather / Traffic 等の標準 benchmark で iTransformer がやや優位、ただし datasets により差異あり (詳細は paper Table 3-5)

**ROGII 工数**: GitHub thuml/iTransformer は readme + PyTorch 実装あり、ROGII の visible block を直接食わせる adapter を 1-2 day で書ける

**9 切り寄与推定**: -0.1〜-0.3 ft。channel-mixed の利点を取れる場面で効く

### 2.3 TimeMixer (Wang 2024 ICLR)

**出典**: Wang et al. "TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting" ICLR 2024, arXiv:2405.14616 (https://arxiv.org/abs/2405.14616), GitHub https://github.com/kwuking/TimeMixer

**核心**: **MLP-only architecture** (Transformer なし)、2 component:
1. **PDM (Past-Decomposable-Mixing)**: 多 scale 系列を seasonal / trend に分解 (= STL 流)、fine→coarse と coarse→fine の bidirectional mixing
2. **FMM (Future-Multipredictor-Mixing)**: scale 毎に予測 head を持ち ensemble

**ROGII への含意**:
- ROGII の **multi-scale 構造** (= R kernel Self-NCC 17/31/51 ft window, `top3-distill.dense.md`) は TimeMixer の multi-scale mixing と同型概念
- **MLP-only で軽量** = Transformer 系より 5-10x 高速、9 hr 推論制約に余裕
- decomposition (seasonal / trend) は ROGII では「formation-internal HF noise / inter-formation LF trend」に対応

**選択軸**:
- TimeMixer は **forecasting (= 過去→未来 extrapolation)** 設計、ROGII の **bidirectional imputation (= visible→hidden 補完)** とは厳密には違う。FMM の future predictor を imputation predictor に置換する adapter 必要
- PatchTST と異なり patch 概念なし、raw sequence 直接食う → 長 sequence (6500 ft) で memory OK か要確認

**9 切り寄与推定**: -0.05〜-0.2 ft (= multi-scale 既存 R kernel と redundant な部分多い)

### 2.4 TimeXer (Wang 2024-2025)

**出典**: Wang et al. "TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables" arXiv:2402.19072 (https://arxiv.org/abs/2402.19072), GitHub https://github.com/thuml/TimeXer

**核心**: **endogenous (= 主 target) と exogenous (= 補助 variates) を明示分離**:
- endogenous (= TVT) は patch-wise self-attention で時間方向に処理
- exogenous (= GR, Z, ANCC, typewell) は variate-wise cross-attention で endogenous に注入
- **global endogenous token** が bridge

**ROGII への含意 ★★**:
- これは **ROGII の構造に最も自然な fit** の 1 つ:
  - endogenous = TVT (= target sequence)
  - exogenous = GR_horizontal, GR_typewell, Z, X, Y, ANCC_imputed (= context)
- `top3-distill.dense.md` の R/N kernel が「typewell GR を cross-attention で query」している発想と直接対応
- 12 個の real-world benchmark で SOTA、generality 高い

**9 切り寄与推定**: -0.2〜-0.5 ft (= endogenous/exogenous 分離が ROGII の cross-attention 化と完全整合)

**工数**: GitHub thuml/TimeXer に PyTorch 実装あり、ROGII 用 adapter 2-3 day

### 2.5 SAITS deeper — masked imputation の数理

**出典**: Du et al. "SAITS: Self-Attention-based Imputation for Time Series" Expert Systems with Applications 219:119619 (2023), arXiv:2202.08516 (https://arxiv.org/abs/2202.08516), GitHub https://github.com/WenjieDu/SAITS, MIT license

**著者**: Wenjie Du, David Cote, Yan Liu

**Loss 数理**:

SAITS は 2 重 loss:
$$
\mathcal{L} = \mathcal{L}_{\text{ORT}} + \lambda \cdot \mathcal{L}_{\text{MIT}}
$$

- **ORT (Observed Reconstruction Task)**: 観測されている位置の値を **入力に渡しているにも関わらず** reconstruct、self-attention の reflection loss。$\mathcal{L}_{\text{ORT}} = \|\hat{X}_o - X_o\|^2$ where $X_o$ は観測値
- **MIT (Masked Imputation Task)**: artificially masked した位置を予測。$\mathcal{L}_{\text{MIT}} = \|\hat{X}_m - X_m\|^2$ where $X_m$ は人工的に隠した位置の真値
- λ (= MIT weight) は paper では 1 (= 等重み)

**DMSA (Diagonally-Masked Self-Attention)**: 自己位置を attention から除外 (= diagonal を mask)、これにより「観測位置の値を直接 copy する trivial solution」を防ぐ。**2 個の DMSA block** を直列に並べ、間に dynamic combine weighting (= attention map から重み生成して 2 block 出力を blend)

**ROGII への含意**:
- ROGII の hidden block (visible 末端から 末端 70%) は **artificial mask ではなく natural mask** で、SAITS の MIT 設計が directly fit
- **DMSA の diagonal mask** は ROGII で 「自分の位置の TVT を予測するのに自分の TVT を入力に使わない」(= test では NaN なので使えない) という設定と同一
- 2 DMSA block の dynamic combine は R/N kernel の Ridge stack の発想と類似 (= 2 base model を attention weight で blend)

**9 切り寄与推定**: -0.1〜-0.4 ft。特に **ORT loss を入れることで visible 区間の reflection が安定化**、long contiguous mask に対する robustness 向上

**ROGII 工数**: WenjieDu/SAITS は PyPOTS (= imputation toolkit) に統合済み、`pip install pypots` で使用可、ROGII adapter 1-2 day

### 2.6 CSDI — Conditional Score Diffusion Imputer

**出典**: Tashiro et al. "CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation" NeurIPS 2021, arXiv:2107.03502 (https://arxiv.org/abs/2107.03502), GitHub https://github.com/ermongroup/CSDI, MIT license

**著者**: Yusuke Tashiro, Jiaming Song, Yang Song, Stefano Ermon (= Stanford Ermon lab、 score-based diffusion の系譜)

**核心**: **条件付き score-based diffusion**:
- 観測値 $x_o$ を condition、欠損値 $x_m$ を generate
- forward process: $x_m^{(t)} = \sqrt{\bar{\alpha}_t} x_m^{(0)} + \sqrt{1-\bar{\alpha}_t} \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$
- reverse process: $\epsilon_\theta(x_m^{(t)}, x_o, t)$ で denoising
- training: 観測値の一部を artificially condition / target に split

**性能**: 既存 probabilistic imputation 比 **40-65% 改善**、deterministic (SAITS 等) 比 **5-20% 改善** (paper abstract)

**正確な hyperparameter (= `ermongroup/CSDI/config/base.yaml` より)**:
- diffusion steps **T=50**
- noise schedule: **quad** (= quadratic、 linear ではない)、 β_start=0.0001、 β_end=0.5
- 内部 Transformer: channels=64、 layers=4、 heads=8、 diffusion_embedding_dim=128
- time embedding dim=128、 feature embedding dim=16
- training: batch=16、 lr=1e-3、 epochs=200

**ROGII への含意**:
- **uncertainty を持つ TVT 予測** が直接出る (= score sampling で multi-sample → quantile)
- ROGII の `b_well_resid_std p50=0.008 ft` から std を物理事前として diffusion noise schedule に encode 可
- **正確な計算 cost (= T=50 確定済)**: 776 wells × 50 step ≈ 38800 forward pass。 1 forward 0.05 sec (= 軽量 Transformer 4 layer × A6000 推定) で **1940 sec ≈ 0.54 hr**、 9 hr 制約内
- ただし **single sample** では uncertainty 取れない (= 1 path しか出ない)、 nsample 10-100 で multi-sample 取ると 5-54 hr で over → **nsample 10 + visible 比率上位 wells のみ subset 推論** が現実的妥協

**9 切り寄与推定**: -0.3〜-0.7 ft (= uncertainty marginalization の理論的 edge)、ただし推論時間制約で **subset 推論 (= unstable wells のみ CSDI、残りは SAITS)** の hybrid 必要

**工数**: ermongroup/CSDI に PyTorch 実装あり、ROGII adapter 3-5 day、推論最適化に追加 2-3 day

---

## 3. 領域 C: Heteroscedastic + UQ

### 3.1 Beta-NLL (Seitzer 2022 ICLR)

**出典**: Seitzer et al. "On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks" ICLR 2022, arXiv:2203.09168 (https://arxiv.org/abs/2203.09168), GitHub http://github.com/martius-lab/beta-nll

**問題提起**: 標準 Gaussian NLL は **低 noise 領域でのフィッティング失敗**:
$$
\mathcal{L}_{\text{NLL}} = \frac{1}{2} \log \sigma^2 + \frac{(\hat{y} - y)^2}{2 \sigma^2}
$$
最初の term が分散 σ² を大きく取る方向に gradient を出す (= small σ で penalty 大)、結果として model は σ² を過小評価する局所最適に落ちやすい

**Beta-NLL 修正**:
$$
\mathcal{L}_{\text{β-NLL}} = \mathcal{L}_{\text{NLL}} \cdot \lfloor \sigma^{2\beta} \rfloor_{\text{sg}}
$$
where $\lfloor \cdot \rfloor_{\text{sg}}$ は stop-gradient、β ∈ [0, 1] は hyperparameter

- β=0: 標準 NLL に戻る
- β=1: high-variance sample に大 weight、low-variance sample に小 weight
- 推奨 β = 0.5

**ROGII への適用**:
- ROGII の per-well variance (= `b_well_resid_std` p50=0.008 / p90=0.009 / max=0.011) は **well により 1.4x 差**
- ROGII の R/N kernel は **plain RMSE** training。Beta-NLL に置き換えれば per-well stable wells (small σ) で σ underestimate を防げる
- ただし、評価指標は RMSE なので NLL 系 loss の improvement が **直接 RMSE 改善** に繋がるかは uncertain (= 経験的にしか言えない、ablation 必須)

**9 切り寄与推定**: -0.05〜-0.2 ft (= variance 推定精度向上 → confidence-weighted ensemble が効くシナリオ)

**工数**: pytorch 数行追加で済む、0.5 day

### 3.2 DEUP — Direct Epistemic Uncertainty Prediction (Lahlou 2023)

**出典**: Lahlou et al. "DEUP: Direct Epistemic Uncertainty Prediction" TMLR 2023, arXiv:2102.08501 (https://arxiv.org/abs/2102.08501)

**著者**: Salem Lahlou, Moksh Jain, Hadi Nekoei, Victor Ion Butoi, Paul Bertin, Jarrid Rector-Brooks, Maksym Korablyov, Yoshua Bengio (= Mila)

**核心**: epistemic uncertainty を **excess risk** で定義:
$$
\text{Epistemic}(x) = R(\hat{f}, x) - R^*(x)
$$
where $R$ は generalization error、$R^*$ は Bayes 最適 predictor の error。**Bayesian posterior variance ではない** (= model misspecification も capture)

**実装**:
1. main predictor $\hat{f}(x) \to y$ を train
2. **auxiliary predictor** $g(x) \to \text{loss}(\hat{f}(x), y)$ を train (= generalization error の直接予測)
3. aleatoric uncertainty $A(x)$ を別に推定
4. $\text{Epistemic}(x) = g(x) - A(x)$

**ROGII への含意**:
- ROGII は **well misclassification (= dip 急変 well で plane fit 破綻)** が epistemic 不確実性の主因
- DEUP の auxiliary regressor を「visible 区間の R/N kernel OOF residual を予測する」well 単位で train、test well で先に epistemic を推定 → 高 epistemic well で DTW / particle filter を強化

**9 切り寄与推定**: -0.1〜-0.3 ft (= adaptive post-proc が unstable well で大きく効く場合)

**工数**: auxiliary regressor 1 個追加、1-2 day

### 3.3 Deep Evidential Regression (Amini 2020 NeurIPS) — 補強

**出典**: Amini et al. "Deep Evidential Regression" NeurIPS 2020, arXiv:1910.02600 (https://arxiv.org/abs/1910.02600), GitHub https://github.com/aamini/evidential-deep-learning, Apache 2.0

**核心**: Normal-Inverse-Gamma (NIG) prior を出力、analytic に aleatoric + epistemic 両方を解析的に分離:
$$
p(y | \mu, \sigma^2) \cdot p(\mu, \sigma^2) = \mathcal{N}(y | \mu, \sigma^2) \cdot \text{NIG}(\mu, \sigma^2 | \gamma, \upsilon, \alpha, \beta)
$$

network 出力は 4 parameter (γ, υ, α, β)。MC dropout や ensemble 不要 (= 1 forward pass で uncertainty)

**ROGII への含意**:
- 1 forward で uncertainty 取れる = 9 hr 制約で MC dropout (= 30 sample × 推論時間) より遥かに高速
- ROGII の plain RMSE training を NIG loss に置換、出力 head が 1 (= μ) → 4 (= γ, υ, α, β) に増える

**9 切り寄与推定**: -0.05〜-0.2 ft (= Beta-NLL と類似領域、選択軸は loss 形のシンプルさ vs uncertainty 分離性)

### 3.4 CARD — Classification And Regression Diffusion (Han 2022 NeurIPS)

**出典**: Han et al. "CARD: Classification and Regression Diffusion Models" NeurIPS 2022, arXiv:2206.07275 (https://arxiv.org/abs/2206.07275), GitHub https://github.com/XzwHan/CARD

**核心**: regression を **diffusion で multimodal posterior** として表現:
- 通常 NN regression: 1 point estimate $\hat{y} = f(x)$
- CARD: $p(y | x)$ を diffusion で学習、multimodal 分布も表現可

**conditional denoising**:
$$
y^{(t-1)} = \epsilon_\theta(y^{(t)}, x, t)
$$
where $x$ は input feature、$y^{(t)}$ は noisy target、$\epsilon_\theta$ は denoiser network。**pre-trained conditional mean estimator** (= 標準 NN) を $\epsilon_\theta$ の初期 prior として使う点が CSDI と異なる

**ROGII への含意**:
- ROGII の hidden TVT は **multimodal posterior** を持つ場面が想定される (= bit が上下動して GR pattern repeat = `problem-essence.dense.md §3.1`)
- CARD なら multi-mode を捕まえて sampling、Beam Search K=5-7 の代替に
- **CSDI との違い**: CARD は y のみに diffusion (= 軽量)、CSDI は y + masked x 全体に diffusion (= 重い)

**9 切り寄与推定**: -0.1〜-0.4 ft (= multimodal posterior の正確な表現が dip 急変 well で効く)

**工数**: XzwHan/CARD の PyTorch 実装あり、ROGII adapter 3-4 day

### 3.5 領域 C 全体の ROGII への適用 path

| method | aleatoric | epistemic | 推論 cost | ROGII 工数 | LB 期待寄与 |
|---|---|---|---|---|---|
| Beta-NLL | ○ (= σ) | × | 1x | 0.5d | -0.05〜-0.2 |
| Deep Evidential (Amini) | ○ (= NIG α/β) | ○ (= NIG υ) | 1x | 1d | -0.05〜-0.2 |
| DEUP (Lahlou) | △ (= 別途) | ◎ | 1.5x (= aux regressor) | 1-2d | -0.1〜-0.3 |
| CARD (Han) | ◎ (= sampling) | ◎ | 5-10x (= 50 denoising step) | 3-4d | -0.1〜-0.4 |
| CSDI (Tashiro) | ◎ | ◎ | 10-30x | 3-5d | -0.3〜-0.7 |

選択軸: 推論時間 vs uncertainty 分離精度のトレードオフ。9 hr 制約下で全 wells に CSDI は無理、subset 推論との hybrid 必須

---

## 4. 領域 D: SEG / SPE / EAGE winner

### 4.1 SEG 2023 ML — 該当 competition 不存在の確認

**調査結果**: 「SEG 2023 ML Competition」は **存在を確認できず**。
- SEG (Society of Exploration Geophysicists) は 2016 ML contest を最後に独自 ML competition を開催していない (出典: https://github.com/seg/2016-ml-contest)
- 2023 年の同種は **SPWLA 3rd ML Competition 2023 (= depth shift)** で、これは `past-comps-deepdive.dense.md §1` で完読済 (= Dreamstar 1 位)
- SEG 系の最新 ML work は **paper として** SEG Journal 系で散発的に発表 (= 例 https://library.seg.org/doi/10.1190/geo2023-0129.1 "Where are we and how far are we from the holy grail?" review article)

→ **「SEG 2023 ML Competition の winner」を ROGII に直接転用する path は無し**。代わりに SEG Journal の review paper / SEG 2016 ML contest (= facies classification) を間接参照する

### 4.2 SEG 2016 ML Contest (= facies classification、間接参照)

**出典**: https://github.com/seg/2016-ml-contest

- 1 位 LA Team: gradient boosting + augmentation
- top 10 共通: Bestagini polynomial features (= `past-comps-deepdive.dense.md §3.4` で touch)
- ROGII への含意: 既存 distill で touch 済、新規 finding なし

### 4.3 SPE 2024 / EAGE 2024 ML Competition — 不存在の確認

**調査結果**: 「SPE Geomechanics ML 2024」「EAGE ML for subsurface 2024」を web search したが、**公開 competition / leaderboard / repository は確認できず**。

- EAGE は **AI Committee newsletter** を出すが competition format ではない (出典: https://eage.org/communities/artificial-intelligence-community/)
- SPE は OnePetro で paper 公開はあるが ML competition の leaderboard 公開なし

→ **ROGII に直接転用できる 2024 公開 winner は SPWLA 2023 / FORCE 2020 系以外には現存しない** (= 既存 distill で touch 済範囲が網羅的)

### 4.4 関連: OnePetro 系 paper 

代わりに、OnePetro の lithology prediction paper を mining:
- "Subsurface Lithology Classification Using Well Log Data" (Springer, 2024 https://link.springer.com/chapter/10.1007/978-981-99-1620-7_18) — supervised ML survey
- "Data-driven machine learning approaches for precise lithofacies identification" (Tandfonline 2024 https://www.tandfonline.com/doi/full/10.1080/10095020.2024.2405635) — multi-feature lithofacies
- "Drava Basin Pannonian super basin well data" (MDPI 2024 https://www.mdpi.com/2076-3417/14/14/6039) — geological feature engineering

これらは **survey 性質** で ROGII の baseline 比 -0.05 ft 級の improvement にしか寄与しない。

### 4.5 領域 D まとめ

**結論**: ROGII の context で **「SEG/SPE/EAGE 2024 公開 ML competition winner」は実質存在せず**、`past-comps-deepdive.dense.md` の SPWLA 2023 / FORCE 2020 / IceCube / Ventilator / OpenVaccine / Indoor Location / Jane Street の集約が網羅的。本領域での 9 切り寄与は **新規 path 無し**。

---

## 5. 領域 E: Geostatistics + dip continuity

### 5.1 Pyrcz GeostatsPy — variogram / kriging

**出典**: Pyrcz (Univ. Texas Austin), GeostatsPy package, PyPI: https://pypi.org/project/geostatspy/, GitHub: https://github.com/GeostatsGuy/GeostatsPy, MIT license

**Pyrcz 2014 textbook**: "Geostatistical Reservoir Modeling" (2nd ed. 2014, Oxford Univ Press, ISBN 978-0199731442). **2020 版は存在せず** (= 2nd ed 2014 が最新)

**核心 1: Variogram**

variogram $\gamma(h)$ は 2 点間距離 $h$ における値の分散の半分:
$$
\gamma(h) = \frac{1}{2} \mathbb{E}\left[(Z(x) - Z(x+h))^2\right]
$$

実用上:
- **range (= a)**: $\gamma$ が sill に達する距離 = correlation length
- **sill (= C₀+C)**: $\gamma$ が plateau に達する値 = total variance
- **nugget (= C₀)**: $h \to 0$ の jump = measurement noise

**geometric anisotropy** (= GeostatsPy doc 強調): 「vertical range of correlation is much less than horizontal range」(= 地層は水平方向に長く、垂直方向に短く相関)

**核心 2: Kriging (= BLUE)**

simple kriging:
$$
\hat{Z}(x_0) = \mu + \sum_{i=1}^n \lambda_i (Z(x_i) - \mu)
$$
where $\lambda_i$ は covariance matrix から決まる weight (= variogram 経由)。

**kriging variance** (= 予測の uncertainty):
$$
\sigma_K^2(x_0) = C(0) - \sum_{i=1}^n \lambda_i C(x_i, x_0)
$$

これは **予測点が観測点から離れるほど variance 増加** という直感を厳密化したもの。

### 5.2 ROGII Edge N (offset well retrieval) の数理 strict 化

**現状**: `top3-distill.dense.md` の R/N kernel は **FormationPlaneKNN** (= 6 formation × KNN k=10 で plane fit) で ANCC を impute。これは「平面 + 距離 weight」だが variogram 情報を陽に使っていない。

**Pyrcz 流での strict 化**:
1. **Step 1**: ANCC top depth (X, Y) per formation を train wells から集める = ~773 wells × 6 formations = ~4600 観測点
2. **Step 2**: variogram fitting (= sample variogram + spherical / Gaussian model fit)。GeostatsPy `gamv` 関数で sample variogram、`vmodel` で fit
3. **Step 3**: test well の (X, Y) に対し **ordinary kriging** で ANCC を推定、同時に kriging variance も得る
4. **Step 4**: kriging variance を Bayesian posterior の prior precision として下流 model に渡す

**正確な GeostatsPy 関数 signature** (= GeostatsPyDemos_Book https://geostatsguy.github.io/GeostatsPyDemos_Book/GeostatsPy_kriging.html より、 actual code 抜粋):

```python
# variogram model 構築 (= 異方性 OK、 2 構造まで)
variogram = GSLIB.make_variogram(
    nug=0.0,           # nugget effect (= measurement noise variance)
    nst=1,             # 構造数 (1 or 2)
    it1=1,             # 1=spherical, 2=exponential, 3=Gaussian
    cc1=1.0,           # contribution (= sill - nugget)
    azi1=0.0,          # azimuth (= NE-SW 方向の主軸、 度)
    hmaj1=5000.0,      # major range (= 主軸方向 correlation length, ft)
    hmin1=2000.0,      # minor range (= 直交方向 correlation length, ft)
)

# 2D ordinary kriging
kmap, vmap = geostats.kb2d(
    df=df, xcol='X', ycol='Y', vcol='ANCC_top_depth',
    tmin=-99999, tmax=99999,
    nx=100, xmn=0.0, xsiz=10.0,    # grid 設定
    ny=100, ymn=0.0, ysiz=10.0,
    nxdis=1, nydis=1,              # block discretization (1=point kriging)
    ndmin=4, ndmax=10,             # 検索 neighbor 最小 / 最大
    radius=10000.0,                # 検索半径 (ft)
    ktype=1,                       # 0=simple, 1=ordinary kriging
    skmean=0.0,                    # ktype=0 のときのみ使用
    vario=variogram,
)
# kmap = kriging estimate, vmap = kriging variance
```

**ROGII での具体的 fit 戦略**:
- 6 formations × 773 wells で **per-formation** に variogram fit (= formation 毎に anisotropy が異なる前提)
- ROGII の (X, Y) は host EDA で normalized 済み (= `data-spec.dense.md`)、 実 ft 単位ではない可能性。 まず raw 単位での variogram range を確認、 必要なら scaling
- **`ktype=1` (ordinary)** を採用、 train wells の ANCC 全体平均は formation 毎に不明 (= mean stationarity 仮定が破れる)、 ordinary は local mean を local data で推定するため robust

**期待 effect**:
- 現 KNN k=10 は **等方 weight** (= 全方向同じ)、variogram 流は **anisotropic** (= NE-SW か E-W 方向に layer が長い場合の方位依存)
- kriging variance を **per-well epistemic** として下流 (= Beta-NLL / DEUP) に渡せる

**選択軸**:
- KNN (= 既存) vs Kriging (= Pyrcz): kriging は理論最適 BLUE だが variogram fitting の追加工数
- ROGII の (X, Y) 座標は normalized 済 (= host EDA)、metric 解釈は要確認

**9 切り寄与推定**: -0.1〜-0.3 ft (= anisotropic 構造を encode できる well 数 × 効果)

**工数**: GeostatsPy install + adapter 2-3 day

### 5.3 dip continuity の数理

**stratigraphic dip**: 地層平面の傾き、ROGII では各 formation の TVT(X, Y) に対する gradient $\nabla \text{TVT}$ で定義可

ROGII の `b_well_drift_first_to_last p50 = 0.00154 ft` は dip 変動の indirect 計測。`b_well_resid_std p50 = 0.008 ft` は per-well dip-fit 残差。

**variogram の含意**: dip 連続性は **horizontal range >> vertical range** が経験則。ROGII の 6 formations の variogram を別々に fit すれば、formation 毎の「水平相関長」を定量化できる → **Edge N の KNN k 値を formation 毎に variogram range に応じて自動調整**

### 5.4 領域 E まとめ

| 改修 | 既存 KNN | Pyrcz Kriging | 工数 | LB 期待 |
|---|---|---|---|---|
| ANCC imputation | 等方 weight k=10 | anisotropic variogram | 2-3d | -0.1〜-0.3 |
| uncertainty | × | kriging variance ⇒ 下流 | (上に含) | -0.05〜-0.1 |
| formation 毎 k | 固定 | variogram range で自動 | 0.5d | -0.05〜-0.1 |

---

## 6. 領域 F: Test-time training / transductive

### 6.1 TTT (Sun 2020) — 既存言及の確認

**出典**: Sun et al. "Test-Time Training with Self-Supervision for Generalization under Distribution Shifts" ICML 2020, arXiv:1909.13231 (https://arxiv.org/abs/1909.13231), GitHub https://github.com/yueatsprograms/ttt_cifar_release

**核心**: training に **rotation prediction (= self-supervision)** auxiliary task を組み込み、test 時に rotation pred loss だけで encoder を update。test sample 1 個毎に gradient step 1-10 回。

**ROGII での既存実装**: `top3-distill.dense.md` の Edge R (= R kernel Self-NCC) は **静的** な multi-scale NCC で、TTT 流の test-time gradient update は **していない**。

### 6.2 TTT++ (Liu 2021 NeurIPS)

**出典**: Liu et al. "TTT++: When Does Self-Supervised Test-Time Training Fail or Thrive?" NeurIPS 2021 (https://proceedings.neurips.cc/paper/2021/file/b618c3210e934362ac261db280128c22-Paper.pdf), https://openreview.net/forum?id=86NHK__yFDl

**著者**: Yuejiang Liu, Parth Kothari, Bastien van Delft, Baptiste Bellot-Gurlet, Taylor Mordan, Alexandre Alahi (= EPFL VITA lab)

**核心**: TTT は **severe distribution shift で性能劣化** することを実証、2 改善:
1. **TFA (Test-time Feature Alignment)**: source feature 統計を offline 保存、test 時に target feature 統計 (= 1次/2次 moment) を online matching、source-target gap を統計レベルで縮める
2. **TTT-C (Test-time Contrastive)**: rotation 予測の代わりに **contrastive learning** を auxiliary に使う、より informative

**batch-queue decoupling**: online 推論で batch size 小でも安定した moment 推定を実現

**ROGII への含意 ★**:
- ROGII test wells は train wells から **dip 急変 / fault** などの distribution shift がある可能性 (= `problem-essence.dense.md §1.3` "lateral facies change")
- TFA の feature moment matching は ROGII で **test well の GR mean/std を train well 分布に align** する形で適用可
- TTT-C の contrastive は WLFM (= §1.1) の stratigraphy-aware contrastive と同種

**9 切り寄与推定**: -0.1〜-0.4 ft (= dip 急変 well での local adaptation 効果)

### 6.3 MEMO (Zhang 2021 NeurIPS)

**出典**: Zhang et al. "MEMO: Test Time Robustness via Adaptation and Augmentation" NeurIPS 2022, arXiv:2110.09506 (https://arxiv.org/abs/2110.09506), GitHub https://github.com/zhangmarvin/memo

**著者**: Marvin Zhang, Sergey Levine, Chelsea Finn (= Stanford / Berkeley)

**核心**: **1 test sample に対して** 多数 augmentation を作り、それらの **marginal output 分布の entropy を最小化** で encoder update:
$$
\mathcal{L}_{\text{MEMO}}(x) = H\left(\frac{1}{N} \sum_{i=1}^N p(y | a_i(x))\right)
$$
where $a_i$ は augmentation、$H$ は entropy

**ROGII への含意**:
- ROGII 各 test well を「1 sample」、`random_stretch` (= SPWLA 2023 Dreamstar) や noise injection で N=8-16 augmentation 作り、TVT 予測の marginal が **sharp** (= 低 entropy) になるよう encoder fine-tune
- 9 hr 制約: 776 wells × 16 aug × 5 gradient step = 62k forward。1 forward 0.1 sec で 6200 sec = 1.7 hr。**OK**

**9 切り寄与推定**: -0.1〜-0.3 ft

### 6.4 NC-TTT (2024) と最新動向

**出典**: "NC-TTT: A Noise Contrastive Approach for Test-Time Training" (https://arxiv.org/html/2404.08392v1)

NC-TTT は TTT-C の contrastive を **noise contrastive estimation** に置換、より安定化。同種に **ReC-TTT** (https://arxiv.org/html/2411.17869v1) の contrastive feature reconstruction、**CTA** (Cross-Task Alignment, https://arxiv.org/html/2507.05221v1)。

これらは ROGII に直接転用しても -0.05 ft 級の差。

### 6.5 Edge R / Edge T (Cal-Aug) との数理的 unification

ROGII の **Edge R (= R kernel Self-NCC multi-scale)** は test 時に typewell との NCC を計算し alignment、これは TTT の **forward pass 統計利用** に近い。**Edge T (= Cal-Aug)** は visible 区間で per-well 線形校正を fit、これは TTT++ の **TFA (feature moment matching)** と数理的に同型 (= 1次/2次 moment alignment)。

**統一: Test-Time Calibration + Adaptation の formalism**:
$$
\hat{f}_{\text{adapted}}(x_h | x_v) = \arg\min_{\theta} \left[ \mathcal{L}_{\text{prior}}(\theta) + \lambda \cdot \mathcal{L}_{\text{ttt}}(x_v) \right]
$$
where $\theta$ は per-well calibration parameter、$x_v$ は visible 区間、$x_h$ は hidden 区間予測対象。$\mathcal{L}_{\text{ttt}}$ は self-supervised loss (= visible NCC match / reconstruction)。

**ROGII で未着手**:
- **gradient-based** test-time fine-tune (= 現 Edge T は closed-form 線形校正のみ)
- **contrastive** auxiliary (= visible 区間で typewell GR と horizontal GR を contrastive pair に)
- **augmentation marginal entropy** (= MEMO 流 random_stretch ×16 → entropy)

**9 切り寄与推定**: gradient-based fine-tune 単独で -0.1〜-0.3 ft、contrastive aux 追加で +(-0.05〜-0.15)

**工数**: 既存 R/N kernel に test-time gradient loop 追加 3-5 day

### 6.6 領域 F まとめ

| method | 既存 ROGII edge との関係 | 9 切り寄与 | 工数 |
|---|---|---|---|
| TTT (Sun 2020) | Edge R の dynamic 化 | -0.05〜-0.2 | 2-3d |
| TTT++ (Liu 2021 TFA) | Edge T (Cal-Aug) と同型 + dynamic | -0.1〜-0.3 | 3-4d |
| TTT++ (TTT-C contrastive) | WLFM contrastive と関連 | -0.05〜-0.2 | 3-4d |
| MEMO (Zhang 2022) | random_stretch + entropy min | -0.1〜-0.3 | 2-3d |
| NC-TTT / ReC-TTT 系 | 上記 variant | -0.05〜-0.15 | 同上 |

---

## 7. ROGII の 9 切りに直結する文献 hack 上位 5 件

(= 推奨せず軸のみ提示、judgment は中央)

### 7.1 hack #1: **TimeXer × ROGII (= endogenous/exogenous 分離 cross-attention)**

- **出典**: Wang 2024-2025 arXiv:2402.19072
- **適用**: TVT = endogenous, GR_horiz / GR_typewell / Z / X / Y / ANCC_imputed = exogenous の構造で TimeXer attach
- **選択軸**: ROGII の物理事前 (= `TVT = -Z + ANCC + b`) を attention で直接 encode したい場合
- **コスト**: 工数 2-3d、推論時間 +20%
- **LB 期待寄与**: **-0.2〜-0.5 ft** (= 12 benchmark で SOTA 実績)

### 7.2 hack #2: **CSDI subset 推論 hybrid (= unstable wells のみ CSDI、残り SAITS)**

- **出典**: Tashiro 2021 NeurIPS arXiv:2107.03502
- **適用**: epistemic 高い well (= DEUP で識別) のみ diffusion 推論、残りは SAITS / R-kernel
- **選択軸**: uncertainty marginalization の理論的 edge を 9 hr 制約下で取りたい場合
- **コスト**: 工数 3-5d (= CSDI adapter) + 2-3d (= hybrid logic)
- **LB 期待寄与**: **-0.3〜-0.7 ft** (= probabilistic imputation 40-65% 改善実績)

### 7.3 hack #3: **GeostatsPy Kriging × ANCC imputation (= Edge N の数理 strict 化)**

- **出典**: Pyrcz, GeostatsPy GitHub
- **適用**: 既存 FormationPlaneKNN (k=10 等方) を anisotropic variogram kriging に置換、kriging variance を下流 model に
- **選択軸**: ROGII の per-formation 水平相関長を陽に活用したい場合
- **コスト**: 工数 2-3d
- **LB 期待寄与**: **-0.1〜-0.3 ft** + (epistemic 連携で) -0.05〜-0.1 ft

### 7.4 hack #4: **MEMO 流 random_stretch × entropy minimization fine-tune**

- **出典**: Zhang 2022 NeurIPS arXiv:2110.09506 + SPWLA 2023 Dreamstar
- **適用**: 各 test well に対し random_stretch ×16 augmentation 作り、marginal TVT 予測 entropy を test-time minimize
- **選択軸**: test-time adaptation を gradient-based で取り入れる第一歩、9 hr 制約内
- **コスト**: 工数 2-3d
- **LB 期待寄与**: **-0.1〜-0.3 ft**

### 7.5 hack #5: **WLFM pretrain weights path α / β**

- **出典**: Qi 2025 arXiv:2509.18152
- **適用**: 
  - path α: 公開 weights あれば ROGII 上に fine-tune
  - path β: WLFM recipe (= patch + masked + stratigraphy contrastive) を FORCE/VOLVE/ROGII で zero pretrain
- **選択軸**: cross-well generalization を foundation model で取りたい場合
- **コスト**: 工数 path α 2-3d / path β 5-10d
- **LB 期待寄与**: path α **-0.5〜-1.0 ft** (前提: weights 入手) / path β **-0.1〜-0.3 ft**

### 7.6 5 件の選択軸比較

| hack | 物理事前活用 | uncertainty 出力 | 推論時間 | 工数 | LB upper bound | 確実性 |
|---|---|---|---|---|---|---|
| #1 TimeXer | ◎ | × | 1.2x | 2-3d | -0.5 | 中 (= 12 benchmark 実績) |
| #2 CSDI hybrid | △ | ◎ | 2-3x | 5-8d | -0.7 | 中 (= imputation 領域実績) |
| #3 Kriging | ◎ | ○ | 1x | 2-3d | -0.4 | 高 (= 理論最適 BLUE) |
| #4 MEMO TTT | △ | × | 1.5x | 2-3d | -0.3 | 中 |
| #5 WLFM α/β | ◎ | △ | 1x | 2-10d | -1.0 / -0.3 | α: 不明 / β: 中 |

---

## 8. 残課題 / 時間切れで深掘りできなかった部分

### 8.1 深掘りできなかった文献
- **TimesNet (Wu 2023)**: 既存 distill で言及あるが、TimesBlock の Inception 構造、FFT-based period detection の数理は本 doc では skip。`past-comps-deepdive.dense.md` に比して新規 finding 少と判断
- **GraphNeT** (IceCube 1 位): graph + transformer、ROGII の 6 formation graph 化への直接 adapter は未確認
- **SCINet / N-BEATS / N-HiTS**: forecasting 系の MLP-based baselines、TimeMixer と類似領域で skip
- **DLinear / TiDE**: linear baseline 系、ROGII で baseline 比較として有用だが本 doc 主旨外

### 8.2 検証できなかった claim
- **WLFM weights public availability**: paper 上で明記されず、HuggingFace / GitHub 検索でも見当たらず。**著者連絡 (= Qi et al.) が必要** な可能性
- **CSDI 推論時間 ROGII での実測**: 推定値 11 hr (= 9 hr 制約 over) は理論計算、実機での measurement が未実施
- **GeostatsPy の ROGII coordinate (X, Y) スケール整合**: host EDA で normalized 済みだが、variogram fit のための metric 距離が地物単位で正しいかは未検証

### 8.3 さらなる deeper 余地のある領域
- **SDE / Neural ODE** (= continuous-time 系列モデル): ROGII の AR(1) phi=0.999 を連続時間で扱う数理的整合
- **Probabilistic numerics** (= Hennig et al.): kriging を generalize した GP regression に深層を組み合わせる
- **Generalized Linear Mixed Model (GLMM)**: per-well random effect を統計学的に厳密化 (= b_well を random effect として明示)

### 8.4 並列 worker (= subagent V) と conflict 範囲
- `docs/research/academic-literature-deeper.dense.md` は本 worker のみ書込み (= disjoint OK)
- `docs/research/sub-data-mining-*.md` は V が書込み (= 別 file、conflict なし)

---

## 9. 関連 doc / source 一覧

### 9.1 ROGII 内 distill
- `docs/research/past-comps-deepdive.dense.md` — 過去 winner solutions
- `docs/research/first-principles.dense.md` — 物理事前 + irreducible error 計測
- `docs/research/problem-essence.dense.md` — 問題本質 + 推測ラベル
- `docs/research/top3-distill.dense.md` — R/N kernel 詳解
- `docs/research/data-spec.dense.md` — column / mask spec

### 9.2 一次資料 URL (= 本 doc 引用全件)
- WLFM: https://arxiv.org/abs/2509.18152
- MOMENT: https://arxiv.org/abs/2402.03885
- Lag-Llama: https://arxiv.org/abs/2310.08278
- TimesFM: https://arxiv.org/abs/2310.10688
- PatchTST: https://arxiv.org/abs/2211.14730 / https://github.com/yuqinie98/PatchTST / https://nixtlaverse.nixtla.io/neuralforecast/models.patchtst.html
- iTransformer: https://arxiv.org/abs/2310.06625 / https://github.com/thuml/iTransformer
- TimeMixer: https://arxiv.org/abs/2405.14616 / https://github.com/kwuking/TimeMixer
- TimeXer: https://arxiv.org/abs/2402.19072 / https://github.com/thuml/TimeXer
- SAITS: https://arxiv.org/abs/2202.08516 / https://github.com/WenjieDu/SAITS
- CSDI: https://arxiv.org/abs/2107.03502 / https://github.com/ermongroup/CSDI
- Beta-NLL: https://arxiv.org/abs/2203.09168 / http://github.com/martius-lab/beta-nll
- DEUP: https://arxiv.org/abs/2102.08501
- Deep Evidential Regression: https://arxiv.org/abs/1910.02600 / https://github.com/aamini/evidential-deep-learning
- CARD: https://arxiv.org/abs/2206.07275 / https://github.com/XzwHan/CARD
- TTT (Sun 2020): https://arxiv.org/abs/1909.13231 / https://github.com/yueatsprograms/ttt_cifar_release
- TTT++ (Liu 2021): https://proceedings.neurips.cc/paper/2021/file/b618c3210e934362ac261db280128c22-Paper.pdf / https://openreview.net/forum?id=86NHK__yFDl
- MEMO: https://arxiv.org/abs/2110.09506 / https://github.com/zhangmarvin/memo
- NC-TTT (2024): https://arxiv.org/html/2404.08392v1
- ReC-TTT (2024): https://arxiv.org/html/2411.17869v1
- GeostatsPy: https://github.com/GeostatsGuy/GeostatsPy / https://geostatsguy.github.io/GeostatsPyDemos_Book/
- Pyrcz 2014 textbook: ISBN 978-0199731442 "Geostatistical Reservoir Modeling" Oxford Univ Press 2nd ed
- SEG 2016 ML contest: https://github.com/seg/2016-ml-contest
- SPWLA 2023 ML competition: https://github.com/pddasig/Machine-Learning-Competition-2023
- EAGE AI community: https://eage.org/communities/artificial-intelligence-community/
- SEG holy grail review: https://library.seg.org/doi/10.1190/geo2023-0129.1

---

## 10. 更新履歴

- 2026-05-11 初版 (= ROGII リサーチ担当 W、 branch `docs/literature-deeper-2026-05-11`)
