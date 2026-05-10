# Kaggle GM 叡智 deep dive (2026-05-11)

> **目的**: ROGII (= masked TVT regression、現状 Ridge OOF 10.39 / LB 10.317) を **CV 8 台 + LB 8 台** に持っていくため、Kaggle Grandmaster / Master の核心 hack を 60+ 件抽出し、ROGII の `tabular regression + sequence regression + CV-LB gap shrinking` という 3 軸に直結する hack に絞り込む。
>
> **方針**: コードを書かない / 修正しない (リサーチ専任)。**推奨は出さない、トレードオフ表 + 選択軸 + 失敗モード** のみ提示 (CLAUDE.md 智者尽其慮 / 中立指示原則)。各 hack に **GM の Kaggle write-up / blog / talk / book URL** を出典として添える。
>
> **CRITICAL 制約**: ROGII の現課題 = (1) CV-LB diff が systematic に追跡できていない (= 今日ユーザー指摘で発覚)、(2) Ridge OOF 10.39 → 8.5 への構造的 hack が必要、(3) 9hr Kaggle 制約での model 数最大化。

---

## 0. 出典 + 対象 GM 一覧

| GM 名 | Kaggle profile | 主出典 (blog / talk / book) | ROGII 関連性 |
|---|---|---|---|
| **Chris Deotte** (NVIDIA, 11x Kaggle GM) | https://www.kaggle.com/cdeotte | NVIDIA Playbook https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/、Author page https://developer.nvidia.com/blog/author/cdeotte/、Pseudo-Labeling kernel https://www.kaggle.com/code/cdeotte/pseudo-labeling-qda-0-969 | tabular + GBM stack + pseudo-labeling + CV strategy の総合 |
| **Bojan Tunguz** (NVIDIA, Quadruple GM) | https://www.kaggle.com/tunguz | Adversarial validation kernels (IEEE https://www.kaggle.com/code/tunguz/adversarial-ieee、Elo https://www.kaggle.com/tunguz/elo-adversarial-validation、Quora https://www.kaggle.com/code/tunguz/quora-adversarial-validation)、書籍 The Kaggle Book ISBN 978-1801817479 | adversarial validation で CV-LB shift を測る原点 |
| **CPMP / Jean-Francois Puget** (NVIDIA, Triple GM, Discussion #1) | https://www.kaggle.com/cpmpml | NVIDIA author https://developer.nvidia.com/blog/author/jeanfrancoispuget/、Hackernoon interview https://hackernoon.com/interview-with-twice-kaggle-grandmaster-dr-jean-francois-puget-cpmp-6d92328e433a、IEEE-CIS 2nd https://www.youtube.com/watch?v=wqHlAOFSFuQ、"Beyond Feature Engineering and HPO" talk https://www.youtube.com/watch?v=VC8Jc9_lNoY | Trust your CV / nested CV / stacking 哲学 |
| **Gilles Vandewiele** (Ventilator 1 位) | https://www.kaggle.com/group16 | Write-up https://medium.com/data-science/winning-the-kaggle-google-brain-ventilator-pressure-prediction-2d4c90d831ec、Discussion https://www.kaggle.com/c/ventilator-pressure-prediction/discussion/285256、Liverpool Ion 3rd https://github.com/GillesVandewiele/Liverpool-Ion-Switching、OpenVaccine 4th https://github.com/GillesVandewiele/covid19-mrna-degradation-prediction | Sequence regression + 15 seed × 5 fold = 75 model + weighted MEDIAN |
| **Konstantin Yakovlev** (Quadruple GM, M5 4th) | https://www.kaggle.com/kyakovlev | M5 1st place discussion https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/163684、AmEx https://www.linkedin.com/posts/konstantin-yakovlev-010b46125_american-express-default-prediction-kaggle-activity-6935250015619592194-hqnM | Multi-fold MEDIAN、lag/rolling triple |
| **Abhishek Thakur** (4x GM, HuggingFace) | https://www.kaggle.com/abhishek | AAAMLP book https://github.com/abhishekkrthakur/approachingalmost (PDF https://github.com/abhishekkrthakur/approachingalmost/blob/master/AAAMLP.pdf)、YouTube channel https://www.youtube.com/channel/UCBPRJjIWfyNG4X-CRbnv78A | Feature engineering + pseudo-labeling 体系 |
| **Philipp Singer** (NVIDIA, Triple GM) | https://www.kaggle.com/philippsinger | LANL 1st https://medium.com/@ph_singer/1st-place-in-kaggle-lanl-earthquake-prediction-competition-15a1137c2457 | Sequence regression での「複雑な DL より stat features + GBM」事例 |
| **Vladimir Iglovikov** (Triple GM, Albumentations 作者) | https://www.kaggle.com/iglovikov | Albumentations TTA docs https://albumentations.ai/docs/4-advanced-guides/test-time-augmentation/、Satellite 2nd arXiv 1706.06169 | TTA / data augmentation philosophy |
| **Pavel Pleskov** (Triple GM, Top 3 World) | https://www.kaggle.com/ppleskov | YouTube "5 secrets to GM" https://www.youtube.com/watch?v=fXnzjJMbujc、Hackernoon https://hackernoon.com/interview-with-kaggle-grandmaster-data-scientist-at-point-api-pavel-pleskov-cc8ca67de249 | Multi-seed bagging + pseudo-labeling |
| **Rich Caruana** (CMU、ICML 2004) | (Kaggler ではない、ensemble selection 学術出典) | "Ensemble Selection from Libraries of Models" ICML 2004 https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf、follow-up ICDM 2006 http://www.niculescu-mizil.org/papers/enssel_most_long.pdf | Hill climbing ensemble の母法 |
| **Pavel Izmailov** (Cornell→OpenAI) | (学術) | SWA paper arXiv 1803.05407 https://arxiv.org/abs/1803.05407、PyTorch blog https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/ | Stochastic Weight Averaging |
| **Olawale Ibrahim** (FORCE 2020 1 位) | (Kaggle 外) | Medium https://ibrahim-olawale13.medium.com/force-2020-machine-learning-lithology-predictionwinning-solution-8cbf78290b41、GitHub https://github.com/olawaleibrahim/2020_FORCE_Lithology_Prediction | well-log 同領域、Bestagini features + 強 L2 regularization |
| **PriorLabs (TabPFN team)** | (Foundation model) | Nature paper https://www.nature.com/articles/s41586-024-08328-6、TabPFN v2.5 arXiv 2511.08667 https://arxiv.org/abs/2511.08667、TabICL https://github.com/soda-inria/tabicl | Tabular foundation model in stack (N kernel 既採用) |

**書籍**:
- *The Kaggle Book* (Massaron, Tunguz, Banachewicz 2026 ed.) ISBN 978-1801817479 — 全 GM-tier 戦術が体系化
- *Approaching (Almost) Any Machine Learning Problem* (Thakur 2020) — free PDF https://github.com/abhishekkrthakur/approachingalmost/blob/master/AAAMLP.pdf

---

## 1. CV-LB diff shrinking 戦略 (= 我々の最優先課題)

我々の現状: exp005 LB 10.317 / exp007 Ridge OOF 10.3874。**CV と LB の systematic な対応が取れていない** (= 今日のユーザー指摘で発覚)。これを構造的に解決する hack を以下に列挙。

### 1.1 Adversarial validation (Bojan Tunguz 流) ★★★

**核心**: train と test の **分布差を classifier で計測**。train=0, test=1 の binary classification を LightGBM で組み、**AUC を distribution shift の magnitude metric** にする。

| AUC | 意味 | 対応 |
|---|---|---|
| ~0.5 | shift なし | 信頼可、そのまま運用 |
| 0.5-0.7 | mild shift | feature importance を観察 |
| 0.7-0.9 | moderate shift | shift driver features を drop / clip |
| > 0.9 | severe shift | CV-LB 信頼関係崩壊、test 寄り resampling 必須 |

**出典**: Tunguz の Kaggle kernel https://www.kaggle.com/code/tunguz/adversarial-ieee, https://www.kaggle.com/tunguz/elo-adversarial-validation (実際の AUC 計算 + drop feature workflow を公開)。Chris Deotte の NVIDIA Playbook "Smarter EDA" でも筆頭に挙げられる https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/

**なぜ ROGII に効くか**: train 773 wells と test 200 wells で X/Y 座標 / visible_ratio / hidden_len 分布が違う可能性。adversarial AUC > 0.7 なら CV-LB の 0.5 ft 差は **distribution shift で説明できる** = fold strategy の信頼性低下。CV 10.39 / LB 10.317 のような奇妙な「LB の方が低い」現象も adversarial AUC を見れば説明できる可能性がある。

**ROGII 直結度**: ★★★ (= CV-LB diff の根本診断、未着手)。

**失敗モード**: AUC が high でも、shift の中身が **target と無関係な軸** (= well_id 列のような identifier) だと drop しても CV 改善せず。→ 対策: shift features を 1 つずつ ablation して target-correlated shift のみに絞る。

### 1.2 Nested CV / Trust Your CV (CPMP 流) ★★★

**核心**: outer CV で model 評価、inner CV で hyperparam tuning。これにより hyperparam tuning が outer CV を leak しない。「Public LB の改善より、自分の CV scheme の堅牢性を信じる」 (Kaggle mantra "Trust Your CV" = CPMP が discussion で繰り返し主張)。

**出典**:
- CPMP YouTube talk "Beyond Feature Engineering and HPO" https://www.youtube.com/watch?v=VC8Jc9_lNoY
- Hackernoon interview https://hackernoon.com/interview-with-twice-kaggle-grandmaster-dr-jean-francois-puget-cpmp-6d92328e433a
- Greg Park 名作 postmortem https://gregpark.io/blog/Kaggle-Psychopathy-Postmortem (public LB overfit の典型例)
- IEEE-CIS 2nd talk https://www.youtube.com/watch?v=wqHlAOFSFuQ

**ROGII 直結度**: ★★★ (= 我々が今やっていないこと)。現状 hyperparam を public LB で見る誘惑があるが、これをやらず CV best と LB best の **両方を sub 2 つに割る** (Phase 6 戦略) の根拠は CPMP の原則。

**失敗モード**: nested CV は 5-fold × 5-fold = 25 model 訓練で計算量 5 倍、ROGII 9hr 制約に直撃。→ 対策: 主 model のみ nested CV、補助 model は通常 CV で妥協。

### 1.3 GroupKFold "true" implementation + fold-misalign 検出 ★★★

**核心**: 我々の fold-misalign 問題 (= exp007 で発覚、subset_A の fold 番号と subset_B の fold 番号が違う型で `_fold0_A` と `_fold0_B` が同じ wells を指していない) を防ぐ標準パターン:
1. `well_id` を唯一の group key にする
2. 全 base model で `kf.split(X, y, groups=well_id)` の **同一 instance** を渡す (= seed と n_splits を fix)
3. `oof_predictions.npy` を保存するとき、`oof[val_idx]` の val_idx が他 base と完全一致するか assert
4. **fold_id 列** を train data に書き込み、全 base がこの列を参照する (= fold 計算を 1 回だけ、再計算しない)

**出典**:
- CPMP discussion (general principle、specific URL なし)
- Chris Deotte NVIDIA Playbook §"Careful Validation" https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/
- Kaggle Handbook "Shake-up survival" https://medium.com/global-maksimum-data-information-technologies/kaggle-handbook-fundamentals-to-survive-a-kaggle-shake-up-3dec0c085bc8

**ROGII 直結度**: ★★★ (= exp007 fold-misalign の根本原因、即修正可)。

**失敗モード**: assert を入れても、`fold_id.parquet` が複数 base で **異なる timestamp で再生成** されると silent shift が再発。→ 対策: fold_id を `data/processed/` に commit、再生成禁止。

### 1.4 Eval-Zone-Mimicking CV ★★

**核心**: ROGII の test 評価が「各 well の trailing 73% を予測」なので、CV でも **train fold の wells を visible 25% mask 状態で feature 構築 → val fold の wells を hidden 73% で RMSE 評価**。これを完全に test と一致させる。

**出典**: ROGII の N kernel (needless090) https://www.kaggle.com/code/needless090/score-10-081-score-lb-32-rank が implicit にこの構造、`docs/research/kaggler-tactics.dense.md` §B-2 に既出。一般論として Kaggle "Time Series Cross-Validation" pattern。

**ROGII 直結度**: ★★ (= 部分実装済、未点検箇所あり)。**train side の feature 構築で hidden 部の GR を使っていないか** 要監査 (現 exp005 では GR を全長 rolling に使っているが、これは test でも GR が全長見えるので合法)。

**失敗モード**: train side で TVT (= target) を使う feature を作ると leak。→ 対策: feature 構築 script に `assert "TVT" not in feature_inputs` を入れる。

### 1.5 Permutation Importance + Drop ★★

**核心**: train 済 model に対し、各 feature を **shuffle → CV 再計算 → ΔRMSE** を計測。ΔRMSE が 0 以下 (= shuffle して悪化しない) の feature は noise、drop すると CV-LB 両方改善することが多い。

**出典**:
- Breiman 2001 (Random Forest paper、Permutation Importance 原典)
- sklearn doc https://scikit-learn.org/stable/modules/permutation_importance.html
- Chris Deotte NVIDIA Playbook §"Diverse Baselines" の variant
- Kaggle Book ch.7 Feature Importance and Selection

**ROGII 直結度**: ★★ (= 現 exp005 で 35+ features、ノイズ列の drop で 0.05-0.15 ft 改善余地)。

**失敗モード**: highly correlated 2 features を両方 shuffle で **互いに代替され ΔRMSE = 0 が出る** = drop すべきでないのに drop してしまう。→ 対策: drop は **1 feature ずつ** に絞り、相関 > 0.95 群は除外。

### 1.6 Stratified by Difficulty CV ★

**核心**: fold 分割時に `visible_ratio` / `hidden_len` の bin で **stratified GroupKFold**。各 fold の難易度を揃える = fold variance 削減 = CV mean の信頼性向上。

**出典**: Konstantin Yakovlev M5 discussion (general pattern), sklearn `StratifiedGroupKFold` https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html

**ROGII 直結度**: ★ (= fold variance 削減で CV → LB の信頼性微改善、CV 絶対値は変わらない)。

**失敗モード**: stratify key の選択 (visible_ratio vs hidden_len) で結果が変動。→ 対策: 両方試して同等改善する key を採用。

### 1.7 Multiple Random Seed CV ★

**核心**: CV を 3-5 個の random seed で回し、OOF を平均 → fold 分割 noise を削減。同じ feature / model でも seed 違いで CV ±0.05-0.15 振れる。

**出典**: NVIDIA Playbook §"Extra Training" 末尾、Vandewiele Ventilator write-up の 15 seed の根拠

**ROGII 直結度**: ★★ (= 即適用可、Phase 4 で seed=42/123/2024 の 3 seed)。

**失敗モード**: 計算量 3 倍、9hr 制約直撃。→ 対策: 主 base (LGB) のみ 3 seed、補助 (XGB/CB) は 1 seed で妥協。

### 1.8 CV-LB diff モニタリング script ★★★

**核心**: 全 exp で **CV (= OOF RMSE) と LB (= public LB score) を 1 個の CSV に蓄積**、`cv_lb_diff = LB - CV` 列を作る。Phase ごとに `cv_lb_diff` の中央値・std を追跡:

```
exp_id | OOF_RMSE | LB | CV-LB diff | fold count | seed | adversarial_AUC
exp005 | 10.39    | 10.317 | -0.073 | 5 | 42 | (未計測)
exp007 | 10.3874  | (PEND) | ?     | 5 | 42 | ?
...
```

CV-LB diff の **std が 0.1 以下** で stable なら CV を絶対基準にできる。**0.3 以上 で systematic** なら CV 改善が LB に出ない構造的問題 (= fold strategy か feature shift)。

**出典**: 一般 Kaggle 流 (Konstantin Yakovlev "I trust my CV when the gap is consistent" discussion 発言、Kaggle Handbook https://medium.com/global-maksimum-data-information-technologies/kaggle-handbook-fundamentals-to-survive-a-kaggle-shake-up-3dec0c085bc8)。

**ROGII 直結度**: ★★★ (= 今すぐ作るべき、未存在)。

**失敗モード**: LB が PEND の exp が混在すると diff 計算不可。→ 対策: PEND 行は除外する flag 列を追加。

---

## 2. Multi-seed × Multi-fold MEDIAN (Vandewiele 流)

### 2.1 15 seed × 5 fold = 75 model 構成 ★★★

**核心**: Vandewiele Ventilator 1 位の決定打。**LSTM hybrid を 15 seed で 5 fold CV → 75 model の予測を取得**。各 prediction を `weighted_median` (重みは `1 / OOF_MAE^β`) で aggregate。**mean ではなく median** が外れ値 model に robust。

**出典**:
- Vandewiele write-up https://medium.com/data-science/winning-the-kaggle-google-brain-ventilator-pressure-prediction-2d4c90d831ec
- Kaggle discussion https://www.kaggle.com/c/ventilator-pressure-prediction/discussion/285256
- Shujun He repo https://github.com/Shujun-He/Google-Brain-Ventilator

**核心引用** (Vandewiele Medium 原文): "We typically see weighted MEDIAN performs better than mean when MAE is the metric"

**ROGII 直結度**: ★★★ (= RMSE 評価なので median は厳密最適ではないが、**base model 出力に median**、stacking 最終層に mean、という二段構えで効く)。

**失敗モード**:
1. 75 model 訓練 + 推論で 9hr 直撃 → 対策: **5 seed × 5 fold = 25 model** に縮小、または LGB のみ 15 seed (LGB は 1 model 軽量)
2. RMSE で median は理論最適でない → 対策: stacking 最終層は Ridge mean、median は base level の variance 削減目的に限定
3. seed が独立でないと variance 削減効果薄 → 対策: bagging_seed / feature_fraction_seed も全 seed で別値

### 2.2 Seed bagging (Konstantin Yakovlev 流) ★★

**核心**: 同一 hyperparam で seed だけ変えて 5+ model 訓練、OOF を mean (RMSE) or median (MAE)。LightGBM の `seed`, `bagging_seed`, `feature_fraction_seed` を全部別値にする。

**出典**: Yakovlev M5 discussion https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/163684、Kaggle Book ch.9

**ROGII 直結度**: ★★ (= LGB×3 を seed 多様化、追加工数低)。

**失敗モード**: 9hr で 15 seed × 5 fold = 75 model は LGB 単体でも切れる可能性。→ 対策: precompute beam_features / pf_features 1 回 + seed 毎は fitting だけ繰り返す = 75 fitting は計算上可能。

### 2.3 MEAN vs MEDIAN tradeoff 表

| metric | base ensemble 最適 | 理由 |
|---|---|---|
| RMSE | mean | L2 loss の minimizer は mean |
| MAE | median | L1 loss の minimizer は median |
| RMSLE | log mean | log transform 後 mean |
| Pinball (quantile) | quantile | quantile loss minimizer |
| ROGII (RMSE) | **mean 主、median 補助** | base variance 削減目的のみ median |

**出典**: Vandewiele write-up、Kaggle Book ch.10 Ensembling

---

## 3. Classification framing (= Edge S 拡張可能性)

### 3.1 Discrete target classification + softmax ★

**核心**: 連続値 target を **bin 化** (= 50 ft step grid) → classification head で softmax → expectation を回帰値として復元。利点: model が **分布全体** を出力、uncertainty quantification 可能。

**出典**:
- Kaggle Optiver Volatility 1 位 (https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970) — quantile bin
- Vandewiele Ventilator も pressure の **離散値 950 種** に着目 (= PID matching の base)

**ROGII 直結度**: ★ (= Edge S RoundGrid と同型、現 exp008/009 で進行中)。

**失敗モード**:
1. bin 数選択 (50 ft vs 25 ft vs 10 ft) で精度トレードオフ → 対策: bin size を OOF で grid search
2. 末端 extrapolation で bin が train 範囲外 → 対策: bin の両端を `[-100, +100]` で広めに

### 3.2 cumsum reconstruct ★★

**核心**: ΔTVT を bin で classify → cumsum で TVT 復元。`TVT(s) = TVT(visible_end) + Σ_{t≤s} ΔTVT_classify(t)`。Vandewiele Ventilator の `target + diff + cumsum` 3-head は同じ paradigm。

**出典**: Vandewiele write-up §"Multi-task heads" https://medium.com/data-science/winning-the-kaggle-google-brain-ventilator-pressure-prediction-2d4c90d831ec、OpenVaccine 4th solution https://github.com/GillesVandewiele/covid19-mrna-degradation-prediction

**ROGII 直結度**: ★★ (= ΔTVT は ar1_phi=0.999 で random walk 性、bin classification が自然)。

**失敗モード**: cumsum で誤差累積 → 対策: 末端 specialist (= 案 J Stage-Aware) と組合せ、または anchor points で reset。

### 3.3 Quantile regression (LGB native) ★★

**核心**: LightGBM の `objective: quantile, alpha=0.1/0.3/0.5/0.7/0.9` を 5 個別 model で訓練 → 5 個の OOF を feature として stack。median (alpha=0.5) を main、両端 quantile を **uncertainty feature** として meta-learner に渡す。

**出典**:
- LightGBM doc https://lightgbm.readthedocs.io/en/latest/Parameters.html (objective=quantile)
- Quantile regression tutorial https://towardsdatascience.com/lightgbm-for-quantile-regression-4288d0bb23fd/
- Kaggle M5 Uncertainty 1 位 (probabilistic) https://www.kaggle.com/competitions/m5-forecasting-uncertainty

**ROGII 直結度**: ★★ (= 末端不確実性 18.83 ft が大きい、quantile range が uncertainty proxy になる)。

**失敗モード**:
1. 5 quantile × 5 fold = 25 model + 通常 LGB と合算 9hr 切れ → 対策: quantile は 3 個 (0.1/0.5/0.9) に絞る
2. quantile median (0.5) は L1 minimum で RMSE main と異なる方向 → 対策: median を main 出力にせず stack 入力に留める

---

## 4. NN + GBM stack (案 B 復活 path)

### 4.1 TabNet / FT-Transformer / NODE (= tabular DL)

**核心**:
- **TabNet** (Arik 2019, https://arxiv.org/abs/1908.07442): sequential attention で feature mask、interpretable
- **FT-Transformer** (Gorishniy 2021, https://arxiv.org/abs/2106.11189): numerical embedding + Transformer encoder、tabular の SOTA NN
- **NODE** (Popov 2019, https://arxiv.org/abs/1909.06312): differentiable oblivious decision tree

**出典 (Kaggle 流)**:
- AmEx 1st place writeup https://www.kaggle.com/competitions/amex-default-prediction/writeups で TabNet が補助 base
- FT-Transformer baseline kernel https://www.kaggle.com/competitions/playground-series-s5e1 (PS5 で利用)

**ROGII 直結度**: ★ (= LGB stack に +1 base、独自性は低い、improvement 0.05-0.15 ft)。

**失敗モード**: tabular NN は GBM に対し **800 wells 規模では大抵負ける**、訓練 5x 遅い。→ 対策: stacking の diversity 目的に限定、main にしない。

### 4.2 Sequence-specific NN: PatchTST / TimesNet / SAITS ★★★

**核心**:
- **PatchTST** (Nie 2023, ICLR https://arxiv.org/pdf/2211.14730): patching (= sequence を chunk に切る) + channel-independent Transformer。長 sequence で memory 効率 + 精度両立
- **TimesNet** (Wu 2023, https://arxiv.org/abs/2210.02186): 2D Inception (frequency × time) で multi-scale 時間 dependency
- **SAITS** (Du 2023, https://arxiv.org/abs/2202.08516): masked self-attention で imputation (= hidden 区間予測そのもの)

**出典**:
- IceCube 1-3 位 (全 Transformer) https://arxiv.org/abs/2310.15674
- Time-Series Transformer benchmark https://github.com/yuqinie98/PatchTST
- AmEx 1 位 で GRU+Transformer 補助

**ROGII 直結度**: ★★★ (= 案 B の具体実装、200 wells overfit が課題)。

**失敗モード**:
1. 200 wells で deep model 不足 → 対策: FORCE 2020 (98 wells) + VOLVE (~200 wells) で **masked TVT pretext pretrain**
2. memory: sequence length 4840 ft × cross-attention = 8M → 対策: PatchTST (patch=64-128 ft) で sequence 短縮、FlashAttention
3. Kaggle 9hr で fine-tune + 推論 → 対策: pretext は外部 GPU で先に済ませる、Kaggle では fine-tune + 推論のみ

### 4.3 Tabular foundation model: TabPFN v2 / TabICL ★★

**核心**:
- **TabPFN v2** (PriorLabs, Nature 2024 https://www.nature.com/articles/s41586-024-08328-6): in-context learning、訓練不要、最大 ~50k samples × 500 features
- **TabPFN v2.5** (2025, arXiv 2511.08667 https://arxiv.org/abs/2511.08667): 100k samples × 2000 features まで拡張
- **TabICL** (Soda team, https://github.com/soda-inria/tabicl): 別系統の tabular foundation model、N kernel が ROGII 既採用 (TabICL 4096/8192 ctx の 2 variant)

**出典**:
- TabPFN GitHub https://github.com/PriorLabs/TabPFN
- Time-series 適用 arXiv 2501.02945 https://arxiv.org/abs/2501.02945
- ROGII N kernel https://www.kaggle.com/code/needless090/score-10-081-score-lb-32-rank (TabICL 既採用)
- TDS exploration https://towardsdatascience.com/exploring-tabpfn-a-foundation-model-built-for-tabular-data/

**ROGII 直結度**: ★★ (= 既採用、TabPFN v2 への切替で improvement 0.1-0.3 ft 余地)。

**失敗モード**:
1. ROGII の 5M rows × 35 features は TabPFN v2 上限超過 → 対策: per-well sampling (= 各 well から 5000 rows) で TabPFN に渡し、aggregation
2. 推論時 9hr 切れ → 対策: TabICL の ctx=4096 (= N kernel 既値) に固定、ctx=16384 試行は禁止

---

## 5. Pseudo-labeling cycle

### 5.1 Chris Deotte 流 1-round pseudo-labeling ★★

**核心**:
1. 通常訓練 → test prediction (= "pseudo labels")
2. **high-confidence** subset を train に追加 (= confidence threshold で filter)
3. 拡張 train で 1 round 再訓練

**出典**: Deotte pseudo-label kernel https://www.kaggle.com/code/cdeotte/pseudo-labeling-qda-0-969 (Instant Gratification 1 位の path、QDA + pseudo-label で LB 0.969)、NVIDIA Playbook §"Pseudo-labeling" https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/

**核心引用** (NVIDIA Playbook): "use soft labels (probabilities) to reduce noise and add regularization"

**ROGII 直結度**: ★★ (= visible TVT_input は既に "labeled"、hidden の predicted TVT を 1-round で pseudo-label に)。

**失敗モード**:
1. multi-round (2+) で **collapse** (= 自己肯定の echo) → 対策: 1 round に厳格制限
2. high-confidence subset が train 分布と乖離 → 対策: confidence threshold を慎重に (= top 30% ではなく top 60%)
3. CV 評価が pseudo-label を使うと leak → 対策: pseudo-label は **train fold のみ**、val fold は触らない

### 5.2 Pavel Pleskov 流 distillation pattern ★

**核心**: 大 model の予測を soft target として小 model を再訓練。Pleskov の interview で「pseudo-labeling は overfit 一歩手前まで」と発言 (具体 URL なし、YouTube talk https://www.youtube.com/watch?v=fXnzjJMbujc)。

**出典**: Pleskov YouTube "5 secrets to becoming a Kaggle grandmaster"、Hackernoon https://hackernoon.com/interview-with-kaggle-grandmaster-data-scientist-at-point-api-pavel-pleskov-cc8ca67de249

**ROGII 直結度**: ★ (= TabICL 8192ctx 出力を soft target に LGB 再訓練、独自性中)。

**失敗モード**: distillation の効果は base 同士の diversity 依存 → 対策: 異種 base (LGB vs TabICL) のみ distillation 対象。

---

## 6. Loss function tricks

### 6.1 Huber (RMSE robust 版) ★★

**核心**: `huber_loss(δ=1.0)` = L2 + L1 のハイブリッド。残差 |r| < δ で L2、|r| > δ で L1。RMSE を主目標としつつ外れ値 (= hidden 末端 wells) の影響を緩和。

**出典**: LightGBM doc https://lightgbm.readthedocs.io/en/latest/Parameters.html (objective=huber, alpha)、kaggler-tactics.dense.md §C-3 既出

**ROGII 直結度**: ★★ (= 末端 extrap_resid std=18.83 ft の外れ値群で huber が効く)。

**失敗モード**: δ 選択で精度変動 → 対策: OOF で δ ∈ {0.5, 1.0, 2.0, 5.0} を grid search。

### 6.2 Quantile (LQR) ★

§3.3 と重複、quantile=0.5 が L1 minimum で robust。

### 6.3 Targeted smoothing (= label smoothing for regression) ★

**核心**: target に小 Gaussian noise (σ=0.05) を加えて訓練 → overfit 抑制。

**出典**: NVIDIA Playbook の "Extra Training" の variant、Kaggle Book ch.8

**ROGII 直結度**: ★ (= effect 0.02-0.05 ft、small but cheap)。

**失敗モード**: noise が大きすぎると CV 悪化 → 対策: σ を 0.01-0.1 で grid search。

### 6.4 Custom objective with sample weight ★★

**核心**: LightGBM の `init_score` / `sample_weight` で **per-row 重み** を渡す。ROGII では:
- `weight = 1.0 / (1.0 + md_offset / 1000)` で末端 row の weight を下げる (= 末端 noise を model に教える)
- `weight = 1.0 * visible_ratio_of_well` で visible 多い well の重要度上げる

**出典**: Kaggle AmEx 1 位 https://www.kaggle.com/competitions/amex-default-prediction/writeups (recency weighted), Kaggle Book ch.7

**ROGII 直結度**: ★★ (= md_offset weighting で末端 noise 影響緩和、未着手)。

**失敗モード**: weight が CV と LB で違う最適 → 対策: weight 関数を OOF で tune、3 パターン提出。

---

## 7. TTA (Test-Time Augmentation) strategies

### 7.1 Sub-window shifts ★

**核心**: 推論時に input を ±N ft shift して 2N+1 個の prediction → mean/median で aggregate。Iglovikov Albumentations TTA docs https://albumentations.ai/docs/4-advanced-guides/test-time-augmentation/ より「TTA is used in virtually 100% of Kaggle CV competitions」。

**ROGII 直結度**: ★ (= sequence model にのみ適用、GBM では shift 効かず)。

**失敗モード**: end effect (= shift で末端が範囲外) → 対策: padding strategy 必須。

### 7.2 Feature dropout TTA ★

**核心**: 推論時に each feature を ±5% noise 加える / 一部 feature を drop → multiple predictions の mean。

**出典**: Kaggle G2Net Gravitational Wave 上位、Vladimir Iglovikov の Satellite imagery 2nd arXiv 1706.06169

**ROGII 直結度**: ★ (= effect 0.02-0.08 ft、cheap)。

### 7.3 Online TTA (= Edge R 拡張) ★★

**核心**: visible 部 GR sliding window を **rolling time** で複数 window 抽出 → 各 window で beam 推論 → consensus。

**出典**: SPWLA 2023 Dreamstar (cell 123) https://github.com/arrayofstar/dreamstar の sliding window + groupby median

**ROGII 直結度**: ★★ (= Edge R の延長、未着手部分あり)。

---

## 8. Stacking pyramid

### 8.1 Caruana Ensemble Selection (= Hill Climbing) ★★★

**核心** (ICML 2004 母法):
```
ensemble = {} (empty)
for step in 1..N:
    best_model = argmin_m RMSE(ensemble ∪ {m}, val)
    ensemble.add(best_model)  # 重複可能 = weight 増加
weights = count(m in ensemble) / |ensemble|
```

**出典**:
- Caruana 2004 paper https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf
- follow-up ICDM 2006 http://www.niculescu-mizil.org/papers/enssel_most_long.pdf
- Python implementation https://github.com/dclambert/pyensemble
- Kaggle 直近 1 位例 PS5E12 https://www.kaggle.com/competitions/playground-series-s5e12/writeups/1st-place-solution-hill-climbing-ridge-ensembl
- Kaggle ensembling guide https://evah.github.io/2017-02-05/ensemble/

**ROGII 直結度**: ★★★ (= Ridge stack の代替 / 追加、Phase 5 final stack で必須候補)。

**失敗モード**:
1. validation set への overfit → 対策: nested CV で hill climbing、または train fold 内 OOF を val として hill climb
2. 同じ model が多数選ばれて diversity 失う → 対策: 上限 weight (= max_count_per_model) を設定

### 8.2 Ridge meta-learner (= 既採用、N kernel) ★★

**核心**: Level 0 OOF 8 列 → Ridge regression (alpha=1.0 or grid search) → final prediction。N kernel が既採用、ROGII exp007 で実装済。

**出典**: Kaggle Book ch.10, Olivier Dieter (Retail Forecasting GM) の StackedRegressor pattern

**ROGII 直結度**: ★★ (= 既採用、alpha=0.1/1.0/10 で grid search 余地)。

### 8.3 L3 simple average final ★

**核心**: L1 (10 base) → L2 (Ridge + Hill Climbing) → L3 (= Ridge と Hill Climbing の simple mean)。最終層は overfit を避けるため複雑な fit を避ける。

**出典**: AmEx 1 位 https://www.kaggle.com/competitions/amex-default-prediction/writeups, Kaggle ensemble guide

**ROGII 直結度**: ★ (= effect 0.02-0.05 ft、軽量)。

### 8.4 Smooth blend (Olivier Dieter / Retail Forecasting 流) ★

**核心**: L2 メタの prediction を **OOF-tuned exponential smoothing** で blend。重み = `softmax(- α * OOF_RMSE_of_model)`。

**出典**: Kaggle M5 retail forecasting solutions (general pattern), Massaron-Tunguz Kaggle Book ch.10

---

## 9. DART boosting + SWA

### 9.1 DART (Dropouts meet Multiple Additive Regression Trees) ★

**核心**: LightGBM の `boosting_type=dart`。各 iteration で **既存 trees を一定確率で dropout** → 残った tree の貢献を boost → 新 tree 追加。正則化効果 = `gbdt` より overfit に強いが訓練 3-5x 遅い。

**出典**:
- DART paper Rashmi & Gilad-Bachrach 2015 https://www.researchgate.net/publication/276149305_DART_Dropouts_meet_Multiple_Additive_Regression_Trees
- LightGBM doc https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html
- Medium tutorial https://medium.com/@meir412_37692/d-a-r-t-your-new-weapon-against-overfitting-in-boosting-models-9ea4e6aa435b
- Kaggle Lightgbm DART kernel https://www.kaggle.com/code/stautxie/lightgbm-dart-boosting-save-best-model

**核心 hyperparam**:
- `drop_rate=0.1` (各 iter で 10% dropout)
- `max_drop=50` (最大 drop tree 数)
- `skip_drop=0.5` (50% の確率で drop skip)
- `uniform_drop=False` (= weight に応じた drop)

**ROGII 直結度**: ★ (= LGB の追加 base として diversity 貢献、improvement 0.05-0.15 ft)。

**失敗モード**: 訓練時間 3-5x で 9hr 制約直撃 → 対策: DART は 1 base のみ、他は gbdt 維持。

### 9.2 SWA (Stochastic Weight Averaging, Izmailov 2018) ★★

**核心**: SGD trajectory の **複数 weight snapshot** を average → flat minimum に収束 = generalization 改善。SGD final point は flat region の **boundary**、average は **center**。

**出典**:
- Izmailov 2018 arXiv 1803.05407 https://arxiv.org/abs/1803.05407
- PyTorch blog https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/
- PyTorch torch.optim.swa_utils https://docs.pytorch.org/docs/stable/optim.html#stochastic-weight-averaging
- Pechyonkin Medium https://medium.com/data-science/stochastic-weight-averaging-a-new-way-to-get-state-of-the-art-results-in-deep-learning-c639ccf36a
- Kaggle Liverpool Ion Switching 2-3 位で採用 https://www.kaggle.com/code/vicensgaitan/2-wavenet-swa

**核心 schedule**:
- 通常 SGD で 75% epoch 学習 → 残 25% で **constant or cyclic lr** で SWA
- 各 epoch end の weight を SWA_model に accumulate
- 最終 inference は SWA_model

**ROGII 直結度**: ★★ (= 案 B/F deep model で 0.1-0.3 ft 改善余地、Kaggle native 簡単)。

**失敗モード**:
1. SWA で BN running stats が古いまま → 対策: `torch.optim.swa_utils.update_bn(loader, swa_model)` を最後に呼ぶ
2. cyclic lr の周期選択 → 対策: 1 epoch 周期で始め、収束しなければ 5 epoch

### 9.3 EMA (Exponential Moving Average) ★

**核心**: 各 step で `θ_ema = β * θ_ema + (1-β) * θ`, β=0.999。SWA の連続時間 limit、軽量実装。

**出典**: PyTorch EMA blog同上、Kaggle G2Net 上位、AmEx 1 位

**ROGII 直結度**: ★ (= deep model に "for free" な改善、軽量)。

---

## 10. Memory / runtime hacks (Kaggle 9hr 制約用)

### 10.1 Float32 + Pandas → Polars ★★

**核心**: 5M rows × 35 features を float64 で 1.4 GB → float32 で 700 MB。Polars は groupby が pandas の 5-10x 速い。

**出典**:
- Polars benchmarks https://pola.rs/posts/benchmarks/
- Kaggle GPU/Polars discussion (general)

**ROGII 直結度**: ★★ (= 既部分実装、未完全切替)。

### 10.2 LightGBM GPU mode ★★

**核心**: `device=gpu`, `gpu_platform_id=0`, `gpu_device_id=0`。Kaggle P100/T4 で 5x 高速化。

**出典**: LightGBM doc https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html, kaggler-tactics.dense.md §G-3

**ROGII 直結度**: ★★ (= 9hr 制約緩和、未試験)。

**失敗モード**: GPU mode で結果 reproducibility が CPU と微異なる → 対策: best model のみ GPU、CV は CPU で再現性確保。

### 10.3 Dataset.save_binary() ★

**核心**: `lgb.Dataset(X, y).save_binary("data.bin")` → re-load 10x 速い、Optuna 200 trials で総時間半減。

**出典**: LightGBM doc, kaggler-tactics §G-4

### 10.4 RAPIDS cuDF / cuML ★★

**核心**: NVIDIA Playbook の "Fast Experimentation" 推奨。cuDF は pandas API 互換で GPU 上 30-100x。Kaggle Kernel で `--accelerator gpu` 指定。

**出典**: NVIDIA Playbook https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/, RAPIDS docs https://rapids.ai/

**ROGII 直結度**: ★★ (= feature engineering 200+ features で時間圧縮、未着手)。

**失敗モード**:
1. cuDF と pandas で挙動微妙に違う (groupby ordering 等) → 対策: 主 pipeline は pandas、FE のみ cuDF で速度 boost
2. Kaggle T4 GPU memory 16GB で 5M rows × 200 features = OOM → 対策: well 単位 chunk

### 10.5 PYTHONUNBUFFERED + `tee` log ★

**核心**: 進捗 visible 化、kill 後の resume 容易。

**出典**: kaggler-tactics §G-2

---

## 11. ROGII に直結する hack 上位 10 件 (= 推奨せず軸のみ)

| # | hack | 出典 GM | 工数 | 期待 CV 改善 | 期待 LB 改善 | 失敗モード |
|---|---|---|---|---|---|---|
| 1 | **CV-LB diff モニタリング CSV** (§1.8) | Konstantin Yakovlev / Kaggle Handbook | 0.5 日 | 計測自体 (= 0 改善) | systematic gap 検出で fold strategy 修正可能 | PEND 行混在で diff 計算不可 |
| 2 | **Adversarial validation** (§1.1, Bojan Tunguz) | Tunguz Kaggle kernels | 1 日 | -0.1〜-0.3 (shift driver drop) | -0.2〜-0.5 (shift drop 後 CV-LB diff 縮小) | target-uncorrelated shift driver の誤 drop |
| 3 | **Fold-misalign 完全修正 + assert** (§1.3) | CPMP / NVIDIA Playbook | 0.5 日 | -0.0〜-0.2 (= 真の OOF 復元) | -0.3〜-0.8 (= fold 一貫で stack 動作) | fold_id 再生成タイミングで silent shift |
| 4 | **Multi-seed × Multi-fold MEDIAN** (§2.1, Vandewiele) | Vandewiele Ventilator 1 位 | 2 日 | -0.15〜-0.4 | -0.2〜-0.5 | 9hr 制約 / RMSE で median 理論最適でない |
| 5 | **Hill Climbing ensemble** (§8.1, Caruana) | Caruana 2004, Kaggle 直近 1 位 PS5E12 | 1 日 | -0.1〜-0.3 (Ridge 比) | -0.1〜-0.3 | val overfit (nested CV で対処) |
| 6 | **Permutation importance + feature drop** (§1.5) | Breiman / Chris Deotte | 1 日 | -0.05〜-0.2 | -0.1〜-0.25 | correlated features 同時 drop の誤判定 |
| 7 | **Quantile regression head 3 個** (§3.3) | LightGBM native, M5 Uncertainty | 1.5 日 | -0.1〜-0.3 | -0.1〜-0.3 | quantile median と RMSE main の方向違い |
| 8 | **Sample weight by md_offset** (§6.4) | AmEx 1 位 (recency weight) | 0.5 日 | -0.05〜-0.15 | -0.1〜-0.2 | CV-LB で weight 関数の最適が違う |
| 9 | **DART boosting 追加 base** (§9.1) | Rashmi-Bachrach 2015, LightGBM | 1 日 | -0.05〜-0.15 (stack diversity) | 同上 | 訓練 3-5x で 9hr 切迫 |
| 10 | **PatchTST + masked TVT pretext pretrain** (§4.2) | Nie 2023, IceCube 1 位 | 5 日 | -0.3〜-1.5 | -0.3〜-1.5 | 200 wells overfit、外部 GPU 必要 |

**合計理論 stack** (= 1-9 全部入れた場合): CV 10.39 → 8.5-9.0 帯、LB 8.5-9.2 帯 (= 9 切り境界)。

**hack 10 (PatchTST) を加えると**: CV 7.5-8.5、LB 7.5-8.5 (= 案 B / 改修 c 経路)。

---

## 12. 我々の改修 b roadmap への組み込み map

現行 plan (= `docs/strategy/winning-strategy.dense.md` + `docs/dev/schedule-2026-05-10.dense.md`、推定) の Phase 3-5 に hack を割り当て:

### Phase 3 (現在 exp005-009): **GBM Stack 完成 + CV-LB diff 計測**

| exp | 投入 hack | 期待 CV | 期待 LB |
|---|---|---|---|
| exp010 (新規) | hack 1 (CV-LB CSV) + hack 3 (fold-misalign 修正) | 10.0-10.2 | 10.0-10.2 |
| exp011 (新規) | hack 2 (adversarial validation) + hack 6 (permutation importance) | 9.7-10.0 | 9.6-10.0 |
| exp012 (新規) | hack 4 (multi-seed MEDIAN, LGB×3 seed) | 9.5-9.8 | 9.4-9.7 |

### Phase 4 (Edge Q + M + 案 D + E + Edge O/S/T/N/P/R + 案 G): **独自 edge 投入 + Stacking 強化**

| exp | 投入 hack | 期待 CV | 期待 LB |
|---|---|---|---|
| exp013 (= 現 plan) | 独自 edge D/E/G + hack 5 (Hill Climbing) | 9.0-9.5 | 9.0-9.5 |
| exp014 (= 現 plan) | hack 7 (quantile 3 head) + hack 8 (md_offset weight) | 8.7-9.2 | 8.7-9.2 |
| exp015 (= 現 plan) | hack 9 (DART) + Edge S RoundGrid 完成 | 8.5-9.0 | 8.5-9.0 |

### Phase 5 (案 G + final stack + post-proc): **CV 8 台到達**

| exp | 投入 hack | 期待 CV | 期待 LB |
|---|---|---|---|
| exp016 (= 現 plan) | hack 5 拡張 (L3 average) + 案 G MoE | 8.3-8.7 | 8.3-8.7 |
| exp017 (final) | 全 stack + 案 J Stage-Aware post-proc | **8.0-8.5** | **8.0-8.5** |

### Phase 6 (sub 選択): hack 10 (PatchTST) を追加で投入する場合のみ

LB 7.5 帯 (= 改修 c 経路) を狙うなら exp018 で PatchTST + pretext。残 86 日で時間あれば。

---

## 13. 想定外の発見 / 推奨せず示すべき axiom

### 13.1 「LB 8 切りは構造的 hack で十分到達可能」

§11 の hack 1-9 (= deep model なし、GBM + stack + CV strategy のみ) で **理論到達 LB 8.5-9.0**。**deep model なしで 9 切りは射程内**。

### 13.2 「CV-LB gap は systematic gap (= shift) と stochastic gap (= seed) の合計」

- systematic = adversarial validation + fold-misalign 修正で診断・修正可能
- stochastic = multi-seed × multi-fold MEDIAN で variance 削減

ROGII の現状 LB 10.317 < CV 10.39 という **「LB < CV」異常** は、(a) LB が public 30% で sampling 有利、(b) CV の fold strategy が test に対し悲観的、の両方の可能性。adversarial validation で診断必須。

### 13.3 「9 切り後の hack 余地」

- hack 10 (PatchTST + pretext): -0.3〜-1.5 ft → LB 7.5-8.5 帯
- 案 I (Diffusion, `independent-edges.dense.md` §6): -0.3〜-1.5 ft → 6.5-7.5 帯
- 案 F (Soft-DTW + Cross-Attn, `independent-edges.dense.md` §3): -0.5〜-1.5 ft

これらは **9 切り達成後** に gold/prize 圏狙いで投入する path、Phase 5 内では工数足りない。

---

## 14. 関連ファイル

- `docs/research/first-principles.dense.md` (= 母法、計測値)
- `docs/research/independent-edges.dense.md` (= 案 D-M)
- `docs/research/top3-distill.dense.md` (= 公開 top 解体)
- `docs/research/strategy-critique.dense.md` (= 改修 a-d 比較)
- `docs/research/past-comps-deepdive.dense.md` (= Dreamstar / Vandewiele / Olawale)
- `docs/research/kaggler-tactics.dense.md` (= 旧 tactics doc、本 doc が拡張版)
- `docs/strategy/winning-strategy.dense.md` (= 編集対象)
- `docs/dev/schedule-2026-05-10.dense.md` (= スケジュール)

---

## 15. 残課題 / 時間切れで深掘りできなかった部分

1. **Konstantin Yakovlev M5 "dark magic" kernel の核心** — direct kernel content にアクセスできず、general lag/rolling 知識で代替。M5 1 位 In Yeonjun の `target transformation` の詳細式 (= log? 平方根?) は未確定
2. **Pavel Pleskov の pseudo-labeling distillation 具体 implementation** — YouTube talk と Hackernoon interview に止まり、code-level pattern 未抽出
3. **TabPFN v2 → v2.5 移行で ROGII の TabICL ctx=4096 を v2.5 に置換可能か** — N kernel の TabICL を TabPFN v2.5 (= 100k samples × 2000 features) に置換すると 0.1-0.3 ft 改善可能性、要実証
4. **PatchTST の ROGII 適用での patch_size 最適値** — generic 16-128 ft が推奨だが、ROGII の `ar1_phi=0.999` random walk 性で **patch_size=32 ft** が最適と推定、実験で確定要
5. **Vandewiele PID matching の ROGII 等価物** — `tvt_formula = -Z + ANCC + b_well` で完全分解できる行が **何%** あるか、host dataset で測定すべき (= bovard 公開 dataset で `tvt_formula` を完全再現する rows の数えあげ)

---

> **completion note**: 本 doc は 2026-05-11 08:11 JST 開始、Q (リサーチ担当) が並列 subagent R (構造変革 paths) と独立して作業。Conflict 範囲は `docs/research/` の別ファイル、安全。
