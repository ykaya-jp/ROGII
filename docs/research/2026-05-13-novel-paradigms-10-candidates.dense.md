# 優勝へ向けた **新規 paradigm 10 候補**: 科学的考察 + 論文 base + データ駆動仮説

> 起点: 2026-05-13 user 指示「新たなアイデアも積極的に検討、 当てずっぽうでなく科学的考察、 実験結果・分析対象データからの仮説、 様々な研究論文に基づいて」
> 既存 plan = `~/.claude/plans/wobbly-moseying-petal.md` (= 5 Phase roadmap)、 本 doc は **Phase 拡張候補** を 10 件 ranked で提示
> 親 docs: 既存 12 件 deep dive (= ~2350 行)

---

## Tier S (= 高 impact + 中 risk + 論文 backed 強)

### S1. Conformal Prediction wrapper (= per-row coverage guarantee)
- **数理本質**: 全 base + post-proc の prediction interval を **conformal prediction** で coverage guarantee 化、 per-row uncertainty を Ridge meta blend に **adaptive weight** として加える。 高 confidence row では point estimate、 低 confidence row では neighbor smoothing
- **データ駆動仮説**: test 3 wells の per-row mean_abs 0.68-1.40 ft は **systematic でない、 row-level variance あり**。 Conformal で per-row α-uncertainty 取得すれば「曖昧 row」 のみ smoothing
- **論文**: Vovk 2005 "Distribution-Free Predictive Inference"、 **Gibbs & Candès 2021 "Adaptive Conformal Inference"** (= time-varying uncertainty)、 Romano 2019 "Conformalized Quantile Regression"
- **期待 lift**: -0.05〜-0.15 ft
- **実装**: `src/rogii/conformal.py` 新規 100-150 行、 per-row α=0.10 split conformal → quantile residual → adaptive blend weight

### S2. Gradient-Boosted Meta-Stack (= Ridge 線形限界 break)
- **数理本質**: 現状 Ridge meta blend (= 線形 closed-form positive 制約) → **LightGBM / XGBoost meta** で non-linear blending。 per-row feature (= visible_ratio、 incl_local、 md_since、 b_cluster ID) を **meta-feature として加える**。 row ごとに optimal base weight が決まる
- **データ駆動仮説**: 親 doc § 4 で「hc correction = block-wise selective、 zero-crossings 1.3%、 P90/P50 ratio 2.8-4.3」 = non-linear 1280 row block correction。 これは Ridge では表現不可、 GBM なら自然に学習
- **論文**: Wolpert 1992 "Stacked Generalization"、 Kaggle 上位解法多数 (= 例: Otto Product 2015 1st、 Numerai 上位)
- **期待 lift**: -0.05〜-0.15 ft
- **実装**: `src/rogii/meta_gbm.py` 新規、 既存 Ridge 9-base + per-row features → LGB meta (= 200-300 round、 depth 4-6)

### S3. Causal adjustment by Propensity Score Matching (= train/test distribution mismatch fix)
- **数理本質**: train 770 wells と test 3 wells で **visible_ratio 分布 systematic differ** (= test 20-33% vs train 多様)。 train wells を test-similar wells に **propensity weight** で reweight、 LGB train が test に近い分布で fit
- **データ駆動仮説**: 親 doc § 8 で「test 3 wells すべて incl_tail ≥ 89° の完全水平 bucket」、 train で同 bucket は 159/773 wells (= 20.6%)、 80% の wells は test distribution 外。 これらが overfit に寄与
- **論文**: Rosenbaum & Rubin 1983 "Propensity Score"、 **Pearl 2009 "Causal Inference"**、 Shimodaira 2000 "Covariate Shift"、 Sugiyama 2007 "Direct Importance Estimation"
- **期待 lift**: -0.03〜-0.10 ft
- **実装**: `src/rogii/causal_adjust.py` 新規、 train wells に propensity_score(visible_ratio, incl_tail, slope_tail) を fit、 LGB の sample_weight に inject

---

## Tier A (= 中 impact + 中 risk + 論文 base)

### A1. Variational Sequential Monte Carlo (= Bayesian PF with uncertainty propagation)
- **数理本質**: ANCC + Z dual PF の posterior distribution を **Variational Inference + Sequential Monte Carlo** で full Bayesian、 uncertainty (= particle variance) を Ridge meta blend weight に inject。 高 uncertainty row では PF weight 削減、 低 uncertainty row では PF weight 増
- **データ駆動仮説**: 親 doc § 9 で「ravaghi cell 6 = ANCC PF + Z dual PF」、 ただし single-point estimate のみ。 Bayesian で variance 取れば per-row adaptive weight 可能
- **論文**: **Naesseth 2018 "Variational Sequential Monte Carlo"**、 Lindsten 2014 "Particle Gibbs with Ancestor Sampling"、 Blei 2017 "Variational Inference: A Review"
- **期待 lift**: -0.05〜-0.15 ft
- **実装**: `src/rogii/vsmc.py` 新規、 既存 `run_pf_ancc` + `run_pf_z` を VSMC で拡張 (= 200-300 行)

### A2. RL-driven Hyperparameter Optimization (= context-aware Optuna)
- **数理本質**: ravaghi Optuna 500-trial = **single global** α/τ/w_pf。 RL agent (= PPO or BO with multi-task) で **per-well features → α/τ/w_pf**。 Well-context aware tuning
- **データ駆動仮説**: 親 doc § 7 で「ours は test 3 wells で systematic に異なる方向 bias」 = 1 つの global hyperparam set では捕捉不可、 per-well config が必要
- **論文**: Zoph 2017 "Neural Architecture Search with RL"、 **Andrychowicz 2016 "Learning to Learn by Gradient Descent"**、 Falkner 2018 "BOHB"
- **期待 lift**: -0.05〜-0.15 ft
- **実装**: `src/rogii/rl_hpo.py` 新規、 well-context vector → α/τ/w_pf を PPO agent で学習

---

## Tier B (= 高 impact だが 高 cost or 高 risk)

### B1. TabTransformer + GMLP base (= NN paradigm 5 拡張)
- **数理本質**: Mamba state-space (= plan §3 Phase C-1) の **代替 / 補完**。 TabTransformer (= attention on column features) で context-aware base、 GMLP (= gated MLP) で long-range row interaction
- **論文**: **Huang 2020 "TabTransformer"**、 Liu 2021 "Pay Attention to MLPs"、 Gu 2022 "Mamba: Linear-Time Sequence Modeling"
- **期待 lift**: -0.10〜-0.25 ft
- **コスト**: Colab A100 30-50 hr (= Mamba と同程度)
- **実装**: `src/rogii/nn_tabtransformer.py` 新規 400-500 行、 Phase C-1 と並走 ablation 推奨

### B2. Graph Neural Network on well 3D topology (= 新軸独占)
- **数理本質**: 全 wells (= 773 + 3 = 776) の (X, Y, Z) 中心点を 3D graph node、 距離 < threshold で edge、 GraphSAGE / GAT で feature propagation。 test wells の周囲 train wells から情報を集約
- **データ駆動仮説**: 親 doc § 9 で「物理 inclination prior」 確証、 ただし「近傍 well の TVT pattern」 は未利用。 GNN で 3D 空間 locality を model 化
- **論文**: **Veličković 2018 "Graph Attention Networks"**、 Hamilton 2017 "GraphSAGE"、 Battaglia 2018 "Relational Inductive Biases"
- **期待 lift**: -0.05〜-0.20 ft
- **コスト**: graph build (= cKDTree で K-nearest) + GNN train、 PyTorch Geometric で 10-20h
- **実装**: `src/rogii/gnn_well.py` 新規 300 行

### B3. Diffusion model for TVT trajectory generation (= conditional generative)
- **数理本質**: TVT trajectory (= visible + hidden) を Diffusion model で generative。 condition = (GR signal, anchor, last_known_tvt)、 sampling で multiple trajectories → uncertainty + best-of-N voting
- **データ駆動仮説**: ravaghi の PF は **deterministic single trajectory**、 diffusion で multiple sample = stochastic ensemble (= aleatoric uncertainty 取得)
- **論文**: **Ho 2020 "Denoising Diffusion Probabilistic Models"**、 Song 2021 "DDIM: Denoising Diffusion Implicit Models"、 Karras 2022 "Elucidating the Design Space"
- **期待 lift**: -0.05〜-0.30 ft
- **コスト**: A100 50-100h、 training + inference 大
- **実装**: `src/rogii/diffusion_traj.py` 新規 500-700 行、 Phase C-3 候補

### B4. Geophysical inversion (= analytic 物理 model)
- **数理本質**: TVT = f(GR, ANCC, geology_model, well_geometry) を **解析的 Bayesian inversion** で解く。 forward model = physics (= GR vs lithology、 ANCC vs depth)、 inverse = posterior over hidden TVT
- **論文**: **Tarantola 2005 "Geophysical Inverse Theory"**、 Iglesias 2014 "Bayesian Inverse Problems"
- **期待 lift**: -0.05〜-0.30 ft (= 上振れ大、 ただし geology domain expert 必要)
- **コスト**: domain knowledge 大、 implementation 200-400h
- **実装**: 専門 consultant 必要、 plan D 候補 (= 終盤 risky slot)

---

## Tier C (= 危険、 推奨せず)

### C1. Active Learning + Self-training on test visible (= 危険、 leak risk)
- **数理本質**: test wells の visible 部分を pseudo-labeled として LGB に追加 fit (= test-time training)
- **問題**: ROGII rule で「test visible は inference のみ可、 training 利用は spirit 違反」 の可能性。 ただし host が明示禁止しているかは未確認、 grey zone
- **論文**: Yarowsky 1995、 Lee 2013 "Pseudo-Label"
- **判断**: §11 「rule fix で死なない static approach」 違反 候補、 **回避**

---

## 推奨実装順序 (= 既存 plan §3 への追加 / 置換)

### Phase B 拡張 (= Week 2-3、 既存 Phase B + 新):
- **Phase B-4**: S2 (= GBM meta-stack) → -0.05〜-0.15、 中 cost
- **Phase B-5**: S3 (= Causal adjustment) → -0.03〜-0.10、 低 cost
- **Phase B-6**: S1 (= Conformal Prediction) → -0.05〜-0.15、 中 cost

### Phase C 拡張 (= Week 3-4、 既存 Phase C と並走):
- **Phase C-3**: A1 (= VSMC) → -0.05〜-0.15、 PF 強化
- **Phase C-4**: A2 (= RL HPO) → -0.05〜-0.15、 hyperparam adaptive

### Phase D 拡張 (= Week 4-6、 高 risk upside):
- **Phase D-1**: B1 (= TabTransformer + GMLP) → -0.10〜-0.25
- **Phase D-2**: B2 (= GNN) → -0.05〜-0.20

### Phase E (= 終盤 risky slot):
- **Phase E-1**: B3 (= Diffusion) → -0.05〜-0.30、 dual final submit の risky slot
- **Phase E-2**: B4 (= Geophysical inversion) → 専門 consultant あれば

---

## 累積期待 LB lift (= 5 Phase 全完了 + 新 paradigm 統合)

```
段階                    目標 LB        確信度
exp016-v2 (= Phase A.5  9.5-9.7        70%
exp020 (= Phase A.5)    9.3-9.5        75%
+ Phase B 既存 (aug+kNN) 9.0-9.3       60%
+ Phase B 新 (S1+S2+S3) 8.7-9.0        55%
+ Phase C 既存 (NN seq) 8.3-8.7        35%
+ Phase C 新 (A1+A2)   8.0-8.4        30%
+ Phase D 新 (B1+B2)   7.5-8.0        20%
+ Phase E 新 (B3)      7.0-7.7        10%
─────────────────────────────────────────
LB 1 位 (= 8.966)      85% (= Phase B 新 で射程)
LB 8.0 切り             40% (= Phase C 新 で射程)
LB 7.0 帯 (= 圧勝)     10% (= Phase E B3 + 全 paradigm 統合)
```

---

## 倫理 / §11 「優勝本質性」 commit 確認

- ✅ **数理本質**: 全候補は paper-backed、 数式 derivation 可能、 軽さ-driven でない
- ✅ **rule 耐性**: Tier S/A/B は全て static feature engineering / model architecture、 rule fix で死なない
- ✅ **transduction + induction mix**: paradigm 多様性 = Conformal (= calibration)、 GBM stack (= ensemble)、 VSMC (= inductive Bayesian)、 GNN (= structured prediction)、 Diffusion (= generative)、 Geophysical (= analytic) で **6 paradigm 以上 mix**
- ❌ Tier C #C1 self-training = leak risk で除外
- ✅ **datapoint 価値**: 各 paradigm が独立 ablation 可能、 effect isolate 

---

## 関連 doc

- 既存 plan: `~/.claude/plans/wobbly-moseying-petal.md` §3 (= 5 Phase roadmap base)
- 既存 12 deep dive: `docs/research/2026-05-13-*.dense.md`
- 公開 NB audit: `docs/research/2026-05-11-public-source-audit.dense.md`
- 既存 Phase A〜C 候補: winning-path A/D/F/G sketches
