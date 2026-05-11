# CV 戦略 4 候補 (C1-C4) — 設計書

> task: `.criteria/kaggle-rogii-cv-strategies-2026-05-11.yaml`
> branch: `feat/cv-strategies-final-2026-05-11`
> ユーザー指示 (2026-05-11):
> 1. 「local CV が LB と相関する、 またかつデータとタスクの本質に沿うようなデータ分割戦略を見つけたうえで、 優勝できると思っている手法の CV が本当に今より改善しているのかを検証しようよまずは」
> 2. 「やるべきことをやるべきように やってくれ 数理的見地に立って」
>
> GM 原則対応: Bestfitting「Good CV is half of success」 + Greg Park「Trust your CV not LB」 + Chris Deotte 警告 #4 (= ratio drift) + #8 (= Day 0 で CV 確定)

---

## 0. なぜ CV 戦略再構築が必要か (= 数理本質)

### 0.1 H10 = Jensen lower bound 天井 (= exp010 起源)

`docs/dev/2026-05-11-h10-postmortem-and-fold-reform.dense.md` で確定:

exp007 per-fold valid RMSE = [10.06, 9.20, **12.07**, 10.55, **11.67**]、 σ_fold = **1.175 ft**、 σ_base = 0.121 ft、 比 **9.7 倍**。

Jensen の不等式から:
$$\text{CV (OOF RMSE)} = \sqrt{\overline{\text{RMSE}^2}} \approx \sqrt{\overline{\text{RMSE}}^2 + \sigma_{\text{fold}}^2}$$

exp007: $\overline{\text{RMSE}} \approx 10.71$、 $\sigma_{\text{fold}} \approx 1.175$ → **CV ≈ 10.77 ft が天井**。

= **base model improve だけでは CV を 10.77 ft 以下に下げられない**。 fold 構造改革で σ_fold を縮小しないと、 LB 9 切り (= base improve で CV ratio 1.0 と仮定すれば LB 8.x 必要) は数理的に不可能。

### 0.2 test 3 wells の本質と plain GroupKFold の rep 失敗

`docs/research/2026-05-11-test-distribution.dense.md` の F1-F8 (= test 3 wells を deepest EDA 130+ metric で実測):

- **F1**: b_ANCC cluster は 2 種 (11373×2、 11855×1)
- **F2**: typewell geology class は 2 modal (10 class×2、 6 class×1)
- **F3**: visible_ratio 20-33% (= w1 が 20% で最難)
- **F4**: n_visible は 1442/1545/2083 = p11/p25/p96 で散らばる
- **F5**: GR signal は w2 だけ extreme outlier (= gr_ac50 p100、 gr_mean p2)
- **F6**: tail dTVT 全 +1 (= train は 50/50、 test は同方向)
- **F7**: typewell_geology_n_labeled は test 全件 p6-8 (= minority)
- **F8**: typewell content-hash duplicate 13 groups (= 34 wells) は同 fold 必須

plain GroupKFold (= 既存 `make_well_folds`) は random shuffle で 5-fold = 各 fold val ≈ 155 wells が train 分布の random sampling。 test 3 wells の特殊性 (= b_cluster 2 種、 GR extreme 1 件、 typewell 2 modal) を rep していない = **CV-LB gap の構造的原因**。

= 単に「validate して数値を信じる」 のではなく、 **test 分布に沿った val 構築** が GM section 1.1「Good CV is half of success」 の本質。

### 0.3 既存 exp010 = fold 構造改革の第 1 歩

`feat/phase-6-exp010-fold-reform` (= cherry-pick済) で:
- v2 = `build_edge_q_folds` (= Edge Q typewell hash GroupKFold) で F8 leak 解消
- v3 = `build_stratified_edge_q_folds` (= single-key stratified、 default `tw_gr_resid_std` quartile) で σ_fold 0.526 → 0.303 = **42% 削減** smoke で確認

ただし:
- single-key stratify は F1-F8 のうち F5 (= GR difficulty proxy `tw_gr_resid_std`) を主に rep、 F1 (b_cluster) + F3 (visible_ratio) を直接 stratify していない
- adversarial validation も kernel inline のみで src 統合は未

= **C1-C4 で 4 構造原理を出し揃え、 全 SCORED で LB との相関を測って 1 つ確定**、 が次手。

---

## 1. 4 CV 候補 — 構造原理が異なる

### 1.1 C1: pseudo-test fold via kNN (= test 分布再現 paradigm)

#### 1.1.1 数理本質
test 3 wells を per-well feature space (= deepest EDA 32 metric) に embed、 train wells から **kNN 距離で test 類似** wells を抽出して **fold 0 = pseudo-test val** として固定。 残り train wells を 4 fold で round-robin。

#### 1.1.2 距離関数 (= 数理 fragility 防止)
1. **z-score 正規化**: train wells の median / mean / std で各 feature を centering
2. **PCA top-10 次元削減**: 32 metric → 10 dim。 cumulative variance ratio ≥ 0.5 を assertion (= 不足なら ValueError raise = fragility 上限保証)
3. **Euclidean 距離**: 各 test well と全 train wells の距離計算
4. **top-K nearest 集約**: 各 test well の top-K (= default 50) を union → fold 0

#### 1.1.3 出典
- Bestfitting "pseudo-test set" workflow (= top-1 GM の常套手段)
- Chris Deotte adversarial validation writeups (= 別 paradigm だが同思想)
- (kaggle blog) https://medium.com/kaggle-blog/profiling-top-kagglers-bestfitting

#### 1.1.4 実装
`src/rogii/cv.py:build_pseudo_test_fold` + `compute_test_distance_features` helper。
`tests/test_cv_strategies.py::test_c1_*` で no-leak + deterministic + PCA fragility guard 確認。

### 1.2 C2: multi-key stratified Edge Q (= 多軸層化 paradigm)

#### 1.2.1 数理本質
typewell hash group atomic 制約 (= F8) を維持しつつ、 group の majority stratum で **多軸 stratified sampling**。 best case の fold variance 削減比は $\sigma^2 \to \sigma^2 / K$ (= K = stratum 数)。

#### 1.2.2 bin 設計 (= fragmenting 解消)
naive な multi-axis cross product (= visible_ratio × b_cluster × typewell_n_classes × tail_dir = 5 × 67 × 2 × 2 = **1340 bin / 776 wells で fragmenting**) は破綻。

数理的に妥当な bin (= default):
- `visible_ratio`: **5 quantile bin**
- `tw_gr_resid_std`: **4 quantile bin** (= 既存 exp010 v3 と同 key)
- = **20 stratum**、 776 wells で stratum 平均 39 wells (= stratify 機能)

拡張可能:
```python
stratify_specs=[
    ("visible_ratio",       "quantile",    5),
    ("tw_gr_resid_std",     "quantile",    4),
    ("ar1_phi",             "quantile",    2),   # 3rd axis 任意
]
```

#### 1.2.3 出典
- sklearn `StratifiedGroupKFold` (= single-target 専用、 multi-axis は自前)
- exp010 v3 (= `build_stratified_edge_q_folds`、 single-key) の generalization

#### 1.2.4 実装
`src/rogii/cv.py:build_multi_key_stratified_edge_q_folds` + `_bin_well_by_keys` helper。 typewell hash group atomic 制約 (= F8) + 多軸 bin 配分 (= F1+F3+F5 直接 stratify)。

### 1.3 C3: adversarial validation drop (= ML-driven test 同定 paradigm)

#### 1.3.1 数理本質
**Domain adaptation** の H-divergence: target risk ≤ source risk + $d_{\mathcal{H}}(P_S, P_T)$。 adversarial classifier の OOF AUC は $2 \cdot d_{\mathcal{H}}(P_S, P_T) - 1$ に近似 = **train ⇔ test の distribution shift** を ML で定量化。

- AUC ≈ 0.5: shift 軽微 (= adversarial 不要)
- AUC ≥ 0.6: shift 検出 → 下位 q (= default 20%) wells = 「test らしくない train wells」 を sample_weight × 0.5 (= 軟 drop) もしくは fold = -1 (= 硬 drop)

#### 1.3.2 leak guard (= 3 層)
1. **Feature 設計**: visible-only per-well stats のみ (= TVT / TVT_input / target / tvt_range 一切非参照、 `FORBIDDEN_FEATURE_COLS` runtime assert)
2. **Classifier OOF**: 5-fold StratifiedKFold (= train+test 結合の binary label) で確率を OOF 取得 (= over-confident 単 fold predict 防止)
3. **return**: oof_prob[:n_train] のみ caller に返す (= test wells を test wells 自身で reweight しない)

#### 1.3.3 出典
- Chris Deotte adversarial validation writeups (= Kaggle 2020-2024 多用)
- Ben-David et al. (2010) "A theory of learning from different domains"
- exp010 kernel inline 実装 (= compute_adversarial_well_features / fit_adversarial_classifier / apply_adversarial_reweight) を src 移管

#### 1.3.4 実装
`src/rogii/adversarial.py` 新規モジュール。 cv.py から `compute_adversarial_classifier_oof` + `apply_adversarial_drop_fold` を re-export。 fold-level drop は base fold 上に -1 mark を載せる方式 (= base = Edge Q v2 or v3 or C2)。

### 1.4 C4: typewell hash + b_cluster post-hoc balance (= leak 解消 + 偶発偏在検出 paradigm)

#### 1.4.1 数理本質
Edge Q v2 = `build_edge_q_folds` (= typewell hash GroupKFold) で F8 leak は解消するが、 b_cluster (= F1、 67 cluster) が偶発的に fold に偏在する可能性。 post-hoc で fold × b_cluster の **chi-square 独立性検定** を行い、 p-value > 0.05 (= "balanced") を audit。

数学:
$$\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}, \quad E_{ij} = \frac{\text{row}_i \times \text{col}_j}{N}$$

p-value < 0.05 → fold ⊥ cluster の null 仮説棄却 = **偶発偏在 detected** → fold 再生成または stratify 追加。

#### 1.4.2 出典
- Pearson's chi-square test of independence (= 古典統計)
- scipy.stats.chi2_contingency

#### 1.4.3 実装
`src/rogii/cv.py:compute_b_cluster_balance` (= pivot table + p_value 返す audit helper)。 fold function 自体は `build_edge_q_folds` (= 既存 v2) を流用、 post-hoc で audit。

---

## 2. ablation 計画 (= 8 SCORED + RUNNING exp010 × 4 CV)

### 2.1 OOF 再生成 pipeline

`scripts/regenerate_oof.py` (= Phase 2 着手):
1. 各 SCORED exp の training script を fold-aware に rerun
2. karnakbaev pretrained base は各 fold の train side で re-fit (= 30+ min × 5 fold × 4 CV = 10 hr/exp)
3. 自前 base (= Edge Q fold で学習済) は from scratch、 同 fold で OOF 取得
4. OOF を `outputs/oof/cv-lb-correlation/{exp}_{cv}_oof.parquet` に保存

Compute: 9 SCORED (= exp002/003/005/005v2/005v3/006/007/008v2/exp010) × 4 CV + baseline = 45 OOF set。 Local CPU で 80+ hr、 Colab/Kaggle Notebook 4-server 並列で wall clock 20-30 hr。

### 2.2 multi-metric correlation

`tools/measure_cv_lb_correlation.py` で各 CV について:
- **Spearman**: rank-based、 outlier robust (= exp003 LB 17.510 が含まれても safe)
- **Pearson**: 線形相関、 absolute 値 informative
- **LOO 平均 Spearman**: 8 SCORED から 1 件除外 → 残 7 で Spearman、 全 8 件で平均
- **Bootstrap 95% CI lower**: 1000-iter リサンプル、 lower 5% percentile

### 2.3 AC-7 target (= multi-metric 全部 pass)
最低 1 CV で:
- Spearman ≥ 0.7
- Pearson ≥ 0.7
- LOO 平均 Spearman ≥ 0.6
- Bootstrap 95% CI lower ≥ 0.5

= 4 条件全部 PASS が「Good CV is half of success」 の操作的閾値。

### 2.4 composite score (= 最良 CV 確定基準)
$$\text{composite} = \frac{\text{Spearman} + \text{Pearson} + \text{LOO\_mean} + \text{CI\_lower}}{4}$$

最大 composite score の CV を「最良」 として確定。 ただし exp003 (= LB 17.510 outlier) を除外した 8→7 件再評価で **rank reversal が無い** ことを robust 性 check。

---

## 3. 既存 exp010 work との関係

| 項目 | 既存 (= feat/phase-6-exp010-fold-reform) | C1-C4 追加 |
|---|---|---|
| plain GroupKFold | `make_well_folds` | (baseline、 変更なし) |
| Edge Q v2 (= typewell hash GroupKFold) | `build_edge_q_folds` | C4 の primary fold function として流用 |
| Edge Q v3 (= single-key stratified) | `build_stratified_edge_q_folds` | C2 multi-key 拡張の baseline、 単一 key も保持 |
| typewell hash 計算 | cv.py 内 inline | **src/rogii/typewell_hash.py に移管** (= snapshot constants 追加) |
| adversarial validation | kernel inline (= exp010_fold_reform.py L1434-1583) | **src/rogii/adversarial.py に移管** + fold-level `apply_adversarial_drop_fold` 追加 (C3) |
| pseudo-test fold | × | **C1 として新規** (= `build_pseudo_test_fold` + `compute_test_distance_features`) |
| multi-key stratify | × (= single key のみ) | **C2 として新規** (= `build_multi_key_stratified_edge_q_folds` + `_bin_well_by_keys`) |
| b_cluster balance audit | × | **C4 helper として新規** (= `compute_b_cluster_balance`) |

= **既存 work を継承 + 残り 3 paradigm を新規追加**、 数理本質的拡張で「やるべきことをやるべきように」 を満たす。

---

## 4. verification (= 2026-05-11 時点で達成済)

- **AC-1 PASS**: 10 fold-related functions (= 既存 6 + 新規 4) all importable
- **AC-2 PASS**: pytest tests/test_cv_strategies.py 10/10 PASS in 3.08s (= no-leak + balance + F1 rep + F8 leak free + deterministic seed)
- **AC-3 PASS**: typewell hash dup group runtime check で **13 groups / 34 wells** 検出 (= snapshot 一致)
- **AC-4 PASS**: 本 doc が存在
- **既存 exp010 smoke 4/4 PASS**: σ_fold 0.526 → 0.303 (-42%) 再現確認 = no regression

残作業:
- **AC-5**: 45 OOF table parquet (= Phase 2 重い、 Colab/Kaggle 並列必要)
- **AC-6**: docs/dev/2026-05-11-cv-lb-correlation.md (= Phase 2)
- **AC-7**: multi-metric 全部 pass、 最低 1 CV (= Phase 2 後)
- **AC-8**: baseline plain 併記 (= Phase 2 後)
- **AC-9**: 最良 CV 選定の reviewer agent 判定 (= Verify stage)
- **AC-10**: σ_fold bootstrap 比較 (= Phase 2 後)
- **AC-11, AC-12**: 再現性 + 次タスク plan draft (= Phase 2 後)

---

## 5. 最良 CV 確定 — 宣言 (= Phase 2 完了後に埋める placeholder)

> このセクションは Phase 2 (= OOF 再生成 + correlation 測定) 完了後に、 reviewer agent 判定を経て埋める。
> 形式:
>
> ```
> 最良 CV = C? (= 戦略名)
>
> | 軸 | C1 | C2 | C3 | C4 | baseline |
> |---|---|---|---|---|---|
> | Spearman | ... | ... | ... | ... | ... |
> | Pearson | ... | ... | ... | ... | ... |
> | LOO 平均 | ... | ... | ... | ... | ... |
> | Bootstrap CI lower | ... | ... | ... | ... | ... |
> | composite | ... | ... | ... | ... | ... |
> | exp003 除外時 rank | ... | ... | ... | ... | ... |
> | 解釈可能性 | ... | ... | ... | ... | ... |
>
> 選定理由 (= 5 観点):
> 1. composite score 最大
> 2. exp003 除外時も上位 (= robust)
> 3. fold balance 確認済
> 4. 解釈可能性 ◯
> 5. 既存 paradigm との互換性
>
> 次タスク = 「優勝路手法 A-E (= NN paradigm / deepest EDA D1+D4+D9 / Sparse GP M=500 / Per-Well MoE / LLM-driven) を本 CV で測定」 (= kaggle-rogii-winning-candidates-cv-test-2026-05-12)。
> ```

---

## 6. 関連 doc / 出典

- `.criteria/kaggle-rogii-cv-strategies-2026-05-11.yaml` — 本タスクの success contract
- `docs/research/2026-05-11-test-distribution.dense.md` — F1-F8 finding 元
- `docs/dev/2026-05-11-h10-postmortem-and-fold-reform.dense.md` — H10 Jensen lower bound の元
- `experiments/exp010/design.md` — stratified Edge Q + adversarial drop の元設計
- `~/projects/kaggle/CLAUDE.md` — GM mindset 共通 instructions
- `~/.claude/skills/kaggle-grandmaster-mindset/SKILL.md` — GM skill
- Bestfitting interview: https://medium.com/kaggle-blog/profiling-top-kagglers-bestfitting-currently-1-in-the-world-58cc0e187b
- Chris Deotte adversarial validation: https://www.kaggle.com/cdeotte
- sklearn StratifiedGroupKFold: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
