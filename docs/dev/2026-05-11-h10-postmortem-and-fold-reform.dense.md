# 2026-05-11 H10 Postmortem + Fold 構造改革 plan

> 起源: subagent V (= `docs/data-mining-2026-05-11` branch) 完了報告で **H10 = fold variance σ=1.18 が exp007 type LB を 10.6 帯に数学的固定** を発見。
> 関連: `submission-data-mining.dense.md` (= subagent V 600 行)、`2026-05-11-state-snapshot.dense.md` (= 中央)
> 目的: 9 切り roadmap の **前提変更**を明文化、5/12 plan を root level で refine、subagent Y dispatch text を準備。

## 1. H10 の数学的本質

### 1.1 数値 evidence (= subagent V 抽出)

exp007 自前 4 base per-fold valid RMSE (Kaggle log):
- fold 0: lgb_own0 = 9.86, lgb_own1 = 10.03, lgb_own2 = 10.19, cb_own = 10.16 → mean 10.06
- fold 1: 9.02, 9.14, 9.40, 9.22 → mean 9.20
- **fold 2: 12.05, 12.02, 12.19, 12.03 → mean 12.07** ⚠️
- fold 3: 10.44, 10.64, 10.67, 10.46 → mean 10.55
- **fold 4: 11.67, 11.70, 11.95, 11.35 → mean 11.67** ⚠️

#### 統計
- fold means: [10.06, 9.20, **12.07**, 10.55, **11.67**]
- σ_fold (= fold 間 std) ≈ **1.175** ft
- σ_base (= 同 fold 内 base 間 std) ≈ **0.121** ft
- 比 σ_fold / σ_base = **9.7 倍**

### 1.2 Jensen 下限 (= 数学的天井)

CV (overall OOF RMSE) = sqrt(平均 squared error)。fold ごとに RMSE が異なるとき:
$$
\text{CV} = \sqrt{\frac{1}{K} \sum_{k=1}^{K} \text{MSE}_k} \ge \sqrt{\overline{\text{MSE}}}
$$

別の見方: $\overline{\text{MSE}} = \overline{\text{RMSE}}^2 + \sigma_{\text{fold}}^2$ (= variance + bias decomposition)。
- $\overline{\text{RMSE}} \approx 10.71$
- $\sigma_{\text{fold}}^2 \approx 1.38$
- $\overline{\text{MSE}} \approx 114.7 + 1.38 = 116.1$
- $\text{CV} \approx \sqrt{116.1} \approx **10.77**$

= **base improve だけで CV を下げても σ_fold が dominant、10.77 が天井**。

### 1.3 LB への波及

CV-LB ratio が ≈ 1 (= subagent T path b で fold-misalign 解消した場合) なら、LB ≈ CV ≈ **10.4-10.6 帯**で固定。

= **9 切り (LB 8.x) には fold 構造改革が必須**。

## 2. fold 構造改革 candidates (= 9 切り path 再構築)

### 2.1 candidate A: stratified Edge Q fold (= fold balance)

fold 2/4 で RMSE 12 帯になる原因 = **困難 wells が偏在**。`first-principles per-well-stats.parquet` の `tw_gr_resid_std` や `hidden_len` で stratify、各 fold に困難 wells を分散配置。

- 数学的根拠: stratified sampling で fold variance を $\sigma_{\text{fold}}^2 \to \sigma_{\text{fold}}^2 / K$ (best case) に削減
- 期待 CV 改善: 10.77 → 10.0-10.3 (= σ_fold を 1.18 → 0.5 に圧縮できれば)
- 工数: 0.5 日 (= `src/rogii/cv.py` 内 `build_edge_q_folds()` に stratify logic 追加)

### 2.2 candidate B: adversarial validation drop (= 困難 wells の reweight)

train vs test の distribution shift を adversarial classifier で計測 (Bojan Tunguz 流、`gm-wisdom.dense.md` §1.1)。test と分布が違う wells を:
- (a) drop (= train に含めない)
- (b) sample_weight を 0.5 に reduce
- (c) target-domain pseudo-label (= test visible TVT_input を fit に投入、= Edge T 拡張)

- 数学的根拠: domain adaptation theory、target risk = source risk + divergence penalty
- 期待 CV 改善: 10.77 → 10.2-10.5
- 工数: 1 日

### 2.3 candidate C: weighted fold loss (= 困難 fold の overfit 抑止)

per-fold で early-stopping num_round を **fold 別に調整**。fold 2/4 で num_round 大なら overfit、減らして robust 化。

- 数学的根拠: fold-adaptive model complexity (= Rademacher complexity per-fold)
- 期待 CV 改善: 10.77 → 10.4-10.6
- 工数: 0.5 日

### 2.4 candidate D: fold-by-fold ensemble (= 困難 fold の weight 削減)

5 fold を Ridge meta で同 weight 投入でなく、**per-fold で weight 学習**。fold 2/4 を低 weight、fold 0-3 を高 weight。

- 数学的根拠: bias-variance trade-off で fold variance を ensemble level で削減
- 期待 CV 改善: 10.77 → 10.5-10.7 (= ensemble なので幅小)
- 工数: 0.5 日

### 2.5 統合 = candidate A + B (= 推奨組合せ、subagent Y dispatch)

A (stratified Edge Q fold) で fold balance + B (adversarial validation drop) で test 距離考慮。両方適用で:
- σ_fold 1.18 → 0.4-0.5 推定
- CV 10.77 → 10.0-10.2 推定
- LB 10.677 → 9.5-9.7 推定

= **base improve なし、fold 構造改革のみで -1.0 ft 改善射程**。

## 3. 5/12 sub plan の根本 refine

旧 plan (= state snapshot §3.2): 加算的 inject 純粋路線で 8.x 帯到達

新 plan (= H10 反映): **fold 構造改革を先、加算的 inject はそれ以降**

| sub | layer | LB 想定 (mid) | 工数 |
|---|---|---|---|
| **exp010** (= 5/12 1 sub 目) | **stratified Edge Q fold + adversarial validation drop** (= candidate A + B、自前 4 base 路線継続) | 9.5-9.7 | 1.5 日 (実装) |
| exp011 | exp010 + Edge S + Edge R (= 加算的 inject 復活) | 9.0-9.3 | 0.5 日 |
| exp012 | exp011 + 案 D Kalman (= AR(1) state-space) | 8.5-8.9 | 1 日 |
| exp013 | exp012 + 案 E GP for ANCC (= posterior + variance) | 8.0-8.5 | 1 日 |
| exp014 | exp013 + Edge N (offset well retrieval) + post-proc | 7.7-8.2 | 1 日 |

= **9 切り達成 = exp012-013 で確実視**、自前 4 base 路線を fold 改革で復活させて Top 1 圏到達。

## 4. subagent Y dispatch text (= exp010 実装)

subagent W/X 完了通知 + exp005 v3 / exp008 v2 / exp009 v2 LB 出揃ったタイミングで、以下 text で dispatch:

```
subagent Y mission: exp010 = stratified Edge Q fold + adversarial validation drop を実装
(= H10 verified after data mining、9 切り path 再構築の core)

base: kaggle_kernels/exp009_case_e_edge_o/exp009_case_e_edge_o.py (= subagent T の v3、Huber+hetero+MEDIAN+path b)
新 branch: feat/phase-6-exp010-fold-reform feat/phase-5-edge-r-online から派生

実装 1: stratified Edge Q fold
- src/rogii/cv.py の build_edge_q_folds() を拡張
- stratify key: tw_gr_resid_std (= per-well-stats.parquet) の bin (= quartile)
- 各 fold で困難 quartile が均等に分布するように
- 5 fold で σ_fold を 1.18 → 0.4-0.5 に圧縮目標

実装 2: adversarial validation drop
- train vs test wells を classify する LightGBM (= 1 model)
- AUC > 0.7 なら distribution shift あり、各 train well の test-likelihood を計算
- test-likelihood 低い train wells を sample_weight 0.5 で reduce or drop

ローカル smoke: 5 wells × 200 row で stratified fold + AV drop が動くか確認
kernel push しない (= 中央が exp012 と統合 push 検討)

完了報告: σ_fold 削減幅 + AV AUC + 5/12 LB 想定
```

= subagent Y 工数 1.5 日想定、subagent W/X 完了次第即 dispatch。

## 5. 残課題 / 未消化

| 未消化 | impact |
|---|---|
| subagent V の 残 9 仮説 (H1-H9) | 各々 +0.05〜+0.30 ft 改善寄与可能性、5/13 以降の追加 dispatch 候補 |
| subagent W (学術文献 deeper) との照合 | adversarial validation の最新 paper、stratified fold の theoretical guarantee |
| subagent X (paradigm deeper) との照合 | Mamba/S4 の hidden state per-fold variance reduction 可能性 |
| **H10 verify**: 中央で Pearson correlation matrix + per-id diff + fold balance 計算 | subagent V output (= `tools/mining/submission_data_mining.py`) を別 branch で実行 |

= subagent W/X 完了 → synthesis で H10 + paradigm deeper + 学術文献 を統合 → exp010 dispatch + 中央 verify scripts 実行 = 5/11 23:00 UTC 頃 final freeze。

## 6. 関連 doc

- `docs/research/submission-data-mining.dense.md` (= subagent V、別 branch)
- `docs/research/academic-literature-deeper.dense.md` (= subagent W、別 branch、進行中)
- `docs/research/paradigm-deeper.dense.md` (= subagent X、現 branch、進行中)
- `docs/dev/2026-05-11-state-snapshot.dense.md` (= 中央 snapshot)
- `docs/dev/submission-postmortems.dense.md` (= postmortem doc)
- `docs/research/gm-wisdom.dense.md` §1.1 (= Adversarial Validation Bojan Tunguz 流)
