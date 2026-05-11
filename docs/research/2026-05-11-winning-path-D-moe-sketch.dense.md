# 優勝路手法 D — Per-Well MoE wrapper (= 予備 sketch、 data evidence 駆動)

> 起源: docs/research/2026-05-11-public-source-audit.dense.md § 1.1.1 (= per-row diff 実測)
> 親 plan: kaggle-rogii-winning-candidates-cv-test-2026-05-12
> 数理本質: 全 train wells を 1 model で均一に学習すると b_cluster minority (= 11855) に under-fit。 visible_ratio × b_cluster で gate して per-cluster expert を学習すれば minority に専門化。 expected LB lift -0.3 〜 -0.5 ft (= 大半は 00bbac68 から)。

---

## 1. data evidence (= 数理本質を data で確認)

### 1.1 Top10 vs 我々の per-well delta (= 2026-05-11 実測)

`docs/research/2026-05-11-public-source-audit.dense.md § 1.1.1` で取得:

| well | b_cluster | visible_ratio | mean abs delta | max delta |
|---|---|---:|---:|---:|
| 000d7d20 | 11373 (cluster 37、 train majority) | 27% | 1.49 ft | 5.31 |
| **00bbac68** | **11855 (cluster 52、 train minority)** | **20%** | **3.91 ft** | **10.78** |
| 00e12e8b | 11373 (cluster 37) | 33% | 2.52 ft | 6.36 |

= **00bbac68 で我々が圧倒的に外している**。 LB 9.957 → 9.728 (= -0.229 ft) の lift 余地は 00bbac68 (= 6014 rows = 42.5% of test) に集中。

### 1.2 なぜ 00bbac68 で外れるか (= 数理本質の仮説)

a. **b_cluster 11855 は train で minority** (= train 773 wells 中 何 wells が 11855 か未確定だが、 cluster 52 ID = sort 順 67/2 ≈ 33%-66% 帯のはず、 実測必要)
b. **visible_ratio 20% = 80% hidden** = データ不足の予測難度最大 well
c. **GR signal が 11373 cluster wells と異なる構造の可能性** (= F1-F8 で 00bbac68 は GR moderate、 ただし typewell_n_classes=6 = majority、 これは GR profile が train 平均的)
d. **typewell match が train majority に偏っている可能性** (= karnakbaev base が 11373 cluster wells で過学習)

つまり (a) + (b) の組み合わせ = 11855 minority かつ visible 20% で、 model 全体が学習データ majority (= 11373 + visible 25-30%) に最適化されて 00bbac68 を under-fit。

### 1.3 一般化 (= 全 train wells での同様 issue)

00bbac68 のような構造 (= cluster minority + low visible) を持つ train wells も同様に under-fit のはず。 ablation: train wells を「同 cluster + 同 visible bin」 でグループ化し、 fold-aware OOF RMSE を比較すれば、 minority + low visible wells が高 RMSE を持つことを実証可能。

これは exp007 fold 2/4 で RMSE 12 帯になった H10 finding (= σ_fold 1.18) の **再解釈**: fold 2/4 = minority cluster + low visible が集中していた fold だった可能性。 exp010 stratified Edge Q で fold を均等化 (= σ_fold 0.30) しても、 **minority well 自体の予測 質** は改善せず、 単に「fold 間の variance」 だけ縮小した。 = **fold reform は半分の解、 残り半分 は per-well 専門化** (= MoE)。

---

## 2. Per-Well MoE 設計

### 2.1 architecture

```
            ┌──── expert_majority_high_visible (= 11373 cluster + visible≥25%)
            │       (= LGB×3 + CB on train wells matching this profile)
            │
gate(well) ─┼──── expert_majority_low_visible (= 11373 cluster + visible<25%)
            │
            └──── expert_minority (= 11855 cluster の wells のみ、 + visible bin)
                    (= LGB×3 + CB on minority train wells)

per test well:
  pred = w_maj_high * expert_maj_high(features) +
         w_maj_low  * expert_maj_low(features) +
         w_min      * expert_minority(features)
  where (w_maj_high, w_maj_low, w_min) = gate(visible_ratio, b_cluster)
```

3 expert × per-fold = 15 model fits、 1 fit ≈ 5-10 min = 75-150 min/CV strategy。

### 2.2 gate 関数

```python
def gate(visible_ratio, b_cluster_id):
    # b_cluster_id: 67 unique values (= D1 feature 経由)
    # b_cluster minority = 11855 周辺 (e.g., cluster_id 52, 51, 53 等)
    minority_cluster_ids = {52, 51, 53}  # tentative
    if b_cluster_id in minority_cluster_ids:
        return (0.1, 0.1, 0.8)  # minority expert に高 weight
    if visible_ratio >= 0.25:
        return (0.7, 0.2, 0.1)
    return (0.2, 0.7, 0.1)
```

soft gate (= 連続関数) version:
```python
import numpy as np
def gate_soft(visible_ratio, b_cluster_dist_from_11855):
    # b_cluster_dist_from_11855: |b_ANCC_med - 11855|
    minority_score = np.exp(-b_cluster_dist_from_11855 / 100)  # 11855 周辺で高
    high_vis_score = 1 / (1 + np.exp(-(visible_ratio - 0.25) * 20))
    w_min = minority_score
    w_maj_high = (1 - minority_score) * high_vis_score
    w_maj_low = (1 - minority_score) * (1 - high_vis_score)
    return w_maj_high, w_maj_low, w_min
```

### 2.3 expert training

各 expert は **all train wells で fit、 sample_weight を gate と一致**:
```python
for fold in folds:
    for expert_name in ['maj_high', 'maj_low', 'minority']:
        sample_weight = compute_gate_weights(train_df, expert_name)
        X = features
        y = target
        model = LGB.train(X, y, weight=sample_weight)
        expert[expert_name] = model
```

これで minority expert は minority wells に偏った学習、 majority expert は majority に偏る。

### 2.4 ensemble との接続

MoE prediction を 既存 9-base Ridge meta に **10 番目 (= expert ensemble blend)** として追加:
```python
moe_oof = sum(gate_weights[k] * expert_oof[k] for k in experts)
Sx = np.column_stack([... 9 base ..., moe_oof])
ridge_ex.fit(Sx, y_kb)
```

---

## 3. risks と mitigation

| Risk | Mitigation |
|---|---|
| minority expert が overfit (= cluster 11855 train wells が少なすぎ) | sample_weight の minority 集中度 を CV で tune、 過度な集中で OOF 悪化なら緩和 |
| gate の hand-tuning overfit | 距離関数 (= visible_ratio + b_cluster) は data driven、 hyperparameter は CV で grid search |
| 既存 9-base に対する diversity ゼロ (= 同じ features 同じ target、 ただし sample_weight 違い) | features subset を expert ごとに変える (= minority expert は formation imputed + typewell に集中) |
| compute cost (= 3 expert × 5 fold × 4 CV = 60 fits) | minority expert のみ実装で start、 promising なら 3-expert に拡張 |

---

## 4. 期待 LB lift (= per-row diff から推定)

仮に 00bbac68 で Top10 並みに予測できると:
- 6014 rows × 0² (= perfect) + 8137 rows × current_err² (= 我々の他 well average ≈ 2.0 ft RMSE)
- 全体 OOF RMSE ≈ √((0 + 8137 × 4) / 14151) ≈ 1.51 ft (= simplification 仮定 LB と相関する場合)
- = 我々 LB 9.957 → 1.51 ft くらい? これは over-simplification (= LB は absolute RMSE)
- 現実的 estimate: **00bbac68 の per-row error を半分 (= 3.91 → 2.0 ft) にできれば LB -0.3 〜 -0.5 ft lift**

これは Top10 (= 9.728) を超え、 **Top 5 賞金圏 (= 9.415) も射程**。

---

## 5. 本タスクとの関係

| layer | 本タスク (= cv-strategies) | 拡張タスク (= winning-path D) |
|---|---|---|
| CV 戦略確定 | ✓ 最良 CV 確定 | (流用) |
| 自前 base OOF | ✓ 4 SCORED × 4 CV | (流用) |
| Per-Well MoE 実装 | × | ✓ 3 expert + gate + ensemble |
| 00bbac68 motivation の verification | × | ✓ data evidence で確証済 (= public-source-audit § 1.1.1) |

= 本タスク完了後、 「最良 CV で MoE expert 構造の改善幅を測る」 = 真の LB lift 評価。

---

## 6. 実装着手の前提

1. 本タスク (= cv-strategies-2026-05-11) AC-7 全 pass + 最良 CV 確定
2. `src/rogii/features.py:compute_b_well_cluster_id` (= 既に統合済) を gate input として使用
3. `.criteria/kaggle-rogii-winning-path-D-moe-2026-05-XX.yaml` 起票

---

## 7. 関連

- `docs/research/2026-05-11-public-source-audit.dense.md` § 1.1.1 = data evidence
- `src/rogii/features.py` D1/D4/D9 = MoE gate の input features (= 統合済)
- `docs/dev/2026-05-11-cv-lb-correlation.md` § 5.2 = winning path D 元 entry
- `~/projects/kaggle/CLAUDE.md` § 11 = 優勝本質性 ✅ 採用 (= data-driven、 軽さ-driven ではない)
- 兄弟 sketch: `docs/research/2026-05-11-winning-path-A-nn-sketch.dense.md` (= paradigm 2 sequence)
