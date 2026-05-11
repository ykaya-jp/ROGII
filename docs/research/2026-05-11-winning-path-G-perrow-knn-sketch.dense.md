# 優勝路手法 G — Per-row kNN inference (= paradigm 6) sketch

> 起源: docs/research/2026-05-11-public-source-audit.dense.md § 1.7 (= enisteper LB 9.960 audit で paradigm 6 大発見)
> 親 plan: kaggle-rogii-winning-candidates-cv-test-2026-05-12
> 期待 LB lift: -0.05 〜 -0.20 ft

---

## 1. 数理本質

### 1.1 enisteper paradigm の data evidence

enisteper LB 9.960 解法 (= rank 54、 we're 53 9.957) は train_engineered.parquet (= 3.78M rows × 121 cols) に **knn_row_*** features を含む:

| feature | 意味 |
|---|---|
| `knn_row_ANCC` | per-row kNN-based ANCC estimate (= 別 well の similar row から) |
| `knn_row_ANCC_dz` | dz delta |
| `knn_row_ANCC_std` | std (= confidence) |
| `knn_row_dist` | kNN distance metric |
| `knn_row_b_well` | b_well estimate from kNN |
| `knn_row_tvt_pred_delta` | TVT prediction delta |
| `fk_vs_row_ANCC_diff` | formation kNN vs per-row kNN delta |

= **per-row level の kNN inference**、 既存 LGB の row-level features を **「別 wells からの近傍情報」** で拡張。

### 1.2 paradigm 6 vs 既存 paradigm

| paradigm | 単位 | source |
|---|---|---|
| ② 自前 compiler (= LGB on row features) | per-row tabular | own training |
| ③ force/hand-craft (= D1 b_cluster ID) | per-well categorical | deepest EDA |
| ⑤ data augmentation (= visible_ratio random) | per-well 6× samples | aeroridge |
| **⑥ per-row kNN inference** | **per-row from other wells** | **enisteper** |

= paradigm 6 は **per-row level で他 wells の情報** = augmentation (= 自 well) + MoE (= cluster) とも違う。

### 1.3 数理本質

各 row $(well_i, t)$ に対し:
1. 全 train wells の visible rows から **GR signature が類似する** $K$ 個 を find (= kNN)
2. 類似 row の **ANCC / b_well / TVT** を集約 (= median or weighted mean)
3. これを「per-row kNN-based estimate」 として feature 化

= **inductive bias の追加**: 「ROGII の hidden TVT は、 GR shape が類似する別 well の visible TVT に類似する」 という仮定。

これは：
- 既存 LGB row features (= GR + depth + formation imputed) は **同 well 内の前後 row** から signal
- paradigm 6 kNN は **別 wells の類似 row** から signal = independent diversity source

---

## 2. 実装方針

### 2.1 kNN index 構築 (= train-time、 hidden 部除外 leak guard 必須)

```python
# train + test の全 wells の **visible region** から row-level features 抽出
visible_rows = []
for well in all_wells:
    df = load(well)
    visible = df[df["TVT_input"].notna()]  # visible のみ
    for _, row in visible.iterrows():
        visible_rows.append({
            "well_id": well,
            "md": row["MD"],
            "gr": row["GR"],
            "gr_roll11": row["GR_roll_11"],
            # ... GR-based features only (= depth-independent signature)
            "ANCC": row["ANCC"] if is_train else None,  # train のみ target
            "b_well": row["b_well"] if is_train else None,
        })

# kNN index = train wells の visible rows のみ (= target known)
# leak guard: 自 well の row は除外 (= self-NCC trap)
```

### 2.2 per-row inference

```python
def compute_knn_row_features(query_row, knn_index, k=5):
    """For one row, find k nearest (in GR signature space) train rows
    from OTHER wells, return aggregated kNN-based estimates."""
    own_well = query_row["well_id"]
    # Filter index = exclude same well
    candidates = knn_index[knn_index["well_id"] != own_well]
    # kNN search on GR-based features (= depth-independent)
    dists, ids = knn_search(query_row[GR_FEATURES], candidates, k=k)
    nbrs = candidates.iloc[ids]
    return {
        "knn_row_ANCC": float(np.nanmedian(nbrs["ANCC"])),
        "knn_row_ANCC_std": float(np.nanstd(nbrs["ANCC"])),
        "knn_row_b_well": float(np.nanmedian(nbrs["b_well"])),
        "knn_row_dist": float(np.nanmedian(dists)),
        # ...
    }
```

### 2.3 leak guard (= 3 層、 CRITICAL)

1. **自 well 除外**: query が train well なら、 kNN index から **自 well の rows を完全除外**
2. **target leak 防止**: query row の `TVT_input` は kNN signature に含めない (= GR + MD + depth のみ)
3. **fold-aware**: 我々 fold strategy で「val side wells」 の rows は kNN index から除外 (= fold leak prevent)

### 2.4 compute cost

- 3.78M rows × 5 neighbors × O(log N) kNN = ~10 min on GPU (= FAISS or PyTorch KNN)
- LGB train 後段は変わらず
- Total: existing train pipeline + 10-20 min for kNN feature compute

---

## 3. 期待 LB lift

enisteper LB 9.960 = kNN paradigm 採用、 ただし他 base 部分は our 9-base より弱い可能性。 つまり:
- **我々 9-base + paradigm 6 kNN features (= 6 new cols)** で blend
- 期待: -0.05 〜 -0.20 ft lift (= LB 9.8-9.9 帯)
- 加えて winning path F (= augmentation) 並行採用なら更に -0.05 〜 -0.10 ft = LB 9.7-9.85 帯 = **Top 10 (Gold cutoff 9.728) 射程**

---

## 4. 「優勝本質性」 ✅ 採用判定

- ✅ **数理本質**: inductive bias = 別 wells からの per-row 類似情報、 既存 LGB が見落とす独自 signal
- ✅ **優勝寄与**: enisteper LB 9.96 の core feature 群、 lift -0.05 〜 -0.20 ft で gold zone 射程
- ✅ **代替比較**: 既存 LGB row features 単独では到達不可能 (= enisteper が証拠)
- ✅ **rule 耐性**: visible region のみ使用、 host fix で死なない
- ✅ **datapoint 価値**: AB 対照群で paradigm 6 単独効果測定可能 (= 我々 baseline + kNN features)

---

## 5. 本タスクとの関係 + 着手前提

| layer | 本タスク (= cv-strategies) | 拡張タスク (= winning-path G) |
|---|---|---|
| CV 戦略確定 | ✓ 最良 CV | (流用、 fold-aware kNN index 必須) |
| kNN index 構築 | × | ✓ visible row features 抽出 + FAISS index |
| per-row kNN features | × | ✓ compute_knn_row_features pipeline |
| ensemble | ✓ 9-base | 拡張: 9-base + 6 paradigm-6 cols |

着手前提:
1. 本タスク AC-7 全 pass + 最良 CV 確定
2. enisteper の kNN implementation 詳細 reverse engineer (= train_engineered.parquet からの distillation)
3. `.criteria/kaggle-rogii-winning-path-G-knn-2026-05-XX.yaml` 起票
4. FAISS or pytorch KNN 利用 (= GPU 推奨)

---

## 6. 関連

- `docs/research/2026-05-11-public-source-audit.dense.md` § 1.7 = enisteper paradigm 6 finding
- `.work/source_audit/enisteper-train/train_engineered.parquet` = 121 cols 含む素材
- 兄弟 sketch: winning path A (NN) / D (MoE) / F (augmentation)
- `~/projects/kaggle/CLAUDE.md` § 4 = 4 paradigm 強制ルール、 paradigm 6 = **新発見の 6 番目**
- `~/projects/kaggle/CLAUDE.md` § 11 = 優勝本質性 ✅ 採用 (= data-driven、 enisteper 9.96 evidence)
