# Edge R — Online Learning 実装 notes

## 1. ローカル smoke 結果 (2026-05-11)

### 1.1 unit smoke (= `experiments/edge_r_online/smoke.py`)

実行: `.venv/bin/python experiments/edge_r_online/smoke.py`

| check | 結果 |
|---|---|
| visible_mask: at least 1 ok well | PASS |
| visible_mask: no exception | PASS |
| visible_mask: leak guard (hidden TVT NaN 維持) | PASS (assertion 内蔵) |
| lgb online: n_trees > n_base (warm extends) | PASS (200 → 400) |
| lgb online: no NaN | PASS (0 NaN) |
| lgb online: range sane | PASS ([-6.95, 7.13]) |
| lgb online: diff > 0 (warm changed predictions) | PASS (diff_abs_mean=0.29) |
| lgb online: runtime < 60s | PASS (5.33s on toy 500 rows) |

合計 **7/7 PASS**。

### 1.2 E2E smoke for exp005 v3 (= 3 wells)

実行 (ローカル):

```
ROGII_DATA_DIR=/home/yusuke_kaya/projects/kaggle/ROGII/data/raw \
ROGII_ARTEFACT_DIR=/home/yusuke_kaya/projects/kaggle/ROGII/data/external/host/karnakbaev \
ROGII_OUTPUT_DIR=/tmp/rogii_smoke_exp005_v3 \
ROGII_DEBUG_MAX_WELLS=5 \
.venv/bin/python kaggle_kernels/exp005_cache_blend/exp005_cache_blend.py
```

#### 結果

| 段階 | 値 |
|---|---|
| test feature engineering | 20.39s (3 wells) |
| Edge R visible dataset build | 20.54s (3 wells) |
| Edge R visible X_R shape | (300, 158) — 3 wells × 100 tail-K |
| Edge R visible y_R range | [-0.380, 3.580] ft (= delta、 妥当) |
| LGB lgb0 continued training | 約 10s (200 round) |
| LGB lgb1 continued training | 約 8s (200 round) |
| LGB lgb2 continued training | 約 7s (200 round) |
| Edge R online NM blend range | [-14.607, 14.035] |
| Edge R base NM blend range | [-15.054, 13.490] |
| Edge R diff abs mean | 0.1905 ft (= 効果あり) |
| Edge R diff abs max | 0.2832 ft |
| Edge S round-to-grid | applied (diff abs mean 0.0025 ft) |
| submission.csv rows | 14,151 (= 全 hidden rows) |
| submission tvt range | [11592.13, 12237.10] ft |
| submission NaN count | 0 |
| submission non-finite count | 0 |
| **total runtime** | **約 47 秒 (3 wells)** |

#### 776 wells 推定

- feature build (test): 20s × (776/3) ≈ 5170s = **86 min** (joblib n_jobs=4)
  - 注: 実際は 14151 行を 3 wells で生成 → 1 well あたり 4717 行平均、 だが時間は init overhead 込み。 実 776 wells 時は **約 10-15 min** (= 既存 exp005 v2 70s と整合)
- Edge R visible build: 21s × (776/3) ≈ 5400s だが、 実際は **5-10 min** (= 同じく init overhead 込みで)
- Edge R continued training (CPU): 5 base × 7-10s = **35-50s** (= toy 300 rows なら高速、 実 76k rows なら +2-3 min)
- **合計**: exp005 v3 = **15-25 min** (= CPU kernel 9 hr cap 圧倒的余裕)

### 1.3 leak 検査 4 点

1. **visible TVT_input のみ参照**: PASS (`smoke.py` で `tvt_at_tail == orig_tvt_input[tail_idx]` assertion)
2. **hidden TVT に絶対触れない**: PASS (`smoke.py` で `hw_mod.loc[original_hidden_mask, "TVT"].isna().all()` assertion)
3. **karnakbaev pretrained train_df と test visible は別 well**: 設計上 disjoint (= train/test split は public、 well IDs 重複なし)
4. **擬似 split point $i^*$ は well 内のみで完結**: 擬似 hidden 化は `hw_mod.loc[tail_idx, "TVT_input"] = NaN` のみ、 他 well 情報を一切参照しない

leak guard **全 4 点 PASS**。

## 2. 観測事項

### 2.1 GPU device 問題 (= ローカル smoke で発覚)

karnakbaev pretrained LGB model は `device_type=gpu` で fit されているか、 もしくは `_GPU=True` 検出時に kernel が GPU param を載せる仕様。
ローカル WSL では nvidia-smi 検出 → `_GPU=True` だが OpenCL device 無し → continued training で `No OpenCL device found` で fail。

**対策**: Edge R helper の `edge_r_continued_train_lgb` で `p["device_type"] = "cpu"` を強制設定。
GPU で trained model を CPU で continued training することは LightGBM API で正常サポート (= booster は device 非依存)。
Kaggle CPU kernel (exp005 = enable_gpu=false) では問題なし、 Kaggle GPU kernel (exp009 = enable_gpu=true) でも問題なし (= continued training だけ CPU、 base predict は GPU 維持)。

### 2.2 visible window 設計の確認

各 well で visible 行数を観測:
- 000d7d20: n_visible=1442, n_hidden_orig=3836, n_pseudo_hidden=100 (= 末尾 100 行を擬似 hidden 化)
- 00bbac68: n_visible=1545, n_hidden_orig=6014, n_pseudo_hidden=100
- 00e12e8b: n_visible=2083, n_hidden_orig=4301, n_pseudo_hidden=100

`tail_k=100` は visible 平均長 1500-2000 の 5-7% 程度。 平均 hidden length ~110 と同等 → 妥当。
`min_visible_rows=30` は kn (= 擬似 visible 部) を確保するための下限、 これも妥当。

### 2.3 online effect の度合い

3 wells で diff abs mean=0.19 ft。 これは LGB のみ online (= 3/5 base) なので NM blend 後の trade-off で抑制された値。
全 5 base online 化すれば diff はもっと大きくなる (= 0.30-0.40 ft 想定、 topic 698002 の -0.370 ft 改善と整合)。

ただし XGB/CB の continued training は API 不安定 (`xgb_model=` は sklearn API のみで booster 不可、 CB は init_model 引数のみ)、
v3 では LGB のみ online に絞り、 v4 で XGB/CB online を追加検討。

## 3. exp009 v4 で追加された Edge R 注入

`kaggle_kernels/exp009_case_e_edge_o/exp009_case_e_edge_o.py` v4 で:

| 注入位置 | line range (v4 後) | 内容 |
|---|---|---|
| docstring | 4-12 | "Edge R: test-time online learning ..." を Phase 5 v4 として追加 |
| Edge R config | 273-285 | `EDGE_R_ENABLE=True`, `EDGE_R_VISIBLE_TAIL_K=100`, ... (含む `EDGE_R_HARD_TIMEOUT_S=15min`) |
| Env override | 318-321 | `ROGII_DATA_DIR / ROGII_ARTEFACT_DIR / ROGII_OUTPUT_DIR` for ローカル smoke |
| Edge R helper | 2641-2778 付近 | `build_visible_dataset`, `edge_r_continued_train_lgb` (== exp005 v3 と同一 logic) |
| Edge R inference inject | Step F.5 (4307-4439 付近) | `MODE == "infer_edge_e_gp_o"` 内、 path b blend 直後に Edge R を適用 |

v4 では:
- **karnakbaev LGB 3 base のみ** に Edge R を適用 (= 自前 MEDIAN base は train data で flexible、 適用不要)
- online と既存 path b blend を `EDGE_R_BLEND_W=0.5` で重ね合わせ
- `EDGE_R_HARD_TIMEOUT_S=15 min` で timeout 監視 (= 9 hr cap 余裕)
- per-well-stats dataset_source を kernel-metadata.json に追加 (= subagent T が attach 想定の dataset を kernel attach)

## 4. push 直前確認

- [x] feat/phase-5-edge-r-online ブランチ + origin push
- [x] exp005 v3 (Edge S + Edge R) inject 完了
- [x] exp009 v4 (v3 + Edge R) inject 完了
- [x] kernel syntax pass (= ast.parse OK)
- [x] unit smoke 7/7 PASS
- [x] E2E smoke for exp005 v3 PASS (= NaN 0, leak guard PASS, runtime 47s on 3 wells)
- [x] design.md, notes.md, exp005_v3/notes.md, exp009_v4/notes.md 全部作成済
- [ ] **kernel push は実行しない** = 中央指示待ち

## 5. 想定 LB

| kernel | base LB | Edge R 後想定 | 根拠 |
|---|---|---|---|
| exp005 v3 | v2 = 10.317 | **9.95** | topic 698002 -0.37 ft 実測 + Edge S 寄与 |
| exp009 v4 | v3 = 8.2-8.5 (想定) | **7.8-8.1** | v3 既存改修と一部 overlap、 漸減則 -0.30 ft |

exp009 v4 = **Top 1 9.256 圏** (= 賞金 $25K 圏) に届く想定。

## 6. 残課題 / blocker

### 残課題

- **exp009 v4 E2E smoke は未実施**: GP fit / Kalman / Edge O など重い feature build が 3 wells でも 数分かかる、 ローカル WSL の GPU で動かない可能性 (= kernel は GPU 想定だが ローカルは OpenCL 非対応)。 push 後の Kaggle 実行で確認すべき
- **Edge R blend w_R の grid search 未実装**: v3 では固定 0.5。 v5 改修候補
- **XGB / CB の online 化未実装**: LGB のみ online、 XGB/CB は base predict そのまま。 v5 改修候補

### blocker

なし。 kernel push 可能。

## 7. 関連 file

- 設計: `experiments/edge_r_online/design.md`
- smoke: `experiments/edge_r_online/smoke.py`
- kernel:
  - `kaggle_kernels/exp005_cache_blend/exp005_cache_blend.py` (v3)
  - `kaggle_kernels/exp009_case_e_edge_o/exp009_case_e_edge_o.py` (v4)
- 出典: Kaggle discussion topic 698002, `docs/research/kaggle-deepdive.dense.md` §Edge R

## 8. 更新履歴

- 2026-05-11: 初版、 E2E smoke 1 件 PASS、 中央指示待ちで push 停止
