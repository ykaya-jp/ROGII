# CV vs public LB correlation — 結果 (進行中)

> task: `.criteria/kaggle-rogii-cv-strategies-2026-05-11.yaml`
> branch: `feat/cv-strategies-final-2026-05-11`
> 目的: 4 CV (C1-C4) と baseline plain GroupKFold の OOF を 8 SCORED (+ RUNNING exp010) で再生成、 LB との Spearman/Pearson/LOO/bootstrap を測定し、 最良 CV を確定する。
> 形式: 段階的に更新、 最終的に AC-6 / AC-7 / AC-9 / AC-10 を満たす。

---

## 1. Phase 2.0 = fold parquet 早期 diagnostic (= 2026-05-11)

`scripts/build_fold_parquets.py` で 5 CV strategy × 773 train wells の fold 配分を事前生成。 **base re-train は未着手** (= compute heavy)、 ここまでは fold 配分の構造的性質のみを評価する数理 diagnostic。

### 1.1 fold size balance

| CV | 戦略 | fold sizes (5 fold) | max/min | dropped | leak check |
|---|---|---|---|---|---|
| baseline | plain GroupKFold by well | 155, 155, 155, 154, 154 | **1.006** | 0 | n/a (well 直 split) |
| **C1** | pseudo-test fold via kNN (= test 類似 wells を fold 0 集約) | 140, 159, 158, 158, 158 | **1.136** | 0 | no leak |
| **C2** | multi-key stratified (visible_ratio × tw_gr_resid_std) | 161, 155, 152, 152, 153 | **1.059** | 0 | no leak (= 752 groups OK) |
| **C3** | adversarial drop (= baseline + 20% test-unlike wells excluded) | 118, 128, 126, 116, 126 | **1.103** | **159** | n/a (= baseline base) |
| **C4** | typewell hash GroupKFold (= Edge Q v2) | 155, 155, 155, 154, 154 | **1.006** | 0 | no leak (= 752 groups OK) |

**観察**:
- baseline + C4 が **完全 balance** (= max/min = 1.006)。 C2 / C3 / C1 は若干 unbalanced だが、 全部 max/min ≤ 1.14 で許容範囲。
- C3 で **159 wells dropped** (= 全 train 773 wells 中 20.6% = 設定値 drop_quantile = 0.20 に整合)。

### 1.2 数理 diagnostic finding

#### F-D1: C3 adversarial AUC = **0.3167** (test n=3 で不安定)

```
fit_adversarial_classifier 5-fold OOF AUC = 0.3167
train test-likelihood: min=0.04 max=0.10 mean=0.06
```

- AUC < 0.5 = **classifier が逆に train を train として識別**、 ただし test n=3 で AUC 推定 unstable
- 数理意味: H-divergence ≈ |2 AUC - 1| = 0.37、 distribution shift は **軽度** (= 0.5 を中心に 0.37 偏り)
- 含意: **adversarial drop の effect は限定的** = C3 alone で大幅 LB 改善は期待薄、 ただし C2 / C4 baseline に combined inject なら marginal lift 可能

#### F-D2: C4 b_cluster balance p_value = **0.0424** (= 5% 有意で偏在)

```
fold × b_cluster pivot shape: (5, 87)
chi-square p-value = 0.0424
```

- p < 0.05 で **fold ⊥ b_cluster の null 棄却** = typewell hash GroupKFold は b_cluster (F1) を偶発的に偏在させる
- 数理意味: C4 単体は **F8 (typewell leak free)** を保証するが、 **F1 (b_cluster 2 種 rep)** は満たさない
- 含意: **C2 (= multi-key stratify、 ただし default は visible_ratio + tw_gr_resid_std で b_cluster 未含)** + b_cluster を stratify_specs に追加した C2.v2 が必要、 もしくは C4 fold 後の post-hoc rebalance

#### F-D3: C1 pseudo-test fold 0 sample = 140 wells (= 5 CV 中最小)

```
C1 fold sizes: 140 (fold 0 = pseudo-test), 159/158/158/158 (其他)
```

- fold 0 (= test 3 wells に kNN 距離で近い train wells、 K=50/test_well × 3 wells で union 後 dedup)
- 140 / 773 = **18.1%** が test 類似 train wells = LB との rank 整合性測定の主要 indicator
- 数理意味: fold 0 OOF RMSE が他 fold より大きく / 小さく動くか = **test 分布での予測難度を直接 surrogate**

### 1.3 暫定仮説 (= base re-train 前の数理推測)

1. **C2 が一番 robust** な可能性 (= 多軸 stratify + F8 leak free)。 ただし **default stratify_specs に b_cluster を含めるべき** (= F-D2 から、 F1 を直接 strat する value)。 → 次 step で `stratify_specs=[("visible_ratio", "quantile", 5), ("tw_gr_resid_std", "quantile", 4), ("b_ANCC_med", "quantile", 3)]` で再評価。
2. **C3 adversarial drop の単独効果は限定的** (= AUC 0.32 で shift 軽度)。 C2/C4 base に combined inject すれば marginal lift。
3. **C1 fold 0 OOF RMSE は LB-proxy として valuable** (= 140 wells が test 類似)。 ただし他 fold の OOF は「典型的 train wells」 で LB-proxy ではない。 全 OOF を pool した overall RMSE は valid だが、 fold 0 単独 を LB proxy として併記すべき。
4. **C4 単独は F1 偏在で risky** (= b_cluster balance p < 0.05)。 C4 を採用するなら必ず b_cluster post-hoc audit を併記。

---

## 2. Phase 2.1 = 各 SCORED の base re-train + OOF 再生成

### 2.1.a kernel patch (= **完了**)

build_edge_q_folds (+ exp010 の build_stratified_edge_q_folds) の冒頭に env var `FOLD_OVERRIDE_PARQUET` から fold partition を読む hook を注入:

- ✓ `kaggle_kernels/exp007_edge_q_m/exp007_edge_q_m.py`
- ✓ `kaggle_kernels/exp008_case_d_kalman/exp008_case_d_kalman.py` (= v2 と v3 共用)
- ✓ `kaggle_kernels/exp009_case_e_edge_o/exp009_case_e_edge_o.py`
- ✓ `kaggle_kernels/exp010_fold_reform/exp010_fold_reform.py` (= 両 fold fn + helper)
- 未: exp002_lgb / exp003_lgb / exp005_cache_blend / exp006_tabicl_pflite (= karnakbaev only や baseline)。 これらは別 task で対応。

### 2.1.b OOF 再生成 pipeline (= **完了**)

`scripts/regenerate_oof.py` 新規 = (exp, cv) ペアごとに:
1. kernel script に CV strategy 冒頭 inject (= `FOLD_OVERRIDE_PARQUET=/kaggle/input/rogii-cv-fold-overrides/{cv}.parquet`)
2. kernel-metadata.json を patch (= id/title 変更 + dataset_sources に fold dataset 追加)
3. `kaggle kernels push` で submit
4. COMPLETE 待ち + output OOF download

dry-run 動作確認済 (= patched kernel + metadata が正しく生成)。

### 2.1.c 実行 (= **ユーザー手元 work**、 未完)

```bash
# 1) ローカルで fold parquet を Kaggle dataset として upload (= 一度きり)
cd outputs/folds/ && kaggle datasets create -p . -u
# → ky7240/rogii-cv-fold-overrides がアップロードされる
# (= dataset slug は scripts/regenerate_oof.py の DEFAULT_FOLD_DATASET と一致)

# 2) 全 20 kernel (= 4 exp × 5 cv) を kaggle に push、 完走待機
.venv/bin/python scripts/regenerate_oof.py --all

# もしくは parallel push (= Kaggle 側で queue に並ぶ、 wall clock 短縮)
.venv/bin/python scripts/regenerate_oof.py --all --parallel-push
```

compute: 1 kernel ≈ 30-60 min Kaggle GPU、 20 kernel sequential で 10-20 hr、 parallel で wall 1-2 hr (= Kaggle scoring queue 依存)。

### 2.1.d aggregate (= **未着手**)

`scripts/regenerate_oof.py` が生成する `outputs/oof/cv-lb-correlation/<exp>__cv-<cv>/submission.csv` 群を集約して `oof_table.parquet` (= AC-5) にまとめる集約 script を別途。

### 2.1.e correlation 計算 (= **未着手**)

`tools/measure_cv_lb_correlation.py` で Spearman/Pearson/LOO 平均/bootstrap 95% CI を計算。 AC-6 + AC-7 達成。

注: 本タスクで OOF を取得できるのは self-base 4 kernel (= exp007 / exp008 v2 / exp009 v2 / exp010) のみ。 karnakbaev only の exp005 系 + exp006 は published OOF を流用しているため、 「真の OOF 再生成」 は karnakbaev pretrained を fold-aware に re-fit する別 pipeline が必要 (= 別 task に分離)。 結果として correlation 計算は n=2-4 (= exp007 + exp008 v2 + exp009 v2 SCORED 後 + exp010 SCORED 後) で実施。

### 2.2 SCORED + PENDING

| exp | submitted | scored? | LB | 用途 |
|---|---|---|---|---|
| exp002 | 5/10 13:25 | ✓ | 14.695 | LGB baseline (= sanity outlier) |
| exp003 | 5/10 15:58 | ✓ | 17.510 | tysig 単独 failure (= outlier) |
| exp005 | 5/10 16:14 | ✓ | 10.317 | karnakbaev base |
| exp006 | 5/10 16:51 | ✓ | 10.503 | + TabICL (失敗) |
| exp007 | 5/10 22:10 | ✓ | 10.677 | + Edge Q + 自前 4 base |
| exp005 v2 | 5/11 00:13 | ✓ | 10.203 | + Edge S round-to-grid |
| exp005 v3 | 5/11 00:32 | ✓ | 10.387 | + Edge R online (失敗) |
| exp008 v2 | 5/11 03:15 | ✓ | **9.957** ✅ | + 案 D Kalman (= 現 best) |
| exp009 v2 | 5/11 05:37 | PENDING | TBD | + Sparse GP + Edge O |
| exp008 v3 | 5/11 08:31 | PENDING | TBD | + Huber + hetero + path b |
| exp010 | 5/11 14:42 | kernel RUNNING | TBD | + stratified Edge Q + adversarial drop |

= 8 SCORED + 3 PENDING/RUNNING。 PENDING 完走で sample size 拡大 (= 9-11 件)。

---

## 3. 最良 CV 確定 — 宣言 (= Phase 2.1 完了後に埋める placeholder)

(Phase 2.1 = OOF 再生成 + correlation 測定が終わるまで未確定)

予定 format:

```
最良 CV = C? (= 戦略名)

multi-metric table (= 5 CV × 4 metric):
| CV | Spearman | Pearson | LOO 平均 | Bootstrap 95% CI lower | composite | exp003 除外 rank |
| baseline | ... | ... | ... | ... | ... | ... |
| C1 | ... | ... | ... | ... | ... | ... |
| C2 | ... | ... | ... | ... | ... | ... |
| C3 | ... | ... | ... | ... | ... | ... |
| C4 | ... | ... | ... | ... | ... | ... |

選定理由 (= 5 観点):
1. composite score 最大
2. exp003 除外時も上位 (= robust)
3. fold balance 確認済
4. 解釈可能性 ◯
5. 既存 paradigm との互換性

Jensen lower bound 改善:
- exp007 σ_fold = 1.175 ft → CV 天井 10.77 ft
- 最良 CV exp007 σ_fold = X.XXX → CV 天井 X.XX ft (= 改善幅 -Δ ft)

次タスク = kaggle-rogii-winning-candidates-cv-test-2026-05-12
  「優勝路手法 A-E (= NN paradigm / deepest EDA D1+D4+D9 /
   Sparse GP M=500 / Per-Well MoE / LLM-driven) を 最良 CV で測定」
```

---

## 4. 関連 doc / 出典

- `docs/research/2026-05-11-cv-strategy-candidates.dense.md` — 設計書 (= 4 CV の数理本質)
- `docs/research/2026-05-11-test-distribution.dense.md` — F1-F8 finding
- `docs/dev/2026-05-11-h10-postmortem-and-fold-reform.dense.md` — H10 = Jensen lower bound 起源
- `scripts/build_fold_parquets.py` — 本 doc § 1 の元 script
- `outputs/folds/summary.json` — fold parquet metadata
- `tools/measure_cv_lb_correlation.py` — 本 doc § 3 (= 結果セクション) 自動生成元
- `scripts/regenerate_oof.py` — § 2.1.b kernel push automation
- `scripts/aggregate_oof.py` — § 2.1.d kernel output 集約 → oof_table.parquet

---

## 5. 次タスク plan draft (= AC-12)

### 5.1 task_id: kaggle-rogii-winning-candidates-cv-test-2026-05-12

本タスク (= cv-strategies-2026-05-11) が確定する **最良 CV を基準** に、 優勝路手法 A-E の CV 改善を測定する。

### 5.2 検証対象 (= 5 候補、 deepest EDA + GM mindset 由来)

| ラベル | 戦略 | 期待 CV 改善 (vs exp008 v2 baseline) | paradigm | compute |
|---|---|---|---|---|
| **A** | **GPU NN sequence model 追加** (= Mamba / Transformer / 1D-CNN on GR+dTVT) | -0.3 〜 -0.6 ft | ② 自前 compiler/solver 拡張 | GPU 必須、 5-10 hr/run |
| **B** | **deepest EDA D1 + D4 + D9 を LGB stack に添加** (= b_well 67 cluster ID + b_jump_max_abs + 6 formation top per-well median) | -0.4 〜 -0.9 ft | ③ force/hand-craft (= 公開上位未使用 path) | CPU 0.5-1 hr/run |
| **C** | **Sparse GP M=200 → M=500-1000** + posterior variance 明示注入 | -0.2 〜 -0.5 ft | ② 既存 GP 強化 | GPU 3-5 hr/run |
| **D** | **Per-Well MoE wrapper** (= visible_ratio / b_cluster で per-well 専用 expert + gating blend) | -0.3 〜 -0.5 ft | ② 構造変更 (= 単一 model → MoE) | CPU 1-2 hr/run |
| **E** | **LLM-driven posterior synthesis** (= Claude / GPT で per-test-well 推論 + ANCC 復元 program synthesis) | ?? (= 0 〜 -1.0 ft) | ④ **新規 paradigm** (= ROGII 前例なし) | CPU 0.5 day prototype |

### 5.3 検証順序 (= 数理的優先度)

1. **B** が最優先 (= compute 軽い + 公開上位未使用 path の独自 lift)、 CV 確定後 0.5-1 day で smoke
2. **A** が次 (= paradigm 新規追加で gold zone 5 必要条件 #3 を満たす)、 GPU 環境前提
3. **E** は探索的 paradigm (= reward unknown だが ROGII で前例なし)、 LLM 試作で早期確認
4. **C** + **D** は既存路線強化 (= base improve)、 priority 低

### 5.4 未統合 host dataset (= 2026-05-11 kaggle datasets list -s rogii で発見)

| dataset | size | last update | 推測内容 | 取り込み判断 |
|---|---|---|---|---|
| `thbdh5765/rogii-v4-aeroridge-train-cache` | 1.57 GB | 2026-05-11 04:53 | AeroRidge 系 train feature cache (= 別 base 候補) | E に統合候補 (= 公開 source 拡張) |
| `akshankrithick/rogii-gold-top10-direction-source` | 138 KB | 2026-05-09 07:03 | Gold Top10 解法の direction source | 内容確認後、 E に統合候補 |
| `buchananliang/rogii-karnak-top2-public-artefacts` | 6.6 MB | 2026-05-08 04:07 | Top2 (= karnakbaev 系) 解法 public artifacts | 既存 karnakbaev base と差分確認、 重複なら不要 |
| `nina2025/rogii-03` / `rogii-07` | 0.5 MB / 1.1 MB | 5/7-11 | 小規模 utility (= 詳細不明) | 確認後判断 |
| `alfaxadeyembe/rogii-lgbm-model-artifacts` | 1.2 MB | 2026-05-09 | LGBM 訓練済み artifacts | E に組み込み候補 |

これらは「優勝路手法 E (= 公開 source 集約)」 の素材。 次タスクで `kaggle datasets download` + 内容 audit + ROGII evaluation pipeline に attach 可能性検討。

### 5.5 plan 起票時の出典 / context

- `.criteria/kaggle-rogii-cv-strategies-2026-05-11.yaml` — 本タスク (= CV 確定)
- 本 doc § 3 「最良 CV 確定」 — 次タスクの基準 CV
- `docs/research/2026-05-11-cv-strategy-candidates.dense.md` § 5 — 候補手法の数理本質
- `docs/research/2026-05-11-deepest-eda.dense.md` § 4 — D1-D10 feature 候補詳細
- `~/projects/kaggle/CLAUDE.md` § 4 — 4 paradigm 強制ルール (= 3+ paradigm mix 必須)

### 5.6 出力 deliverables (= 次タスクの想定)

- `.criteria/kaggle-rogii-winning-candidates-cv-test-2026-05-12.yaml`
- 各候補に対する exp ブランチ + smoke + OOF (= 最良 CV で測定)
- `docs/dev/2026-05-12-winning-candidates-cv-results.md` (= 候補 5 つの CV 改善実測)
- 「優勝路手法 確定」 宣言 (= 最も CV 改善幅大きい 1-2 候補を submit 路線として確定)

---
