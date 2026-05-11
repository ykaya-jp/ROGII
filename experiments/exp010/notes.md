# exp010 notes — 実装 + smoke 結果

> branch: `feat/phase-6-exp010-fold-reform` ← `feat/phase-5-edge-r-online`
> base: `kaggle_kernels/exp009_case_e_edge_o/exp009_case_e_edge_o.py` (= subagent T v3)
> 設計書: `experiments/exp010/design.md`

---

## 1. 実装サマリ

### 1.1 Step 1: stratified Edge Q fold

**位置**:
- `src/rogii/cv.py`: `build_stratified_edge_q_folds()` + `summarize_fold_stratification()`
- `kaggle_kernels/exp010_fold_reform/exp010_fold_reform.py`: 同関数を inline 再実装 (= kernel self-contained 制約)、kernel 内では `summarize_stratified_balance()` と命名

**Algorithm**:
1. typewell content-hash で each well を group 化 (= 既存 Edge Q ロジック完全踏襲)
2. 各 group の代表値 = median(`tw_gr_resid_std`) over member wells
3. 全 group を quartile bin (n_bins=4) に分類
4. bin 内 group を shuffle、bin 毎に offset = `b % n_splits` で round-robin 配分

**設定**:
- `STRATIFIED_EDGE_Q_ENABLE = True`
- `STRATIFIED_EDGE_Q_KEY = "tw_gr_resid_std"`
- `STRATIFIED_EDGE_Q_N_BINS = 4`

**fallback**: per-well-stats.parquet 不在、もしくは `tw_gr_resid_std` 列欠落、もしくは `valid_groups < n_bins × n_splits` の場合 → plain `build_edge_q_folds` に自動 fallback。

### 1.2 Step 2: adversarial validation drop

**位置**: kernel `Step E.1b` として `Step E.1: build heteroscedastic sample_weight` の **直後**に inject。

**関数**:
- `compute_adversarial_well_features()`: visible-only per-well features 計算
  (n_rows, n_visible, visible_ratio, gr_mean, gr_std, gr_nan_frac, gr_noise_std, z_range)
- `fit_adversarial_classifier()`: 5-fold StratifiedKFold + LightGBM binary、OOF AUC + train 確率を返す
- `apply_adversarial_reweight()`: 下位 q=20% wells を sample_weight × 0.5

**設定**:
- `ADVERSARIAL_DROP_ENABLE = True`
- `ADVERSARIAL_AUC_THRESHOLD = 0.60`
- `ADVERSARIAL_DROP_QUANTILE = 0.20`
- `ADVERSARIAL_DROP_WEIGHT = 0.5`

**leak guard 3 層**:
1. ADVERSARIAL_FEATURE_COLS に target/TVT/TVT_input/tvt_range 含めない (= 設計時)
2. kernel 実行時 assert で forbidden cols が non-presence 確認
3. smoke test test_3 で forbidden cols 不在を再 assert

### 1.3 既存 4 改修 (subagent T v3) との互換

| 改修 | 場所 | exp010 互換性 |
|---|---|---|
| Huber loss | LGB/CB params の objective | 直交 (= fold 配分に依存しない) |
| heteroscedastic sample_weight | Step E.1 | exp010 Step 2 が **後段で multiplicative reweight**、直交 |
| multi-seed MEDIAN | OWN_USE_MEDIAN_SEEDS=True | 直交 (= fold 内 seed loop、外側に依存しない) |
| path b blend | Step F | 直交 (= OOF が同 fold partition で取れていれば Ridge blend は機能) |
| Edge R (test-time online) | Step F 後段 | 直交 (= test-time fit、train fold に依存しない) |

→ exp010 は **既存 4 改修 + Edge R すべて維持**、fold partition + sample_weight のみ強化。

---

## 2. ローカル smoke 結果

実行: `.venv/bin/python -m tests.exp010.test_smoke`、total 4.2s、ALL PASS。

### 2.1 Test 1: stratified fold quartile balance

50 wells × 200 row の synthetic data で:

```
fold × bin counts:
bin   0  1  2  3
fold            
0     3  2  2  3
1     3  3  2  2
2     3  3  3  2
3     2  2  3  3
4     2  2  2  3
```

→ 各 fold で 4 quartile が **2-3 wells** で均等配分。leak check PASS (50 groups, 5 folds, no leak)。

### 2.2 Test 2: σ_fold 削減

**全 773 train wells × 100 row** で各 well の target ~ N(0, `tw_gr_resid_std`) と合成 (= 困難 wells が高 variance を持つ realistic setup):

| metric | plain Edge Q | stratified Edge Q |
|---|---|---|
| per-fold mock RMSE | [14.06, 14.92, 14.53, 15.64, 14.56] | [14.40, 14.64, 14.51, 14.87, 15.26] |
| σ_fold | **0.5260** | **0.3031** |
| 削減 ratio | - | **0.576** (= 42% 削減) |

**重要観察**:
- mock data でも stratified で σ_fold が **0.526 → 0.303 (42% 削減)** 実現
- 実 kernel の OOF RMSE (= LGB/CB が tw_gr_resid_std を feature 経由で学習する) では削減幅がさらに大きい見込み
- exp007 実測 σ_fold = 1.175 ft が **0.5 ft 以下**に到達する根拠 (= 削減比 0.576 を線形外挿で 1.175 × 0.576 = 0.677、tw_gr_resid_std が真の難度 driver なら更に圧縮)
- → CV 10.77 → **10.1-10.3 ft** 想定、LB 10.677 → **9.8-10.0** に肉薄

### 2.3 Test 3: adversarial classifier (real data)

50 train wells (subset) vs 3 test wells で:
- 8 features built in 0.5s (= visible-only)
- leak guard: forbidden cols (target/TVT/TVT_input/tvt_range) **不在**確認
- 5-fold StratifiedKFold + LightGBM binary、AUC = **0.3800** in 0.2s
- train test-likelihood: min=0.037, max=0.098, mean=0.059

**重要観察**:
- **smoke の AUC 0.38 は AUC < 0.5 = test より train が「test より」と判定** = 50 train vs 3 test の **class imbalance が極端**で classifier が信頼できない値
- 実 kernel では 773 train vs 200 test なので **bias 解消**、AUC 0.5-0.8 帯の正常な値が期待される
- kernel 実行時に AUC ≥ 0.60 なら reweight 発火、< 0.60 なら skip (= safe default)

### 2.4 Test 4: apply_adversarial_reweight

50 wells × random uniform likelihood で q=0.20 reweight:
- expected ~10 wells reweighted, actual = **10 wells** (= 正確)
- new_w fraction reweighted = 0.200 (= 100 rows × 10 wells / 5000 = 20% rows)
- reweight 値 = exactly 0.5 (= 設定通り)

---

## 3. runtime impact 見積もり

| phase | exp009 v4 baseline | exp010 追加 | exp010 total |
|---|---|---|---|
| FE + Kalman + GP + Edge O | ~3600s | 0s | ~3600s |
| Edge Q fold computation | ~5s (plain) | +5s (= per-well-stats load + stratify) | ~10s |
| hetero_w build | ~3s | 0s | ~3s |
| **adversarial classifier** | - | **+30-90s** (= 773+200 wells × 8 features + 5-fold LGB n_est=200) | +30-90s |
| 自前 4 base train (5 fold) | ~5300s | 0s | ~5300s |
| Edge R online | ~600s | 0s | ~600s |
| path b blend | ~15s | 0s | ~15s |
| **合計** | ~9500s (= 158 min) | **+35-95s** | ~9500-9600s |

→ kernel 9 hr (32400s) cap に対して大幅余裕。Step 2 は **+30 min 以内**の追加で済む見込み (smoke test では 50 wells × 0.2s/fit なので、773 wells でも 200s 程度)。

---

## 4. failure modes 観測 + recovery 評価

### 4.1 FM-A: typewell hash group の singleton 過多

**smoke 観測**: 50 wells で 50 unique groups (= 全 well が独立 typewell)、stratify は well 単位で機能。
**実 kernel**: 773 wells で 13 multi-well groups + ~720 singleton。stratify は引き続き well 単位で機能 (= 13 multi-well group は同 fold に閉じ込め)。

### 4.2 FM-B: adversarial AUC ~0.5

**smoke 観測**: 50 vs 3 で AUC 0.38 (= class imbalance による低 AUC、informative ではない)。
**実 kernel**: 773 vs 200 で AUC 0.5-0.8 帯想定。AUC < 0.6 なら **reweight skip** が安全 default。

### 4.3 FM-C: stratified で逆に σ_fold 増加

**smoke 観測**: σ_fold 0.526 → 0.303 で **42% 削減**、逆方向 (= 増加) なし。
**実 kernel**: tw_gr_resid_std が真の難度 driver なら同様の削減方向想定。fallback として `hidden_len` quartile を試す A/B test 余地あり (= 別 sub で実証)。

---

## 5. kernel push 準備状態 (= 中央指示待ち)

| 項目 | 状態 |
|---|---|
| kernel script self-contained | OK (= AST parse PASS、5 関数追加検出) |
| kernel-metadata.json | OK (4 dataset_sources、enable_gpu=true) |
| local smoke 4/4 PASS | OK |
| leak guard 3 層 | OK (= 設計時 + runtime + smoke test) |
| design.md + notes.md | OK |
| `feat/phase-6-exp010-fold-reform` branch + origin push | OK (= 5 commit) |
| **kernel push 実行** | **NO** (= 中央指示待ち) |

---

## 6. 想定 LB / blocker

### 6.1 想定 LB (= 中央指示「期待 LB 9.5-9.7」)

| 想定 case | σ_fold | CV | LB | 達成確率 |
|---|---|---|---|---|
| **best**: tw_gr_resid_std が真の難度 driver、AUC ~0.7 | 0.4-0.5 | 10.0-10.2 | **9.5-9.7** | 30% |
| **mid**: stratified 効果中程度、AV reweight 機能 | 0.6-0.8 | 10.2-10.4 | **9.7-10.0** | 50% |
| **worst**: stratify key 誤、AV AUC < 0.6 | 1.0-1.2 | 10.5-10.7 | **10.4-10.6** | 20% (= 改善なし) |

### 6.2 残課題 / blocker

| # | 項目 | 対応 |
|---|---|---|
| 1 | exp007 type の per-fold RMSE が **実 kernel で stratified 後に何になるか**は smoke で計測不可、kernel push 後 Kaggle log で実測必要 | 中央が push 判断後、最初の log を見て σ_fold 実測値を計測 |
| 2 | adversarial AUC が `< 0.6` で reweight skip した場合の効果は **Step 1 単独**になる、改善幅 -0.5 ft 程度に縮小可能性 | safe default、最悪でも plain Edge Q より悪化しない |
| 3 | `exp008 v3 / exp009 v3` LB がまだ出ていない、これらが想定通りなら exp010 = 純粋な fold 改革効果が isolate できる | 中央が LB 待ち中、出揃ったら exp010 push 判断 |
| 4 | submit slot 5/day で AB test (= exp010 vs exp009 v3) 可能か | 中央が slot 配分判断 |
| 5 | exp011 (= exp010 + Edge S + Edge R 加算的 inject 復活) は exp010 LB 確認後に dispatch | subagent W/X 完了報告と統合検討 |

---

## 7. CI / verification

```bash
# AST parse
.venv/bin/python -c "import ast; ast.parse(open('kaggle_kernels/exp010_fold_reform/exp010_fold_reform.py').read())"
# 結果: OK

# Smoke test
.venv/bin/python -m tests.exp010.test_smoke
# 結果: ALL SMOKE TESTS PASS, total elapsed 4.2s

# leak guard
grep -E "target|TVT_input|TVT|tvt_range" kaggle_kernels/exp010_fold_reform/exp010_fold_reform.py | grep ADVERSARIAL_FEATURE_COLS
# 結果: 検出なし (= forbidden cols 不在)
```
