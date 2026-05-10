# exp008 v3 設計書 — ROI 3 layer + fold-misalign 解消 (Phase 5)

## 0. 目的

`feat/phase-4-edge-s-roundgrid` の `kaggle_kernels/exp008_case_d_kalman/exp008_case_d_kalman.py` (= exp008 v2 base、Edge S round-to-grid 注入済) に対し、
**3 doc 共通推奨 4 改修**を inject し、CV を 9.6 → 8.5 帯、LB を 8.7-9.0 帯まで押し込む。

ユーザー指示「9 切らないと優勝は無理、8 台のスコアを取ってほしい、CV 8 台到達が必要条件」に対する**直接の解**。

## 1. 出典の論拠 (= 何を根拠に 4 改修を選んだか)

| doc | section | claim |
|---|---|---|
| `docs/research/mathematical-formulation.dense.md` | §1.3 A | Huber loss は fat-tail noise (= dtvt_p95 / std = 2.4 > Gaussian 1.645) で minimax 最適 |
| 〃 | §1.3 B | Heteroscedastic likelihood は per-well で $\hat{\sigma}$ 異なる場合 (= `b_well_resid_std` ∈ [0.0061, 0.0111]) の MLE |
| 〃 | §2.2 | Multi-seed × Multi-fold MEDIAN は $\sigma_{\text{ens}}^2 = \sigma^2 / (NK)$ で variance ↓、mean より fat-tail robust |
| 〃 | §3.4 | exp007 fold-misalign (kb 元 fold ≠ Edge Q fold) → Ridge meta leak。path (b) = kb OOF Ridge から除外、別系統 simple-average で blend |
| `docs/research/gm-wisdom.dense.md` | §1.1, §1.3 | Adversarial Validation + fold_id 列 commit、GM Vandewiele 流 Multi-seed MEDIAN |
| `docs/research/cv-breakthrough.dense.md` | Track 4, 6 | AV + Loss redesign は CV 8 達成への "即時 ROI 3 layer" |
| `docs/dev/leaderboard.dense.md` | CV-LB Trend §exp007 | fold-misalign 発覚記録 |
| `docs/research/first-principles.dense.md` | §2.2 | per-well-stats.parquet (n=773) 実測値: `b_well_resid_std` p50=0.00835 |

## 2. 4 改修の inject 位置

base: `kaggle_kernels/exp008_case_d_kalman/exp008_case_d_kalman.py` (= 3431 行)

### 改修 1: Huber loss + δ tuning

- 対象 line: `LGB_BASE` dict (`296: objective = "regression"`), `CB_PARAMS` dict (`330: loss_function = "RMSE"`)
- 変更:
  - `LGB_BASE["objective"] = "huber"`、`LGB_BASE["alpha"] = 0.9` (= LightGBM の `alpha` は Huber δ 比率パラメータ)
  - `CB_PARAMS["loss_function"] = "Huber:delta=0.5"`
- 数理:
  - Huber: $L_\delta(r) = \frac{1}{2} r^2$ if $|r| \le \delta$、$\delta (|r| - \frac{\delta}{2})$ otherwise
  - δ ≈ p50(dtvt_std) ≈ 0.42、conservative に **0.5** を採用 (= per-well dtvt_std 75% 分位 0.443 と整合)
  - LightGBM `alpha` は internal 自動 scaling、`0.9` で test loss 上位 ~10% を Laplace 扱い、残り Gaussian
- 期待 CV 改善: -0.2 〜 -0.5 ft (= fat-tail tail 抑制)
- 失敗 mode 3 件:
  1. LightGBM Huber は GPU で挙動が CPU と微妙にずれる (= `device_type=gpu` で精度劣化 1e-3 ft) → fallback CPU retry を維持
  2. CatBoost `Huber:delta=0.5` は init 時 string parse、`delta` 値の typo で silent fallback (= RMSE 同等) → smoke でログ確認
  3. final retrain で `n_estimators` が Huber では早 stop しすぎる傾向 → `FINAL_ITER_SCALE = 1.10` のままで OK、`early_stopping_rounds = 150` 維持

### 改修 2: Heteroscedastic sample_weight (`b_well_resid_std` 由来)

- 対象 line: 自前 4 base train の `model.fit(Xt.values, yt, eval_set=...)` (= line 3259, 3289) + `Pool(Xt.values, yt)` (= line 3290)
- 変更:
  - train 直前で `train_df_kb["b_well_resid_std"]` を per-well stats parquet から merge
  - `w_train[i] = 1 / sigma_w(i)` の **less aggressive** 版を採用 (= $1/\sigma^2$ は extreme 重み過剰の懸念)
  - 正規化 + clip: `w = w / w.mean()`、`w = np.clip(w, 0.5, 2.0)`
  - LGB: `lgb.LGBMRegressor.fit(Xt, yt, sample_weight=w_tr, eval_set=[(Xv, yv)], eval_sample_weight=[w_va])`
  - CB: `Pool(Xt.values, yt, weight=w_tr)`、`eval_set=Pool(Xv.values, yv, weight=w_va)`
- 数理:
  - heteroscedastic Gaussian MLE: $\mathcal{L} = \sum (y - \hat{y})^2 / (2\sigma_w^2)$
  - sample_weight ∝ 1/σ は MLE の equivalent (Gaussian assumption 下)
  - per-well-stats.parquet 実測: `b_well_resid_std` ∈ [0.0061, 0.0111]、ratio max/min ≈ 1.82 → clip [0.5, 2.0] で覆える
- 期待 CV 改善: -0.1 〜 -0.3 ft (= variance 適正配分)
- 失敗 mode 3 件:
  1. `b_well_resid_std` が 766/773 wells しか埋まらない (= 7 well NaN) → fillna(global p50 = 0.00835) で対処
  2. weight 適用が train のみで eval/test 不要なので `eval_sample_weight` 渡し忘れで early-stop が biased → smoke で fold OOF RMSE を sample_weight 込みで eval 確認
  3. extreme weight (clip 前) で gradient explode → clip 0.5-2.0 で抑える、smoke で `(w.min(), w.max())` を assert

### 改修 3: Multi-seed × 5-fold MEDIAN (= 9 hr cap 内に工夫)

- 対象 line: 自前 4 base train の seed loop (= line 3247 OWN_LGB_SEEDS, 3287 OWN_CB_SEED) + ensemble 集約
- 変更:
  - **戦略**: 既存 `LGB lr=0.02/0.05/0.10` 3-LGB を捨て、**LGB lr=0.05 単一 + CB lr=0.05 単一 = 2 model** × **3 seed (42, 123, 2024)** × **5 fold** = **30 train run**
  - 各 fold で 2 model × 3 seed = 6 predictor、val 上で MEDIAN aggregate して own_oof[fold] に格納
  - test 上も同様に 6 predictor × 5 fold = 30 prediction を MEDIAN
  - `OWN_LGB_LRS = (0.05,)`, `OWN_LGB_N_ESTS = (4000,)`, `OWN_LGB_SEEDS = (42, 123, 2024)`, `OWN_CB_SEEDS = (42, 123, 2024)` に変更
  - own base key: `lgb_own_med` + `cb_own_med` (= 2 key、合算)
- 数理:
  - $\hat{y}_{\text{ens}} = \text{median}\{\hat{y}_{n,k}\}$、$\sigma^2 = \sigma^2_{\text{base}} / (3 \times 5 \times 2)$ for variance 削減
  - MEDIAN は mean より fat-tail robust (= Huber と相性 ◎)
- 工数 + runtime:
  - 既存 5 fold × 4 base (LGB×3 + CB) = 20 run、6 hr 実測 (subagent K exp007 log)
  - 新 5 fold × 2 model × 3 seed = 30 run、各 5-10 min/run = 150-300 min = **2.5-5 hr** (cap 内)
  - CB GPU が co-occupies なら CB は 2 seed に絞る fallback も用意
- 期待 CV 改善: -0.2 〜 -0.5 ft
- 失敗 mode 3 件:
  1. CB GPU で seed loop 中 OOM (= 6 model × 5 fold concurrent state) → seed 単位で `del cb_model; gc.collect()` を fold 内に入れる
  2. 30 run が 9 hr cap を超過した場合の早期 break → 既存 `OWN_TRAIN_HARD_TIMEOUT_S = 6 * 3600` で fail-safe、Multi-seed の場合 partial seed の MEDIAN を使う
  3. MEDIAN aggregate を予測時に `np.median(stack, axis=0)` でやるのは fold ごとに別配列なので fold loop の中で aggregate しない (= fold 完了後に一括) → 実装時に注意

### 改修 4: fold-misalign 解消 (path b = kb-side simple-average separate blend)

- 対象 line: 9-base Ridge meta fit (= line 3322-3390)
- 変更:
  - 9-base Ridge を解体、**2-stage blend** に refactor:
    - Stage 1: own 4-base (= 改修 3 後は 2-base med) を **own-Ridge meta** で fit (= 自前 OOF が真の OOF なので safe)
    - Stage 2: $\hat{y}_{\text{final}} = w_{\text{kb}} \cdot \bar{y}_{\text{kb-5-mean}} + w_{\text{own}} \cdot \text{Ridge}(\hat{y}_{\text{own}})$
    - $w_{\text{kb}}, w_{\text{own}}$ は OOF-grid search で 1D 探索: $w_{\text{kb}} \in [0.0, 1.0]$ step 0.05、$w_{\text{own}} = 1 - w_{\text{kb}}$
  - 注: kb 側 OOF は karnakbaev published OOF (= 元 karnakbaev fold の真の OOF) を使う、ただし **Ridge には入れない** = fold-misalign leak 回避
  - kb 側 5 base は `Sx_kb.mean(axis=1)` で simple-average、これを 1 列として扱う
- 数理:
  - 元 9-base Ridge: 異なる fold partition の OOF を同 row index で Ridge fit → kb base の予測が own fold の train データを既に観測 → leak
  - path b: kb 側を「単一 column の external predictor」として扱い、own Ridge とは独立な 1-D 線形 blend → fold-misalign leak が消える
  - $w_{\text{kb}}^* = \arg\min \|y - (w_{\text{kb}} \bar{y}_{\text{kb}} + (1-w_{\text{kb}}) \hat{y}_{\text{own-Ridge}})\|_2^2$ は閉形式解 (= 1D 線形回帰)
- 工数: 1-2 hr (= 100 行 refactor)
- runtime: 既存と同等 (= Ridge fit 数秒、grid search 1 秒)
- 期待 CV 改善: -0.2 〜 -0.4 ft (= leak 解消による OOF 推定の bias-correction)
- 失敗 mode 3 件:
  1. own Ridge が degenerate (= own_total_w > 0.5 limit を OOF で hit) → STACK_OWN_WEIGHT_DEGRADE_LIMIT 0.5 → 0.8 に緩和、own-base diversity が下がっても backup として保持
  2. $w_{\text{kb}}^*$ が boundary (0 or 1) に張り付く → boundary なら片方単独使用、ログで warn
  3. kb published OOF が元 fold で fit されているので val data が own fold の train に交差し、`w_{\text{kb}}` の OOF estimate は依然 optimistic biased → LB と CV の gap を monitor、本格的解消は v4 (path a) で

## 3. 累積期待効果

| 改修 | CV Δ | LB Δ | runtime Δ |
|---|---|---|---|
| 1. Huber | -0.30 | -0.30 | ±0 |
| 2. Hetero | -0.20 | -0.20 | +5% |
| 3. MEDIAN | -0.30 | -0.30 | +0% (= 4 base → 2 model × 3 seed、ほぼ同 wall) |
| 4. path b | -0.30 | -0.30 | -1% (= Ridge を 1D grid search に置換) |
| 合計 (mid) | -1.10 | -1.10 | 9 hr cap 内 |
| **想定 CV** | exp008 v2 base CV 9.6 → **8.5** | exp008 v2 LB ?.? → **8.7-9.0** | OK |

## 4. push 後の検証チェック

Kaggle log で確認:
- `[exp008] Huber loss enabled (LGB alpha=0.9, CB delta=0.5)` のログが出ること
- `[exp008] heteroscedastic weight: min=0.50, max=2.00, mean=1.00` のログが範囲内
- `[exp008] Multi-seed MEDIAN: 2 model × 3 seed × 5 fold = 30 train runs`
- `[exp008] path b blend: w_kb=0.XX, w_own=0.XX, blend OOF RMSE=8.X`

Kaggle 上で実 run は中央が判断 (= exp007 LB + exp008 v2 LB + exp009 v2 LB 結果待ち)。

## 5. 残課題 (v4 以降)

- path a (= karnakbaev pretrained を Edge Q fold で真の OOF 再生成) を v4 で実装、kb 側 OOF も真の OOF にして Ridge を 9-base に戻す
- Adversarial Validation (= GM §1.1) は本 v3 に未含、AUC > 0.7 なら test 分布 shift driver を feature_importance で特定 → v4
- Discrete dTVT classification + cumsum (= 数理 §1.3 C、Track 4 構造変革) は v5 以降
