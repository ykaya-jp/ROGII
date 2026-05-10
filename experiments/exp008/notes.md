# exp008 — 実装ノート (subagent L)

## 0. 状況 (2026-05-11)

- 担当: subagent L (= 案 D Kalman/PF on dTVT 投入、Phase 4 first edge)
- branch: `feat/phase-4-exp008-case-d-kalman` (= `feat/phase-3-exp007-edge-q-m` から派生)
- 目的: exp007 base camp (LB 9.5-9.7 想定) に AR(1) Kalman 7 features を追加し、9 切り路線 (LB 8.x) への第 1 段を作る
- 制約: ローカル smoke 完了まで実行、kernel push は中央指示待ち

## 1. 実装の要点

### 1.1 案 D 7 features の実装位置と計算順序

`build_well_features()` 内で以下の順番:

| # | 計算 | line (exp008) |
|---|---|---|
| 1 | visible/hidden split, anchors, GR features | ~1530-1620 |
| 2 | Beam features (`bf` 含む `beam_cons_d`) | ~1626 |
| 3 | Particle Filter ANCC (`pf_a_sel`) + PF Z (`pf_z_sel`) | ~1631 |
| 4 | self-correlation NCC, spatial features | ~1644-1660 |
| 5 | **Edge M (4 features)** | ~1854-1879 |
| 6 | **★ Edge D Kalman (7 features) ★** | **~1881-1907** |
| 7 | output dict assemble | ~1910-2010 |

挿入位置の根拠: `pf_a_sel` (PF ANCC) と `bf["beam_cons_d"]` (Beam consensus) が既に scope に在ることが必須 (= disagreement features 計算のため)、かつ output dict 直前であること。

### 1.2 per-well φ + σ_ε MLE

- 採用: **Yule-Walker AR(1) closed form** (= `dt[1:] · dt[:-1] / (dt[:-1])²`)、純 numpy
- 不採用: `statsmodels.tsa.ARIMA` (= 高機能だが overhead 大、773 wells × 5 fold で重い)
- 数値ガード: `phi_clip = 0.99999` (= 1 - phi² > 0 確保)、`sigma_floor = 1e-4`
- 短い visible (n < 4) では NaN を返し、shrinkage で global prior に丸投げ

### 1.3 hierarchical Bayesian shrinkage

- formula: `phi = w * phi_local + (1 - w) * phi_global`、`w = n_eff / (n_eff + λ × pseudo)`
- λ_phi = λ_sigma = **0.3** (= 初期値、smoke で挙動確認済)
- pseudo_count = 100 (= n=100 visible で local と global が同等重み、n=1000 で local 97%)
- global prior:
  - `phi_global = 0.99887` ← `outputs/eda/first_principles/per-well-stats.parquet` ar1_phi p50
  - `sigma_global = 0.01556` ← parquet ar1_eps_std p50
  - parquet 計測値と kernel hardcoded の一致を Phase 3 smoke で assert

### 1.4 Kalman forward pass

- state = dTVT(s) (1D scalar)
- propagate: `dt_{k+1} = phi * dt_k` (mean-zero noise), `cum_dt += dt_k`
- variance: `cum_var += sigma²` per step (= random walk 近似、phi ≈ 1 で正確)
- step unit: `median_dmd` (= visible MD diff の median)、hidden MD distance を steps に換算
- 各 hidden row で `kalman_d = cum_dt`, `kalman_std = sqrt(cum_var)` 出力

### 1.5 features_cols の分離 (= karnakbaev model 互換性)

- `feature_cols_kb` (= 154 cols) ← karnakbaev pretrained 5 base predict 専用 (schema 固定)
- `feature_cols_own` (= 154 + 4 Edge M + 7 Kalman = **165 cols**) ← 自前 4 base train + predict 専用
- karnakbaev `train_df.parquet` には Edge M / Kalman 列が無いので、kernel 内で `build_dataset(TRAIN_DIR, is_train=True)` で再構築し `(well, id)` で merge して inject する logic を Step B' として追加

## 2. ローカル smoke 結果 (Phase 1 + 2 + 3 全 PASS)

`.venv/bin/python experiments/exp008/smoke_kalman.py` 実行ログより:

### Phase 1: unit test (compute_kalman_features 単独)

| 検査 | 結果 |
|---|---|
| feature_names 7 cols | ✓ |
| YW MLE on synthetic AR(1) (phi=0.95, sigma=0.5) | ✓ phi_hat=0.9401, sigma_hat=0.4946 |
| Kalman forward pass nk=1500, n_hidden=3000 | ✓ runtime **6.1 ms** (= 1 well 数十 ms 確認) |
| 全 7 cols populated, NaN=0, Inf=0 | ✓ |
| kalman_std monotonic increasing | ✓ first=0.0157, last=0.8589 |
| 短い visible (nk=3) | ✓ shrinkage で global prior に fall back |
| 空 hidden | ✓ shape (0,) を返す |
| **leak guard 1** (source 内 hidden TVT 参照無し) | ✓ |

### Phase 2: integration smoke (build_well_features 経由、2 wells)

| well | rows | runtime | NaN | std_mono | const_per_well | kalman_phi | kalman_sigma | kalman_d range |
|---|---|---|---|---|---|---|---|---|
| 000d7d20 | 3836 | 2.37 sec | 0 | True | True | 0.98967 | 0.05819 | [0.010, 0.935] |
| 00bbac68 | 6014 | 3.86 sec | 0 | True | True | 0.99845 | 0.02416 | [0.020, 13.245] |

per-well φ が 0.989 vs 0.998 と well 間で variation を持っており、shrinkage が正常に機能。
`kalman_vs_pf_a_d` も非ゼロ値範囲 ([-7.9, 11.7]) で差分シグナルが出ている。

### Phase 3: parquet sanity

- `outputs/eda/first_principles/per-well-stats.parquet` (n=773) の p50 が kernel hardcoded global prior と一致:
  - `ar1_phi p50 = 0.99887` (kernel = 0.99887) ✓
  - `ar1_eps_std p50 = 0.01556` (kernel = 0.01556) ✓

### leak guard 4 点 assert (まとめ)

1. **visible only**: `compute_kalman_features` 引数に hidden TVT を渡さない (= ktvt, kmd, hmd[MD のみ], pf_a_d, beam_cons_d) ✓
2. **no hidden TVT**: 関数 source code 内に `TVT_input`, `ev["TVT"]`, `ev['TVT'`, `ev.TVT` の string 不在を assert ✓
3. **no future MD**: hmd は forward iteration のみ (= no smoother) ✓
4. **per-well isolation**: φ, σ_ε は当該 well visible のみで MLE、global prior は precomputed 773 wells p50 (contamination 無し) ✓

## 3. 想定 LB と 9 切りへの追加 path

| シナリオ | 仮定 | LB |
|---|---|---|
| Best | exp007 base 9.5 + Kalman -0.30 + disagreement -0.10 | **9.10** |
| Mid (期待値) | exp007 base 9.6 + Kalman -0.20 + disagreement -0.05 | **9.35** |
| Worst (Kalman 寄与なし) | exp007 base 9.7 + Ridge weight 微小 | **9.70** (悪化なし) |

**9 切りまで残 0.0-0.1**: exp008 単独で 9.0 帯到達は楽観、9.3-9.5 帯が現実的。

9 切り達成 path:
- exp008 期待値 = 9.35
- exp009 (= 案 E: Bayesian GP for ANCC + posterior variance feature) → -0.30〜-0.50 ⇒ **9.0 帯到達**
- exp010 (= 案 G: Sparse Transformer/SSM 直予測) → -0.20〜-0.40 ⇒ 8.7-8.8 帯
- exp011-013 (= 残 6 edges + final stack tuning) → 8.0-8.5

## 4. 失敗回避ログ (= 設計時に対策した failure mode)

### 4.1 末端 drift (φ < 1 で 0 へ regress)

- **設計対策**: shrinkage で `phi → 0.99887` global prior に λ=0.3 で pull、+ smoke で per-well phi を log
- **smoke 観測**: 000d7d20 で phi=0.989, 00bbac68 で phi=0.998 と well 間 variation 確認済、いずれも 0.99 以上で実用上 random walk として機能

### 4.2 非ガウス noise (fat tail)

- **設計対策**: 本 exp008 は Kalman のみ実装、Particle Filter version は v2 候補 (compute_kalman_features docstring に TODO 記載)
- **smoke 観測**: GBM が disagreement features (`kalman_vs_pf_a_d`) を見て fat tail zone で Kalman 信頼を下げる挙動に期待 (= 直接 train でなく feature 提供のみ)

### 4.3 visible 末端 σ_ε MLE が短い visible で不安定

- **設計対策**: hierarchical shrinkage + sigma_floor=1e-4
- **smoke 観測**: short visible (nk=3) でも NaN/Inf 発生せず、global prior 0.01556 に shrink して動作確認済

### 4.4 [追加発見] phi clip 必要性

- **smoke 中の懸念**: synthetic AR(1) の phi_true=1.0 だと `1 - phi²` が 0 になり ill-conditioned
- **対策**: KALMAN_PHI_CLIP = 0.99999 で確実に閾値以下にクリップ、Var formula を `cum_var += sigma²` per step (= random walk 近似) に統一して closed-form `(1 - phi^2t)/(1 - phi^2)` を回避

### 4.5 [実装中発見] karnakbaev train_df に Edge M/Kalman 列が無い

- **症状**: `train_df.parquet` (= karnakbaev artefact) は 154 cols のみ、自前 4 base が Kalman を学習できない
- **対策**: kernel 内で Step B' として `build_dataset(TRAIN_DIR, is_train=True)` を再実行し `(well, id)` で merge inject。`feature_cols_kb` (154) と `feature_cols_own` (165) を分離して karnakbaev predict は 154 cols に温存

## 5. 残課題 / blocker

1. **kernel push 未実行** (= 中央が exp007 LB 結果待ち + 判断後に push 指示)
2. **Step B' (train rebuild) の runtime cost** が exp007 比 +20-40 sec 想定。773 wells × 自前 train Step B' で **+10-15 min** 追加される可能性。9 hr cap 内余裕あり
3. **Particle Filter version (v2)** は dormant: 案 D の真の差分化 (= fat tail に強い PF on dTVT) は本 exp008 で未投入。exp008 LB が想定下限 (Worst 9.7) に近い場合は v2 切替を検討
4. **smoke は train データのみ**: test 6 wells での実動作は kernel push 時の Kaggle session で初めて確認
5. **Step B' で生成される train extras parquet を artefact 保存しない**: 現状毎回再計算。runtime は 1 回 +20-40 sec で許容範囲

## 6. 完了条件 (本 subagent の)

- [x] `feat/phase-4-exp008-case-d-kalman` ブランチ + origin push (= 4 中間 commit)
- [x] `kaggle_kernels/exp008_case_d_kalman/` 完成 (= self-contained kernel script + metadata)
- [x] ローカル smoke pass (1-3 wells で動作確認、NaN 0、leak guard 4 点 PASS、Kalman std MD で grow 確認)
- [x] `experiments/exp008/design.md` (= case D 設計、7 features、leak check、failure recovery)
- [x] `experiments/exp008/notes.md` (本 file)
- [ ] **kernel push は実行しない** (= 中央指示待ち) ← 仕様通り

## 7. References

- `experiments/exp008/design.md` (= 設計 doc)
- `experiments/exp008/smoke_kalman.py` (= smoke runner、再現用)
- `kaggle_kernels/exp008_case_d_kalman/exp008_case_d_kalman.py` line 1162-1340 (= compute_kalman_features 関数)
- `kaggle_kernels/exp008_case_d_kalman/exp008_case_d_kalman.py` line 1881-1907 (= build_well_features 内挿入)
- `outputs/eda/first_principles/per-well-stats.parquet` (= 773 wells × ar1_phi/eps_std 計測元)
- `docs/research/independent-edges.dense.md` §1
- `docs/research/first-principles.dense.md` §2.1
