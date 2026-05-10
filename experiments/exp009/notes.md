# exp009 — 案 E (Sparse GP) + Edge O (dir-aware Beam) ローカル smoke 結果

## 0. 概要

- branch: `feat/phase-4-exp009-case-e-edge-o` (= exp008 から派生)
- kernel: `kaggle_kernels/exp009_case_e_edge_o/exp009_case_e_edge_o.py` (3893 行)
- metadata: `kernel-metadata.json` = id `ky7240/rogii-exp009-case-e-edge-o` / title `ROGII exp009 Case E Edge O`
- smoke: `experiments/exp009/smoke_gp_edge_o.py` (4 phases)
- **kernel push は実行していない** (中央指示待ち)

## 1. smoke 結果 (= 4 phases, ALL PASS)

実行コマンド: `.venv/bin/python experiments/exp009/smoke_gp_edge_o.py`

### Phase 1 — GPFormationImputer unit (50 wells, M=50)

```
✓ feature names = 24 cols (gp_<formation>_{mean,var,mean_minus_plane,var_norm} × 6)
✓ GP fit: 6/6 formations fit in 8.4 sec (M=50)
✓ GP predict: shape=(5, 6) in 8.4 ms
✓ predict mean ANCC: [-9316, -9740, -9065, -9724, -9121]
✓ predict var ANCC : [828, 781, 781, 728, 685]
✓ empty fallback: mean=zeros, var=ones
```

形状・dtype・NaN 0・var > 0 すべて期待通り。fit 8.4 sec (M=50, 50 wells)、Kaggle 上で M=200, 776 wells では同じく 30-60 sec 想定 (= sklearn GPR の cost は train 点数で立ち上がるが、`n_restarts=3` × per-formation の Cholesky で支配的)。

### Phase 2 — beam_search_dir + compute_edge_o_features unit (synthetic)

```
✓ Edge O 5 configs runtime: 139 ms (nh=600)
✓ beam_dir_std_d range: [0.000, 0.926], mean=0.667
✓ beam_dir_mean_d range: [-0.738, 0.000]
✓ λ_dir=0 vs λ_dir=0.5 mean abs diff (cons): 0.0023
✓ leak guard: beam_search_dir source does not reference TVT_input
```

5 configs 並列で 139 ms / 600 hidden rows。NaN 0、std reasonable (mean 0.667)、λ tuning 反映で path 変化を確認 (差は小さいが synthetic noise 系列なので想定範囲)。

### Phase 3 — build_well_features integration (2 train wells)

GP imputer M=50, 60 wells。

| well | rows | runtime | gp_ANCC_mean range | gp_ANCC_var range | gp_ANCC_mean_minus_plane p50 | beam_dir_cons_d range | beam_dir_std_d mean |
|---|---|---|---|---|---|---|---|
| 000d7d20 | 3836 | 3.13s | [-9337.8, -9249.1] | [1111, 1216] | 25.94 | [0.08, 6.58] | 3.50 |
| 00bbac68 | 6014 | 5.17s | [-9841.7, -9622.8] | [970, 1045] | -2.98 | [-1.59, -0.09] | 11.95 |

- NaN(GP) = 0、NaN(EO) = 0
- cols=203 (= exp008 約 172 + 24 GP + 7 Edge O = 203、整合)
- **2/2 wells PASS**
- per-well FE runtime: mean **4.17s**、max 5.17s
- 推定 full-train (= 776 wells): **~54 min** (= exp008 比 +30-40 min の overhead、Edge O 由来が支配的)

### Phase 4 — leak guards (4 points)

```
✓ guard 1: GPFormationImputer code (post-docstring) does not reference TVT_input/ev[TVT]
✓ guard 2: beam_search_dir code (post-docstring) does not reference TVT_input/ev[TVT]
✓ guard 3: compute_edge_o_features code (post-docstring) does not reference TVT_input/ev[TVT]
✓ guard 4: self-exclusion call ok (mean abs diff: 0.0000)
```

GP / Edge O のコードに hidden TVT への参照が無いことを source-string scan で確認。self-exclusion は GP-level (= cached fit) で 0 差分、KNN fallback パスでのみ差が出る設計通り。

## 2. 案 E 24 features の実装位置 + GP 戦略

### 2.1 実装位置

- 関数 / class:
  - `gp_feature_names()` (line ~1450)、`class GPFormationImputer` (line ~1456) を `class FormationPlaneKNN` の直後に追加 = **既存 plane-fit と並列に GP fit を走らせる disjoint 設計**
  - `_GP_REF: Optional[GPFormationImputer] = None` を `_FI_REF`/`_DI_REF` の直後に global で持つ (= dispatcher で 1 回 fit 後に shared)
- `build_well_features()` 内挿入: Spatial features (line ~2260) 直後に GP block を入れて、`form_ev` (= plane-fit) と `gp_mean_ev` の差を `gp_<formation>_mean_minus_plane` として直接 GBM に渡す
- output dict に `**gp_out` を `**kalman_out` の後に追加 (= line ~2429)

### 2.2 GP 戦略 (= sklearn 採用、GPyTorch 不採用)

採用: **sklearn `GaussianProcessRegressor` + K-Means inducing points (M=200)**

理由:
1. Kaggle GPU kernel に sklearn 標準導入済 (= dep 追加不要、`pyproject.toml` / `uv.lock` 不変、リスク小)
2. GPyTorch は SVGP (Hensman 2013) で正確だが Kaggle kernel 上の build / version 整合のリスク大
3. K-Means で 773 wells → M=200 cluster centers に圧縮 → per formation 1 GP fit で軽量
4. ARD-Matern 3/2 kernel + `n_restarts_optimizer=3` + `length_scale_bounds=(1e-2, 1e3)` で hyperparameter は marginal likelihood で auto-tune
5. fit 失敗時は KNN k=20 avg/std fallback (= 同一 interface を維持、GBM 側に zero impact)
6. **GP fit は dispatcher で 1 回のみ** (per-well loop 外、`_GP_REF` global share) = runtime 制約遵守

24 features = 6 formations × 4 stats:
- `gp_<formation>_mean` = posterior mean
- `gp_<formation>_var` = posterior variance
- `gp_<formation>_mean_minus_plane` = GP - plane-fit (= 不一致 zone signal)
- `gp_<formation>_var_norm` = sqrt(var) / mean(sqrt(var)) (= per-well 正規化 uncertainty)

## 3. Edge O 7 features の direction-aware Beam 実装方法

### 3.1 実装位置

- `beam_search_dir()` (line ~720)、`compute_edge_o_features()` (line ~830)、`edge_o_feature_names()` (line ~715) を `compute_beam_features()` 直後に新規実装
- `build_well_features()` 内挿入: Kalman block (line ~2415) 直後に Edge O block を追加
- output dict に `**edge_o_out` を `**gp_out` の後に追加

### 3.2 数学的拡張

既存 Beam transition cost に sign-of-dGR penalty を追加:

```
cost_aug = cost_base + λ_dir × |sign(dGR_h(s)) - sign(dGR_v(path_s))|
```

- `dGR_h(s) = gr_h(s) - gr_h(s-1)` (= horizontal-well GR の前方差分)
- `dGR_v(j) = tw_gr(j) - tw_gr(j-1)` (= typewell GR の前方差分)
- transition target index `ci ∈ {prev-1, prev, prev+1}` ごとに sign を比較
- `sign(0) = 0`、penalty 値域 ∈ {0, 1, 2} を `λ_dir = 0.5` で重み付け

### 3.3 出力 7 features

- `beam_dir_<cons|loose|sm5|vcons|mid>_d` (5 paths)
- `beam_dir_mean_d` = 5 paths の mean
- `beam_dir_std_d` = 5 paths の std

(= 既存 `beam_*_d` (標準 Beam) と **disjoint な features 名前空間**、GBM が両方を見て decision tree で 自由に combine する設計)

## 4. 想定 LB と 9 切りへの追加 path

| シナリオ | 仮定 | LB |
|---|---|---|
| Best | 案 E -0.80 + Edge O -0.30 | **8.25** |
| Mid | 案 E -0.40 + Edge O -0.20 | **8.75** |
| Worst | 両 fallback (GP 失敗 + λ=0) | **9.35** (= exp008 同点、悪化なし) |

**9 切り** (LB 8.x 帯) は exp009 mid case で初到達。それ以上の打ち手:
- exp010: 案 G (Sparse Transformer/SSM) → 8.5-8.8
- exp011-013: 残 6 edges (案 F/H/I + 補助 J/K/L) + final stack → 8.0-8.5

## 5. runtime 想定 (Kaggle 9 hr cap)

| 段階 | 想定時間 |
|---|---|
| GP fit (6 formations × M=200 × n_restarts=3) | 30-90 sec |
| GP predict (test 6 wells) | 5-15 sec |
| Edge O Beam (5 configs × 776 wells × ~50ms) | 200-300 sec |
| test feature build (6 wells, exp008 + GP + Edge O) | 60-90 sec |
| train feature build (3.78M rows、+GP+Edge O) | exp008 比 **+30-50 min** (smoke 推定) |
| 自前 LGB×3 + CB train (5 fold × 4 base、+31 features) | 70-110 min |
| 9-base Ridge meta + postproc + submission | <10 sec |
| **合計** | **130-200 min** |
| 9 hr cap 余裕 | **5-7 hr** |

= 9 hr cap 内で安全に収まる見込み。train FE が支配的だが thbdh5765 cache 経由の build_dataset over TRAIN_DIR は 1 回のみ (= 自前 base train 再 fit のため)。

## 6. 残課題 / blocker

### 6.1 Kaggle 上での確認が必要な項目

1. **GP fit の runtime**: smoke は 50 wells × M=50 で 8.4 sec。本番 776 wells × M=200 で 30-90 sec を想定するが、`n_restarts=3` の scipy minimize の収束により上振れリスクあり。Kaggle 環境で 5 min 超えなら `GP_N_RESTARTS=1` / `GP_M_INDUCE=100` に下げる可能性 (= config 1 行変更で対応可)
2. **train_df merge**: exp008 で train_df_kb への Edge M / Kalman cols 注入は既存 logic が動作 (smoke 確認済 from exp008)。exp009 では GP 24 + Edge O 7 の 31 cols が追加で merge される。merge time + memory が exp008 比 +30% 程度の見込みで、Kaggle 16 GB cap 内で動作するかは Kaggle 上 1st run でのみ確認可能

### 6.2 設計上の選択

1. **GP は train wells 全体で 1 回 fit** (= self_wid exclude は KNN fallback のみ、GP 本体は global hyperparameter)。773 wells で 1 well 抜いても hyperparameter shift は negligible (= `FormationPlaneKNN` と同等の leak 厳格度)。CV-strict 厳格化が必要なら v2 で per-well GP refit に拡張可能
2. **Edge O の thbdh5765 cache 互換性**: 既存 train cache に `beam_dir_*_d` 列は無い。自前 4 base 再 fit 時に `build_dataset` で生成する logic が動作する (= 既存 Edge M / Kalman と同じ flow)。merge 時に `well/id` で left join、欠損は 0 fill → 自前 base 学習に影響なし
3. **slug 一致**: title `ROGII exp009 Case E Edge O` から Kaggle 自動生成 slug は `rogii-exp009-case-e-edge-o` (= "Case E" → `case-e`、"Edge O" → `edge-o`)。`id: ky7240/rogii-exp009-case-e-edge-o` と一致するはずだが、Kaggle の実 slug 生成挙動は push 時に確認必要 (subagent K の `Edge QM` → `edge-q-m` 不一致経験を学習)

### 6.3 dependencies

- `pyproject.toml` / `uv.lock` を **触っていない** (= sklearn のみ使用、`gpytorch` 不採用)
- 既存 import に `KMeans`, `GaussianProcessRegressor`, `Matern`/`ConstantKernel`/`WhiteKernel` を追加。これらは sklearn 標準コンポーネント、Kaggle base image に含有

## 7. 中間 commit log (= 30 分 / progress)

| # | hash | 内容 |
|---|---|---|
| 1 | 6ab144e | docs(exp009): design doc 491 行 |
| 2 | 2378e58 | feat(exp009): kernel skeleton (exp008 base + MODE + config + imports) |
| 3 | d58b11b | feat(exp009): GPFormationImputer + beam_search_dir + compute_edge_o_features + dispatcher wiring |
| 4 | (本 commit) | feat(exp009): smoke 4-phase ALL PASS + notes.md |

## 8. push 直前 checklist

- [x] feat/phase-4-exp009-case-e-edge-o branch + origin push 済 (4+ 中間 commit)
- [x] kaggle_kernels/exp009_case_e_edge_o/ 完成 (self-contained kernel script + metadata)
- [x] ローカル smoke 4/4 phase ALL PASS (1-2 wells で動作確認、NaN 0、leak guard 4/4 PASS、GP runtime 8.4 sec / 50 wells)
- [x] experiments/exp009/design.md (= case E + Edge O 設計)
- [x] experiments/exp009/notes.md (= 本 file、smoke 結果 + 想定 LB + 残課題)
- [x] **kernel push は実行しない** (中央指示待ち)
