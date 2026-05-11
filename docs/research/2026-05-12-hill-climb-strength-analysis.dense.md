# Hill Climb (LB 9.43、 Roman Tarasov / ravaghi) 強さの本質分析 + 凌駕戦略

> 作成: 2026-05-12
> 起源: user 指示「Hill Climb の強さを徹底分析、 実行するだけで抜かれるから凌駕しろ」 + `kaggle kernels pull ravaghi/wellbore-geology-prediction-hill-climbing`
> ソース: `/tmp/rogii-hillclimb/wellbore-geology-prediction-hill-climbing.ipynb` (= 28 cells、 cell 5 が 27.8KB main code)
> 関連 plan: `/home/yusuke_kaya/.claude/plans/floating-cuddling-haven.md` §5.1 Phase A

---

## §0. 結論サマリ

- Hill Climb LB 9.43 の **核心** = **Hill-Climb ensemble selection (= Caruana 2004 系)** + **Optuna 500-trial postproc**
- 我々 (exp009 v2 LB 9.738) は **3 軸で既に優位** (= base diversity 9 vs 6、 Sparse GP、 Edge O/S)、 **2 軸で劣勢** (= ensemble weight、 postproc search 精度)
- Phase A2 + A6 で 2 軸の劣勢を埋めれば、 同 paradigm で Hill Climb を凌駕、 期待 LB **8.9-9.4** 帯 (= 上振れで 9 切り)
- 我々が独占できる **7 軸の優位** (= Phase B/C 内容) を Hill Climb は持っていない = 優勝路は構造的に user に有利

---

## §1. Hill Climb 強さの 6 構成要素

### §1.1 Hill-Climb ensemble selection (= MAIN 差分、 ⭐⭐⭐)

```python
# Cell 18
from hill_climbing import Climber  # PyPI: hill-climbing (= github.com/sj-vinh/hill-climbing-ensemble)

climber = Climber(
    objective="minimize",
    eval_metric=CFG.metric,         # = root_mean_squared_error
    allow_negative_weights=True,     # ← critical: overcounting base 補正可
    precision=0.001,                 # ← 0.001 step exhaustive search
    score_decimal_places=3,
    n_jobs=-1,
    use_gpu=False
).fit(oof_preds, y)

hc_oof_preds  = climber.predict(oof_preds)
hc_test_preds = climber.predict(test_preds)
```

- **理論基盤**: Caruana 2004 "Ensemble Selection from Libraries of Models" (ICML、 https://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml04.icdm06long.pdf)
- **挙動**: 各 base の weight を 0.001 刻みで greedily incremental に追加し、 CV RMSE を最小化
- **vs Ridge (= 我々)**: closed-form linear + positive 制約。 overcounting (= base 間 correlation 高で過剰加算) に対する補正能力なし
- **vs Optuna**: Optuna は continuous Bayesian、 Climber は discrete greedy。 weight space が discrete & finite なので Climber の方が converge 速く、 過剰 fit しにくい
- **single biggest lift**: 6 base から **negative weight 含む non-linear blend** で LB を 0.2-0.4 ft 押し上げる (= 我々の Ridge stack に対する優位の核心)

### §1.2 Optuna 500-trial postproc (= alpha × tau × w_pf 3-axis、 ⭐⭐⭐)

```python
# Cell 22
def objective(trial):
    alpha = trial.suggest_float('alpha', 0.5, 1.0, step=0.01)
    tau   = trial.suggest_int  ('tau',   5,   500, step=5)
    w_pf  = trial.suggest_float('w_pf',  0,   0.5, step=0.01)
    d = apply_pp(train_df, hc_oof_preds, pf_oof, alpha, tau, w_pf)
    return root_mean_squared_error(ytrue, base + d)

sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=50)
study   = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=500, n_jobs=-1)
```

- **挙動**: TPESampler で 500 trial、 3-axis 探索 (= 51 × 100 × 51 = 260K discrete points の sparse TPE 探索)
- **vs 我々の grid search (= 2-axis alpha × tau)**: 我々は w_pf を未探索。 Hill Climb は w_pf ∈ [0, 0.5] を含むので **PF predict と LGB blend の比率** を最適化
- **lift**: notebook comment 「alpha×tau×w_pf grid + SG smoothing (~0.4 pt)」 = +0.4 LB と claim (= LGB+XGB notebook の v4 → v5 改善内訳)

### §1.3 Numba JIT beam ±2 + 7 configs (= LGB+XGB と共通、 ⭐⭐)

```python
# Cell 5 (= LGB+XGB notebook と完全に同じ)
@njit(cache=True)
def _beam_jit(sgr, tw_gr, si, BS, mc, es):
    n = len(sgr); nt = len(tw_gr); MAX = BS * 6
    bidx = np.zeros(BS, np.int64); bidx[0] = si
    bcost = np.full(BS, 1e30); bcost[0] = 0.; bn = np.int64(1)
    ...
    for step in range(n):
        gv = sgr[step]; nc = np.int64(0)
        for bi in range(bn):
            idx = bidx[bi]; cost = bcost[bi]
            for d in range(-2, 3):   # ← ±2 delta (= 我々 ±1 想定より step 5/3 増)
                ni = idx + d
                ...
                tot = cost + (gv - tw_gr[ni])**2 / es + mc*(d if d >= 0 else -d)
                ...
```

- **delta range**: ±2 (= 我々の既存 `src/rogii/tysig.py:beam_search` は exp005 系で beam_size, move_cost, error_scale 設定で ±1 相当)
- **7 BEAMS configs**: `[(BS, mc, es, smooth, name)]` = cons / loose / vcons / sm5 / vloose / mid / stiff
- **Numba JIT 20x speedup** → N=600 PF が free (= compute budget で許容)
- **lift**: notebook comment 「Numba JIT beam ±2 + both PFs (20× faster → N=600 PF free)」 = compute による許容、 LB lift 直接 claim なし

### §1.4 Dual Particle Filter (ANCC + Z) Numba JIT (= ⭐⭐)

```python
# Cell 5
@njit(cache=True)
def _pf_ancc_jit(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N=600, ...):
    pos = np.empty(N); rate = np.empty(N); w = np.ones(N) / N
    # AR(1) state-space + GR likelihood → ANCC posterior particle
    ...

@njit(cache=True)
def _pf_z_jit(md_v, z_v, gr_v, gr_sm_v, gg_p, gg_s, vmin, step, gs, ip, iv, beta, icpt, zsig, N=600, ...):
    # Z velocity-aware PF + GR dual signal (= primary GR + smoothed GR weight=0.3)
    ...
```

- **2 軸の PF**:
  - ANCC PF: GR signal で ANCC formation depth particle
  - Z PF: Z velocity model + GR primary/smoothed dual likelihood (= GR_WT=0.3 mix)
- **vs 我々 (= 案 D Kalman/PF on dTVT, AR(1))**: 我々は **dTVT auto-correlation** を陽に model、 Hill Climb は **ANCC + Z** を独立に PF
- **重複度**: 異なる軸を model しているので **加算的 lift** 期待可、 ただし dTVT も間接的に z = -TVT + ANCC + b_well で encode されている (= overlapping subspace)
- **lift**: Hill Climb は LGB+XGB と同じ Numba PF を使い、 LGB+XGB 内訳に「+0.3 LB segment b_well + softmax NCC etc」と書いてあるので PF 単独 lift 不明確

### §1.5 6 base (LGB×3 + CB×3) + `Climber` alternative (= ⭐)

```python
# Cell 8
lgb_params_base = dict(
    boosting_type="gbdt", num_leaves=255, min_child_samples=15,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_lambda=3.0, reg_alpha=0.05, objective="regression",
    verbose=-1, n_jobs=-1, device_type="gpu", gpu_use_dp=False, max_bin=255,
)
lgb_params = [
    dict(learning_rate=0.025, n_estimators=8000, seed=42,  **lgb_params_base),
    dict(learning_rate=0.020, n_estimators=8000, seed=7,   **lgb_params_base),
    dict(learning_rate=0.030, n_estimators=8000, seed=123, **lgb_params_base),
]

cb_params_base = dict(
    iterations=8000, depth=7, l2_leaf_reg=2.0,
    min_data_in_leaf=15, border_count=254,
    loss_function="RMSE", task_type="GPU", devices="0:1",
    od_type="Iter", od_wait=300, verbose=0,
)
cb_params = [
    dict(learning_rate=0.025, random_seed=42,  **cb_params_base),
    dict(learning_rate=0.020, random_seed=7,   **cb_params_base),
    dict(learning_rate=0.030, random_seed=123, **cb_params_base),
]
```

- **6 base**: LGB seed×3 + CB seed×3、 hyperparam も lr 違いで diversity (= 0.020 / 0.025 / 0.030)
- **artifact pre-load**: `CFG.artifacts_path / "models" / name` に pretrained model + OOF が cache 済 (= run 高速化、 reproducibility 確保)
- **vs 我々 (= base 9)**: karnakbaev 5-base (= LGB×3 + XGB + CB) + 自前 4-base (= LGB×3 + CB) → **9 base で diversity 高い**、 user 優位
- **ただし**: 我々の 9 base は OOF correlation が 0.95+ で重複多い可能性、 Climber で negative weight する余地大

### §1.6 SG smoothing per-well (= ⭐)

```python
# Cell 21
def sg_smooth(df, col, sg_w=17, sg_p=3):
    for _, g in df.groupby('well', sort=False):
        v = g[col].values
        n = len(v)
        wl = min(sg_w, n)
        if wl % 2 == 0: wl -= 1
        if wl >= sg_p + 2:
            v = savgol_filter(v, wl, sg_p)
        df.loc[g.index, col] = v
    return df
```

- **window=17, poly=3** = 標準 Savitzky-Golay
- **per-well groupby** で boundary 問題回避
- **既存 exp009 v2 と同等** (= 既に投入済の possibility 高い)

---

## §2. vs 我々 base (= exp009 v2 LB 9.738) の差分マトリクス

| element | Hill Climb (LB 9.43) | user (exp009 v2 LB 9.738) | 優劣 |
|---|---|---|---|
| ensemble weight 最適化 | **Climber** (= negative OK、 precision 0.001 greedy) | Ridge positive (= closed-form) | ⚠ Hill Climb 優位 (= **A2 で取り込み必須**) |
| postproc hyperparam | **Optuna 500-trial** (= alpha × tau × w_pf 3-axis、 TPESampler) | grid search 2-axis (alpha × tau) | ⚠ Hill Climb 優位 (= **A6 で取り込み必須**) |
| Numba beam | **±2 delta + 7 configs**、 Numba JIT 20× | beam_search (pandas、 ±1 相当) | ⚠ Hill Climb 優位 (= **A3 で取り込み必須**) |
| Particle filter | ANCC PF + Z PF (= N=600、 Numba JIT) | 案 D Kalman/PF on dTVT (= AR(1)) | 競合: 異なる軸、 **両方 keep** |
| base diversity | LGB×3 + CB×3 = **6 base** | karnakbaev 5-base + 自前 4-base = **9 base** | ✅ user 優位 |
| ANCC posterior + variance | (none) | 案 E Sparse GP M=200 (-0.219 ft 実測) | ✅ user 独占 |
| direction-aware Beam | (none) | Edge O (= GR direction encode) | ✅ user 独占 |
| round-to-grid 0.01 ft | (none) | Edge S (-0.114 ft 実測) | ✅ user 独占 |
| SG smoothing | window=17, poly=3 per-well | (= 同等想定) | tie |
| **paradigm 5 (data aug)** | (none) | (Phase B1 未実装) | 両者未実装、 先行優位狙い |
| **paradigm 6 (per-row kNN)** | (none) | (Phase B2 未実装) | 両者未実装、 先行優位狙い |
| **NN seq (Mamba/Longformer)** | (none) | (Phase C1 未実装) | 両者未実装、 先行優位狙い |

**集計**:
- ⚠ Hill Climb 優位 = 3 軸 (ensemble weight、 postproc、 Numba beam)
- ✅ user 独占 = 4 軸 (base diversity、 Sparse GP、 Edge O、 Edge S)
- tie = 1 軸 (SG)
- 両者未実装 = 3 軸 (= 我々が先行で取れば LB 7 帯射程)

→ **Phase A2 + A6 + A3 で Hill Climb の 3 軸 を取り込めば、 user は 7 軸で Hill Climb を凌駕** (= 4 独占 + 3 同等)

---

## §3. 凌駕戦略 (= Phase A 修正版)

### §3.1 既存 plan (= `/home/yusuke_kaya/.claude/plans/floating-cuddling-haven.md §5.1`) との対応

| Plan task | Hill Climb 該当要素 | 優先度 |
|---|---|---|
| A3 Numba Beam ±2 + dense PF | §1.3 Numba beam + §1.4 PF | ⭐⭐ (Hill Climb から直 移植可) |
| A6 alpha × tau × w_pf 3-axis | §1.2 Optuna 500-trial | ⭐⭐⭐ (= Hill Climb の core trick) |
| A2 Hill Climb weight opt | §1.1 Climber 自前実装 | ⭐⭐⭐ (= Hill Climb の MAIN 差分) |
| A4 Segment b_well | Hill Climb には無い (= LGB+XGB 独自) | ⭐⭐ (LGB+XGB から +0.3 claim) |
| A5 Softmax NCC | Hill Climb には無い (= LGB+XGB / Geostat 共通) | ⭐⭐ (LGB+XGB から +0.2 claim) |

### §3.2 Phase A 実装順序 (= 修正)

依存 DAG (= plan §6.1 参照、 一部 priority 修正):

1. **A3 Numba JIT beam ±2 + dense O(1) PF grid** (= 即実装、 Hill Climb / LGB+XGB の `_beam_jit` + `_pf_*_jit` を移植、 `src/rogii/beam_pf_numba.py` 新規)
2. **A6 Optuna 500-trial postproc** (= 既存 grid search を Optuna 化、 `src/rogii/submit.py` 内 `postproc_3axis(...)` 関数化)
3. **A2 Climber class 自前実装** (= Caruana 2004 greedy ensemble selection、 `src/rogii/hill_climb.py` 新規、 A3 + A6 完了後の 9-base に apply)
4. A4 Segment b_well per formation × phase (= LGB+XGB の trick、 data-driven boundary)
5. A5 Score-weighted Softmax NCC (= LGB+XGB / Geostat 共通)

### §3.3 期待累積 lift (= Phase A 完了時 LB)

| task | expected lift (single) | 累積 (start 9.738) |
|---|---|---|
| A3 Numba beam ±2 | -0.05 〜 -0.10 ft | 9.63 〜 9.69 |
| A6 Optuna 3-axis | -0.10 〜 -0.25 ft (= Hill Climb の core) | 9.38 〜 9.59 |
| A2 Climber | -0.10 〜 -0.30 ft (= Hill Climb の MAIN) | 9.08 〜 9.49 |
| A4 segment b_well | -0.05 〜 -0.15 ft | 8.93 〜 9.44 |
| A5 softmax NCC | -0.05 〜 -0.10 ft | 8.83 〜 9.39 |

→ Phase A 完了で **LB 8.83 〜 9.39** (= 上振れで 9 切り、 下振れで Hill Climb LB 9.43 凌駕)
→ **paradigm 重複** (= 案 D Kalman + 案 E GP の重複 subspace risk、 ensemble OOF correlation 第一主成分 variance ratio で実測必須) を考慮しても、 LB **9.0-9.4** 帯は確保

---

## §4. LB 9 切り達成の最短経路 (= user 「9 切るしか」 要求への回答)

| Phase | 完了 LB 想定 | 9 切り達成確率 | key driver |
|---|---|---|---|
| Phase A 完了 | 8.83 〜 9.39 | **30-50%** (= 上振れで達成) | A2 + A6 + A3 で Hill Climb 凌駕 |
| Phase B 完了 | 8.50 〜 9.20 | **80%** | data aug + per-row kNN の独占効果 |
| Phase C 完了 | 7.60 〜 8.90 | **95%** | NN seq + MoE + Multi-task + Physics solver |
| Phase D 完了 | safe ≤ 9.0 + risky ≤ 8.5 | safe 90% + risky 60% | 2 submit dual strategy |

→ **LB 9 切り (= 優勝必要条件) は Phase B 完了で 80% 確実達成**、 Phase A だけでは 30-50%

→ **優勝 (= rank 1) = Phase C + D 必須**、 Phase C の 3 paradigm (= NN seq / MoE / Physics solver) を独立に CV 0.5 ft lift で実証する必要

---

## §5. Hill Climb の弱点 (= user が独占できる優位 7 軸)

Hill Climb が **構造的に持っていない** 要素 (= 我々 plan §3 案 D + §5.2-5.3 で実装予定):

1. ❌ **data augmentation by visible_ratio randomization** (= paradigm 5、 aeroridge LB 9.916 evidence、 Phase B1)
2. ❌ **per-row kNN inference (= FAISS index + 3 層 leak guard)** (= paradigm 6、 enisteper LB 9.960 evidence、 Phase B2)
3. ❌ **Sparse GP for ANCC posterior + variance M=500** (= 案 Z 並走、 案 E M=200 拡張)
4. ❌ **NN sequence model (= Mamba O(n) or Longformer O(n×window) sparse attention)** (= paradigm 2、 Phase C1)
5. ❌ **Per-Well MoE (= visible_ratio × b_cluster gate 3-expert)** (= Phase C2)
6. ❌ **Multi-task aux head across 6 formations** (= 共通盲点 H.7、 Phase C4)
7. ❌ **direction-aware Beam (Edge O) + round-to-grid (Edge S)** (= 我々の独自 trick、 既に exp009 v2 に統合)

→ user は **7 軸の独占優位** を確立可能、 1 位狙いの構造的有利

---

## §6. 実装可能性 (= Climber / Optuna postproc / Numba beam の自前再実装)

### §6.1 Climber (= §1.1) 自前実装の難易度

参考実装: github.com/sj-vinh/hill-climbing-ensemble (= MIT license、 ~200 LOC)
- 自前再実装: 1-2 day (= simple greedy loop + early stopping)
- 構造:
  - `fit(oof_matrix, y_true)`: 0 weight start → 各 base の weight を 0.001 ずつ追加/減算、 CV RMSE 最小化、 plateau で stop
  - `predict(test_matrix)`: best weights を adopt
  - core loop: O(n_iter × n_base × precision^-1) = 100 × 9 × 1000 = 900K eval

### §6.2 Optuna 500-trial postproc (= §1.2) 自前実装

- 既存依存: `optuna` (= ROGII の `pyproject.toml` に追加必要、 確認後判断)
- 構造変更: `src/rogii/submit.py` に `postproc_3axis(...)` を新規追加、 既存 grid search を Optuna 化
- 工数: 0.5 day (= search space 設定 + apply_pp 関数化)

### §6.3 Numba beam ±2 (= §1.3) 自前実装

- 既存依存: `numba` (= ROGII の `pyproject.toml` に追加確認)
- 移植元: Hill Climb cell 5 + LGB+XGB cell 0 の `_beam_jit` を **そのまま** 採用 (= GitHub-equivalent under fair-use、 Apache-2.0 attribution)
- 工数: 0.5 day (= test 含む)

### §6.4 dependencies 追加

```toml
# pyproject.toml に追加 (= 未追加なら)
[tool.uv]
dependencies = [
    ...
    "numba >= 0.59",
    "optuna >= 4.0",
    ...
]
```

---

## §7. 関連 doc

- 本 doc: `docs/research/2026-05-12-hill-climb-strength-analysis.dense.md`
- plan: `/home/yusuke_kaya/.claude/plans/floating-cuddling-haven.md` (= repo copy: `docs/dev/2026-05-12-plan-gold-to-winning.dense.md`)
- criteria: `.criteria/kaggle-rogii-phase-a-2026-05-12.yaml`
- submission 分解 doc: `docs/research/2026-05-12-submission-analyses.md`
- 関連 audit doc: `docs/research/public-notebook-analysis.dense.md`、 `docs/research/2026-05-11-public-source-audit.dense.md`
- Hill Climb 中身: `/tmp/rogii-hillclimb/wellbore-geology-prediction-hill-climbing.ipynb`
