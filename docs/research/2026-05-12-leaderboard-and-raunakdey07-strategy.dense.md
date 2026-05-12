# LB top30 audit + raunakdey07 "Sub-9 RMSE" 戦略 refine

> 作成: 2026-05-12、 user 直 指示「最新 CODES で公開カーネルに負けまくってる、 研究と分析して取り込んで 1 位を取れるよう戦略を練ってくれ、 プロ Kaggle GM として優勝しに行って」
> 参照 doc: 2026-05-12-hill-climb-strength-analysis.dense.md / 2026-05-12-submission-analyses.md / /home/yusuke_kaya/.claude/plans/floating-cuddling-haven.md
> snapshot: 2026-05-12 13:40 UTC leaderboard

---

## §0. 結論サマリ

- 公開 LB **1 位 = Virtute 8.966** (= 既に 9 切り達成)、 user 目標 LB ≤ 8.5 まで残り 0.466 ft
- 公開 NB **真の最強 = `raunakdey07/rogii-ultra-sub-9-rmse`** (= title "Sub-9 RMSE"、 LB < 9.0 推定 70%+ 確度)
- raunakdey07 の核心 3 層 = (1) Hill Climb 10,000 iter + patience 1,500 (2) α × τ × w_pf 3-axis 2,530 cell exhaustive grid (3) 5-base + Softmax NCC + segment b_well + Numba JIT + Savgol の完成度高い combo (= 全 self-compiled)
- user の既実装 5 module (= A2 Climber + A3 Numba beam ±2 + A4 seg_b_well + A5 softmax NCC + A6 Optuna 500-trial) は **方向性 OK**、 ただし **詳細パラメタ調整** で raunakdey07 並みの aggressive search に必要
- 新規盲点 H.8 〜 H.10 (= ensemble weight softmax parametrization / Shapley base contribution / formation-wise weight split) は **公開 NB 全部 miss**、 user 独占機会
- Phase A 統合 kernel **exp016** は Phase A 5 module を inline inject、 ablation 5 step で raunakdey07 を凌駕

---

## §1. LB top30 (= 2026-05-12 13:40 UTC snapshot)

| Rank | Team | LB | Submission count | Last Submission | 注目 |
|---|---|---|---|---|---|
| 1 | Virtute | **8.966** | 35 | 2026-05-12 10:21 | 既に LB 9 切り達成、 user との差 -0.772 ft |
| 2 | Silogram | 9.301 | 20 | 2026-05-12 11:55 | |
| 3 | lingyu07 | 9.321 | 35 | 2026-05-12 12:26 | |
| 4 | Mr.キノコ | 9.346 | 24 | 2026-05-12 04:09 | (= Hill Climb fork 公開) |
| 5 | Qiwei | 9.356 | 15 | 2026-05-12 00:16 | |
| 6 | anshul3501 | 9.374 | 12 | 2026-05-11 04:06 | |
| 7 | **Chris Deotte** | **9.392** | 32 | 2026-05-12 01:51 | **Kaggle GM 参戦中**、 NN starter CV 15.5 / XGB starter 公開 author |
| 8 | eiheychan | 9.430 | 35 | 2026-05-12 11:00 | |
| 9 | NobelK5342 | 9.439 | 13 | 2026-05-12 05:08 | |
| 10 | Ogurtsov | 9.472 | 6 | 2026-05-12 07:26 | |
| 11 | LJL♥ | 9.493 | 7 | 2026-05-12 04:49 | |
| 12 | Takahiro Saito | 9.498 | 11 | 2026-05-12 02:18 | |
| 13 | Pretty Boring Model | 9.498 | 11 | 2026-05-11 16:24 | |
| 14 | kif | 9.523 | 10 | 2026-05-12 09:50 | |
| 15 | hoang_phuc_6868 | 9.537 | 9 | 2026-05-12 09:59 | (= thbdh5765/rogii-v11 公開 author) |
| 16 | Berkay | 9.552 | 8 | 2026-05-12 07:05 | |
| 17 | kindasomethin | 9.559 | 23 | 2026-05-12 09:30 | |
| 18 | Luke Stafford | 9.564 | 10 | 2026-05-12 01:01 | |
| 19 | Lwen1243 | 9.567 | 1 | 2026-05-12 09:11 | 1 submit で rank 19、 多分 raunakdey07 fork |
| 20 | zihao qi | 9.575 | 26 | 2026-05-11 17:17 | |
| 21 | Tillotson | 9.576 | 13 | 2026-05-12 12:34 | |
| 22 | Ravi Ramakrishnan | 9.577 | 8 | 2026-05-11 21:53 | (= ravi20076 / public-blend-v1 author) |
| 23 | Athar Sayed | 9.579 | 10 | 2026-05-12 02:41 | |
| 24 | lucataco | 9.581 | 14 | 2026-05-11 09:15 | |
| 25 | Will | 9.594 | 15 | 2026-05-12 11:56 | |
| 26 | Kazuki Harada | 9.597 | 5 | 2026-05-11 14:21 | |
| 27 | **Mahdi Ravaghi (Hill Climb 作者)** | 9.619 | 34 | 2026-05-12 12:59 | **Hill Climb 公開 LB 9.43 だが personal best 9.619、 = 公開 NB は personal best より下** |
| 28 | Hem Viramgama | 9.631 | 7 | 2026-05-12 09:28 | |
| 29 | Андрей Четверяков | 9.633 | 22 | 2026-05-12 09:03 | |
| 30 | Andrey Zhabsky | 9.645 | 9 | 2026-05-12 04:30 | |
| 31 | Dr_F | 9.646 | 15 | 2026-05-12 09:38 | |
| — | **user (ky7240)** | **9.738** | 10 | 2026-05-11 08:31 | **rank 50+ 後退**、 5/11 から submit 停止中 |

### 1.1 観察
- **Top 1 Virtute 8.966** = 既に 9 切り、 user 目標 LB ≤ 8.5 まで -0.466 ft の差
- **Top 7 Chris Deotte 9.392** = 著名 Kaggle GM、 NN starter / XGB starter author。 私たち独自 plan の paradigm 2 (NN seq) で **追い抜く** 必要
- **rank 4 Mr.キノコ 9.346** = `mrkinoko/rogii-ravaghi-hill-climbing-as-is` (= Hill Climb 単純 fork) → **Hill Climb fork だけで rank 4 入る** = 強力
- **rank 19 Lwen1243** = 1 submit で rank 19、 raunakdey07 公開 NB の fork 高確率
- **user rank 17 → 50+ 後退** は 5/11 以降 30 件以上の上位 submission で押し下げられた結果

---

## §2. raunakdey07/rogii-ultra-sub-9-rmse 核心 3 層分析 (= GM-level)

Source: `/tmp/rogii-new-audit/raunakdey07/rogii-ultra-sub-9-rmse.ipynb` (= pull 済)、 48895 bytes、 14 cells

### §2.1 層 1: Hill Climb weight optimization の数理本質

**raunakdey07 cell-12**:
```python
def optimize_blend_hill_climb(predictions, target, iterations=10000):
    num_models = predictions.shape[1]
    best_w = np.ones(num_models) / num_models
    best_score = root_mean_squared_error(target, np.dot(predictions, best_w))
    no_improve = 0
    for _ in range(iterations):
        w = best_w + np.random.normal(0, 0.02, num_models)  # Gaussian σ=0.02
        w = np.clip(w, 0.0, 1.0)
        w /= w.sum()                                          # hard normalization
        score = root_mean_squared_error(target, np.dot(predictions, w))
        if score < best_score:
            best_score = score; best_w = w; no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 1500: break                      # patience 1500
    return best_w, best_score
```

- **vs user Ridge** (= closed-form linear least-squares、 positive 制約): non-convex stochastic exploration で local optimum escape、 Gaussian perturbation で weight space を **continuous exploration**
- **vs user 既実装 Climber (= src/rogii/hill_climb.py)**: 
  - 我々の Climber は **discrete +-precision update**、 raunakdey07 は **Gaussian perturbation continuous**
  - 我々の Climber max_iter=10000、 patience=10 (default)
  - **raunakdey07 patience=1500** = 1500 連続 no-improve まで stop しない aggressive
- **lift 機構**: base 5 個の correlation matrix の top principal component 以外の residual variance を clip+perturb で squeeze、 Ridge は this を 取れない
- **実測**: cell-12 で `print(f"Simple Avg OOF: {r_avg:.4f} | Hill Climb OOF: {r_stk:.4f}")` 差分は raunakdey07 で -0.10〜-0.20 ft 期待

### §2.2 層 2: α × τ × w_pf 3-axis 2,530 cell exhaustive grid

**raunakdey07 cell-14**:
```python
alphas = np.arange(0.60, 1.05, 0.02)              # 23 values
taus = [None, 20., 40., 60., 80., 100., 150., 200., 300., 400.]  # 10 values
w_pfs = np.arange(0.0, 0.21, 0.02)                # 11 values
# total: 23 × 10 × 11 = 2,530 cells

def apply_pp(md_since, model_pred, pf_pred, alpha, tau, w_pf):
    d = model_pred * (1 - w_pf) + pf_pred * w_pf
    if tau:
        d *= (1.0 - np.exp(-np.maximum(md_since, 0.0) / tau))
    return d * alpha

best_rmse = np.inf
for alpha in alphas:
    for tau in taus:
        for w_pf in w_pfs:
            d = apply_pp(md_since, hc_oof, pf_oof, alpha, tau, w_pf)
            rmse = root_mean_squared_error(y_true_delta, d)
            if rmse < best_rmse:
                best_rmse = rmse; best_params = (alpha, tau, w_pf)
```

- **vs user 既実装 postproc_optuna** (= src/rogii/postproc_optuna.py): TPE Sampler 500-trial、 search space は同等だが **exhaustive ではない**
  - 500 trial sparse vs 2,530 cell dense
  - Optuna は warm-up 50 trial 後の Bayesian 探索、 raunakdey07 は **brute-force exhaustive**
- **物理直観**: τ = exponential fade-in time constant
  - τ 小 (= 20) → early in well で PF を強く weight (= aggressive correction)
  - τ 大 (= 400) → late in well まで conservative
  - τ=None → fade-in なし、 alpha × (1-w_pf)*model + w_pf*pf
- **lift**: -0.15〜-0.30 ft (= 3-axis 累積、 Hill Climb -0.10〜-0.20 + grid -0.05〜-0.10)

### §2.3 層 3: 5-base + Softmax NCC + segment b_well + Numba JIT + Savgol の完成度

- **5 base**: LGB×3 (seed 42/7/123) + CB×1 + XGB×1 (cell-2)
  - vs user **9 base** (= karnakbaev 5 + 自前 4): user **base diversity 優位**
  - ただし base 数より **correlation 低い base** の方が effective、 user 9 base の OOF correlation 第一主成分 variance ratio で実測必須
- **Numba JIT**: Beam ±2、 PF N=600 (ANCC) + N=600 (Z) dual (cell-4)
  - **user 既実装 src/rogii/beam_pf_numba.py で match**、 ただし dual PF (= ANCC + Z 両方) 確認必要
- **Softmax NCC**: hws=(8,15,25) softmax-weighted ensemble (cell-8)
  - **user 既実装 src/rogii/segment_features.py:multi_scale_ncc で match**
- **segment b_well**: per-formation early/mid/late/wls 4 segment (cell-8)
  - **user 既実装 src/rogii/segment_features.py:seg_b_well で match**
- **Savgol**: window=17, poly=3 trajectory smoothing (cell-14 末尾)
  - **user 既実装 src/rogii/postproc_optuna.py:sg_smooth_inplace で match**
- **artifact**: ❌ 外部 dataset 非依存 (= self-compiled、 user §11 commit と合致)

---

## §3. 我々の既実装 5 module の細部 missing parameters

| module | 既実装 | raunakdey07 |
|---|---|---|
| A2 Climber (= src/rogii/hill_climb.py) | max_iter=10000、 patience=**10**、 precision=0.001 (discrete ±) | patience=**1500**、 Gaussian perturbation σ=0.02 (continuous) |
| A3 Numba Beam (= src/rogii/beam_pf_numba.py) | ±2 delta、 7 BEAMS configs、 ANCC PF only | dual PF (= ANCC + Z N=600 each) ← Z PF 追加必要 |
| A4 seg_b_well (= src/rogii/segment_features.py:seg_b_well) | early/mid/late/wls 4 phase | ✅ match |
| A5 Softmax NCC (= src/rogii/segment_features.py:multi_scale_ncc) | hws=(8,15,25)、 softmax weighted | ✅ match (sw=exp(3·score)) |
| A6 postproc_optuna (= src/rogii/postproc_optuna.py) | Optuna TPE 500-trial、 α × τ × w_pf 3-axis | **exhaustive grid 2,530 cell** ← 追加必要 |
| SG smoothing (= src/rogii/postproc_optuna.py:sg_smooth_inplace) | window=17、 poly=3 | ✅ match |

### 3.1 必須 patches

1. **A2.1** (= 既存 task #13): `src/rogii/hill_climb.py` の Climber に **Gaussian perturbation continuous mode** + **patience 1500** option 追加
2. **A3.1**: `src/rogii/beam_pf_numba.py` に **`_pf_z_jit` + run_pf_z` 追加** (= 既存 ANCC PF と並走で dual PF 構成、 user 6 hours 残)
3. **A6.1** (= 既存 task #14): `src/rogii/postproc_optuna.py` に `optimize_postproc_grid(...)` exhaustive grid 関数追加 (= TPE と並列、 2,530 cell sweep)

---

## §4. 新規盲点 H.8〜H.10 (= 5 NB + user 全員 miss、 独占機会)

| ID | 盲点 | raunakdey07 状況 | user 独占機会 |
|---|---|---|---|
| H.8 | **ensemble weight normalization soft parametrization** | hard `w /= w.sum()` (clip 0,1) | softmax parametrization (= raw logit 通して softmax で実数空間 search、 hard constraint 不要)、 variance 追加削減 |
| H.9 | **per-base Shapley value ablation** | 測定無し、 単純 weight sweep | LOO-OOF (= leave-one-base-out) で各 base の真寄与を測定、 weak base 除外で SNR 向上 |
| H.10 | **formation-wise ensemble weight** | global weight only | ANCC / ASTNU / ASTNL の per-formation 別 weight (= LGB init_score trick)、 formation 別 personalization |

これらは **Phase B/C で取り組む** = paradigm として新規、 user 独占で 1 位狙い

---

## §5. Phase A 統合 kernel exp016 refine 設計

### 5.1 inject 順序 (= ablation 5 step、 各 1 submit で単独 lift 実測)

| step | 内容 | base | expected lift | priority |
|---|---|---|---|---|
| exp016a | exp009 v2 完全 copy (= base reproduction、 LB 9.738 確認) | — | 0 ft (= safety baseline) | high |
| exp016b | + A3 Numba beam ±2 + A3.1 PF Z dual | exp016a | -0.05〜-0.10 ft | high |
| exp016c | + A4 seg_b_well + A5 softmax NCC | exp016b | -0.10〜-0.20 ft | high |
| exp016d | + A6 Optuna 3-axis + A6.1 exhaustive grid 2,530 cell | exp016c | -0.10〜-0.25 ft (= raunakdey07 core) | **critical** |
| exp016e | + A2 Climber Gaussian patience 1500 (= raunakdey07 core) | exp016d | -0.10〜-0.30 ft | **critical** |

期待累積: **9.738 → 9.0-9.4 帯** (Hill Climb 9.43 + raunakdey07 9.0 圏接触)

### 5.2 quota 制約

ROGII rolling 24h で **5 件**、 今 24h で 既 2 件 (= exp010 + exp013) 使用、 残 **3 件**。
ablation 5 step は 5 件必要 → **2 day 分散**:
- Day 1 (= 5/12 残): exp016a (safety) + exp016b (A3+A3.1) + exp016d (A6+A6.1) = 3 件 (= raunakdey07 core を最速 verify)
- Day 2 (= 5/13): exp016c (A4+A5) + exp016e (A2 patience 1500) + 統合 final = 3 件

ただし **exp016a (safety baseline) は LB 9.738 再現で 1 quota 使うのは無駄**、 skip して exp016b から開始可。

### 5.3 並走 Phase B (= local 実装、 Kaggle quota 浪費せず)

- B1 winning-path F (= data augmentation aug_k 6 段) ← local OOF で ablation 実測、 final 1 件のみ Kaggle submit
- B2 winning-path G (= per-row kNN FAISS) ← 同
- Z1 (= Sparse GP M=500 multi-formation) ← Local GPU で fit、 final blend で Kaggle submit

---

## §6. Phase C 修正 (= NN seq paradigm、 GM 級独占機会)

- Chris Deotte が rank 7 LB 9.392 = NN starter / XGB starter author、 NN seq paradigm に詳しい
- ただし **Mamba / Longformer 投入は誰もしていない** (= leaderboard top 30 で NN-only solution unlikely、 ほぼ全員 GBM stack)
- user は paradigm 2 induction (= NN seq) で **Chris Deotte を上回る** 機会、 ただし overfit risk 高

優先順序 (refine):
1. exp016 ablation 完了 (= LB 9.0-9.4 帯確保)
2. winning-path F + G 並走 (= LB 8.5-9.2 帯)
3. winning-path A (= NN seq Mamba 5-fold × 5-seed) ← **Chris Deotte 凌駕の最重要 path**
4. winning-path D (= Per-Well MoE) + H.10 formation-wise weight

---

## §7. 即実行 action 6 件 (= 残 24h、 priority 順)

1. **A2.1 Climber Gaussian perturbation + patience 1500 拡張** (= src/rogii/hill_climb.py modify、 1h、 local)
2. **A6.1 exhaustive grid 2,530 cell 追加** (= src/rogii/postproc_optuna.py modify、 0.5h、 local)
3. **A3.1 PF Z dual 追加** (= src/rogii/beam_pf_numba.py modify、 1.5h、 local)
4. **exp016b kernel skeleton 構築** (= exp009 v2 base に inline inject、 4-6h、 local)
5. **exp016b Kaggle push + submit** (= 1h、 quota 1 件消費)
6. **exp016d 即追加 submit** (= raunakdey07 core 直撃、 quota 1 件)

24h 後の expected: exp016b + exp016d 完了で **LB ~9.2-9.5 帯** (= 上振れで raunakdey07 凌駕、 下振れで Hill Climb 9.43 接近)

---

## §8. 関連 doc

- 親 plan: `/home/yusuke_kaya/.claude/plans/floating-cuddling-haven.md`
- repo copy: `docs/dev/2026-05-12-plan-gold-to-winning.dense.md`
- Hill Climb 強さ分析: `docs/research/2026-05-12-hill-climb-strength-analysis.dense.md`
- submission 分解 (GM §8): `docs/research/2026-05-12-submission-analyses.md`
- criteria: `.criteria/kaggle-rogii-phase-a-2026-05-12.yaml`
- HANDOFF: `docs/dev/HANDOFF-2026-05-12.md`
