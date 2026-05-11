# PySR Implementation Spec for ROGII (2026-05-11)

> 範囲: subagent X 推奨組合せ #1 (= `docs/research/paradigm-deeper.dense.md` §5.3 scenario B, branch `docs/paradigm-deeper-2026-05-11`) で「短工数 2-3 日 / 低 risk / -0.1〜-0.3 ft 加算 / b_well_symbolic を全 head に注入可能」と確証された PySR (= Cranmer 2023, arXiv:2305.01582) を ROGII で運用するための **実装 design + smoke 計画**。実装作業 (= コード生成 / 修正) は本 doc 範囲外、 spec only。
> 出典規律 (~/.claude/CLAUDE.md "Links, not verdicts"): 各 claim に arXiv / GitHub URL / `file:line` を必ず添える。
> 中立指示 (~/.claude/CLAUDE.md 中立指示原則): 推奨は出さない、 トレードオフ + 選択軸のみ提示する。
> 日本語規律 (~/.claude/CLAUDE.md "Plan Mode 出力ルール (Kaggle 関連のみ、 CRITICAL)"): 本文は日本語、 コード識別子・URL・file:line・数式は英数字のまま。
> branch: `docs/pysr-spec-2026-05-11` (= 本 doc 専有、 subagent W / X / Y branch を避ける)。

---

## 0. 序: subagent X §5.3 #1 推奨の解像度を上げる

### 0.1 既存知見 (= 確証済 input)

| source | 数値 / 主張 | file:line |
|---|---|---|
| paradigm-deeper §4.4.1 | PySR 単独 LB 寄与 **-0.1〜-0.3 ft** | `docs/research/paradigm-deeper.dense.md:566-570` (branch `docs/paradigm-deeper-2026-05-11`) |
| paradigm-deeper §5.1 | 工数 **2-3 日**、 priority ★ (= 補助・短工数) | `docs/research/paradigm-deeper.dense.md:590-593` |
| paradigm-deeper §5.3 scenario B | PySR → PINN → TabPFN v2 → Mamba の順次 path | `docs/research/paradigm-deeper.dense.md:612-617` |
| paradigm-deeper §5.4 | 「PySR と他全部: b_well feature 提供のみ → 加法性 +95% 想定 (= 重複ほぼ無)」 | `docs/research/paradigm-deeper.dense.md:630` |
| paradigm-deeper §6.7 | PySR offline-export ワークフロー (= Julia install 不安回避) を強く推奨 | `docs/research/paradigm-deeper.dense.md:788-819` |
| first-principles §0.2 D | `b_drift_first_to_last_p50_abs = 0.00154 ft`、 b_well は MD-linear weak drift | `docs/research/first-principles.dense.md:168-169` |
| first-principles §0.2 (formula oracle) | `formula_oracle_rmse_overall = 0.006 ft`、 真値 formula で評価したときの limit | `docs/research/first-principles.dense.md:38` |
| mathematical-formulation §0.2 | $TVT_w(s) = -Z_w(s) + ANCC_w(s) + b_w(s) + \varepsilon_w(s)$、 $\varepsilon$ fat-tail、 b は MD-linear weak drift | `docs/research/mathematical-formulation.dense.md:64-65` (branch `7b225ce`) |
| per-well-stats parquet 実測 | n=773、 `b_well_mean` median ≈ **10934 ft**、 IQR 10566-11574、 `b_well_drift_first_to_last` abs p50 = **0.0014 ft**、 `b_well_resid_std` p50 = **0.0083 ft** | `outputs/eda/first_principles/per-well-stats.parquet` (= `notebooks/_first_principles_measure.py` 出力) |
| 既存 wls_b_well 実装 | exponential decay weighted、 `decay=0.02` | `src/rogii/tysig.py:400-416` |
| 既存 build_well_features 中 b_well 計算 | `b_F_all`, `b_F_50`, `b_F_wls` の 3 種を per-formation 計算済 | `src/rogii/features.py:147-171` |
| FORMATIONS 一覧 (= 6 件) | `["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]` | `src/rogii/imputers.py:20` |
| LB ベースライン | exp005 v2 = **10.317 ft** (= karnakbaev blend)、 9 切り (= 8.x 帯) まで -1.317 ft 以上必要 | `docs/dev/leaderboard.dense.md:12, :74-76` |
| exp005 v3 結果 | Edge R 単独投入で +0.184 悪化、 base model 依存・w_R 過大の可能性 | `docs/dev/leaderboard.dense.md:16` |

### 0.2 本 spec の役割 (= subagent X からの refine 差分)

subagent X §4-§6 は paradigm 比較の **解像度低い design** (= 推奨 hyperparameter 概略 / failure mode 3 件 / offline-export ひな形)。 本 spec はそれを ROGII 固有 数値 (= `b_well_mean` 実測分布、 6 formation 個別 fit、 既存 `build_well_features` 注入ポイント、 exp015 AB test design) に落として **per-formation × per-well の closed-form 式探索計画** を確定する。

### 0.3 「優勝本質性」 criterion 適用 (~/projects/kaggle/CLAUDE.md §11.2)

5 問に答える:

1. **数理本質**: $b_w(s)$ の per-well constant を symbolic 関数 $b_w = f(X, Y, \text{dip}, \text{thickness}, Z, \ldots)$ で置換。 generative model `TVT = -Z + ANCC + b_well` (= mathematical-formulation §0.2) の **`b_well` head を closed-form 化**、 hidden tail (= MD 4000+ ft 末端) の extrapolation 安定性を上げる
2. **優勝寄与**: 単独 -0.1〜-0.3 ft で 9 切り (= -1.3 ft 必要) のうち **7-23%** を担当。 ★ priority (= 補助)、 PINN / Mamba / TabPFN v2 と併用前提
3. **代替比較**: 軽量 alternative = (a) `wls_b_well` を per-formation polynomial で grid search、 (b) MD-linear drift 1-line 追加。 PySR は (a) の **空間探索を自動化**、 (b) は drift 観測値 0.0014 ft / 5000 row = 1e-7 ft/ft で寄与極小。 PySR は本質的に上位互換
4. **rule 耐性**: PySR offline-export ワークフロー (= §6.7 paradigm-deeper) では Kaggle kernel は **純 Python/numpy 式 hardcode**、 Julia ランタイム不要 → host rule 変更 / Kaggle 環境 update に対する耐性最大
5. **datapoint 価値**: exp015 で AB test 設計 (= exp014a vs exp014_no_pysr で symbolic feature 効果分離)。 noise 範囲 0.05 ft 想定、 -0.1 ft 寄与なら 2σ で検出可能 (= submit 価値あり)

→ 5 問全て pass、 **本 spec は 「優勝本質性」 を満たす**。

---

## 1. PySR install + Kaggle 互換性

### 1.1 数値 + 出典

| 数値 | source | 用途 |
|---|---|---|
| PySR Python package: `pip install pysr` | `https://github.com/MilesCranmer/PySR` README | local install |
| Julia auto-install: `python -c "import pysr; pysr.install()"` | 同上 | first import で juliaup 経由 julia + SymbolicRegression.jl 取得 |
| Docker option: `docker build -t pysr --build-arg JLVERSION=1.10.0 --build-arg PYVERSION=3.11.6 .` | PySR README (context7 query 結果) | reproducible env |
| `LD_LIBRARY_PATH=$HOME/.julia/juliaup/julia-1.10.0+0.x64.linux.gnu/lib/julia/` | PySR README troubleshooting | GLIBCXX 衝突回避 (= Kaggle GPU image で発生 reported) |
| Kaggle Internet disabled (submission kernel) | `~/projects/kaggle/CLAUDE.md` §4.1、 Kaggle 公式 docs `https://www.kaggle.com/docs/efficient-gpu-usage` | submission kernel は internet 不可、 train kernel は enable 可 |

### 1.2 install path 3 案 (= 構造原理が異なる)

#### 1.2.1 案 A: full Julia install on submission kernel (= paradigm-deeper §4.3.1 ひな形)

- 流れ: kernel 冒頭で `pysr.install()` → 初回 7-15 min かかり Julia + SymbolicRegression.jl install → fit 開始
- 工数: 0.5 日 (= notebook 1 セル)
- Kaggle 互換: **submission kernel は internet disabled なので不可**。 train kernel (= internet enable 可) でのみ実行可能
- LB 寄与: ゼロ (= 実行不能)
- 選択軸: production 不採用、 train kernel での fit 用途のみ

#### 1.2.2 案 B: offline-export ワークフロー (= paradigm-deeper §6.7、 推奨 default)

- 流れ:
  1. local machine で PySR fit (= internet 制約なし)
  2. `model.equations_` Pareto front を `experiments/pysr/b_well_formulas.json` に export
  3. submission kernel は **純 Python/numpy 関数を hardcode** (= 例: `0.0021*X_mean + 0.0034*Y_mean - 0.012*np.sin(dip_azimuth)*Z_last + 0.45`)
- 工数: 1.5 日 (= local fit 0.5 日 + Pareto front 選定 0.5 日 + kernel 注入 0.5 日)
- Kaggle 互換: **100%** (= Julia 依存ゼロ、 pip 追加なし)
- LB 寄与: フル (= 式が hardcode されるので train/inference で同一)
- 選択軸: 短工数 + 高互換 + Julia install 失敗 risk ゼロ

#### 1.2.3 案 C: Kaggle private dataset として Julia sysimage upload

- 流れ:
  1. local machine で `juliaup add 1.10.0 + Pkg.add("SymbolicRegression")` + sysimage compile
  2. `~/.julia/` 全体を `ky7240/rogii-pysr-runtime` dataset (= private) として upload
  3. submission kernel で `JULIA_DEPOT_PATH=/kaggle/input/rogii-pysr-runtime/julia` を export
- 工数: 1.5 日 (= sysimage 1 日 + dataset upload + kernel boot 検証 0.5 日)
- Kaggle 互換: △ (= dataset サイズ 1-2 GB、 kernel boot に +30-60 s、 Julia version mismatch risk あり)
- LB 寄与: フル
- 選択軸: 「式更新を kernel 内で再 fit したい」 (= 新 test well 到来時の online refit) シナリオでのみ有効。 我々の AB test には不要

### 1.3 案比較 + 選択軸

| 軸 | 案 A (full install) | 案 B (offline-export) | 案 C (sysimage upload) |
|---|---|---|---|
| Kaggle submission 互換 | ✗ (internet 不可) | ◎ | ○ (boot 遅延あり) |
| 工数 | 0.5 日 | **1.5 日** | 1.5 日 |
| Julia install 失敗 risk | 高 (= GLIBCXX 衝突 historical) | **ゼロ** | 中 (= sysimage corruption) |
| 式更新柔軟性 | △ | △ (= offline で再 fit) | ◎ (= kernel 内 refit) |
| 9 hr cap 影響 | +15 min (= install) | **0** | +60 s (= boot) |

→ 推奨は出さない。 9 hr cap + Julia install 失敗 risk を重視するなら案 B、 online refit が必要なら案 C。

### 1.4 dependency risk + Kaggle environment 確認事項

- **Kaggle base image Python**: 3.11.x (= 2026-05 時点)、 `pysr==1.5.9` 互換確認必要 (= 公式 doc `https://ai.damtp.cam.ac.uk/pysr/v1.5.9/`)
- **numpy / scipy version**: PySR は numpy ≥1.20、 scipy ≥1.7。 既存 kernel と同等
- **sympy 依存**: 式 export 時に sympy 必須 (`model.sympy()`)。 Kaggle base image に既に入っている (= 確認: `import sympy` で version 出力)
- **CUDA 不要**: PySR は CPU GA 探索 (= paradigm-deeper §4.3.3)、 GPU 利用なし

---

## 2. per-formation b_well 計算 (= train data 設計)

### 2.1 既存 WLS の流用 (= 既実装)

`src/rogii/features.py:147-171` で 6 formation × per-well の b_well を以下 3 種計算済:

| feature | 数式 | 用途 |
|---|---|---|
| `bw_<F>` | `np.median(TVT_input[visible] + Z[visible] - f_imp[visible])` | naive median |
| `bw50_<F>` | 上記 visible 末尾 50 行に絞った median | recent-biased |
| `bww_<F>` | `wls_b_well(decay=0.02)` (= `src/rogii/tysig.py:403-416`) | exponential-weighted |

→ PySR の **target** = `bww_<F>` (= per-formation 6 種、 per-well 773 行)。 計算済 feature を再利用、 新規 WLS 計算は不要。

### 2.2 PySR 入力 feature 一覧 (= per-well 集約値)

per-well 1 row、 計 773 サンプル × 6 formation = **4638 row** (= ANCC fit 1 つ + ASTNU fit 1 つ + ... の 6 並列 fit)。 ただし「per-formation 独立 fit」 (= §3.1.2) を採るので **773 row per fit × 6 fit**。

| feature | source | 物理意味 |
|---|---|---|
| `X_first`, `X_last`, `X_mean`, `X_std` | `data/raw/train/<well_id>__horizontal_well.csv` の `X` 列 | well lateral 位置 (= 東西、 ft) |
| `Y_first`, `Y_last`, `Y_mean`, `Y_std` | 同 `Y` 列 | well lateral 位置 (= 南北、 ft) |
| `Z_first`, `Z_last`, `Z_mean`, `Z_std` | 同 `Z` 列 | well TVD (= 真深度、 ft) |
| `MD_first`, `MD_last`, `MD_range` | 同 `MD` 列 | 測長 (= measured depth、 ft) |
| `formation_<F>_thickness` | `f_imp_<F>` (= FormationPlaneKNN imputed)、 hidden 末端 - visible 始端 | formation 層厚 (= ft) |
| `dip_azimuth` | `arctan2(dY/dMD, dX/dMD)` の visible 区間 mean | 地理的 dip 方位 (= rad) |
| `dip_angle` | `arctan2(dZ/dMD, lateral_velocity)` の visible 区間 mean (= `src/rogii/features.py:136` 既算出 `inclination` を per-well 集約) | 鉛直 dip (= rad) |
| `tvt_visible_mean`, `tvt_visible_std`, `tvt_visible_slope` | TVT_input の visible 区間統計 (= visible 末端は last_known_TVT) | TVT signal level |
| `gr_visible_mean`, `gr_visible_std`, `gr_noise_std` | per-well-stats parquet 既算出 (= `gr_noise_std`) | well 個性 (= geology profile) |
| `n_visible`, `visible_ratio` | per-well-stats parquet 既算出 | sample weight prior |
| `b_well_resid_std` | per-well-stats parquet 既算出 (= 0.0083 p50) | per-well noise level (= heteroscedastic prior) |

計 ~24 feature。 PySR の探索空間は `select_k_features=4` で自動 down-select (= context7 query 結果)、 もしくは手動で 8-10 feature 厳選。

### 2.3 target stats (= per-well-stats 実測再掲)

| target | 数値 | source |
|---|---|---|
| `b_well_mean` median | 10934 ft | `outputs/eda/first_principles/per-well-stats.parquet` (n=766) |
| `b_well_mean` IQR | 10566 - 11574 ft (= range 1000 ft) | 同上 |
| `b_well_drift_first_to_last` abs p50 | 0.0014 ft | 同上 |
| `b_well_resid_std` p50 | 0.0083 ft | 同上 |

→ target std (= per-well b_well の sample 間分散) ≈ **637 ft** (= `b_well_mean.std()`)、 mean ≈ 11125 ft。 これは **6 formation 通算**、 per-formation で fit 分離するときは formation 別 std を計測 (= §4.2 で実測)。

### 2.4 fold-aware filter (= train/valid leak 防止)

PySR fit は **per-fold 独立** で実施 (= GroupKFold 5-fold、 well_id base、 既存 `src/rogii/cv.py:build_stratified_edge_q_folds` 流用検討)。

- fold 0-3: 各 fold で train wells (= 618 well) で PySR fit → val wells (= 155 well) で hold-out RMSE 計測
- fold 4: similar
- final production: 全 773 well で 1 回再 fit (= test inference 用)

→ 「過学習を CV で検知」 (= §4.4.2 paradigm-deeper の failure mode #2 対策)。

---

## 3. PySR fit (= offline、 local CPU、 6 formation 独立)

### 3.1 hyperparameters (= ROGII 適応)

#### 3.1.1 paradigm-deeper §4.1.3 base + 修正

| param | paradigm-deeper 推奨 | 本 spec 採用 | 修正根拠 |
|---|---|---|---|
| `niterations` | 100-200 | **100** | per-formation 6 並列 = 工数 budget 厳しい、 100 で elbow 到達想定 (= PySR Springer review §4.1.3) |
| `populations` | 30-60 | **30** | local CPU 4-8 thread 想定、 60 は parallel saturate |
| `population_size` | 50 | **50** | default |
| `ncycles_per_iteration` | 500 (default) | **500** | default 維持 |
| `maxsize` | 25 | **20** | 6 formation 全式の **可読性 + overfit 抑制 ((773 sample - leave-out 155) → train 618)**、 maxsize 20 で AIC 最良 想定 |
| `parsimony` | 0.001-0.01 (paradigm-deeper) | **2e-4** | `min_loss / 5-10` rule (= PySR docs)、 b_well noise 0.0083 ft / std 637 ft → relative noise ratio から 1e-4 〜 1e-3、 中央値 2e-4 採用 |
| `model_selection` | "score" (elbow) | **"score"** | 「best」は overfit (= max accuracy)、 「score」 は complexity vs loss の elbow = generalization 安定 |
| `binary_operators` | +, -, ×, / | **["+", "-", "*", "/"]** | 4 算術 |
| `unary_operators` | sin, cos, tan, exp, log, sqrt | **["sin", "cos", "exp", "log", "sqrt"]** | tan は asymptote 発散で fit hang risk、 除外 |
| `select_k_features` | n/a | **8** | 24 feature から auto down-select、 overfit 抑制 |
| `precision` | 64 (default) | **64** | b_well 値 11000 ft range、 32-bit では catastrophic cancellation risk |
| `procs` | 4 | **4-8** | local CPU thread 数依存 |
| `multithreading` | True | **True** | default |
| `early_stop_condition` | none | **`stop_if(loss, complexity) = loss < 1e-5 && complexity < 15`** | b_well_resid_std 0.0083 ft の 1/800 = 1e-5 ft² を AIC 下限、 complexity 15 で simplicity 保証 |

#### 3.1.2 per-formation 独立 fit (= 6 並列)

```python
# 概念図 (= 実装ではない、 spec 説明)
FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
formulas = {}
for F in FORMATIONS:
    # train: 773 well × 24 feature, target = bww_<F>
    sr = PySRRegressor(niterations=100, populations=30, ..., model_selection="score")
    sr.fit(X_per_well_features, y_bww_F)
    formulas[F] = {
        "pareto_front": sr.equations_.to_dict("records"),
        "best": sr.get_best().to_dict(),
    }
```

→ paradigm-deeper §4.2.1 の「per-well 376 sample」記述は **古い数値** (= subagent X が train 数を低めに見積もり)。 ROGII 実 train = **773 well** (`data/raw/train/` の `*__horizontal_well.csv` 数)、 これで fit。

### 3.2 runtime budget (= local CPU)

per-formation 1 fit:
- 773 sample × 8 feature × 100 iter × 30 pop × CPU 4 thread → **5-10 min** (= paradigm-deeper §4.3.3 計算式から外挿、 サンプル数 2x で線形)
- 6 formation × 8 min ≈ **30-48 min**

→ local 1 セッション (= 1 hr 以内) で全 6 formation の Pareto front 揃う。 Kaggle 9 hr cap 影響ゼロ (= offline)。

### 3.3 Pareto front 選定基準

| 選定軸 | 何を重視 | 選択候補 |
|---|---|---|
| AIC 最良 | accuracy 重視、 overfit risk あり | `model_selection="best"` の equation |
| simplicity 最良 | complexity 低、 bias risk | Pareto front の complexity ≤ 5 |
| elbow (= score) | trade-off | `model_selection="score"` (= 採用 default) |
| 物理整合性 | 単位整合 + 既知物理項 (= ANCC ≈ aX+bY+c plane fit) | 手選定、 §4.1 で詳述 |

各 formation で **3 候補** を保存 (= best / second / simplest)、 §4 で final select。

---

## 4. 式選定 + validation

### 4.1 best 3 式 / formation の選定手順

各 formation について:

1. **Pareto front 全件** を `equations_.csv` 形式で export (= context7 query 結果: `model.equations_[["equation", "loss", "complexity", "score"]]` 列)
2. 以下 3 種を抽出:
   - **best_score**: `model.get_best()` (= elbow 選択)
   - **best_accuracy**: loss 最小、 ただし complexity ≤ maxsize (= 20)
   - **simplest_acceptable**: loss が `b_well_resid_std` (= 0.0083 ft) の 5x = 0.041 ft² 以下を満たす最小 complexity
3. **物理整合性チェック** (= 手判定):
   - 単位整合 (= X, Y は ft、 dip は rad、 thickness は ft)
   - 既知物理項 (= ANCC plane fit `aX+bY+c` `docs/research/problem-essence.dense.md:27`) が現れるか
   - 末端外挿安定性 (= b_well_drift 0.0014 ft の方向と一致するか)
4. 物理整合性違反式は drop、 残り上位 3 件を採用

### 4.2 5-fold cross-valid (= 既存 GroupKFold 流用)

- fold split: `src/rogii/cv.py` 既存 GroupKFold by well_id (= 5-fold)、 paradigm-deeper §4.4.2 failure mode #2 対策
- per-fold metric:
  - **b_well_symbolic_RMSE_ft²**: PySR 式 prediction vs `bww_<F>` ground truth (= WLS) で per-row RMSE
  - **TVT_implied_RMSE_ft**: $\hat{TVT} = -Z + ANCC + \hat{b}_{\text{symbolic}}$ で TVT_input ground truth (= hidden region) との RMSE。 これが本命 metric (= LB と直結)
  - **WLS baseline 比**: 同 fold で `bww_<F>` を constant 採用した baseline と比較、 delta RMSE
- 採用判定: TVT_implied_RMSE が WLS baseline 比 **-0.05 ft 以上改善** + CV 5-fold で σ ≤ 0.03 ft (= stable)

### 4.3 出力 file 設計

```
experiments/pysr/b_well_formulas.json
{
  "version": "v0_2026-05-11",
  "pysr_version": "1.5.9",
  "fit_seed": 42,
  "fit_data_source": "outputs/eda/first_principles/per-well-stats.parquet + data/raw/train/<wid>__horizontal_well.csv",
  "fit_runtime_sec": 1800,
  "fit_n_wells": 773,
  "formulas": {
    "ANCC": {
      "best_score": {
        "equation_str": "0.0021 * X_mean + 0.0034 * Y_mean + ...",
        "sympy_repr": "0.0021*X_mean + 0.0034*Y_mean + ...",
        "complexity": 8,
        "loss_train": 0.00012,
        "loss_cv5_mean": 0.00018,
        "loss_cv5_std": 0.00003,
        "tvt_implied_rmse_cv5_mean": 1.42,
        "tvt_implied_rmse_baseline": 1.51,
        "delta_vs_baseline": -0.09
      },
      "best_accuracy": { ... },
      "simplest_acceptable": { ... }
    },
    "ASTNU": { ... },
    ...
  }
}
```

→ json 1 ファイルで 6 formation × 3 candidate = 18 式を保存、 kernel 側で choose by `<formation>_<variant>` 指定。

---

## 5. kernel script への注入

### 5.1 `compute_b_well_symbolic()` 仕様

```python
# 仕様 (= 実装ではない、 spec 説明)
def compute_b_well_symbolic(
    features: dict[str, np.ndarray],
    formation: str,
    variant: str = "best_score",  # "best_score" | "best_accuracy" | "simplest_acceptable"
) -> np.ndarray:
    """
    Returns per-row b_well_symbolic[<formation>] vector (n_rows,).

    Input features dict must include keys used by the selected formula
    (e.g., X_mean, Y_mean, dip_azimuth, Z_last, ...).

    Formulas are pre-fitted offline and hardcoded as numpy expressions
    (NO Julia / NO PySR runtime dependency at inference time).
    """
    ...
```

実装方針:
- **形式 A: hardcoded if-elif** (= paradigm-deeper §6.7 ひな形)
  - 全 18 式を Python 関数として書き下し
  - 利点: dependency ゼロ、 fast
  - 欠点: 式更新時に再 manual paste 必要
- **形式 B: sympy + lambdify on-the-fly**
  - `b_well_formulas.json` の `sympy_repr` を `sp.sympify(s)` + `sp.lambdify(free_symbols, expr, modules="numpy")` で評価
  - 利点: 式更新が json 編集だけ
  - 欠点: kernel cold start +50 ms / formula、 sympy 依存 (= Kaggle base にあり問題なし)

→ 推奨は出さない。 「式を頻繁に再 fit」 重視なら形式 B、 「kernel I/O 最小化」 重視なら形式 A。

### 5.2 注入位置 (= 既存 `build_well_features` への加算)

`src/rogii/features.py:147-171` の per-formation loop に **並列**で追加 (= 既存 `tvtF_<F>`, `tvtF_<F>_d`, `bw_<F>`, `bw50_<F>`, `bww_<F>` の 5 系列に加えて新たに 4 系列):

```
out[f"b_sym_{fname}_best"]       # symbolic b_well prediction (best_score variant), per-row constant per well
out[f"b_sym_{fname}_simple"]     # symbolic b_well prediction (simplest_acceptable variant)
out[f"tvtF_sym_{fname}"]         # = -Z + f_imp + b_sym_<F>_best, per-row TVT prediction
out[f"tvtF_sym_{fname}_d"]       # = tvtF_sym_<F> - last_tvt, displacement feature
```

→ 6 formation × 4 column = **24 新 features**。 既存 `tvtF_*_d` 6 個 + `bww_*` 6 個 + 既存 4 系列 と並列、 Ridge meta blend が weight 自動学習。

### 5.3 features 出力形式 + dtype

| name | dtype | shape | nan handling |
|---|---|---|---|
| `b_sym_<F>_best` | float32 | (n_rows,) | per-well 定数 (= 全 row 同値)、 fit 失敗 well は `b_F_wls` で fallback |
| `b_sym_<F>_simple` | float32 | (n_rows,) | 同上 |
| `tvtF_sym_<F>` | float32 | (n_rows,) | $-Z + \hat{ANCC} + b_{\text{sym}}$、 row-wise |
| `tvtF_sym_<F>_d` | float32 | (n_rows,) | $tvtF_{\text{sym}} - \text{last\_tvt}$ |

memory cost: 24 col × 5278 row mean × float32 = ~500 KB / well、 全 773 well で **~400 MB** = 9 hr cap 内余裕。

### 5.4 既存 base model との互換性

| base model | 互換 | 注入経路 |
|---|---|---|
| karnakbaev LGB3 + XGB + CB (= exp005) | ◎ | `build_well_features` 出力に追加、 features 名重複なし |
| 自前 LGB (= exp003 系) | ◎ | 同上 |
| TimeXer (= 文献 deeper W) | ○ | per-row scalar feature として inject 可能 |
| Mamba (= paradigm-deeper §1) | ○ | per-row scalar feature として inject 可能 |
| TabPFN v2 (= paradigm-deeper §2) | ○ | per-row column 追加で in-context Bayesian に渡る |
| PINN (= paradigm-deeper §3) | ◎ | `b_well` head の **prior として** loss に追加 (= soft anchor) |
| Ridge meta blend | ◎ | feature 1 つ追加で済む |

→ paradigm-deeper §5.4 主張「PySR と他全部: 加法性 +95% 想定 (= 重複ほぼ無)」と整合。 PySR は **feature 提供 paradigm**、 他 paradigm の predictor 構造を変えない。

---

## 6. 5/13 以降の exp 設計 (= AB test plan)

### 6.1 exp015 design (= PySR features 投入 AB test)

#### 6.1.1 構造

| variant | base | symbolic features | 期待 LB |
|---|---|---|---|
| exp015a (control) | exp014a (= karnakbaev blend + Edge S + Edge Q stratified、 想定 LB 10.0-10.2) | なし | exp014a と同等 |
| exp015b (PySR best_score) | 同上 + `b_sym_<F>_best`, `tvtF_sym_<F>` 12 features | best_score variant | -0.1〜-0.2 ft |
| exp015c (PySR simplest) | 同上 + `b_sym_<F>_simple`, `tvtF_sym_<F>` 12 features (simple 式) | simplest_acceptable variant | -0.05〜-0.15 ft (= bias risk あり) |
| exp015d (PySR full 24 features) | 同上 + 24 features 全部 (= best + simple 両方) | both | -0.1〜-0.3 ft (paradigm-deeper §4.4.1 上限値) |

→ 4 variant の同時 submit で **symbolic feature 効果** + **variant 比較** を 1 batch で測定。 Kaggle daily 5 submit quota 内 (= ROGII は 5/day と想定、 要確認)。

#### 6.1.2 datapoint isolate

- exp015a - exp005 v2 (LB 10.317) = Edge S + Edge Q stratified の効果
- exp015b - exp015a = PySR best_score の純効果
- exp015c - exp015a = simplest 式の純効果
- exp015d - exp015b = 「best + simple 両方」の追加効果

→ 4 datapoint で 3 つの 1 次効果 + 1 つの 2 次効果を分離。 ROGII の「`docs/dev/leaderboard.dense.md` ratio drift 監視」 ルール (= ~/projects/kaggle/CLAUDE.md §8.1 #9) 適用。

#### 6.1.3 失敗時の rollback

- exp015b LB > exp015a LB (= 悪化) なら、 paradigm-deeper §4.4.2 failure mode #1 (= trivial 式 hit) を疑い `parsimony` 1 桁下げて再 fit
- exp015c LB > exp015b LB (= simple 式が劣化) なら、 「複雑式が overfit、 simple が generalize」 で simple 採用
- 全 variant 失敗なら exp005 v2 に戻して PySR 路線 drop

### 6.2 効果分離計画 (= ratio + delta 監視)

- `docs/research/2026-05-11-local-vs-lb-correlation.md` (= ~/projects/kaggle/CLAUDE.md §8.1 #3 ルール、 未作成なら本 spec 採択時に新規起票) に 4 datapoint を append
- `LB_obs - LB_expected` の delta σ で confidence interval、 noise 範囲 ≤ 0.05 ft なら有意性確認

### 6.3 並列開発との衝突回避

- subagent Y kernel run (= 推定: exp008/009 v2 系) と本 exp015 は **branch 分離** で衝突回避
- 本 spec の implement 担当 worker は `feat/phase-7-exp015-pysr` branch で作業、 merge は中央

---

## 7. failure modes 3 件 + recovery

### 7.1 PySR install 失敗 (= Julia / GLIBCXX 衝突)

| 兆候 | recovery |
|---|---|
| `julia: command not found` / `juliaup` install で hang | (a) 案 B offline-export (= §1.2.2) に切替、 local で fit のみ → kernel は hardcode<br>(b) `LD_LIBRARY_PATH=$HOME/.julia/juliaup/julia-*/lib/julia/:$LD_LIBRARY_PATH` 設定 (= PySR README troubleshooting)<br>(c) docker fallback: `docker build -t pysr --build-arg JLVERSION=1.10.0 --build-arg PYVERSION=3.11.6 .` (= PySR README) |
| `pysr.install()` で network timeout | (a) Kaggle private dataset として Julia depot upload (= §1.2.3 案 C)<br>(b) local install 後、 `~/.julia/` zip → dataset upload |
| GLIBCXX_3.4.30 not found | `conda install -c conda-forge libstdcxx-ng=12` (= context7 query 結果) |

最悪 case: 案 B (= offline-export) は **Julia 不要**、 local Python で sympy + scipy GA を hand-roll しても近似可能 (= paradigm-deeper §4.4.2 failure mode #3 対策)。

### 7.2 PySR 式が overfit (= train R² 高、 CV R² 低)

| 兆候 | recovery |
|---|---|
| best_score variant の cv5_std > 0.05 ft | (a) `maxsize=15` に削減、 (b) `parsimony` 1 桁上げ、 (c) simplest_acceptable variant 採用 |
| best_score の TVT_implied delta が WLS baseline 比 -0.02 ft 未満 (= 無意味) | (a) `select_k_features=4` で feature 制限、 (b) per-formation でなく per-formation × per-cluster (= 例: dip_angle 二分) で fit |
| best_accuracy variant の complexity > 20 (= maxsize 飽和) | maxsize=30 に上げて再 fit (= 工数 +1 日)、 ただし overfit risk 増大 |

paradigm-deeper §4.4.2 failure mode #2 と同一、 Pareto front から **simpler 式選択** が一次対策。

### 7.3 ROGII LB に効かない (= exp015b - exp015a ≥ 0、 noise 範囲埋没)

| 兆候 | recovery |
|---|---|
| exp015b LB - exp015a LB が ±0.05 ft 以内 (= noise 圏) | (a) **既存 `bww_<F>` で blend が既に b_well variance を吸収済** → 純 feature 追加で寄与なし<br>(b) variant: PySR 式を `bww_<F>` の **置換** (= 既存 5 系列から `bww_<F>` を drop、 b_sym で置換) として再 submit<br>(c) Ridge meta blend の weight 出力を log → b_sym 系列が weight 0 を取られている確認、 ある場合は feature 順序 / scaling 調整 |
| 6 formation で 1-2 formation のみ improve、 残り 4-5 formation は劣化 | formation 別 `weight` を Ridge meta で個別学習 (= feature scaling 修正)<br>もしくは `b_sym_<F>` を 「劣化 formation で 0」 に mask |
| ratio drift (= CV vs LB) > 0.3 ft | (a) PySR fit の fold split が CV (= GroupKFold) と一致しているか確認、 leak 可能性<br>(b) symbolic 式が外挿 unstable (= test wells で X, Y が train wells の range 外)、 well 分布の coverage 確認 |

最悪 case: 全 6 formation で改善ゼロなら PySR feature を drop、 exp005 v2 ベースに戻す (= rollback コスト最小、 1 feature 削除のみ)。

---

## 8. 工数 + LB 寄与推定の確度

### 8.1 工数 (= local CPU、 spec to ship)

| phase | 期間 | 担当 |
|---|---|---|
| §1 install 案選定 + local PySR setup | 0.3 日 (= 案 B 採用想定) | implement worker |
| §2 train data 構築 (= 既存 `bww_<F>` 流用) | 0.3 日 | implement worker |
| §3 PySR fit 6 formation 独立 | 0.5 日 (= 1 hr 実行 + 確認) | implement worker |
| §4 Pareto front 選定 + 5-fold cv valid | 0.5 日 | implement worker |
| §5 kernel への注入 (= `compute_b_well_symbolic` + `build_well_features` patch) | 0.5 日 | implement worker |
| §6 exp015 4 variant submit | 0.2 日 (= 4 submit 並列) | submit worker |
| **合計** | **2.3 日** | |

→ paradigm-deeper §5.1 「2-3 日」 と整合。

### 8.2 LB 寄与確度 (= 3 シナリオ)

| シナリオ | 確率 | LB delta vs exp005 v2 (= 10.317) | 説明 |
|---|---|---|---|
| optimistic | 25% | -0.3 ft → **10.02** | paradigm-deeper §4.4.1 上限値、 全 6 formation で b_well noise 吸収、 Ridge weight 適切 |
| central | 50% | -0.1 ft → **10.22** | 中央値、 4-5 formation で improve、 1-2 formation で no-op |
| pessimistic | 25% | 0 ft → **10.32** | b_well variance を 既存 `bww_<F>` が吸収済、 純加算ゼロ |

期待値: -0.125 ft (= 25%×0.3 + 50%×0.1 + 25%×0)。

→ paradigm-deeper §5.4 主張「合成 4 paradigm hit で 7.5-8.5 帯」 のうち PySR 単独貢献は確度高くは 0.1-0.2 ft、 上限 0.3 ft (= optimistic case)。

### 8.3 9 切り (= LB 8.x 帯) 戦略における位置づけ

`docs/dev/leaderboard.dense.md:67-72` で「9 切らないと優勝は無理」、 exp005 v2 (10.317) → 8.5 帯まで **-1.817 ft** 必要。 PySR は単独で 7-23% (= 0.13-0.42 ÷ 1.817) 担当、 残り 1.4-1.7 ft は Mamba / PINN / TabPFN v2 / Edge R 改修などに依存。

→ PySR は **9 切り戦略の中で「補助・短工数・低 risk」** position、 メインエンジン (= Mamba / PINN) と並列 development の前哨として有効。

---

## 9. 残課題

### 9.1 本 spec で未解決

1. **PySR fit の reproducibility**
   - `random_state=42` 固定で 6 formation × 3 fold で結果が一致するか実機検証 (= local fit 必要)
   - PySR は内部 multi-thread GA、 thread schedule で seed 固定でも結果 drift する可能性 (= PySR docs `https://ai.damtp.cam.ac.uk/pysr/v1.5.9/reproducibility/` 参照、 fetch 失敗で内容未取得)

2. **per-formation fit vs joint fit**
   - 6 formation 独立 fit (= §3.1.2) vs 6 formation joint fit (= multi-output PySR、 PySRRegressor 対応未確認)
   - joint fit なら formation 間共通項 (= ANCC plane fit など) が抽出可能、 ただし PySR の multi-output 対応は `https://github.com/MilesCranmer/PySR/issues?q=multioutput` で要確認

3. **train wells の coverage**
   - 773 train wells の X, Y 分布が test wells の X, Y 範囲を cover しているか (= extrapolation risk)
   - per-well-stats parquet に test wells 統計が無い (= test data 未触れ)、 EDA 必要

4. **既存 `bww_<F>` との redundancy 確証**
   - Ridge meta blend で `bww_<F>` が既に b_well variance を吸収済の場合、 PySR feature 純加算ゼロ (= §7.3 failure mode)
   - feature importance 比較 (= SHAP / permutation importance) で `bww_<F>` の effect size 事前測定

5. **Kaggle Julia install の最新状況**
   - 2026 年 5 月時点で Kaggle base image が Python 3.11 + 自動 Julia install を許容するか、 公式 forum の最新 thread (= `https://www.kaggle.com/discussions/general?search=pysr+julia`) 未確認
   - install 失敗 example が複数報告されている場合、 案 B (offline-export) を default 化

### 9.2 5/13 以降への引継ぎ事項

- 本 spec の implement 担当 worker は **`docs/research/pysr-implementation-spec.dense.md`** (= 本 doc) を起点に **`experiments/pysr/` ディレクトリ + `b_well_formulas.json` を作成**
- exp015 4 variant submit 結果 (= LB) を `docs/dev/leaderboard.dense.md` に append、 ratio doc 更新
- 残課題 #1-#5 のうち #3 (= train/test coverage) は EDA worker (= subagent T / U) と協調、 paradigm-deeper §7.1 Mamba 残課題と同時に処理

### 9.3 関連 doc / 一次資料

| ref | URL / `file:line` | 用途 |
|---|---|---|
| PySR (Cranmer 2023) | `https://arxiv.org/abs/2305.01582` | §0.2 paradigm 採用根拠 |
| PySR official docs | `https://ai.damtp.cam.ac.uk/pysr/v1.5.9/` | §1.1, §3.1 hyperparameter |
| PySR GitHub | `https://github.com/MilesCranmer/PySR` | §1.2 install, §3.1 API |
| PySR Springer review (Cranmer 2024) | `https://link.springer.com/article/10.1007/s10710-024-09503-4` | §3.1 niter elbow rule |
| paradigm-deeper §4-§6 | `docs/research/paradigm-deeper.dense.md` (branch `docs/paradigm-deeper-2026-05-11`) | §0 base spec |
| first-principles §0.2 | `docs/research/first-principles.dense.md:38, :164-175` | §0.1 数値根拠 (= b_drift, formula_oracle) |
| mathematical-formulation §0.2 | `docs/research/mathematical-formulation.dense.md:64-65` (branch `7b225ce`) | §0.1 generative model |
| per-well-stats parquet 実測 | `outputs/eda/first_principles/per-well-stats.parquet` | §0.1, §2.3 数値 |
| 既存 `wls_b_well` | `src/rogii/tysig.py:400-416` | §2.1 流用元 |
| 既存 `build_well_features` per-formation loop | `src/rogii/features.py:147-171` | §5.2 注入位置 |
| FORMATIONS list | `src/rogii/imputers.py:20` | §2.1 6 formation |
| leaderboard | `docs/dev/leaderboard.dense.md:12, :16, :67-76` | §0.1 LB ベースライン |
| Kaggle workflow rule | `~/projects/kaggle/CLAUDE.md` §4.1 (= 5/day quota) | §6.1 AB test design |
| 中立指示 + 出典規律 | `~/.claude/CLAUDE.md` (= 主道 §中立指示原則 + Links not verdicts) | 全章 |

---

## 10. 並列衝突回避 (= 中央への報告)

本 spec は branch `docs/pysr-spec-2026-05-11` 専有、 `git add docs/research/pysr-implementation-spec.dense.md` 限定で commit。 subagent Y kernel run (= exp008/009 v2) や subagent W (academic-literature-deeper) との同時 push で 衝突した場合の rollback:

- `git reflog --date=iso` で SHA 確認 → `git branch <recover-name> <SHA>` で永続化 (= ~/.claude/CLAUDE.md [2026-04-30] orphan-recover lesson)
- 30 分以内に中間 commit (= ~/.claude/CLAUDE.md "30 分毎の中間 commit 厳守")
