# Kaggler の叡智カタログ — ROGII 1 位狙い用 transferable pattern 集

> 過去の金 medal solution / NVIDIA Kaggle Grandmaster Playbook / 関連 winning writeup から、ROGII (TVT 回帰) に転用できるテクニックを徹底収集する。
> 各 exp{NNN} で **当たり前のように適用** すべき工夫の checklist。

## A. 特徴量設計 (Feature Engineering)

### A-1. Bestagini Polynomial Features (FORCE 2020 1 位採用)

水平・近傍点の特徴量を 2 次多項式に展開する geophysics 慣習。
出典: [Bestagini et al. 2017 SEG](https://library.seg.org/doi/10.1190/segam2017-17688551.1) — well-log lithology 用に開発、FORCE 2020 winner Olawale Ibrahim も採用。

ROGII への適用:
- 各点の neighbor (±N ft) で gradient / second derivative を作る
- `GR * MD`, `X^2`, `Y^2`, `GR^2` を追加
- formation transitions を増幅する効果

### A-2. Per-Group Aggregations + Differences (M5 / AmEx 1 位流)

各 well 単位で `mean / std / min / max / median / quantile / skew / kurt / nunique` を計算 → row に broadcast → row との `diff` `ratio` を取る。

ROGII への適用:
- per-well `gr_mean`, `gr_std` (実装済み exp001)
- 加えて `gr_p25`, `gr_p75`, `gr_skew`, `gr_kurt`
- `MD - md_visible_max`, `Z - z_visible_end`, `GR - gr_mean`
- visible window 内の TVT_input の trend (slope, intercept)

### A-3. Lag / Rolling / Cumulative Triple Set (M5 winner 流)

時系列・sequence では **lag, rolling, cumulative** の 3 種を組み合わせる。
ROGII での具体: lag (実装済み 1/5/20), rolling mean+std (実装済み 10/50/200), **cumulative** (cumsum of GR, cumprod of dZ) は未実装。

### A-4. Target Encoding (leak-free)

categorical → target mean を fold-wise に算出。well_id を target encode するときは GroupKFold で漏れ防止。

ROGII での適用:
- well_id 単位の `TVT_input.mean()` (visible 部分のみ) を target proxy として row に付与
- ただし well_id は test 側に未知 wells があるので **OOF target encoding** 必須

### A-5. Multi-Resolution / Multi-Scale Features

短期 (10ft), 中期 (50ft), 長期 (200ft) の rolling を別 column として全部入れる。

ROGII での適用: 既に GR_roll_mean_{10,50,200} 実装。**追加で 1000ft (≈ well 全長の 15%) や typewell-aligned window も入れる**。

### A-6. Interaction Features (Otto / Optiver 流)

```
GR_per_MD = GR / (MD + epsilon)
GR_minus_planefit_baseline = GR - alpha * tvt_planefit_pred
X_times_Y, dX/dZ, lateral_velocity * GR
```

ROGII では trajectory × GR の cross term が **layer 反復 (going up/down strata)** を encode する可能性。

### A-7. Domain-Specific Physics-Informed Features (geophysics)

- **dip angle** (= arctan2(dZ, lateral_velocity)) ← 実装済み
- **dogleg severity** (= 隣接 trajectory の角度変化、SLB 慣習)
- **distance to typewell axis** (well の典型 X/Y からの偏差)
- **estimated formation top** (visible TVT_input から線形外挿)

## B. CV 戦略 (Cross-Validation)

### B-1. GroupKFold by Group (= well_id)

leak 防止の最重要事項。実装済み exp001。

### B-2. **Eval-Zone-Mimicking CV** ★ ROGII 必須

LB 評価は「each test well の trailing 73% を予測」なので、CV でも同じ構造を再現する:
- train fold の wells から visible 25% だけを mask 解除した状態で feature 構築
- val fold の wells は visible 25% で feature 構築 → trailing 75% で RMSE 評価

実装済み exp001 (val metric を hidden rows only で算出)、ただし **train side でも feature 構築時に "future leakage"** が無いかは要点検 (現状は visible+hidden の GR を rolling に使ってる、これは test でも GR は全長見えるので合法)。

### B-3. Stratified by Difficulty

`visible_ratio` や `n_rows` の分布で 5 等分し fold に均等配分。
これにより各 fold の難易度を揃える。

### B-4. Multiple Random Seed CV

CV を 3-5 個の seed で回し、OOF を平均して variance 削減。
スコア再現性確認にも使える。

### B-5. Adversarial Validation

train と test の特徴量分布差を classifier で測定 (AUC が 0.5 から離れたら distribution shift)。
重要 features を特定 → 該当 features を `drop` または `clip` する。

ROGII での具体: 200 test wells (本番) と 773 train wells で X/Y 座標分布が異なる可能性。adversarial AUC > 0.7 なら要対応。

## C. LightGBM / XGBoost / CatBoost のベストプラクティス

### C-1. Early Stopping + Best Iteration

`lgb.early_stopping(round_patience)` で val RMSE 監視、best_iteration を booster.best_iteration で取得して predict。
実装済み exp001 (200 round patience)。

### C-2. Hyperparameter Tuning (Optuna)

```python
import optuna
def objective(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 31, 511),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 32, 1024, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 1),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": 0.05,
    }
    # 5-fold CV → return mean RMSE
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=200, n_jobs=1)
```

NVIDIA Grandmaster Playbook 推奨: TPE sampler + 100-200 trials。

### C-3. Multi-Objective (RMSE + MAE / Huber)

LightGBM は `objective: regression` (= L2/RMSE) 以外に:
- `regression_l1` (MAE) ← 外れ値に強い
- `huber` (delta=1.0) ← L2 と L1 のハイブリッド
- `fair` ← さらに robust

ROGII での適用: `regression` と `huber` を別 model で訓練 → blend
(Ventilator 1 位は MAE optimization、ROGII は RMSE 評価なので主軸は L2、補助で L1/Huber)

### C-4. Boosting Type 多様化

`boosting_type`:
- `gbdt` (default, 標準)
- `goss` (Gradient One-Side Sampling, 大規模データ用、訓練 2-3 倍速)
- `dart` (Drop-out Additive Regression Trees, 正則化強い、遅い)

5M rows で goss を試す価値あり (ROGII)。

### C-5. Monotonic Constraints

特徴量と target の単調関係を model に強制:
```python
"monotone_constraints": [+1 if "MD" in c else 0 for c in feature_cols]
```
ROGII で TVT は MD と単調増加じゃないので NG (lateral では TVT は変動)。
ただし `tvt_planefit_pred` は target を近似するので **+1** monotonic を入れる価値あり。

### C-6. Categorical Feature Handling

CatBoost の native categorical は ordered boosting で leak-free。
LGB の `categorical_feature=[...]` も同様に special-cased。

ROGII での適用: well_id は test に新 wells あるので使えない。formation 系は train only なので使えない。**categorical なし** で進む。

### C-7. Multi-Seed Ensemble (variance 削減)

```python
seeds = [42, 123, 2024, 7, 99]
oofs = []
for s in seeds:
    LGB_PARAMS["seed"] = s
    LGB_PARAMS["bagging_seed"] = s
    LGB_PARAMS["feature_fraction_seed"] = s
    oof_s, test_s = train_kfold(LGB_PARAMS)
    oofs.append((oof_s, test_s))
final_oof = np.median(np.stack([o for o,_ in oofs]), axis=0)
final_test = np.median(np.stack([t for _,t in oofs]), axis=0)
```

**weighted MEDIAN ではなく単純 median** が外れ値に強い (Ventilator 1 位採用)。

### C-8. XGBoost / CatBoost を別系統で並走

LGB と XGB は内部実装が違う (histogram vs exact split, leaf-wise vs depth-wise) ので、同じ features でも違う fit。CatBoost はさらに ordered boosting / SymmetricTree で違う性質。

3 model の OOF を blend するだけで通常 -0.1〜0.3 の改善。

## D. アンサンブル / Stacking

### D-1. OOF Stacking with Ridge Meta-Learner

```python
# Level 0: LGB / XGB / CB / Linear → OOFs (5 OOFs)
# Level 1: Ridge regression on Level 0 OOFs → final prediction
from sklearn.linear_model import Ridge
meta = Ridge(alpha=1.0)
meta.fit(np.stack([oof_lgb, oof_xgb, oof_cb, oof_linear, oof_seq], axis=1), y_train)
final = meta.predict(np.stack([test_lgb, test_xgb, test_cb, test_linear, test_seq], axis=1))
```

`kagglib.stacking` に実装済み。Level 0 数を増やすほど効くが過学習に注意 (5-7 model が最適)。

### D-2. Hill Climbing (= greedy ensemble)

過去の AmEx 1 位、Optiver 1 位採用:
```python
weights = [0]*N
best_rmse = inf
for step in range(100):
    for k in range(N):
        for delta in [-0.05, +0.05]:
            new_w = weights.copy(); new_w[k] += delta
            new_oof = np.einsum("nk,k->n", oof_stack, new_w)
            rmse_new = compute_rmse(y, new_oof)
            if rmse_new < best_rmse: best_rmse, weights = rmse_new, new_w
```

スタート weights 0 から greedy に上げる。Ridge より柔軟だが overfit に注意 (validation 分離して early stopping)。

### D-3. Weighted Median Ensemble (Ventilator 1 位流)

mean ではなく **median** を使う:
```python
final = np.median(np.stack([pred_seed_1, ..., pred_seed_15], axis=0), axis=0)
```

外れ値 fold の影響を排除。ROGII でも multi-seed × multi-fold で必須。

### D-4. Multi-Stage Training

stage 1 model → predict → stage 2 model with stage 1 prediction as feature。pseudo-labeling の一般形。
ROGII での具体: stage 1 で粗い TVT 予測 → stage 2 で stage 1 prediction を feature に追加して fine-tune。

## E. データ拡張 / Pseudo-Labeling

### E-1. Pseudo-Labeling (Ventilator / G2Net / OTTO 流)

1. 通常訓練 → test prediction
2. test prediction の高 confidence 部分を train に追加
3. 拡張 train で再訓練

ROGII では:
- visible TVT_input (test) は既に正解 (TVT_input == TVT on visible) なので "labeled" 扱い → 既に train に追加できる
- **hidden 区間の predicted TVT を pseudo-label として再訓練** に使う、ただし 1 round のみ (multi-round は collapse 危険)

### E-2. Label Smoothing for Regression

target に小さい Gaussian noise を加えて訓練 (label smoothing の regression 版)。過学習防止。

### E-3. Mixup / CutMix

`(x_i, y_i)` と `(x_j, y_j)` を `(λ x_i + (1-λ) x_j, λ y_i + (1-λ) y_j)` に混合。
DL では効果大、GBM では効果薄。Phase 4 Transformer で採用。

### E-4. Window Augmentation (sequence 用)

random window mask で hidden region を simulate:
```python
# 各 well で trailing X% を hidden 化 (X は uniform sample)
mask_ratio = np.random.uniform(0.4, 0.85)
mask_start = int(len(well) * (1 - mask_ratio))
well["TVT_input"][mask_start:] = np.nan
```

ROGII では **CV 設計と一致** させて再構築するのが本質。

## F. Post-Processing

### F-1. Physics-Constrained Snap

予測値を物理的に妥当な範囲に snap:
- TVT は typewell の TVT 範囲内に clip (ただし 1.7% は外側、上下 100ft 程度は許容)
- TVT は MD 増加方向で滑らか → cubic spline で smoothing

### F-2. PID-Style Smoothing (Ventilator 1 位流)

予測の高周波 noise を取り除く savgol_filter / Gaussian smoothing:
```python
from scipy.signal import savgol_filter
smoothed = savgol_filter(pred, window_length=11, polyorder=3)
```

ROGII では各 well 内で 1D smoothing (window 5-21 ft 程度)。

### F-3. Geology Boundary Snap

typewell の Geology label が変わる TVT 値 (= layer boundary) を抽出 → 予測値が boundary 近傍 (±10 ft) なら boundary に snap。
出典: SLB Petrel automatic well-tie の発想。
ROGII で typewell Geology は train 時のみ可だが、layer boundary の TVT 値は train 時に取得できれば test の typewell GR からも逆推定可能。

## G. 実装上の効率化

### G-1. Float32 + Pandas → Polars

5M rows × 35 features を float64 で持つと 1.4 GB、float32 で 700 MB。
更に polars に置き換えると groupby が 5-10x 速い。

### G-2. PYTHONUNBUFFERED=1

学習進捗を tee で見るには:
```bash
PYTHONUNBUFFERED=1 uv run python -m rogii.train_baseline 2>&1 | tee log.log
```
または `-u` flag。
exp001 で stdout flush されない問題あり、exp002 以降で必須。

### G-3. lightgbm `device=gpu` 

NVIDIA GPU 環境では:
```python
LGB_PARAMS["device"] = "gpu"
LGB_PARAMS["gpu_platform_id"] = 0
LGB_PARAMS["gpu_device_id"] = 0
```
5x 高速化。Kaggle Notebook の P100 でも有効。

### G-4. `lightgbm.Dataset.save_binary()`

訓練データを binary に保存しておくと re-load が 10 倍速い (Optuna の繰り返しに有効)。

## H. Sequence DL 用 (Phase 4 Transformer)

### H-1. PatchTST + Cross-Attention

ICLR 2023 SOTA。出典: [arxiv 2211.14730](https://arxiv.org/pdf/2211.14730)
ROGII 実装 (Plan 案 B): query=horizontal, key/val=typewell。

### H-2. Masked Modeling Pretrain

BERT 流の 15% mask + reconstruct で self-supervised pretrain (FORCE 2020 / VOLVE で大規模訓練 → ROGII fine-tune)。

### H-3. Multi-Task Heads (Ventilator 1 位流)

main task (TVT) + auxiliary (ΔTVT, Geology label) の同時訓練で representation を richer にする。

### H-4. Gradient Accumulation + Mixed Precision

長 sequence の DL で memory 節約:
```python
torch.cuda.amp.autocast()
scaler = torch.cuda.amp.GradScaler()
optimizer.step() を毎 N batch
```

## 推奨優先順位 (ROGII 適用順)

| 段階 | 必須 (exp001-005) | 推奨 (exp006-010) | 余裕あれば (exp011+) |
|---|---|---|---|
| **特徴** | A-2, A-3, A-5, A-6, A-7 | A-1 (Bestagini), A-4 (target enc) | DTW well-tie (independent script) |
| **CV** | B-1, B-2 | B-3, B-4 | B-5 (adversarial) |
| **GBM** | C-1, C-2 (Optuna), C-3 (multi-obj) | C-4 (goss), C-7 (multi-seed) | C-5 (monotonic) |
| **3-model** | (LGB only) | C-7 (LGB+XGB+CB) | --- |
| **Ensemble** | --- | D-1 (Ridge stack) | D-2 (hill climb), D-3 (median) |
| **Pseudo** | --- | E-1 (1 round) | --- |
| **Post** | --- | F-1, F-2 | F-3 (Geology snap, hard) |

## 参考リソース (随時拡充)

- NVIDIA Kaggle Grandmaster Playbook 2024: https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/
- State of ML Competitions 2025: https://mlcontests.com/state-of-machine-learning-competitions-2025/
- AmEx Default Prediction 1 位 writeup: https://www.kaggle.com/competitions/amex-default-prediction/writeups
- M5 Forecasting Accuracy 1 位: https://www.kaggle.com/competitions/m5-forecasting-accuracy/discussion/163684
- Ventilator Pressure Prediction 1 位: https://medium.com/data-science/winning-the-kaggle-google-brain-ventilator-pressure-prediction-2d4c90d831ec
- FORCE 2020 1 位 (Olawale Ibrahim): https://ibrahim-olawale13.medium.com/force-2020-machine-learning-lithology-predictionwinning-solution-8cbf78290b41
- SPWLA 2023 Depth Shift 1 位 (Dreamstar): https://github.com/pddasig/Machine-Learning-Competition-2023
- Bestagini polynomial features (SEG 2017): https://library.seg.org/doi/10.1190/segam2017-17688551.1
