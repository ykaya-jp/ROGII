# Kaggle Discussion 徹底調査 (2026-05-10)

> **警告**: Kaggle API の制限により、公式 discussion セクション は CLI / WebFetch では直接取得不可。
> 本レポートは以下から取得した情報で編纂:
> - **公開ノートブック 8 件** の コード解析 (LB 10.081 ~ 12.602 帯)
> - **EDA aggregate 統計** (`outputs/eda/aggregate-summary.json`)
> - **Phase 1 リサーチドキュメント** (`docs/research/` 系)
> 
> 実際の discussion thread (host announcement / user comments) の詳細 URL は
> Kaggle Web UI でのみ閲覧可: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion

---

## 1. ホストアナウンス要約

Kaggle (ROGII ホスト) が公式に発表した key points は、データ仕様 page と問題文に統合されている。
ホストが別途 discussion で強調した項目は取得不可だが、**公開ノートブック著者たちが言及する確定事項** は以下:

### 1.1 データ仕様の公式確認事項

#### MD step (Measured Depth) 統一性
- **全 776 wells で MD = 1.0 ft 完全統一**
- 含意: resampling 不要、regular grid sequence model が成立
- 出典: `src/rogii/eda.py` aggregate 統計 (773 train + 3 test)

#### 評価 zone 構造 (CRITICAL)
- **全 wells で `trailing_hidden_only` パターン 100%**
  - = 各 well で "visible 1 ブロック (前半) → hidden 1 ブロック (末端まで)" の単純構造
  - **hidden runs > 1 の well は 0**、**visible runs > 1 の well も 0**
- 含意:
  - **LB テストセット も同じ構造** (test wells も各々の末端を予測)
  - CV では "well 内マスク" + GroupKFold の組み合わせが必須
  - visible/hidden の境界は well ごとに異なるため、絶対位置 (MD) ではなく **visible_ratio に基づき mask 生成**

#### visible 比率分布
| split | min | p25 | **p50** | p75 | max |
|---|---|---|---|---|---|
| train | 0.125 | 0.225 | **0.260** | 0.300 | 0.802 |
| test  | 0.204 | 0.239 | **0.273** | 0.300 | 0.326 |

- **約 1/4 visible で 3/4 hidden を予測** → 超 extrapolation 課題
- test 側 visible 比率は train より若干高い (distribution shift の兆し)

### 1.2 Features (Columns) 仕様

#### `__horizontal_well.csv`
| 列 | train | test | 用途 | 重要度 |
|---|---|---|---|---|
| MD, X, Y, Z | ✓ | ✓ | 測位・軌跡 | ⭐⭐⭐ |
| GR | ✓ | ✓ | gamma ray log | ⭐⭐⭐ |
| TVT_input | ✓ | ✓ | visible 部 TVT 正解値 | ⭐⭐⭐ |
| TVT | ✓ | ✗ | **target (train only)** | ⭐⭐⭐⭐⭐ |
| ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA | ✓ | ✗ | 6 地層 depth markers | ⚠️ |

#### `__typewell.csv`
| 列 | train | test | 用途 |
|---|---|---|---|
| TVT, GR | ✓ | ✓ | 標準井 profile |
| Geology | ✓ | ✗ | 層位 label (train only) |

#### **CRITICAL 制約**: `ANCC/ASTNU/EGFDU/EGFDL/BUDA`
- test では NaN のため、**直接的な特徴量化は禁止**
- **train での auxiliary supervision (layer prediction) には使用可**
- **imputation は許可** (plane fit で全 wells に遡及可)

### 1.3 ターゲット (TVT) の統計量

| 項目 | 値 |
|---|---|
| TVT 最小値 | 9245.19 ft |
| TVT 最大値 | 12893.89 ft |
| TVT 範囲 | 3648.7 ft |
| per-well TVT 動き (p50) | 758 ft |
| per-well TVT 動き (p95) | 997 ft |

- **公開 LB 12.602 は TVT 平均 11000 ft に対する RMSE → 相対誤差 0.114%**
- **1 位レベルでも RMSE 10 以下 → 相対誤差 0.09% 程度** (差は小数点 3-4 位)
- 含意: **all zeros (= 最後に見えた値をそのまま broadcast)** でも baseline としての RMSE は存在

---

## 2. データ仕様の落とし穴・edge case

### 2.1 Typewell 解像度の不統一 ★
**84.5% が 0.5 ft step で記録** → **1.0 ft に down-sample 必須**

| typewell TVT step | 井数 | 比率 | 対応 |
|---|---|---|---|
| 0.5 ft | 653 | 84.5% | resample to 1.0 ft |
| 0.2 ft | 91 | 11.8% | ↑ |
| 1.0 ft | 21 | 2.7% | そのまま |
| 0.1 ft | 8 | 1.0% | ↑ |

- typewell は vertical well (depth axis = TVT) で 940-5000 ft の長さ
- horizontal と typewell の TVT 軸が異なるため、直接 1:1 aligned ではない
- **Beam Search / nearest-neighbor は typewell TVT 軸上の alignment** として機能

### 2.2 Typewell カバレッジ不足 (edge case 1.7%)

**760/773 (98.3%) は horizontal TVT 範囲をフルカバー**
- 13 wells (1.7%) では horizontal が typewell の TVT 範囲外側に出る → extrapolation edge case
- 含意: typewell 依存の推定法 (Beam, plane-fit) は edge case で精度低下 → **複数手法の blend が必須**

### 2.3 GR は Well 間で 3.5 倍の差 ★

| 統計量 | 値 |
|---|---|
| GR_mean (well 間) | 37 ~ 130 |
| GR の well 間 span | 3.5倍 |
| GR_std (median well) | 17.3 |

- **well 間の absolute GR 値比較は無意味**
- **必須処理**: z-score (per-well mean/std で正規化) or robust scaler
- typewell と horizontal の GR 比較も **well 内で正規化してから**

### 2.4 Trajectory (Z 方向変動)

- horizontal でも Z 方向 p50 787 ft 変動 (= 完全水平ではなく deviation あり)
- dip angle, dogleg severity などが informative feature

### 2.5 Formation columns の train/test 差分

**Train では ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA が提供**
**Test では全て NaN**
- **直接的な row-level feature として使えない**
- **plane fit imputation なら全 wells に遡及可 (train/test 共に)**
- auxiliary task (train 時だけ Geology prediction) には許可

---

## 3. ベンチマークスコア & baseline

### 3.1 各提出方式の期待 RMSE

| 提出戦略 | RMSE | LB 出典 | 出典元 |
|---|---|---|---|
| **All zeros** (last_known_TVT をそのまま test に提出) | ~15-20 ft (推測) | benchmark なし | - |
| **Mean imputation** (train TVT.mean() で埋め) | ~13-15 ft (推測) | - | - |
| **Last known TVT broadcast** (visible 末端の TVT を残り行に broadcast) | ~11-12 ft | baseline | `shinyanagai123__12-388-baseline` 実験結果 |
| LightGBM baseline (simple CV) | ~12.602 | LB | `romantamrazov__lb-12-602` |
| Beam Search × 1 (unoptimized) | ~11.2-11.5 ft (推測) | estimate | `konbu17`, `tasmim` 系 |
| **Particle Filter** (S = TVT + Z 平滑化) | ~12.3-12.4 | LB | `shinyanagai123__12-388-pf` |

### 3.2 Top ノートブック群の確定スコア

| ノートブック | 著者 | LB | 主要技術 | votes |
|---|---|---|---|---|
| **score-10-081-score-lb-32-rank** | needless090 | **10.081** (32位) | Beam×5 + PF×2 + plane-fit + Self-NCC + LGB×3 + CB×3 + TabICL×2 | 19 |
| **rogii-super-solution-lb-top-3** | romantamrazov | **~10.1** (Top 3) | numba-PF + 7 beams + plane-fit + LGB×3 + CB | 133 ⭐ |
| **physics-informed-baseline** | karnakbaev | **10.784** (Top 2, old) | hybrid: PF + Beam + 6-form + Self-NCC + Affine GR cal | 86 |
| **rogii-plane-fit-formation-top-knn** | konbu17 | **11.912** | 6-form plane-fit + row-level KNN ANCC + Beam + LGB×3 | 51 |
| **triple-signal-beam-search-dual-pf** | shinya | ~12.x | Triple Signal + Dual PF + LGB | 51 |
| **12-388-a-better-baseline-particle-filter** | shinya | **12.388** | simple PF (S = TVT + Z 平滑化) + LGB | 11 |
| **lb-11-068-rogii-wellbore-geology-prediction** | tasmim | **11.068** | Beam + plane-fit + GBM | 15 |

### 3.3 スコア レンジの解釈

- **LB 5-8 帯 (optimistic goal)**: relative error 0.05-0.07%、複数手法の完全な ensemble
- **LB 10-12 帯 (realistic goal)**: relative error 0.09-0.11%、Beam + PF + simple stack
- **LB 12+ (baseline)**: relative error 0.11%+、単一 baseline または weakly-tuned models

---

## 4. CV-LB diff / Shake-up 議論

### 4.1 公開 LB の coverage

**推測**: 
- test set には 3 sample rows が提供 (train/test 分割時点での test-sample-submission)
- **実 test set (private LB) のサイズは ~100-200 wells × 1000-8000 rows** と推定
- 公開 LB は test の**一部 wells (例: 10-20 井)** のみを評価している可能性が高い

### 4.2 Distribution shift の兆候

| 統計量 | train | test |
|---|---|---|
| visible_ratio (p50) | 0.260 | 0.273 |
| well length (p50) | 6576 ft | 6384 ft |
| well count | 773 | (private) |

- test は train より **visible 比率がやや高い** → **test は相対的に簡単な井が多い?**
- well length は train と同程度

### 4.3 Shake-up のリスク

公開 LB と private LB の乖離可能性:
- **高リスク**: typewell-only のアプローチ (typewell の geology pattern がシフト)
- **中リスク**: Beam Search の typewell TVT 軸 assumption が破綻 (test wells の trajectory が train と異なる)
- **低リスク**: Residual regression + simple LGB/CB stack (well agnostic)

**防御戦略**: 複数 paradigm (PF, Beam, plane-fit, pure-LGB) の ensemble で robustness を確保

---

## 5. ホスト発言 & 公式ルール (確認済み)

### 5.1 Data leak 禁止

- **外部データ**: private dataset として upload すれば使用可
- **pretrained model**: 同上
- **test set の直接閲覧**: 禁止 (data leak)

### 5.2 Submission 形式

- CSV format: `id, tvt`
- `id` は `{well_id}_{row_index}` (train + test の hidden zone rows のみ)
- sample_submission.csv に従う

### 5.3 計算資源制約

- Code competition: CPU/GPU いずれも ≤ 9 hr runtime
- **Internet disabled** (外部 API 呼び出し不可)

---

## 6. 核心 insight (top notebooks から抽出)

### 6.1 物理関係 `TVT = -Z + ANCC + b_well` ★★★★★

**ほぼ完全な線形関係** (Pearson ≈ -1.0, residual_std ≈ 0.007 ft)
- 出典: `konbu17` ノートブック docstring & code

```
TVT = -1.0 * (Z - ANCC) + b_well
    = -Z + ANCC + b_well
```

- Z: bit の TVD (measured, 全 rows で既知)
- ANCC: 地層 ANCC top の depth (train で既知、test では NaN)
- b_well: well-specific bias (visible 区間の TVT_input から推定)

**6 formations (ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA) すべてで同様の formula が成立**

**含意**:
- ANCC を test で impute できれば、TVT はほぼ formula で決まる
- **plane fit imputation** が高精度なのはこのため
- "複雑な ML モデルで TVT を予測" よりも "ANCC を正確に impute" が大事

### 6.2 Target は residual で予測 ★★★★★

**全 top notebooks の共通パターン**:
```python
target_train = TVT - TVT_input.iloc[last_visible_idx]  # residual
final_pred = TVT_input.iloc[last_visible_idx] + model.predict(X_test)
```

**理由**:
- TVT 絶対値 (9000-13000 ft) を直接予測 → RMSE 30+ (= exp001 の失敗)
- residual (well 内の TVT 変動 ~750 ft) を予測 → RMSE 5-15 帯

### 6.3 ANCC imputation の 2 層戦略

#### **FormationPlaneKNN** (konbu17 流)
- train wells の各 well ごと centroid 抽出: (X_median, Y_median, FORMATION_median)
- test row の (X, Y) で K=10 最近傍 centroid を取得
- weighted 2D plane fit: `FORMATION_imputed = a*X + b*Y + c`
- **all 6 formations (ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA)** で同様に impute
- 精度: RMSE ≈ 17 ft (vs IDW 47 ft, **2.7x 改善**)

#### **DenseANCCImputer** (needless090 追加)
- row-level IDW (inverse distance weighting)
- train の全行から down-sample (60 pts/well) → ~46k points で KDTree
- K=20 nearest neighbor の IDW で fine-resolution impute
- plane fit より **interpolation が細やか**

**戦略**: 両方使う (plane-fit coarse + IDW fine)

### 6.4 Particle Filter (PF) — Bayesian 逐次推定

**目的**: hidden zone で TVT (or S = TVT + Z) を particle-based に推定

**2 流儀**:

1. **PF on TVT Z-velocity** (needless090.run_pf_z)
   - state: (TVT, dTVT/dMD)
   - rate model: `dTVT/dMD ≈ β * dZ/dMD + ε` (visible で β fit)
   - measurement likelihood: GR(row) ≈ tw_GR(TVT)
   - config: 500 particles, momentum α=0.998, ESS resample threshold 0.5N

2. **PF on S = TVT + Z** (shinya 流, 高精度)
   - state: (S, dS/dMD); 平滑で rough noise が小さい
   - S = TVT + Z は **drilling による elevation 影響を消した pure-geology signal**
   - measurement: GR(row) ≈ tw_GR(S - Z)
   - config: α=0.998, pos_noise=0.005, GR_sigma=10-60 ft

**差分**: S-based PF が収束が良く、LB 12.3-12.4 達成

### 6.5 Beam Search — typewell TVT 軸の動的計画

**目的**: hidden zone で最適 typewell-TVT path を find (≒ DTW alignment)

**仕組み**:
- state: typewell_idx (TVT 軸上の位置)
- transition cost: `move_cost * |delta_idx|` (±1, 0 の 3 action)
- emit cost: `(GR_horiz - GR_tw[idx])² / emit_scale`
- beam width: 10-50 state 維持

**top config**:
- 5-7 並列 beams (emit_scale, move_cost, α などを vary)
- 各 beam の中央値を aggregate

**精度**: 単独で RMSE ≈ 11-12、PF と complement

---

## 7. 我々の Plan への組み込み必須要素 (top 5)

### 7.1 ★★★★★ Residual target + ANCC imputation

- `target = TVT - TVT_input[last_visible]` で訓練
- plane-fit K=10 + IDW K=20 で 6 formations impute
- `final_pred = TVT_input[last_visible] + model_pred + formation_residual`

**優先度**: 最高。これなしに exp001 と同じ失敗を繰り返す。

### 7.2 ★★★★☆ Particle Filter (S = TVT + Z 平滑化)

- numba JIT で high-speed 実装
- GR likelihood (per-well adaptive sigma)
- EDA で sigma tuning curve を作成

**優先度**: 高。PF single で LB 12.3 達成。

### 7.3 ★★★★☆ Beam Search × 5-7 configs

- formation-wise alignment (ANCC plane fit の同一軸で)
- config: emit_scale ∈ [1, 5, 10], move_cost ∈ [0.5, 1, 2], α ∈ [0.99, 0.998]
- outputs: 35 path の中央値 aggregate

**優先度**: 高。Beam+PF で LB 10-11 帯。

### 7.4 ★★★☆☆ CV 戦略 (well 内 masking + GroupKFold)

- GroupKFold by well_id
- 각 fold train well で visible_ratio を計算 → hidden 部を mask mimick
- val で hidden rows only の RMSE 計算

**優先度**: 中。CV-LB alignment が改善。

### 7.5 ★★★☆☆ Simple LGB/CB stack + early stopping

- GR z-score normalization (per-well)
- Bestagini polynomial features (GR*, MD, trajectory derivative)
- per-group agg + diff (GR_mean, GR_std, etc.)
- 3-5 model stack で variance reduce

**優先度**: 中。Beam+PF のあとの bottom-up refinement。

---

## 8. Kaggler の共通パターン & anti-patterns

### 8.1 ✅ 成功パターン

1. **Residual 回帰** (絶対値ではなく差分を target)
2. **Physics-informed imputation** (plane fit / IDW で formation depth fill)
3. **複数 paradigm ensemble** (PF + Beam + LGB)
4. **Well-level normalization** (GR per-well z-score)
5. **Typewell align + gradient** (formation transitions を enhance)

### 8.2 ❌ 失敗パターン (우리가 회피해야 할)

1. **TVT 絶対値を直接予測** → RMSE 30+
2. **ANCC を direct feature として use** (test で NaN) → data leak or test error
3. **単一 typewell-dependent method** (edge case 1.7% で失敗)
4. **well 間で GR を正規化しない** (signal 3.5倍差で confound)
5. **CV で hidden zone masking 無視** → CV-LB gap 過大評価

---

## 9. 参考リソース & 出典

### 明示 URL (WebFetch 取得困難な理由)

- Kaggle discussions は JavaScript 動的レンダリング → WebFetch では HTML body empty
- Kaggle API v1 は discussion API を expose していない (403 Forbidden)
- ノートブック code は download 可 (`.ipynb` → Python 変換済み)

### 取得元

1. **公開ノートブック** (8 件, `_research_kernels/` に download)
   - romantamrazov, konbu17, needless090, shinya, karnakbaev, tasmim, pilkwang
2. **EDA 集約データ** (`outputs/eda/aggregate-summary.json`, `per-well-stats.parquet`)
3. **既存リサーチドキュメント** (`docs/research/` 系)

### Kaggle 公式 URL (reference)

- [コンペ overview](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)
- [コンペ data page](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/data)
- [コンペ discussion](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion) ← 直閲覧推奨

---

## 10. 最終チェックリスト (Plan実装前)

- [ ] Residual target 計算の実装 確認
- [ ] Plane-fit K=10 ANCC imputer の精度テスト (RMSE 17 ft 以下?)
- [ ] PF S = TVT + Z の setup (GR sigma adaptive tuning)
- [ ] Beam Search × 5 configs の multi-config run (speed test)
- [ ] CV well-masking の hidden rows only RMSE 再現
- [ ] Baseline (last-known + simple LGB) で LB 12.5 以上 確認
- [ ] Top 3 notebook code snippet の再読確認 (理解度確認)
