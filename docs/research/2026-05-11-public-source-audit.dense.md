# 公開 source dataset audit (= 優勝路手法 E 素材確認)

> 起源: docs/dev/2026-05-11-cv-lb-correlation.md § 5.4 (= 未統合 host dataset 5 件 list)
> 目的: 各 dataset の **数理本質的価値** を audit、 「優勝本質性」 criterion で採用判定
> 適用 criterion: `~/projects/kaggle/CLAUDE.md` § 11 (= 数理本質 / 優勝寄与 / rule 耐性 / datapoint 価値)

---

## 1. audit 済 dataset (= 2026-05-11)

### 1.1 `akshankrithick/rogii-gold-top10-direction-source` (138 KB、 5/9 公開)

**内容**:
- `gold_top10_submission.csv` のみ (= 464 KB、 test wells に対する predictions)
- schema: `id, tvt` (= row 単位)
- source code / model artifacts なし

**例 (= test well 000d7d20 の予測値)**:
```
000d7d20_1442,11746.069818072147
000d7d20_1443,11746.156777143762
000d7d20_1444,11746.171364689839
000d7d20_1445,11746.230501516411
```

**判定**:
- ❌ **集約 source として不採用** (= 「優勝本質性」 §11.1 違反: 「公開 source 集約で +X」 = paradigm 1 transduction、 他人解法依存)
- ✅ **diagnostic data として採用候補** (= per-row 比較で「我々がどの row で大きく外しているか」 を pinpoint、 数理本質ある analysis)

**diagnostic 用途の sketch**:
```python
# 我々の exp008 v2 (= LB 9.957) submission vs Gold Top10 (= LB 9.728)
ours = pd.read_csv("submissions/exp008_v2.csv")
gold = pd.read_csv("gold_top10_submission.csv")
diff = ours.merge(gold, on="id", suffixes=("_ours", "_gold"))
diff["abs_err"] = (diff["tvt_ours"] - diff["tvt_gold"]).abs()
# 大きく外している top-N rows を抽出 → どの well / 何 ft 深さで失敗しているか
worst = diff.sort_values("abs_err", ascending=False).head(100)
```

これは「Gold Top10 と我々の delta」 = 数理的 valuable (= 「Top10 が見つけている signal を我々が見落としている箇所」 を pinpoint)。

#### 1.1.1 実測 diagnostic (= 2026-05-11、 14151 hidden rows)

実際に我々 exp008 v2 (= LB 9.957) 出力 (= kaggle kernels output で取得) と Gold Top10 submission を per-row 比較:

**overall**:
- delta mean abs = **2.83 ft**
- delta std = 2.26 ft、 p50 = 2.20、 p90 = 6.29、 p99 = 8.91、 **max = 10.78 ft**

**per-well**:

| well | b_ANCC_med (cluster) | visible_ratio | mean abs | std | max |
|---|---|---:|---:|---:|---:|
| 000d7d20 | 11373 (cluster 37) | 27% | 1.49 | 1.10 | 5.31 |
| **00bbac68** | **11855 (cluster 52)** | **20%** | **3.91** | **2.73** | **10.78** |
| 00e12e8b | 11373 (cluster 37) | 33% | 2.52 | 1.43 | 6.36 |

= **00bbac68 で圧倒的に外している** (= mean abs 3.91 ft、 max 10.78 ft)。 これは F1 finding (= 00bbac68 は b_cluster 11855 = train minority、 visible_ratio 20% = 最難 well) と完全に整合する **data evidence**。

**数理的含意 (= 重要)**:
- LB 9.957 → 9.728 = -0.229 ft 改善必要
- 14151 rows のうち 00bbac68 = 6014 rows (= 42.5%)、 mean abs 3.91 ft
- 仮に 00bbac68 で Top10 並みに予測できれば、 全体 OOF RMSE は √((6014 × 0² + 8137 × 1.5²) / 14151) ≈ 1.13 ft (= simplification、 実際は y_diff 2 乗の平均)
- = **LB lift は 00bbac68 への集中投資で大幅可能**

**戦略仮説 (= 優勝路手法 D Per-Well MoE の data evidence)**:
- 00bbac68 (= b_cluster 11855) は train で minority (= 11855 cluster の train wells は少数、 学習データ不足)
- LGB / Ridge は majority cluster (= 11373) を well 学習、 minority (= 11855) を under-fit
- = **per-well or per-cluster MoE wrapper** が直接効く paradigm
- 具体的: `gate(visible_ratio, b_cluster) → expert_for_low_visibility_minority_cluster` を追加すれば 00bbac68 で大幅 lift

これは次タスク (= 優勝路手法 D) の **データ駆動 motivation**。 「軽さ-driven」 ではなく **「数理本質に基づく LB lift path」** = 優勝本質性 §11.1 ✅ 採用。

---

### 1.4 `nina2025/rogii-03` (495 KB、 5/11 公開) — ★ **超 valuable**

**内容**:
- `9.956.csv` (464 KB) — **LB 9.956 submission** (= 我々 9.957 と same band!)
- `10.142.csv` (464 KB) — LB 10.142
- `11.284.csv` (424 KB) — LB 11.284
- `11.338.csv` (463 KB) — LB 11.338
- author F.A.Nina = rank 147/779, **LB 10.252** (= 個人 LB は 10.25 帯、 ただし 4 sub を ascending で公開)

**判定 = ★ ensemble blend 候補**:
- ❌ akshankrithick / buchananliang / alfaxa とは違って **9.956.csv が我々 9.957 と same band**
- ✅ **paradigm 1 (= submission only) だが 数理本質 OK**:
  * 我々 exp008 v2 (= karnakbaev + 自前 4 base + 案 D Kalman + Edge S)
  * nina 9.956 (= 別 path、 詳細不明だが LB 同帯)
  * = 2 解法の **error 分布が独立** ならば ensemble で MSE 半減効果 (= classical Bagging theorem)
- 期待 LB lift: **-0.05 〜 -0.3 ft** (= rank average of independent errors)
- これは「他の人がやってないが理論で優位」 (= 優勝本質性 §11.1 ✅) = ensemble theorem に基づく数理本質

**simple blend sketch**:
```python
ours = pd.read_csv("submissions/exp008_v2.csv")  # LB 9.957
nina = pd.read_csv(".work/source_audit/nina-03/9.956.csv")  # LB 9.956
merged = ours.merge(nina, on="id", suffixes=("_ours", "_nina"))
# simple average
blend = (merged["tvt_ours"] + merged["tvt_nina"]) / 2
# rank average (= more robust for non-aligned distributions)
rank_blend = (merged["tvt_ours"].rank() + merged["tvt_nina"].rank()) / 2
```

これは **1 submit quota で試せる軽量 try** = LB 0.05-0.3 ft lift 可能性で submit 価値高い。 ただし「他人 submission の blend」 paradigm = 優勝路として弱い (= self-compile では無い) → **本タスク完了後の安全 sub の 1 候補** に整理。

**注意**:
- nina の 4 sub (= 9.956 + 10.142 + 11.284 + 11.338) の **submission date がすべて 5/11 06:19** = 同時公開 (= 同 dataset 内に複数 LB 帯の sub を投下、 おそらく ensembling 用素材として公開した意図)
- 9.956 < 10.142 < 11.284 < 11.338 = 改善 history (= 同 author の自前 LB progression、 model architecture 進化の trace)
- 4 sub を **rank-based ensemble** で blend すれば single-author の variance reduction (= 我々の lift より limited、 nina 自身が既に best 9.956 を出してる)

→ **9.956.csv のみを我々と blend** が最適 path (= other 3 sub は弱いので avoid)

**diversity 実測 (= 2026-05-11)**:

我々 exp008 v2 (= LB 9.957) と nina 9.956 を per-row 比較:

| metric | 値 |
|---|---:|
| **correlation** | **0.999923** |
| abs diff mean | 2.89 ft |
| abs diff p50 | 1.71 ft |
| abs diff p90 | 7.81 ft |
| abs diff max | 13.41 ft |

= **correlation 0.9999 (ほぼ完全 correlated)**。 ensemble variance reduction:
$$\sigma_{\text{blend}}^2 = \frac{\sigma_1^2 + \sigma_2^2 + 2\rho \sigma_1 \sigma_2}{4}$$
ρ = 0.9999、 σ_1 ≈ σ_2 とすれば σ_blend² ≈ σ² (= **benefit ほぼゼロ**)。

**修正 判定**:
- ❌ **採用不要** = correlation 0.9999 で blend benefit limited
- ただし **重要 diagnostic**:
  * 「LB 9.95-9.96 帯の 2 解法は ほぼ同 prediction」 = この帯は **local optimum neighborhood**
  * 真の lift には **architecture change** (= NN sequence A / MoE D / 別 paradigm E) が必須
  * = winning path A/D の direction を data evidence で支持

**整理**:
- abs diff mean 2.89 ft は **Top10 vs 我々の delta (= 2.83 ft) とほぼ同帯**
- 同 LB band 解法間の delta は ROGII の hidden ground truth から見た「予測ばらつき」 の lower bound に近い (= 9.95 帯は近似 best、 個別 row では 2-3 ft 揺れる)
- = **真の LB 8.x 帯到達には per-row error 構造の根本的変更** (= MoE for minority cluster) が必要

---

### 1.3 `alfaxadeyembe/rogii-lgbm-model-artifacts` (1.2 MB、 5/9 公開)

**内容**:
- `lightgbm_booster.txt` (3.1 MB) — trained LightGBM
- `feature_columns.json` (2 KB) — 100+ feature names
- `ensemble_weights.json` — `{"base_b0_last": 0.109, "lightgbm_prediction": 0.891, "oof_rmse": 14.959}`
- `rogii_infer_runtime.py` (14.7 KB) — **inference source code** (= 数理本質読める)

**author LB**:
- Alfaxad rank 224/779, **LB 10.561** (= 我々 -0.6 ft 悪)
- **OOF RMSE = 14.96 ft** → CV-LB gap = **-4.4 ft** (= 重要 evidence)

**architecture**:
- Blend = `0.11 × b_well_last_value + 0.89 × LGB(features)`
- 100+ features:
  - 基本: MD, X, Y, Z, GR (+ d_, step_, azi sin/cos, inclination, curvature)
  - rolling **5 scales × 4 stats** = gr_roll_{mean,std,min,max}_{5,11,25,51,101} + z_roll_slope_*
  - prefix features: prefix_tvt_{min/max/range/std/rows}, prefix_slope_md_{25,50,100,200,500}
  - last_known: last_tvt, last_md, last_xyz, last_gr

**判定**:
- ❌ **採用不要** = LB 10.56 < 我々 9.957、 +0.6 ft 下位
- ✅ **valuable artifacts**:
  - `rogii_infer_runtime.py` = 解法を読める source = paradigm 2 流用候補 (= 「我々が見逃している simple base」 発見可能)
  - **OOF 14.96 ft → LB 10.56 = CV-LB gap -4.4 ft** = ROGII の general CV-LB diff evidence (= 我々 exp008 v2 の CV-LB gap と比較する baseline)
  - 「Blend = 0.11 × b_well_last + 0.89 × LGB」 = **b_well last value alone が LB の 11% 寄与する** simple baseline (= 我々 9-base に簡単な inject 可能性)

**ablation candidate**: 我々の base 群 + `b_well_last` simple feature を 10 番目 base として追加 → blend、 0.1-0.3 ft lift 可能性 (= 数理本質弱いが軽量 try)

---

### 1.2 `buchananliang/rogii-karnak-top2-public-artefacts` (6.6 MB、 5/8 公開)

**内容**:
- `final_lgb.txt` (8.6 MB) — trained LightGBM model
- `final_xgb.json` (8.6 MB) — trained XGBoost model
- `final_cb.cbm` (1.95 MB) — trained CatBoost model
- `features.json` (1.7 KB) — 101 feature names list
- `ensemble_weights.json` (74 bytes) — `{"lgb": 0.0, "xgb": 0.473, "cb": 0.527}`

**features 構成 (= 101 cols)**:
- 基本: last_known_tvt, known/hidden_len, frac_hidden, md, z, dx/dy/dz, dist_xy/xyz
- GR: gr_roll{3,5,11,21,51,151}, gr_std{5,21}, gr_min/max{5,21}, gr_range{5,21}, gr_grad, gr_lag/lead{1,5,10}
- typewell: prefix_tw_rmse/mae/bias, tw_gr_at_last, tw_gr_std_local, tw_tvt_range
- **Beam (Top2 独自)**: beam_{tight,cons,loose,vloose}_delta + beam_mean/std/spread/gap
- **PF (Top2 独自)**: pf_delta, pf_std, pf_beam_cons_diff, pf_beam_loose_diff
- **ANCC (Top2 独自)**: ancc_delta, ancc_std, ancc_beam_cons_diff, ancc_pf_diff
- **tw_diff at 13 offsets**: -120/-80/-40/-20/-10/-5/0/+5/+10/+20/+40/+80/+120

**判定**:
- ✓ **paradigm 2 流用候補** (= trained XGB + CB artifacts = 数理本質を理解可能、 「他の人がやってないが理論で優位」 ではないが diversity ある base 追加候補)
- ❌ LGB は ensemble weight = 0 で drop されている (= Top2 解法でも LGB 単独は劣化 contribution = 同 paradigm の LGB×3 を使う karnakbaev とは別の learning signal の可能性)
- ✓ **features.json は我々の自前 4 base 165 features の subset baseline として valuable** (= 我々が「追加で増やしている 64 features」 が本質的に lift しているか ablation 可能)

**統合 sketch**:
```python
# kaggle_kernels に load
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

top2_xgb = xgb.Booster()
top2_xgb.load_model("/kaggle/input/rogii-karnak-top2-public-artefacts/final_xgb.json")
top2_cb = cb.CatBoostRegressor().load_model("/kaggle/input/rogii-karnak-top2-public-artefacts/final_cb.cbm")

# 我々の features (= 165 cols) から top2 features (= 101 cols) を抽出して predict
top2_xgb_pred = top2_xgb.predict(X[top2_features])
top2_cb_pred = top2_cb.predict(X[top2_features])
top2_ensemble = 0.473 * top2_xgb_pred + 0.527 * top2_cb_pred

# 既存 9-base Ridge meta に 10 番目 base として追加
Sx = np.column_stack([... 9 base ..., top2_ensemble])
```

期待 LB lift: **-0.05 〜 -0.2 ft** (= 同 paradigm GBM diversity だが Top2 LB 値次第)。 ただし Top2 解法 (= LB ?) と我々 (= LB 9.957) が同帯なら lift 限定的、 Top2 LB 9.0-9.5 帯なら有意な lift 可能。

→ Top2 LB 値の確認が前提。 kaggle competition で「buchananliang」 author の submission score を leaderboard で探す必要。

**LB 値確認 (= 2026-05-11 実測)**:

| 項目 | 値 |
|---|---:|
| **Buchan Liang (= dataset author)** | **rank 205/779, LB 10.463** |
| 我々 Reexel | rank 53/779, LB 9.957 |
| 差 | **+0.506 ft (= Top2 author の方が悪い)** |

**修正 判定** (= 数理本質):
- dataset 名「karnak-top2」 は誤解を招く。 karnakbaev (= ROGII Top2 解法 LB 10.78、 別 dataset で公開) ではなく、 **Buchan Liang 自身の LB 10.463 解法 trained artifacts**
- 我々 9.957 より +0.5 ft 悪い model を加えても LB lift は期待薄 (= diminishing returns、 同 paradigm GBM の subset version)
- ❌ **採用不要** = 優勝本質性 §11.1 で「他人解法、 しかも下位」 = 数理本質弱い

ただし **features.json (= 101 features list)** は valuable diagnostic:
- 我々自前 4 base 165 features の **subset baseline** 確認に有用
- 「我々が増やした 64 features (= 165 - 101) は本当に LB lift しているか」 を ablation 可能
- これは **feature ablation 用** に保持 (= base model artifacts は使わない)

---

## 2. 未 audit dataset (= 次タスクで個別 audit)

| dataset | size | last update | 推測内容 | audit priority |
|---|---|---|---|---|
| `thbdh5765/rogii-v4-aeroridge-train-cache` | 1.57 GB | 5/11 04:53 | AeroRidge 系 train feature cache (= 別 base 候補) | **高** (= 5/11 最新 + AeroRidge は別 paradigm のヒント) |
| `buchananliang/rogii-karnak-top2-public-artefacts` | 6.6 MB | 5/8 04:07 | Top2 (= karnakbaev 系) 解法 artifacts | **中** (= 既存 karnakbaev dataset と差分確認、 重複なら不要) |
| `alfaxadeyembe/rogii-lgbm-model-artifacts` | 1.2 MB | 5/9 12:49 | LGBM 訓練済 artifacts | **中** (= LGBM 系 pretrained 流用候補) |
| `nina2025/rogii-03` | 507 KB | 5/11 06:19 | 小規模 utility (= 詳細不明) | 低 |
| `nina2025/rogii-07` | 1.1 MB | 5/7 22:26 | 別 version | 低 |
| `iathar/rogii-wellbore-models` | 2.2 MB | 5/7 12:46 | wellbore models | 低 |
| `geeknik/rogii-blend-public` | 71 KB | 5/8 01:36 | blend predictions | 低 (= 同様 submission-only の可能性) |
| `enisteper1/rogii-train-dataset` | 937 MB | 5/9 07:30 | 大容量、 詳細不明 | 中 |
| `innerf1re/rogii-blend-srcs-public` | 430 KB | 5/10 08:54 | blend sources | 中 |

---

## 3. audit プロセス (= 次タスク proceduralization)

各 dataset に対し:
1. `kaggle datasets download <slug> -p .work/source_audit/<short>/` で download
2. `unzip` + `ls -la` + 各 file の type 確認 (= source code / parquet / submission / model artifacts)
3. file 内容を 5-10 行 head で確認
4. 数理本質 判定:
   - source code あり = paradigm 1 でも valuable (= 解法 reverse engineer 可能)
   - submission only = transduction、 diagnostic 用途のみ
   - parquet (= train feature cache) = base model 改善の素材
   - model artifacts = 直接 blend に使用可能、 ただし licence 確認必須
5. 「優勝本質性」 criterion で採用判定
6. 本 doc § 1 に audit entry 追加

---

## 4. 「優勝本質性」 で篩った後の戦略

優勝本質性 §11.1 の binary table:

| ❌ 採るな | ✅ 採れ |
|---|---|
| "他の人がやってるから" | "他の人がやってないが理論で優位" |
| "公開 source 集約で +X 取れるから" | "self-compiled で独自 lift" |
| "コードが書きやすいから" | "数理本質で問題を解く" |

= **submission-only dataset は不採用 (= transduction paradigm)**、 source code / feature cache / model artifacts は **数理本質を理解できる場合のみ採用**。

逆に重要:
- **diagnostic data として常に valuable** (= Top10 submission との per-row diff は「自分の error 分布」 を data 由来で確認可能)
- これは「自前 self-compile」 の方向性決定の入力

---

## 5. 関連

- `docs/dev/2026-05-11-cv-lb-correlation.md` § 5.4 = 元 host dataset list
- `~/projects/kaggle/CLAUDE.md` § 11 = 優勝本質性 criterion
- `~/projects/kaggle/CLAUDE.md` § 4 = 4 paradigm 強制ルール (= paradigm 1 単独 では gold 不可)
- 別タスク draft: `docs/research/2026-05-11-karnakbaev-refit-pipeline-draft.dense.md` (= paradigm 2 拡張)
- 別タスク draft: `docs/research/2026-05-11-winning-path-A-nn-sketch.dense.md` (= paradigm 2 sequence model)
