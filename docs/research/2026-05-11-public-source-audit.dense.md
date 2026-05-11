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
