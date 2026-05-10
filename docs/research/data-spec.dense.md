# データ仕様 (確定情報) — 2026-05-10 bootstrap 段階

> このファイルは Phase 0 (bootstrap) で実データを展開し、`000d7d20` 1 wells のみで確認した内容。**Phase 1 で全 wells 統計を取り直して書き換える**こと。
> 出典: `data/raw/{train,test}/*.csv` を直接読んだ結果 (出力は `notebooks/00_eda.ipynb` に再現)。

## 1. ファイル構成

- `data/raw/`
  - `sample_submission.csv` — 14151 rows, columns: `id, tvt`. id format: `{8charhash}_{row_index}`
  - `train/` — `{HASH}__horizontal_well.csv`, `{HASH}__typewell.csv`, `{HASH}.png` の 3 ファイル/well
  - `test/` — 同上構成。**展開時点では sample 3 wells のみ** (`000d7d20`, `00bbac68`, `00e12e8b`)。本番リーダーボード時に約 200 wells に置き換わる
  - `AI_wellbore_geology_prediction_task_en.pptx` — 主催者提供の task 説明スライド (要レビュー)

## 2. wells 数 (確定)

- **train wells: 773** (`ls data/raw/train/*__horizontal_well.csv | wc -l`)
- **test wells (sample): 3** — 本番では約 200 wells (Kaggle Overview より)

## 3. horizontal_well の columns

### train 側 (13 columns)

```
MD, X, Y, Z, ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA, TVT, GR, TVT_input
```

### test 側 (6 columns) — train から **`ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA, TVT` の 7 列が削除**

```
MD, X, Y, Z, GR, TVT_input
```

> **重要含意**: `ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA` (formation depth feature) は **train only**。これらに依存する model は test 推論時に再現できないため、特徴量として直接使うことは禁止。ただし train 時の **auxiliary supervision target** としては使える (multi-task で formation depth も予測させ、main task の representation を richer にする)。

## 4. typewell の columns

### train 側 (3 columns)

```
TVT, GR, Geology
```

`Geology` 例: 空欄 (NaN) が混在。確認: `head -3 data/raw/train/000d7d20__typewell.csv` ⇒ 最初 2 行は Geology=空。

### test 側 (2 columns) — `Geology` 列が削除

```
TVT, GR
```

> **重要含意**: typewell は **test 側でも与えられる** ⇒ 案 B (Cross-Attention to typewell) と案 C (DTW well-tie) の前提は成立。
> ただし **`Geology` label は test では無い** ⇒ Plan の案 B `Multi-task: TVT + ΔTVT + Geology label aux` の Geology 部は train 時の auxiliary supervision には使えても、test 推論時に必要な入力にはできない。設計修正必要。

## 5. 評価 zone (1 well のみ確認、`000d7d20`)

- **horizontal_well rows**: 5278 (MD 11467.0 〜 16744.0, step = 1.0 ft 確認)
- **TVT_input visible**: 1442 rows (MD 11467 〜 12908)
- **TVT_input NaN (= eval zone)**: 3836 rows (MD 12909 〜 16744) ← lateral 全長の **後半 73%**
- 評価 zone は **lateral の後半連続 hidden** という構造
- sample_submission の 3 wells も同様: 3836 + 6014 + 4301 = 14151 rows = 各 wells の hidden zone を全部足したもの

> **重要含意**:
> 1. CV 設計: 訓練時に visible 区間で学習 → hidden 区間 (= MD 後半) を予測する pattern を再現する **leave-one-well-out + 後半 mask CV** を組む。GroupKFold by well_id 単独では不十分。well 内の visible/hidden 分割が test 構造に整合する必要あり。
> 2. 案 C (Plane fit + residual ML) の Plane fit は **visible 区間 = lateral 前半 27%** で計算 → 後半に外挿する形になる。前半が短いほど extrapolation 誤差が累積する。
> 3. TVT_input は **後半全部 NaN**。可視化されているのは前半だけ。これに過剰依存すると hidden zone で破綻する。

## 6. TVT 値域 (1 well のみ確認、`000d7d20`)

- TVT 最小値: 11236.02 (MD 11467 時点)
- TVT は MD と同じスケール (10000ft オーダー)。MD と TVT のスケール感を踏まえると **公開 LB baseline RMSE 12.602 ft は TVT 値の 0.1% 誤差程度**

> **重要含意**: Plan で「LB 5-8 帯を最終目標」と書いたが、絶対値 5-8 ft というのは TVT 11000ft オーダーに対し相対誤差 0.05-0.07% の世界。改善の余地は数 ft 単位の精度勝負。差別化は微小な精度の積み上げが効く。Phase 1 で全 wells の TVT 値域分布を確認したうえで再校正する。

## 7. id format

- `{WELLNAME}_{row_index}`
- 例: `000d7d20_1442` → well `000d7d20`, row index 1442 (0-origin で hidden zone の最初の行に対応 = MD 12909.0)

## 8. Phase 1 EDA で必ず確認する項目

Bootstrap 段階で 1 wells (`000d7d20`) だけ見たので、全 train wells / 全 test sample wells で以下を統計的に検証する:

1. ✅ MD step は **常に 1.0 ft** か (train/test 全 wells で一定か検証)
2. ✅ 評価 zone は **常に lateral 後半連続** か (途中に visible が挟まる pattern が無いか)
3. visible 区間の長さ分布 (well ごとの最初の MD カウント比率)
4. TVT 値域分布 (TVT min / max / range / std を well ごとに集計)
5. GR スケール統一性 (well ごとの mean/std/min/max を箱ひげで)
6. typewell の TVT 範囲が horizontal の TVT 範囲をカバーしているか
7. Geology label の出現頻度 (train typewell のみ)
8. trajectory の dip/azimuth/dogleg 分布

## 9. Plan へのフィードバック (修正候補)

Phase 0 で得られた事実から Plan を以下のように修正したい:

| 修正点 | 元 plan | 新 plan |
|---|---|---|
| 案 B の aux task | "Multi-task: TVT + ΔTVT + Geology label aux" | Geology label は **train typewell からのみ pretext / aux supervision**。test 推論時には不要 (= forward pass で Geology head を捨てる) |
| 案 C で使う visible 区間 | "visible TVT_input の最小二乗 plane fit" | visible = lateral 前半 27% のみという前提を明示。**plane fit の信頼区間** を考慮した extrapolation 誤差を Phase 1 で実測 |
| CV 戦略 | "GroupKFold by well_id, 10-fold" | well_id GroupKFold + **well 内の後半 mask 模倣** の組み合わせに変更。hold-out well の visible 部だけを学習し、hidden 部で評価 |
| Risk 1 (データ仕様の誤読) | 想定していた | `ANCC/ASTNU/...` が test で無いこと、Geology が test typewell に無いことは Phase 0 で確認済み。Risk 1 をクローズ |
| 期待 LB 校正 | "LB 5-8 帯" | TVT 値域 (10000ft オーダー) を踏まえて Phase 1 で再校正 |

> Plan 本体 (`docs/strategy/winning-strategy.dense.md` および `~/.claude/plans/rogii-wellbore-geology-modular-shannon.md`) を Phase 1 完了時に上記反映で更新する。
