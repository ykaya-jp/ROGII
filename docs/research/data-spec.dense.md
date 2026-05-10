# データ仕様 — Phase 1 EDA 確定版 (2026-05-10)

> Phase 1 EDA で全 776 wells (773 train + 3 test sample) を集計した結果。
> スクリプト: `src/rogii/eda.py` (`uv run python -m rogii.eda` で再現可能)
> 生成物: `outputs/eda/aggregate-summary.json`, `outputs/eda/per-well-stats.parquet` (776 rows)

## 1. 形状の確定

### MD step

- **全 776 wells で MD step = 1.0 ft 完全統一** (`MD_step_modal_unique = [(1.0, 773)]` for train, `[(1.0, 3)]` for test)
- 含意: **regular grid sequence model が成立**。resampling 不要

### well 長 (`n_rows` = lateral 区間 ft 数)

| split | min | p50 | p95 | max |
|---|---|---|---|---|
| train | 2058 | **6576** | 8614 | 12141 |
| test  | 5278 | 6384 | 7441 | 7559 |

- 含意: 案 B Sequence Transformer の patch サイズは **128 ft 〜 256 ft** が妥当 (= p50 6576 を 25-50 patch に分割)

### 評価 zone 構造 ★ 全 wells 統一

- **`wells_with_trailing_hidden_only`: 773 / 773 (100%)**
- **`wells_with_visible_runs_gt_1`: 0**, **`wells_with_hidden_runs_gt_1`: 0**
- ⇒ 全 wells で **「visible 1 ブロック → hidden 1 ブロック (末端まで)」** の単純パターン
- 含意:
  - CV 設計は「**well 内の終盤を hidden 模倣**」で OK (各 well で末端 ~75% を mask して訓練)
  - GroupKFold by well_id だけでは不十分。**well 内マスク** も併用
  - **更に typewell hash で stratify** が必要 (= 同一 typewell を共有する 13 group / 34 wells、§10 参照、`data/processed/typewell_groups.parquet` の `group_id` を sklearn `GroupKFold(groups=group_id)` に渡す)

### visible 比率

| split | min | p25 | p50 | p75 | max |
|---|---|---|---|---|---|
| train | 0.125 | 0.225 | **0.260** | 0.300 | 0.802 |
| test  | 0.204 | 0.239 | 0.273 | 0.300 | 0.326 |

- 含意:
  - **約 1/4 (visible) で 3/4 (hidden) を予測** する超 extrapolation 課題
  - 案 C plane fit は **visible 25% (= lateral 前半)** で fit → 残り 75% に外挿。前半が短いほど extrapolation 誤差が累積
  - 案 B Transformer も visible 部分が短いので **typewell との cross-attention** が決め手

## 2. TVT 値域 (train, target 列)

### 2.0 TVT 公式定義 (2026-05-10 更新、出典: discussion 698282 ROGII 公式回答 + PowerPoint Slide 5-7)

- **TVT = "geology of the wellbore"** = **virtual / imaginary reference line までの vertical distance** (PowerPoint Slide 3 + discussion 698282 msg4)
- **TVT=0 が ground level かは未確定** (固定ではなく地質構造に追随する)
- **lateral と typewell の TVT 軸は対応する** (lateral-typewell pair ごと)。typewell は同一 reference line を **vertical 視点** で観測したもの
- **物理モデル**: `TVT(s) = -Z(s) + b_well + ε(s)` の `b_well` は **well 表面 (= 地表点) における virtual reference line の offset**。`first-principles.dense.md` §1.4 の `b_well` と同一
- **PowerPoint Slide 6 の核心知見**: 同じ GR signature が typewell と match しても **TVT 増加方向と減少方向の 2 通り** がある (drillhead が horizon を上下どちら方向に切るかで反対、discussion 697431 msg8 の DTW reverse index と整合)
- **Slide 7**: GR signature が constant な lateral 区間では TVT も constant (= 同一 layer 内 lateral 進行)
- **含意**:
  - `b_well` は per-well 1 定数で扱う近似が物理的に正当 (`first-principles.dense.md` §5)
  - pure xcorr alignment は **方向情報を別途 prior** として与えないと曖昧 (visible 末端 dTVT 符号を local prior にする提案 = `host-resources.dense.md` H1)
  - PatrickAIForFun (discussion 698282 msg2) の独立検証 `ANCC - Z = TVT + offset_per_well` は本物理モデルの帰結

### 2.1 値域

- TVT min: **9245.19** ft
- TVT max: **12893.89** ft
- per-well TVT range: p50 **758 ft**, p95 997 ft (1 well 内で TVT は ~800 ft 動く)
- visible 区間で **TVT_input == TVT (全 773 wells で True 検証済み)** ⇒ TVT_input は visible 部の正解 TVT そのもの

> 含意: 公開 LB 12.602 = TVT 平均 11000 ft に対する **相対誤差 0.114%**。1 位は更に小さい誤差勝負。Plan の「LB 5-8 帯目標」は TVT スケールから見て妥当 (相対誤差 0.05-0.07%)。

## 3. typewell の解像度差 ★

| typewell TVT step | wells | 比率 |
|---|---|---|
| **0.5 ft** | 653 | **84.5%** |
| 0.2 ft | 91 | 11.8% |
| 1.0 ft | 21 | 2.7% |
| 0.1 ft | 8 | 1.0% |

- horizontal は 1.0 ft step、typewell の 84.5% は 0.5 ft step ⇒ **typewell 側を 1.0 ft に resample (downsample) が必要**
- typewell `n_rows`: p50 1874 → 約 940 ft の縦深 (0.5 step 仮定)、最大 10043 (= 5000 ft の長い縦穴も)

## 4. typewell カバレッジ ★

- **typewell が horizontal の TVT 範囲を完全カバー: 760 / 773 (98.3%)**
- 13 wells (1.7%) で horizontal が typewell の TVT 範囲外側に出る ⇒ extrapolation edge case
- 含意: 案 B/C は edge case の 1.7% で精度低下。**案 A (typewell 非依存) との blend** で底上げ

## 5. typewell の Geology label 分布 (train のみ。test 側 typewell に Geology 列無し)

| 指標 | min | p50 | max |
|---|---|---|---|
| `typewell_geology_unique_count` (label 種類) | 4 | **6** | 21 |
| `typewell_geology_n_labeled` (label 付き行数) | 434 | 1182 | 6339 |

- 含意:
  - 案 B aux task の Geology classification は **6 layer median** で扱う
  - test 側で Geology 列が無いので、**train 時の auxiliary supervision 専用**。test 推論時は forward pass で Geology head を捨てる
  - 21 layer wells (max) は最大 class 数として preserve するか、希少 class を merge するか別途検討

## 6. GR (Gamma Ray) 正規化 ★

- GR_mean range (well 間): **37 〜 130** ⇒ **3.5 倍の差**
- GR_std (median): 17.3
- 含意:
  - well 間で **必ず正規化** (z-score / robust scaler)
  - well ごとの absolute GR 値比較は無意味
  - typewell GR と horizontal GR の比較も well 内で正規化してからやる

## 7. trajectory (vertical extent)

- Z_range (well 内): p50 **787 ft**, p95 1069 ft
- horizontal でも Z 方向に 800 ft 動く ⇒ 完全水平ではなく deviation あり
- 含意: trajectory derivative (dip, azimuth, dogleg) は十分 informative な特徴量になる

## 8. column 定義 (train / test 差分)

### `__horizontal_well.csv`

| 列 | train | test | 用途 |
|---|---|---|---|
| `MD` | ✓ | ✓ | 測長深度 |
| `X` `Y` `Z` | ✓ | ✓ | 3D 座標 |
| `GR` | ✓ | ✓ | Gamma Ray |
| `TVT_input` | ✓ | ✓ | 部分マスク TVT (visible 部のみ非 NaN) |
| `TVT` | ✓ | ✗ | **target (train only)** |
| `ANCC` `ASTNU` `ASTNL` `EGFDU` `EGFDL` `BUDA` | ✓ | ✗ | 各地層の predicted depth (**train only**) — 直接特徴量化禁止、ただし aux supervision には使える |

### `__typewell.csv`

| 列 | train | test | 用途 |
|---|---|---|---|
| `TVT` | ✓ | ✓ | 縦穴の depth index |
| `GR` | ✓ | ✓ | 縦穴 GR signature |
| `Geology` | ✓ | ✗ | layer label (**train only**) |

## 9. id format

- `{WELLNAME}_{row_index}` (8 文字 hash + underscore + 整数)
- 例: `000d7d20_1442` → well `000d7d20`, row index 1442 (= hidden zone 内の最初の行 = MD 12909.0)
- sample_submission の id は **hidden zone 内の行のみ**

## 10. Plan への反映 (確定版、Plan を別途更新する)

| 修正点 | 元 plan の表現 | 新 plan の表現 |
|---|---|---|
| MD step | 「1 ft step 想定」 | **全 wells で 1.0 ft 完全統一** 確定 |
| 評価 zone 構造 | 「中央? 末端?」 | **全 wells trailing hidden only** 確定 |
| visible 比率 | 不明 | **median 26%** 確定 (= 残り 74% を予測) |
| typewell 解像度 | 不明 | **0.5 ft step が 84.5%** ⇒ resample to 1.0 ft 必須 |
| TVT 値域 | 不明 | **9245-12894 ft** ⇒ LB 5-8 ft 目標は相対誤差 0.05% 妥当 |
| GR 正規化 | 言及無し | **必須** (well 間で 3.5 倍差) |
| typewell カバレッジ | 不明 | **98.3% カバー、1.7% (13 wells) は edge case** |
| Geology 列 | aux | train 時のみ aux supervision、6 layer median |
| `ANCC/ASTNU/EGFDU/EGFDL/BUDA` | train only と仮定 | **確定**: test では無し ⇒ 直接特徴禁止、train 時 aux supervision としては使用可 |
| CV 戦略 | GroupKFold | **GroupKFold by `group_id` (= typewell hash group, §10b) + 各 well 内で末端 visible_ratio (= median 74%) を hidden 模倣** |

## 10b. typewell duplicate group (2026-05-10 追加)

> **発火元**: discussion topic 698449 (ROGII 公式回答 v=2、2026-05-10) 「project 内 typewells の一部は **pseudo-typewell** (隣接 lateral から作った interpretation)」
> **検証**: 全 773 train typewell を `md5(TVT, GR, Geology bytes, NaN-normalised, 4 桁丸め)` で fingerprint → group 化
> **再現スクリプト**: `notebooks/_typewell_groups_build.py` (`uv run python notebooks/_typewell_groups_build.py`)
> **生成物**: `data/processed/typewell_groups.parquet` (gitignored、773 rows × 9 cols)

### 10b.1 統計

| 指標 | 値 |
|---|---|
| 全 train wells | 773 |
| unique typewell hash | 752 |
| duplicate group 数 | **13** (= discussion 698449 の手動発見と完全一致 13/13) |
| duplicate group 内の wells 合計 | 34 / 773 = **4.4%** |

### 10b.2 group_size 分布

| group_size | group 数 | wells 合計 |
|---|---|---|
| 1 (unique) | 739 | 739 |
| 2 | 12 | 24 |
| **10** | 1 | 10 |

最大 group (size=10) は **TVT 11782.62-12285.10 ft, n_rows=1006** の typewell を `02e7fe5a, 10b89021, 3417285d, 6ae68655, 7993a768, bc4381e2, ecdab904, f021b650, f49fdea3, f88ddb26` の 10 lateral wells が共有。

### 10b.3 CV 設計への含意

- **leakage source**: well_id GroupKFold で fold 分割すると、同一 typewell の lateral 群が train/val 両側に出現 → val の「typewell 答え」が train から漏れる
- **正しい運用**: `data/processed/typewell_groups.parquet` の `group_id` 列を `sklearn.model_selection.GroupKFold(groups=group_id)` に渡す
  ```python
  import pandas as pd
  from sklearn.model_selection import GroupKFold
  tg = pd.read_parquet("data/processed/typewell_groups.parquet")
  groups = df_train.merge(tg[["well_id", "group_id"]], on="well_id")["group_id"].values
  for tr, va in GroupKFold(n_splits=5).split(df_train, groups=groups):
      ...
  ```
- **loss weight 候補**: `is_duplicate=True` (= pseudo-typewell の可能性が高い 34 wells) は **ground truth `manualTVT` 自体が他 lateral からの interpretation** で誤差が乗っている (PowerPoint Slide 14 = `manualTVT` は人手 interpretation) → loss を 0.7-0.8x に縮小する変種を試す価値あり (= `host-resources.dense.md` H6 と整合)

### 10b.4 column 仕様 (`typewell_groups.parquet`, v2 = 14 cols)

| 列 | 型 | 説明 |
|---|---|---|
| `well_id` | str | horizontal well ID (8 chars) |
| `typewell_hash` | str | md5 of (TVT, GR, Geology bytes), 32 chars |
| `group_id` | str | 同 hash 内で min(well_id) を group 代表 ID として使用 |
| `group_size` | int | 当該 hash を共有する well 数 (1, 2, または 10) |
| `is_duplicate` | bool | `group_size > 1` |
| `n_rows`, `tvt_min`, `tvt_max`, `has_geology` | misc | typewell content sanity 用 |
| `azimuth_visible_pca_deg` | float | visible-zone XY trajectory の SVD 第一主軸 azimuth (degrees clockwise from north) |
| `azimuth_full_deg` | float | full lateral end-minus-start direction の azimuth (degrees) |
| `straightness_visible` | float | visible XY の `S2/S1`、低いほど直線的 (p50=0.015, p95=0.053) |
| `azimuth_circ_std_deg` | float | 同 group 内の `azimuth_full_deg` の **circular std** |
| **`is_pseudo_likely`** | bool | `is_duplicate AND azimuth_circ_std_deg >= 60deg` ⇒ 反対方向の lateral が同 typewell を共有 = pseudo-typewell の強い疑い (10 wells / 5 groups: `071d7b45`, `7b38844c`, `b977be4a`, `cd7f1687`, `f321a31c`) |

### 10b.5 loss weight 自動調整 (winning strategy 接続)

- `is_pseudo_likely=True` の 10 wells は ROGII `manualTVT` (= human interpretation) が **過去 lateral から伝播した interpretation** で誤差が乗っている可能性が高い
- 推奨 variant: 該当 wells の loss weight を **0.5-0.7x** に縮小、CV/LB の感度を計測 → 効果あれば baseline に取り込み
- 詳細出典: `docs/research/host-resources.dense.md` §3.5.3, `docs/research/independent-edges.dense.md` 案 M (visible-as-typewell との合成可能性)

## 11. 詳細データ

- per-well 統計の全カラム: `outputs/eda/per-well-stats.parquet` (776 rows × 35 cols)
- 横断統計: `outputs/eda/aggregate-summary.json`
- スクリプト: `src/rogii/eda.py`
