# ROGII Host 公式 pptx 要約 (2026-05-11 取得)

> 出典: `data/raw/AI_wellbore_geology_prediction_task_en.pptx` (28.8MB、14 slides、host が `kaggle competitions files rogii-wellbore-geology-prediction` で公開)
> 最終更新: 2026-05-01 18:40 UTC (= コンペ開始日)
> 抽出方法: `zipfile` で xml dump + 正規表現で plain text 化

## 1. 既知情報 (= 戦略 doc / data-spec と一致)

- **goal**: horizontal well の lateral 区間 (PS = Prediction Start point 以降) の TVT を回帰予測
- **入力**: horizontal well の `XYZ`, `MD`, `GR` (NaN 含む), `TVT_input` (PS 前のみ visible)、formation top depth (train only)、typewell の `TVT`, `GR`, `Geology label`
- **評価**: $\text{dTVT} = \text{manualTVT} - \text{predictedTVT}$ の RMSE (slide 14)
- **単位**: feet
- **MD 1 ft step** (戦略 doc / data-spec で確定済)

## 2. 新発見 1 — slide 9: **horizontal visible GR > typewell GR (resolution)** ★★★

### 2.1 引用 (slide 9 から)

> "GR from the vertical well has better resolution than GR from the typewell"
> "The green GR correlates better with the red GR than with the typewell GR (black)"
> "**It may be better to use GR data from the horizontal well before the PS point, combined with deeper TVT data, to correlate the rest of the lateral**"

### 2.2 解釈

- **horizontal well の PS 前 GR** = high resolution (1 ft step)
- **typewell GR** = lower resolution (0.5-1.0 ft step、戦略 doc 既知)
- **host が自前 horizontal の visible GR を reference に使え** と明示している
- 公開 Top 3/N は typewell-horizontal alignment を Beam Search / Self-NCC で取るが、**自前 horizontal の PS 前 GR との alignment** はあまり強調されてない paradigm

### 2.3 戦略への impact

- これは `independent-edges.dense.md` 系列の独自 edge **M (visible-as-typewell)** に直結
- subagent G が `537fa9f docs: H-T1/T2/T6/T7 + edge M (visible-as-typewell)` で既に発見済 (host pptx を読んだ可能性高い)
- 実装 path:
  1. PS 前の horizontal GR を typewell GR の **追加 reference** として使う
  2. Beam Search / Self-NCC を **horizontal-self alignment** にも拡張 (= sliding window で PS 前 vs PS 後 の matching)
  3. typewell match と horizontal-self match の **conflict feature** (= disagreement zone は不確実) を GBM に渡す
  - 推定工数 1 日、期待 LB -0.3〜0.6 ft
- `pilkwang/rogii-eda-v4-same-matrix-super-stack` の PF-lite が「複数手法の意見合致度」を符号化 (`pilkwang-distill.dense.md`) しているが、典型的に typewell との agreement のみ。**horizontal-self との agreement** を加えれば paradigm が拡張される

## 3. 新発見 2 — slide 12-13: **offset well + dip continuity** ★★★

### 3.1 引用 (slide 12-13 から)

> "The azimuth of horizontal drilling affects the expected geology dip"
> "The geology of an offset well can help predict the geology of the current well"
> "**Geological dips behave similarly in neighboring wells**"
> "Dip Dip Flat Flat" (= 隣接 wells で dip パターンが共有)

### 3.2 解釈

- **offset well (= 近接 well)** が **dip continuity** を持つ
- 我々の計測 (`first-principles.dense.md` §1.1): drilling trajectory は X(s), Y(s), Z(s), MD(s) で表される空間曲線、dip 角度と well 位置で TVT が決まる
- **隣接 wells で dip が連続** = X/Y 距離 + drilling azimuth で**近接 well retrieval** すると、その well の TVT pattern を借りられる

### 3.3 戦略への impact

- **`independent-edges.dense.md` §9.2 の案 K (Graph Neural Network on Well-Pair Similarity)** に直結
- 公開 top の `FormationPlaneKNN` (X/Y centroid から K=10 で formation 別 plane fit) は **point-wise imputer** であって well-pair pattern retrieval ではない
- 我々の独自 edge: **near-neighbor well TVT pattern retrieval** (= K=5-10 wells を X/Y + azimuth で選び、それらの末端 TVT delta pattern を current well の予測に bias として加える)
- 実装 path:
  1. test well の X/Y centroid + azimuth (= drilling direction) で近接 train wells を K=10 retrieve
  2. 各 retrieved well の hidden zone TVT delta pattern (= last_known_TVT からの relative delta) を取り出す
  3. これらを **average + std** (= mean dTVT shape + uncertainty) として features 化
  4. GBM に渡す or Ridge meta-learner で blend
  - 推定工数 1.5 日、期待 LB -0.4〜0.8 ft (= near-neighbor retrieval は典型的に強い)

## 4. 新発見 3 — slide 6-8: GR signature matching の物理的根拠

### 4.1 引用 (slide 6-7 から)

> "GR projected on TVT track" + "GR signature matches Typewell GR"
> "TVT is increasing / decreasing / constant" (= 3 状態)
> "GR signature is constant" + "TVT is constant" (= visible 領域での dTVT=0 zone は GR も flat)

### 4.2 解釈

- GR と TVT は **stratigraphic correlation** で結ばれる (= 地層の鉛直構造が dip 角度と well 位置で TVT 値を決める)
- TVT 増減方向は GR signature の "走り方" で読める (= **direction-aware matching**)
- 公開 top の Beam Search / Self-NCC は 「GR signature の **相関値**」を取るが、TVT 増減方向の **sign awareness** は弱い

### 4.3 戦略への impact

- **direction-aware Beam / NCC**: 標準 Beam が `corr(GR_h, GR_v)` を最大化するなら、direction-aware は `corr(dGR_h/dt, dGR_v/dt)` (= 微分相関) を additionally check
- これは **slide 6-7 の "TVT is increasing while GR signature matches Typewell GR"** という physical principle を model に教える
- 実装 path: Beam Search の cost function に `lambda * |sign(dGR_h) - sign(dGR_v)|` の penalty を追加
- 推定工数 0.5 日、期待 LB -0.1〜0.3 ft

## 5. 統合: 新発見 3 件の独自 edge への寄与

| edge | 起源 | 推定工数 | 期待 LB | 既存案合成 |
|---|---|---|---|---|
| **M: visible-as-typewell** (slide 9) | 既存 `independent-edges.dense.md` + host pptx 公式 endorsement | 1 日 | -0.3〜0.6 | exp006 (= subagent G/I の流れ) で追加可能 |
| **N: offset well retrieval** (slide 12-13) | 既存 `independent-edges.dense.md` §9.2 案 K + host pptx 公式 endorsement | 1.5 日 | -0.4〜0.8 | exp007-008 で扱う、独立 paradigm |
| **O: direction-aware Beam/NCC** (slide 6-7) | 新規 (独自 edge 候補に追加すべき) | 0.5 日 | -0.1〜0.3 | 既存 Beam / Self-NCC を augment |

3 個合計工数 3 日、期待 LB -0.8〜-1.7 ft = **exp005 (= LB 10.x 想定) + 上記 = LB 9.0-9.5 帯射程**。

## 6. 戦略 doc / independent-edges.dense.md の update 候補

### 6.1 `docs/research/independent-edges.dense.md`

- §9.2 案 K (GNN) の前提に **slide 12-13 host endorsement** を追記
- 新規 §9.5 案 N (= visible-as-typewell の合成可能性) を追加
- 新規 §9.6 案 O (= direction-aware Beam/NCC) を追加

### 6.2 `docs/strategy/winning-strategy.dense.md`

- §「優勝戦略の構造原理」に「**slide 9 host endorsement = horizontal visible GR を reference に使え**」を明記
- 改修 b の中身に **edge M を追加** (= 元改修 b は A+D+E+G の 4 案、edge M を加えると 5 案)

### 6.3 `docs/dev/schedule-2026-05-10.dense.md`

- exp006 (TabICL + PF-lite) の improve 候補に edge M を inline 追加
- exp007-008 で edge N (offset well retrieval) を扱う旨を update

判断 (= update する / しない / 部分 update) は中央 (= 私) に委ねる。

## 7. 関連ファイル

- `data/raw/AI_wellbore_geology_prediction_task_en.pptx` (= 元 pptx)
- `docs/research/data-spec.dense.md` (= データ仕様、本 doc は補完)
- `docs/research/first-principles.dense.md` (= 計測値、本 doc の slide 12-13 dip continuity と整合)
- `docs/research/independent-edges.dense.md` (= 独自 edge 集、本 doc の edge M/N/O を update 候補)
- `docs/research/pilkwang-distill.dense.md` (= PF-lite の詳細、本 doc の slide 9 = visible-as-typewell に拡張)
- `docs/research/top3-distill.dense.md` (= 公開 top の解体、本 doc は新規 paradigm 提示)
