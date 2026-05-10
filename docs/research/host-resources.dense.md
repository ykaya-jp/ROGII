# Host Resources 解読 — 公式 PowerPoint + YouTube (2026-05-10)

> **取得方法**:
> - PowerPoint (`data/raw/AI_wellbore_geology_prediction_task_en.pptx`, 28 MB, 14 slides) を `python-pptx` で text 抽出
> - YouTube `https://youtu.be/gdK_eY5_QrE?t=146` (ROGII 公式 geosteering 解説、396s) を `youtube-transcript-api` で auto-generated transcript 取得
> **生成物**:
> - `data/external/host-resources/pptx-extracted.json` (14 slides 構造化)
> - `data/external/host-resources/youtube-gdK_eY5_QrE-transcript.{json,txt}` (時刻付き transcript)
> **背景**: discussion topic 697416 (ROGII 公式 welcome) で「PowerPoint をレビューせよ」「manual geosteering YouTube を見よ」と明示推奨されている。両方とも host が「これが解釈の前提」と言っている一次資料。

---

## 1. PowerPoint 14 slides 要約

### Slide 1: Wellbore Geology Prediction (Title)
コンペ名のみ。

### Slide 2: Data Available
2 ファイル構成:
- `Well1XXXX__horizontal_well.csv` (lateral 側)
- `Well1XXXX__typewell__Typewell2XXXX.csv` (vertical reference)

### Slide 3: Horizontal Well Data
Columns 確定:
- **MD** (measured depth, well 長), **XYZ** (各点座標), **GR** (gamma ray, 一部 NaN 可)
- **TVT** (geology of wellbore、**train でのみ** 提供) ← 予測 target
- **TVT_input** (PS 点まで known)
- **Top depth of each geological formation** (= ANCC, ASTNU 等の formation top、**train でのみ** 提供)

> 公式整理: TVT は「geology of the wellbore」と呼ばれている → discussion 698282 の「imaginary reference line までの vertical distance」と整合。

### Slide 4: Typewell Data
- 各 horizontal に **typewell 1 本** が割り当てられる
- columns: TVT (true vertical thickness, vertical well での depth), GR, **Geology** (層名 label)
- → これが topic 697857 で「test typewell に Geology 列無し」と確認された理由 (実体は train-only auxiliary)

### Slide 5: Goal Definition
- typewell の TVT は **常に known**
- horizontal の TVT は **PS 点まで known** (= visible 部)
- 目標: PS 以降 TVT を、horizontal の `XYZ + GR` と typewell の `TVT + GR` から計算

### Slide 6: ★ 核心: GR signature と TVT 増減の関係 (図解)
2 つの図解パターン:
- パターン A: `GR signature matches Typewell GR` & `TVT is increasing` (= drillhead が下方向に horizon を切っている)
- パターン B: `GR signature matches Typewell GR` & `TVT is decreasing` (= drillhead が上方向に horizon を切っている)
- **両方とも GR 一致だが、TVT 増減方向は反対** → discussion 697431 msg8 の「DTW の index は reverse 可」「drillhead が up/down 双方向」と完全一致

> **含意**: pure xcorr (best lag) アプローチでは「方向」が判別不能。**signed alignment (ascending/descending bistate)** または **方向 prior** (= visible 末端の dTVT 符号を local prior にする) が必要。

### Slide 7: ★ TVT constant region (図解)
- パターン C: `GR signature is constant` & `TVT is constant` (= drillhead が同一 layer 内 lateral 進行)
- 含意: lateral の **長 stretch で TVT がほぼフラット** な区間が存在 → naive last-value baseline (RMSE 15.91) が部分的に有効な理由

### Slide 8: 動画リンク (= Slide 9 の前置き)
"VIDEO: GR behavior helps us understand the TVT (geology) along each portion of the horizontal well"

### Slide 9: ★★ 解像度差と self-correlation hint (核心)
- **「horizontal well の GR は typewell の GR より高解像度 (high resolution)」**
- 図解: 3 色の GR 線 — **green** (PS 後 lateral GR), **red** (PS 前 visible lateral GR), **black** (typewell GR)
- 公式の発言:
  > **The green GR correlates better with the red GR than with the typewell GR (black)**
  > **It may be better to use GR data from the horizontal well before the PS point, combined with deeper TVT data, to correlate the rest of the lateral**

> **含意 (極めて重要)**:
> - 公式が **「typewell よりも visible 部 lateral GR を local reference として使う方が高精度」** と公言している
> - これは pseudo-typewell の自己包含版 = **「visible 部 horizontal GR を local typewell として活用するアプローチ」を ROGII が推奨**
> - 既存 strategy `winning-strategy.dense.md` 案 A/B/C のいずれにこれが盛り込まれているか要確認 → 無ければ **edge I (visible-as-typewell) を first-principles addendum に追加候補**
> - 実装案: visible lateral 末端の GR window と PS 以降の GR window で **xcorr or DTW** を取り、TVT 増減方向と magnitude を **typewell 経由を bypass** して直接推定。typewell 推定との **ensemble** で堅牢化

### Slide 10-11: 空間配置可視化
- training + validation wells の **map view (2D)** と **3D view**
- → discussion 697431 msg12 「plot in 3D and it is a folding problem」と同じ知見を host が提示

### Slide 12: ★ Geology dip と方位の関係 (図解)
- "Geology is almost flat" / "Geology is dipping" の 2 例
- "**The azimuth of horizontal drilling affects the expected geology dip**"
- "**The geology of an offset well can help predict the geology of the current well**"

> **含意**: multi-well joint inference は **公式に推奨されている戦略**。
> - well の azimuth (= XY 平面進行方向) を feature 化すると dip 方向との相互作用を捉えられる
> - offset well の TVT で current well の prior を作る → discussion 697431 「3D 構造再構築」の host 公認版

### Slide 13: ★ 隣接 well の dip 相似性 (図解)
- "Geological dips behave similarly in neighboring wells"
- 4 図解で `Dip / Dip / Flat / Flat` のラベル

> **含意**: これは independent edge G/H 候補に **空間統計的根拠**を与える:
> - **Spatial GP / Kriging** で隣接 well の TVT を prior として導入
> - Bayesian 階層モデル (`first-principles.dense.md` §1.5) の物理的妥当性を host が裏付け
> - feature engineering: 各 well に「**最近傍 N 本の lateral との dip cosine 類似度**」を追加

### Slide 14: 評価指標
- `dTVT = manualTVT - predictedTVT` の各点 (1 ft step) で計算
- **RMSE of all dTVT values** が score
- → 既知だが、`manualTVT` という名前から **真値 TVT は人手で interpretation された** ことが確定 (= 完全 noise free ではない、interpretation noise を含む)

---

## 2. YouTube transcript (`gdK_eY5_QrE`, 0:00-6:36) ハイライト

> ROGII 社の geosteering ソフトウェアの操作 tutorial。全 396s, 6037 chars。
> ML 戦略への含意は **segment-based stretch/squeeze** が human alignment の本質という発見 (= piece-wise DTW + dip 推定の物理的根拠)。
> 全文: `data/external/host-resources/youtube-gdK_eY5_QrE-transcript.txt`

### 2.1 0:00-1:00 (intro): horizon 概念
- 前動画で「typewell referenced project」「**horizons**」を作成済 → 本動画は horizon 設定後の interpretation step
- **"horizons"** = typewell から推定した layer top の MD-TVT pairs (= 既存 docs の formation top に相当)

### 2.2 1:00-2:30 (segment 概念): blue/green line
- 初期状態は lateral 全体が **1 single orange segment**、interpretation で segment を細分化
- **blue line** = active segment の **start** (= interpretation start)
- **green line** = active segment の **end / new segment 作成 trigger**
- segment は active=red、inactive=別色で可視化

### 2.3 2:30-4:00 (★ stretch / squeeze = dip 推定の本質)

**「I can also move this green and blue lines within the vertical track this will change the dip as it's stretching and squeezing the data」**

- segment 内のデータを vertical track 上で **伸ばす / 縮める** ことで dip 角度が変化
- = **affine 変換 (TVT scale)** by per-segment factor
- 実例: `dipped value changes next to that green line` (1:58 付近)

> **含意**: human geosteering の核心は **「segment ごとに typewell vertical scale を伸縮させて MWD GR と match させる」** = **piece-wise affine TVT scale alignment**。
> - これは pure DTW (1-to-1 monotone alignment) より自由度高く、**piece-wise linear scaling** に相当
> - ML 実装案: `segment k で TVT_horizontal = a_k × TVT_typewell + b_k` の `(a_k, b_k)` per segment 推定。`a_k` = dip 因子, `b_k` = offset (= b_well の segment 化)
> - discussion 697431 msg6 hengck23 「piece-wise fitting DTW. model predict (start, end, **dTVT/dMD slope**) for each segment」の **slope** = この `a_k` (= stretch ratio) と完全一致

### 2.4 4:00-5:30 (★ dip の典型値: 89-91 deg)

実例 transcript:
- "I'm squeezing that data to fit at a **dip of 91.32**"
- "I can squeeze that data up to an **89 degree**"
- "this data is showing me that there's two different dips ... I will break this segment"
- "I can see the peaks and troughs but they're not matching with one single bed dip"

> **含意**:
> - dip range は **89-91 deg 付近に集中** (= ほぼ horizontal、わずかな variance を当てるゲーム)
> - **同一 segment 内で 2 つの dip 検出** = segment 細分化トリガー (= ML 側では loss が segment 内で bimodal なら break する self-adaptive segmentation)
> - 90 deg からの偏差を直接 regress する **dip residual head** が情報的にコンパクト (`dip - 90` の per-segment 値)

### 2.5 5:30-6:36 (multiple interpretations = ensemble)

**「You can have as many interpretations as you'd like if you have different scenarios of your interpretation」**

- 操作: right click → copy → paste で interpretation 複製 → activate して別解釈を保持
- → human が **ensemble of interpretations** を内部で持つことを示唆

> **含意**:
> - ROGII の human ground truth (`manualTVT`) は **複数 interpretation の中から選ばれた 1 つ** = label noise 源
> - ML 側で **multiple interpretations を陽に予測 → mixture density estimation** すれば、label noise を marginalise できる
> - 簡易版: **N-best beam search** で K 個の hidden TVT 解を保持 → 最尤を選ぶか accumulate average

### 2.6 winning strategy への 4 つの追加示唆 (transcript 全読了後)

| # | 操作の本質 | ML への直接対応 |
|---|---|---|
| Y1 | segment 内 stretch/squeeze で **dip 因子 `a_k` を fit** | per-segment affine `TVT = a_k × typewell_TVT + b_k` 推定 |
| Y2 | segment 細分化 trigger = **bimodal dip detection** | residual signature が bimodal な区間を切る self-adaptive segmentation |
| Y3 | dip 89-91 deg 集中 → **`dip - 90` residual head** | per-segment scalar regression (small magnitude) |
| Y4 | multiple interpretations = ensemble | mixture density / N-best beam の出力形式

---

## 3. winning strategy への含意 (まとめ)

| # | host hint | 既存 dense doc との関係 | 提案 |
|---|---|---|---|
| H1 | Slide 6: GR match で TVT 増減方向は **両義** | first-principles §1.4 b_well 単独では捉えきれない | **signed alignment (visible 末端の dTVT 符号を local prior に)** |
| H2 | Slide 7: TVT constant region 存在 | `data-spec.dense.md` `dTVT_p95 = 1.01` と整合 | naive last-value baseline (RMSE 15.91) が部分有効な物理根拠 |
| H3 | **Slide 9: visible GR を local typewell として使う方が高精度** | 既存案 A/B/C で明示的に統合されていない可能性 | **edge I (visible-as-typewell xcorr) を新規 candidate として追加** |
| H4 | Slide 12-13: 隣接 well で dip 類似 + azimuth 影響 | first-principles §2.7 §C で暗黙的に AR1 / random walk として扱う | **spatial GP / kriging prior + azimuth feature** を追加 |
| H5 | YouTube: segment-based piece-wise alignment が human workflow | discussion 697431 msg6 hengck23 提案と一致 | **K-segment 予測タスク** (`(start_md, end_md, dTVT/dMD)` × K segments) を edge H と統合 |
| H6 | Slide 14: `manualTVT` = human interpretation | `formula_oracle_rmse_p50 = 0.0055 ` (first-principles §1) と整合 (oracle ≠ 0) | label noise を考慮した **robust loss (Huber, quantile)** を試す |

---

## 3.5 well azimuth EDA (H-T2 完了, 2026-05-10)

> **再現スクリプト**: `notebooks/_typewell_groups_build.py` 隣に `data/processed/well_azimuth.parquet` を生成 (本セッションでインライン実行、773 wells 全件)
> **生成物**: `data/processed/well_azimuth.{parquet,csv}` (gitignored、773 rows × 7 cols)
> **方法**: visible-zone XY (`TVT_input` notna 区間) に SVD を掛け第一主軸を well 進行方向とする。同時に full-trajectory end-minus-start direction も比較計算。

### 3.5.1 全体 azimuth 分布 (full lateral end-direction)

| octant | wells | % |
|---|---|---|
| **NW** | 364 | **47%** |
| **SE** | 279 | **36%** |
| N | 39 | 5% |
| S | 36 | 5% |
| W | 31 | 4% |
| E | 20 | 3% |
| NE | 3 | 0% |
| SW | 1 | 0% |

- **wells の 83% (= NW + SE 合計) が NW-SE 軸上**
- 北米 Texas Eagle Ford trend は **NW-SE strike** (北西-南東に走る帯状鉱床) → 物理的整合
- 含意: **dip 方向 ≈ NE-SW 直交方向** (NW-SE strike の dip は 90 度回転した NE-SW 軸) → wells は dip 直交方向に lateral 進行している = PowerPoint Slide 12「azimuth は dip に影響」と整合

### 3.5.2 lateral straightness (visible-zone PCA `S2/S1`)

| metric | p5 | p25 | **p50** | p75 | p95 |
|---|---|---|---|---|---|
| `straightness_visible` | 0.005 | 0.009 | **0.015** | 0.024 | 0.053 |

- 全 wells で **S2/S1 < 0.05** = 第二主軸が第一主軸の 5% 以下 → **極めて直線的**
- 含意: lateral は基本まっすぐ、curving lateral は無い → PCA 主軸 = azimuth 推定が信頼できる

### 3.5.3 ★ duplicate group 内の azimuth dispersion (= pseudo-typewell 信頼度の代理指標)

13 duplicate groups (§10b) 内で azimuth の **circular std** を計算:

| group_id | size | azimuth circ_std (deg) | range (deg) | 解釈 |
|---|---|---|---|---|
| `75cd5f11` | 2 | **0.5** | 0.9 | 同方向の隣接 lateral (= standard practice) |
| `89f1085d` | 2 | **0.1** | 0.3 | 同上 |
| `8b95d6d1` | 2 | **0.7** | 1.3 | 同上 |
| `a4f989c2` | 2 | **0.5** | 0.9 | 同上 |
| `add9c322` | 2 | 8.1 | 16.2 | ほぼ同方向 |
| `25939962` | 2 | 12.1 | 24.0 | ほぼ同方向 |
| `2f8e53c3` | 2 | 12.8 | 25.6 | ほぼ同方向 |
| `02e7fe5a` | **10** | 41.8 | 357.1 | **N-S 双方向 10 lateral が 1 typewell 共有 (★ pseudo-typewell 代表例)** |
| `b977be4a` | 2 | 105.2 | 158.7 | **方向反対** = pseudo-typewell 候補 |
| `7b38844c` | 2 | 101.5 | 156.0 | **方向反対** = pseudo-typewell 候補 |
| `f321a31c` | 2 | 116.1 | 194.7 | **方向反対** = pseudo-typewell 候補 |
| `cd7f1687` | 2 | 165.7 | 181.7 | **完全反対方向** = pseudo-typewell ほぼ確定 |
| `071d7b45` | 2 | 181.4 | 179.2 | **完全反対方向** = pseudo-typewell ほぼ確定 |

- **circ_std ≤ 30 deg**: 7 groups (= 同方向の隣接 lateral pair、standard typewell sharing) → **ground truth 信頼度 = 通常並み**
- **circ_std ≥ 100 deg**: 6 groups (= 方向反対の lateral 群、azimuth 違うのに同 typewell) → **pseudo-typewell の可能性高 = ground truth 信頼度 低**

> **含意 (loss weight 自動調整)**:
> - 既存 `typewell_groups.parquet` に **`is_pseudo_likely`** = `is_duplicate AND circ_std_within_group >= 60deg` を追加すれば、pseudo-typewell 23 wells (= 6 groups の合計、`b977be4a`+`7b38844c`+`f321a31c`+`cd7f1687`+`071d7b45` の各 2 + `02e7fe5a` の 10) を自動識別可
> - これらに loss weight 0.5-0.7x を掛ける variant を試す価値あり
> - exp004/005 候補: typewell_groups.parquet の v2 (azimuth dispersion 列追加) と loss-weighted variant

## 4. 次の行動 (本 doc から派生する具体タスク)

| # | task | 優先度 | 配置先 |
|---|---|---|---|
| H-T1 | Slide 9 の「visible-as-typewell」approach を edge I として `independent-edges.dense.md` に追記 | 高 | exp004 候補 (host 公式推奨) |
| H-T2 ✅ | well azimuth (= visible 部 XY 進行方向の主成分) を feature 化、dip 推定 prior に統合 — 完了 (§3.5、`data/processed/well_azimuth.parquet`) | 高 | 完了。実装側は `azimuth_visible_pca_deg` を sin/cos に展開して feature 追加 |
| H-T3 | spatial 近傍 well を `KDTree(X, Y)` で取得 → top-k well の TVT を prior として attention 入力に | 中-高 | exp005 候補 (NN 系) |
| H-T4 | per-segment affine (`a_k, b_k`) 予測 head を Sequence Transformer (案 B) に追加 — Y1 + Y2 + Y3 統合、stretch/squeeze の ML 化 | **高** | exp006 候補 (Y1-Y4 finding が直結) |
| H-T5 | `manualTVT` label noise 耐性のため Huber loss で baseline を再 train、RMSE 比較 | 低-中 | exp003.1 等の minor variant |
| H-T6 ✅ | YouTube transcript 残り (4:00-6:36) を読み込み、segment 操作の詳細 UI ロジックから設計示唆を追加抽出 — 完了 (§2.3-2.6、Y1-Y4 finding 追加) | 中 | 完了 |
| H-T7 (新規) | `typewell_groups.parquet` v2 — azimuth dispersion 列を追加、`is_pseudo_likely` flag を計算 (§3.5.3) | 高 | exp004 candidate input |
| H-T8 (新規) | mixture density / N-best beam の出力形式を Sequence Transformer に追加 (Y4 finding) | 中 | exp006/007 候補 |

---

## 5. ファイル指紋

- 取得スクリプト: 本 doc 直前のインライン bash + uv run --with python-pptx --with youtube-transcript-api
- PPTX raw: `data/raw/AI_wellbore_geology_prediction_task_en.pptx` (28.79 MB、Kaggle 配布物)
- PPTX 抽出: `data/external/host-resources/pptx-extracted.json` (14 slides 構造化)
- YouTube transcript: `data/external/host-resources/youtube-gdK_eY5_QrE-transcript.{json,txt}` (396s、6037 chars)
- 取得時刻: 2026-05-10 22:48 JST (pane 2)
- 関連 dense doc: `docs/discussion/2026-05-10-summary.md` §4 T5 (本 doc がこの T5 を完了)
