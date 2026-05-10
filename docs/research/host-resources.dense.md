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

> ROGII 社の geosteering ソフトウェアの操作 tutorial。前半 (0:00-2:00) は「typewell referenced project の load」「mode 切替で geosteering に入る」など UI 操作。
> 中盤 (2:00-4:30) で **segment-based interpretation** の概念が出る (= piece-wise DTW の物理対応物)。
> 重要箇所のみ抽出。

### 0:00-1:00 (intro)
- 前 video で「typewell referenced our project」「horizons を作った」とあるので、本動画は **horizon 設定後の interpretation step**
- "horizons" = typewell から推定した layer top の MD-TVT pairs

### 1:00-2:30 (segment 概念)
- **「we haven't broken our data into smaller segments so we have one single orange segment」**
- = 初期状態は lateral 全体が 1 segment、interpretation 進行に従って segment を分割していく
- → discussion 697431 msg6 「piece-wise fitting DTW. model predict (start, end, dTVT/dMD slope) for each segment」と完全一致
- **blue line** = active segment の **start** (= 過去の interpretation start)
- **green line** = active segment の **end / new segment 作成 trigger**

> **含意**: ROGII 社の human geosteering workflow が **segment-based piece-wise alignment** を取っているなら、**segment 検出 (= TVT 増減方向の切替点検出)** を予測する補助タスクが有効。
> - subtask 案: 「PS 以降の hidden zone を **K segments に分割**、各 segment で `(start MD, end MD, dTVT/dMD slope)` を予測」(= hengck23 提案 msg6 の実装ガイド)

### 2:30-4:00 (interpretation 操作)
- "if I move that blue line ... more data populate on the vertical track" → segment start を動かすと typewell vertical track の対応範囲が変わる UI 動作
- "if I move my green line I will create a new segment"
- → これは **interactive geosteering**: human が segment を切ったり延ばしたりして RMSE 最小化する操作。ML が代替する core process そのもの

### 4:00-6:36 (終盤、要約は別途生 transcript 参照)
- 詳細は `data/external/host-resources/youtube-gdK_eY5_QrE-transcript.txt` を参照
- 重要キーワード: `horizon`, `segment`, `interpretation start`, `geosteering cross section`

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

## 4. 次の行動 (本 doc から派生する具体タスク)

| # | task | 優先度 | 配置先 |
|---|---|---|---|
| H-T1 | Slide 9 の「visible-as-typewell」approach を edge I として `independent-edges.dense.md` に追記 | 高 | exp004 候補 (host 公式推奨) |
| H-T2 | well azimuth (= visible 部 XY 進行方向の主成分) を feature 化、dip 推定 prior に統合 | 高 | exp004/005 候補 |
| H-T3 | spatial 近傍 well を `KDTree(X, Y)` で取得 → top-k well の TVT を prior として attention 入力に | 中-高 | exp005 候補 (NN 系) |
| H-T4 | K-segment 予測 head を Sequence Transformer (案 B) に追加 — `(start_md, end_md, dTVT/dMD slope)` × K | 中 | exp006 候補 |
| H-T5 | `manualTVT` label noise 耐性のため Huber loss で baseline を再 train、RMSE 比較 | 低-中 | exp003.1 等の minor variant |
| H-T6 | YouTube transcript 残り (4:00-6:36) を読み込み、segment 操作の詳細 UI ロジックから設計示唆を追加抽出 | 低 | future |

---

## 5. ファイル指紋

- 取得スクリプト: 本 doc 直前のインライン bash + uv run --with python-pptx --with youtube-transcript-api
- PPTX raw: `data/raw/AI_wellbore_geology_prediction_task_en.pptx` (28.79 MB、Kaggle 配布物)
- PPTX 抽出: `data/external/host-resources/pptx-extracted.json` (14 slides 構造化)
- YouTube transcript: `data/external/host-resources/youtube-gdK_eY5_QrE-transcript.{json,txt}` (396s、6037 chars)
- 取得時刻: 2026-05-10 22:48 JST (pane 2)
- 関連 dense doc: `docs/discussion/2026-05-10-summary.md` §4 T5 (本 doc がこの T5 を完了)
