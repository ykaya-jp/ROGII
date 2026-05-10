# ROGII Kaggle Deep Dive — Discussions のみ (2026-05-11、scope 縮小版)

> **担当**: subagent J (2026-05-11)
> **対象**: Kaggle 公式 discussion thread 全 20 件 (= `kaggle competitions topic-messages` 経由で取得済)
> **生データ**: `docs/research/raw/discussions_topics.json` (20 topics) + `docs/research/raw/discussions_messages.json` (top-level 70 件 + 多段返信)
> **取得方法**: `tools/discussion_fetch.py` (公式 Python API: `KaggleApi.competition_list_topics` / `competition_list_topic_messages`)。
> 既存 doc `docs/research/discussions.dense.md` の「discussion は WebFetch では取れない」前提は CLI v1 の topic-messages サブコマンド開放で**現在は false**、これが本 doc の存在意義。

---

## 0. 既存リサーチとの差分

既存 doc で **既に把握済み (= 本 doc では再掲しない)**:

- `discussions.dense.md` Phase 6.1 / 6.2 — `TVT = -Z + ANCC + b_well` の物理関係、residual target、ANCC plane-fit imputation
- `discussions.dense.md` §1.1, §2 — MD=1 ft 統一、`trailing_hidden_only` 100%、visible_ratio 0.26 中央値、typewell step 不統一 84.5%
- `top3-distill.dense.md` / `pilkwang-distill.dense.md` — 上位公開 kernel (needless090 / romantamrazov / karnakbaev / pilkwang) の内部構造
- `independent-edges.dense.md` / `first-principles.dense.md` — エッジ M / N / O など我々が独自に発掘した方向性
- `data-spec.dense.md` — column 仕様

本 doc が**新規に追加するもの**は以下に絞る:

1. **host (ROGII / Kaggle staff) からの一次声明**で既存 doc になかったもの
2. **参加者議論で初めて公けになった発見** (= 公開 kernel に未反映、Discussion で口頭共有されたもの)
3. **共通質問の頻度感覚** (= 多数派が困っている領域、戦術アドバンテージの源泉)

---

## 1. Host 公式アナウンス (CRITICAL: 戦略の前提条件)

20 thread のうち host (organizer ROGII の `@maria` / Kaggle staff `@RyanHolbrook` `@addisonhoward` / その他 organizer) が直接書き込んでいる thread と、その確定情報を全列挙する。

### 1.1 競技構造に関する公式声明

| topic | URL | 発信者 | 確定事項 | 我々の戦略への影響 |
|---|---|---|---|---|
| 697416 | https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697416 | ROGII organizer (welcome post) | 「データは経験ある interpreter が confident に解釈可能」「PowerPoint 添付資料に key info」「youtube 補足動画」 (= https://youtu.be/gdK_eY5_QrE?si=0zvqh87AtgtToRE1&t=146 ) | 動画は manual geosteering の実例、`host-pptx-summary.dense.md` 担当の subagent が pptx を分析中。**動画の geosteering 操作 = 人間プロが TVT-GR の縦シフト + 横ストレッチで visual に合わせる、これが本 task の domain truth** |
| 697400 | https://www.kaggle.com/.../discussion/697400 | host @maria | 2026-05-05 22:46 に submission 一時停止 → 22:57 に **backend 修正のみで完了、再 DL 不要** と確定 | ⚠️ **ローカルの train/test data を再 DL しないと壊れる、という心配は不要**。current snapshot で OK |
| 698282 reply | https://www.kaggle.com/.../discussion/698282#PatrickAIForFun への返信 | host (organizer 相当の anonymous account) | **「TVT は virtual/imaginary reference line への vertical distance、TVT=0 が ground level とは限らない」**「lateral の TVT は対応する typewell の TVT に一致 (lateral-typewell pair で対応関係が定義される)」「lateral の XYZ は true spatial、typewell は geology の vertical 定義 (GR values)」 | **TVT 定義の最も明示的な host 確認**。我々の前提 `TVT = -Z + ANCC + b_well` は host 確認では**式そのものは語られていない**が、「typewell TVT 軸 = lateral TVT 軸」が明示されたため、**Beam Search / DTW / PF が typewell-TVT 軸上で動作する根拠は確定** |
| 698449 reply | https://www.kaggle.com/.../discussion/698449 | host (organizer 風) | **「typewell 選定は manual。lateral 近接 + drilling 当時に available なものを geologist が選ぶ。**10 年に渡る drilling のため、過去はあっても現在は close でない typewell も含まれる。**typewell の一部は隣接 lateral の interpretation そのもの = "pseudo-typewell"」** | ⭐⭐⭐⭐⭐ **新規最重要事実**。typewell は **iid な vertical well ではなく**、過去 lateral からの interpolation を含む場合がある。詳細は §2.2 |

### 1.2 提出仕様に関する公式声明

| topic | 発信者 | 確定事項 | 影響 |
|---|---|---|---|
| 697329 ([@RyanHolbrook 返信、2026-05-08 14:47](https://www.kaggle.com/.../discussion/697329)) | Kaggle staff @RyanHolbrook | **「Data ページに見える test set は example のみ。submit 時に notebook が rerun され、example data が hidden test set に置換される。submission scoring error の主因は『local CSV を upload する人』 = 自分のローカル test 予測を上げても hidden 側 ID と一致しない」** | ⭐⭐⭐⭐⭐ **このコンペは code competition 専用、CSV 単体 submit 不可**。Kaggle notebook 内で predict-and-write しなければ scored できない |
| 697329 ([@RyanHolbrook 返信、2026-05-08 11:50](https://www.kaggle.com/.../discussion/697329)) | Kaggle staff | submit すると **published data 用 run + hidden test rerun の 2 並列 run** が走る、submission cap を消費するのは **hidden 側のみ**。notebook 上で commit してから「submit from viewer」すれば hidden 単独 run も可能 | dev cycle で「自分の TVT 値が public LB に反映されるまでの時間」読みに直接影響。**published 側 run が完了しないと hidden 側 run も開始しない** ことを **明言していない** が、実際の経験談 (4–5 hr) と整合 |
| 697329 ([@RyanHolbrook 返信、2026-05-08 11:47](https://www.kaggle.com/.../discussion/697329)) | Kaggle staff | **scorer の不具合は 2026-05-05 launch 直後の数時間のみで既に解消済**。今 submit error が出るのは「published example test 上では動くが、hidden test 上では ID 不一致でコケる」コードだから | ローカル CSV upload (= R で生成して dataset 化) は**仕組み上完全に NG**、これは何度も participants が同じハマり方をしている |
| 697552 ([@RyanHolbrook など](https://www.kaggle.com/.../discussion/697552)) | participant + staff | **submission rerun の wallclock 期待値: 30 min runtime → 2 hr で結果 / 2.5 hr runtime → 7+ hr / 11 hr 超で timeout**。9 hr 制限は documented | **runtime 3 hr 以下推奨**、4–5 hr は実用上限。我々の exp003/exp006 が 3.5 hr 圏なのは limit 接近、TabICL+PF-lite で 4-5 hr 構成は **危険ゾーン** |

### 1.3 ルール解釈に関する **未確答の質問** (= 重要、host から明示返答なし)

| topic | 質問 | 現状 |
|---|---|---|
| 698002 [Is online learning / test-time fine-tuning allowed?](https://www.kaggle.com/.../discussion/698002) | 「rerun 中に hidden test の TVT_input + features を peek して self-supervised に fine-tune するのは rule 適合か?」 | **host から返答なし** (2026-05-11 時点)。一方で別 participant が「**online: 10.953 / offline: 11.323**、test-time adaptation で **0.37 RMSE 改善**」と実測共有。改善幅は exp003 vs exp005 の差より大きい |
| 697857 [Is 'Geology' feature included in actual test set?](https://www.kaggle.com/.../discussion/697857) | 「test 側 sample に Geology 列がない、本番もそうか?」 | host 返答なし。複数 participant が「train only feature。本番 test も無いはず、ground truth は **derived**」と返答、これが **事実上の確定**。我々は exp 系で Geology を train auxiliary としか使っていないので影響無 |

---

## 2. 参加者間の重要発見

### 2.1 データの罠

#### 2.1.1 Test 側 visible 3 wells の hidden row 数 (topic 697507)

`@konbu17` と思われる participant が以下を共有:

```
000d7d20: 3836 hidden rows
00bbac68: 6014 hidden rows
00e12e8b: 4301 hidden rows
合計: 14,151 rows = sample_submission.csv の row 数
```

合計 14,151 と一致 → **example test の 3 wells のみで sample_submission が構成**。本番 hidden test は別 wells、行数は不明。**これに依存する CV / 推論ロジックは hidden で全く別の wells に直面する**ことを再確認。

#### 2.1.2 Test well `000d7d20` が train にも存在する (topic 697507)

participant `@maherelouahabi` が指摘、host 返答:

> 「test/ wells は example のみ、submit 時に hidden test (train に含まれない wells) に置換される。**hidden test には train wells は含まれない**」

確認できた事実:
- `000d7d20` (train) と `000d7d20` (test) は MD/X/Y/Z が **5,278 rows 完全一致**、GR は ~43% rows で異なる (= **同じ trajectory に異なる GR ノイズ**)
- → example test は train から sample されている、**train の特定 well を test 的に処理する CV が valid**

我々の戦略への impact:
- ✅ exp001/002/003/005/006 で train wells を val として使う CV は host stance と整合
- ✅ `000d7d20` のような train↔test 重複は**わざわざ排除する必要なし** (= leak ではなく example として正規)

#### 2.1.3 Train ↔ sample_submission ID 重複は leak ではない (topic 698266)

別 participant が「train file 内に sample_submission の (well, row) target ID が存在する場合、TVT 値を直接 lookup していいか?」と質問 → 別 participant が「sample_submission も example のみ、本番では hidden に置換される。peek しても無意味」と説明 → host 返答無し、**事実上 close**。

我々の戦略への impact:
- ✅ submission CSV を train から lookup する hack は不可能 (= rerun 時に ID が変わる)
- ✅ **inference notebook 内で hidden test を読んで predict する設計を変える必要なし**

### 2.2 評価 zone の特殊事情 — Pseudo-typewell の存在 ⭐⭐⭐⭐⭐

topic 698449 で participant が **13 グループ計 35 wells で typewell ファイル完全一致**を発見:

```python
duplicate_groups = [
    ['02e7fe5a', '10b89021', '3417285d', '6ae68655', '7993a768',
     'bc4381e2', 'ecdab904', 'f021b650', 'f49fdea3', 'f88ddb26'],  # 10 wells
    ['071d7b45', '4463446c'],
    ['25939962', '8050c789'],
    ['2f8e53c3', '91db7070'],
    ['75cd5f11', 'be83e781'],
    ['7b38844c', 'ed6e6e54'],
    ['89f1085d', 'aed44918'],
    ['8b95d6d1', 'a2e8e7f6'],
    ['a4f989c2', 'd011f41b'],
    ['add9c322', 'cbe62450'],
    ['b977be4a', 'c908edd0'],
    ['cd7f1687', 'fcfcc902'],
    ['f321a31c', 'fa667be2'],
]
```

これに対して **host が直接返答**:

> 「typewell selection は manual process。geologist が lateral に近く、当時 (drilling 時点で) available なものを選ぶ。**10 年スパンの drilling のため、当時 closest だったが今は close でない typewell も含まれる**。**さらに**、geologist は近隣 lateral の interpretation 結果を typewell として使う場合がある (= "**pseudo-typewell**")。**プロジェクト内 typewell の一部は実は近隣 lateral からの interpretation**」

**含意**:
1. **typewell GR profile は完全に independent ではない** — pseudo-typewell ならば近隣 lateral と相関する
2. **同じ typewell を共有する wells は地質的に相関 group** — train wells で同じ typewell を持つもの同士は CV fold で**同じ fold に置く**べき (well leak 防止)
3. **typewell が近隣 lateral 由来なら、**近隣 lateral の hidden TVT を埋める情報源として使える可能性 — ただし test wells (新規 wells) の typewell が train lateral の interpretation である場合、その train lateral の TVT は train にあるので **間接的に強い prior** になる
4. **逆に** train↔train 内で typewell duplicate group が GroupKFold (by well_id) を**そのまま信じると leak**: `02e7fe5a` group の 10 wells が異なる fold に分かれると、val fold の "正解" に近い情報が train fold の typewell 経由で流入する

我々の戦略への impact (⭐⭐⭐⭐⭐):
- ⚠️ **CV 戦略に直撃**: GroupKFold(by=well_id) ではなく **GroupKFold(by=typewell_signature_hash) で nest** が必要
- ✅ **typewell embedding** (= typewell GR を fixed-length vector 化して well 間距離計算) が**地質 cluster の prior** として valid → KNN 推論の strong feature
- ✅ test inference 時、test の typewell が train lateral の pseudo-typewell な場合、**その train lateral の TVT を直接 prior に使える**

### 2.3 Online learning / Test-time fine-tuning の効果実測 ⭐⭐⭐⭐

topic 698002 で participant `@maherelouahabi` (推定) が **同 setup で実測比較**:

| 設定 | Public LB |
|---|---|
| online training (= rerun 時に hidden test の features を読んで fine-tune) | **10.953** |
| no online training | 11.323 |

差分: **-0.370 RMSE**。base notebook は `triple-signal-beam-search-dual-pf-lightgbm` (shinyanagai123)。

**含意**:
- LB 9 帯に到達するには test-time adaptation が現実的に必要
- host から禁止アナウンス無し → 現状 fair game
- 我々の exp003 (10.34) → exp006 (TabICL+PF-lite) は offline only、**online TTA の上積み余地が +0.4 ある可能性**

リスク:
- host が遡及的に「これは spirit 違反」と判断するリスク (= 過去コンペで類似事例あり、rule 修正の可能性)
- ただし「test-time に hidden test の features を読む」は code competition の前提構造 (notebook が hidden test を read できる) と表裏一体、禁止すると competition が成立しない

### 2.4 TVT 定義の混乱 (topic 697418, 698282)

複数の participant が「TVT > Z な行が多い、TVT が地層 top depth より大きい」と当惑。host (anonymous organizer) が **698282 の reply で**:

> 「TVT は virtual/imaginary reference line への vertical distance。TVT=0 が ground level に対応するとは限らない。**lateral の TVT は対応 typewell の TVT と一致する**(per pair で定義)。lateral XYZ は true spatial、typewell は geology の vertical definition」

**確定事実**:
- TVT は signed depth ではなく "geology に対する仮想座標"
- typewell の TVT 軸と lateral の TVT 軸は **pair 内で一致** (= matching の物理的根拠)
- Z (TVD) と TVT は **直接対応しない**、formation tilt 込みの relation

参加者 `@PatrickAIForFun` の独立 EDA: 「`ANCC - Z = TVT + offset`、offset は per-well constant、つまり **TVT=0 reference は地質構造に沿って tilt している**」 → host が reply で「**ground level reference は不確定だが TVT=virtual reference 説は正しい**」と部分肯定。

我々の戦略への impact:
- ✅ 既存仮説 `TVT = -Z + ANCC + b_well` は **host から完全には confirm されていない** (= ANCC を介さない formulation の可能性)、ただし konbu17 / needless090 の高 LB が経験的に確証
- ✅ **typewell TVT 軸 = lateral TVT 軸 (per pair)** が host 公認 → Beam / DTW / PF の typewell-TVT alignment は domain truth に乗っている

### 2.5 戦術 hint (topic 697431, 697500)

`hengck23` (= Kaggle Grandmaster) を含む参加者が discussion で雑談的に放出した hint:

| hint | 出典 | 我々の解釈 |
|---|---|---|
| 「3D geological site を train + test で再構築する。同じ site 内なら test の formation 構造は train から推定可能」 | 697431 hengck23 推定 | **edge O / O+ クラスのアプローチ**。pilkwang は plane fit、karnakbaev は Self-NCC で local 平面を fit、しかし **全 wells を 1 つの 3D field として joint 推定**は未実装 |
| 「md vs dTVT plot で dy が constant に見える → synthetic data の可能性」 | 697431 | dTVT/dMD ≈ const な segment が多い = piecewise linear。**piecewise DTW** の有効性示唆 |
| 「PS (= Pilot Section anchor point) から transformer を grow させる、anchor point を先に見つけるべき」 | 697431 hengck23 | visible 末端を anchor として、hidden 全体を seq2seq で生成する **transformer baseline** の hint。我々は exp002 で MLP residual、exp006 で TabICL — transformer の seq decode は未試行 |
| 「forward model: typewell TVT 軸を horizontal の TVT (visible 部) で interp、hidden 部は extrapolate」 | 697431 ([code snippet](https://www.kaggle.com/.../discussion/697431) ) | **karnakbaev の Self-NCC affine GR cal の本質をシンプル化した式**。1 行 numpy: `query_gr = np.interp(h_tvt, tw_tvt, tw_gr)` |
| 「ローカル train で大きい計算 + Kaggle dataset として model を保存 → notebook では prediction のみ」 | 697500 | 我々の exp005 stack 戦略と整合、ただし **online TTA を入れる場合は Kaggle 内で fit 必要** |

---

## 3. 議論が分かれた設計判断

### 3.1 GR 正規化方法 — discussion では明示議論なし

discussion 上では GR z-score / robust scaler / typewell affine cal の比較議論は**まったく出ていない**。`pilkwang-distill.dense.md` `top3-distill.dense.md` で公開 kernel 内に各方式が並列実装されているが、participants 間で「どれが best か」を比較する thread は**存在しない**。

我々の戦略への impact:
- 公開議論で勝者が定まっていない領域 = **edge にしやすい**
- exp001-006 で z-score / robust / NCC を内部比較するべき

### 3.2 CV 戦略 — discussion では明示議論なし

GroupKFold(by well) vs StratifiedKFold(by formation) vs typewell-aware fold の比較議論は thread には存在しない。topic 697300 [Some question](https://www.kaggle.com/.../discussion/697300) で「data augmentation OK か」の質問はあったが、具体的 CV 構造には触れず。

ただし§2.2 の **pseudo-typewell 共有 13 group** を踏まえると、GroupKFold(by typewell_hash) が**本来正しい** (公開議論にはまだ出ていない):
- 我々が**先に typewell-grouped CV を試して LB-CV gap を測れば、edge を取れる可能性**

### 3.3 External data 使用の是非 — discussion では明示議論なし

外部 well log データ (例: 既存公開 Texas well データ) を補助 dataset として使う議論は無い。topic 697406 で `Geological Formations on well Geology (region: Texas)` の wikipedia 引用解説のみ。Texas Eagle Ford / Austin Chalk が地質的なターゲット領域と明示。

参考潜在 dataset:
- USGS / Texas RRC public well log database
- TexNet / Eagle Ford GR profile public datasets
- 競技 rule 上 "private dataset として upload なら OK" (= `discussions.dense.md` §5.1 既知)

我々の戦略への impact:
- 試行価値はあるが、competition 残 86 日 / Gold cutoff 9.92 で **R&D 優先度は中** (= TTA / typewell-aware CV / 3D field reconstruction の方が ROI 高)

---

## 4. Trending 質問 (= 多数派が困っている領域)

discussion 全 20 topic で**繰り返し出てくる common 質問**を頻度順に:

| 頻度 | 質問 | thread 数 | 解釈 |
|---|---|---|---|
| ⭐⭐⭐⭐⭐ | **Submission Scoring Error が出る** (Kaggle notebook 上で動くが score されない) | 697329 (15 messages), 698185 (3), 697552 (4), 697500 (2) | 多数派が **code competition の仕組みを誤解** (= ローカル CSV upload しようとしている) → **これに気づいた参加者は提出経験で先行** |
| ⭐⭐⭐⭐ | **TVT の物理的意味が分からない** | 697418 (4), 698282 (3), 697431 一部 | 多数派が TVT = absolute depth と誤解、`TVT - Z` の関係を取り違え。**正しい理解 = lateral と typewell pair で TVT 軸が共通** がやっと共有されてきた |
| ⭐⭐⭐⭐ | **Typewell の役割が分からない** | 697418, 697431, 698282 | typewell が ground truth reference として機能する理由 (= TVT 軸経由の lookup) を理解できていない participants 多数 |
| ⭐⭐⭐ | **Geology / formation 列が test に無い** | 697857 (3) | 「test に Geology 無いから model が壊れる」と慌てる participant 多数。**train auxiliary 専用と理解した participant が先行** |
| ⭐⭐⭐ | **9–10 帯から進めない** | 698562 (1) | 9.944 stuck の助けを求める投稿あり。**Gold cutoff 9.92 直前で多数停滞** = 競合が強い、上に出れば一気にメダル圏 |
| ⭐⭐ | **Submit に何時間かかる?** | 697552 (4) | 4–5 hr が参加者経験値、9 hr で timeout |
| ⭐⭐ | **Local 学習 OK か?** | 697500 (2) | OK、ただし predict は notebook 内 |
| ⭐⭐ | **Data augmentation OK か?** | 697300 (3) | 標準的 augment は効かない、GR noise / mask length augment は効くかも、と参加者 reply |

我々の戦略への impact:
- 多数派が **code competition の rerun 構造を誤解**している = 我々の inference notebook (exp006_v2) 設計 (Kaggle dataset → notebook → submission.csv 生成) は**仕組み的にはアドバンテージ**
- TVT 物理的意味の混乱が多い = **我々が host pptx 解析 / first-principles で確立した理解はまだ広まっていない**、edge を保てる

---

## 5. 我々の戦略への impact (= 推奨せず軸のみ)

> 注: 推奨ではなく **判断軸** を提示。最終決定は開発者。

### 5.1 直近 (= 残 86 日) で評価すべき軸

| 軸 | 期待 RMSE 上積み (= 単独試行時) | 実装コスト (= 残時間との比) | リスク | 出典 thread |
|---|---|---|---|---|
| **A. Typewell-aware GroupKFold** (= pseudo-typewell 共有 13 group を 1 fold に集約) | CV-LB gap 縮小 → Public LB の信頼性 ↑、LB 単体で +0.05 〜 +0.20 程度 (= 過学習低減) | 低 (= fold builder 関数 1 つ追加) | low (= 既存 CV と並列に実装可) | 698449 (host pseudo-typewell 確認) |
| **B. Online learning / TTA** (= rerun 中に hidden test の visible TVT_input + GR で fine-tune) | LB +0.30 〜 +0.40 (実測値, 698002 → 11.323→10.953) | 中 (= notebook 内 incremental training loop、3 hr runtime 制約と衝突注意) | medium (= host の遡及的 rule 適用、4–5 hr runtime で timeout) | 698002 |
| **C. Pseudo-typewell linkage feature** (= train lateral と test lateral が typewell file 共有していたら、その train lateral の TVT を direct prior) | LB +0.10 〜 +0.30 (= うまくマッチした場合) | 中 (= typewell hash → train well_id mapping 構築 + KNN prior) | low (= 既存 plane-fit / KNN とは独立に動く) | 698449 |
| **D. 3D geological field 共同推定** (= train+test 全 wells を 1 つの 3D point cloud として ANCC plane を超えた 2D grid で fit) | 不明 (実装と実験次第)、potentially high | 高 (= 新規開発、実装 1-2 weeks) | high (= over-fit、複雑度) | 697431 hengck23 hint |
| **E. Piecewise DTW** (= dTVT/dMD が const な segment が dominant、piecewise linear DTW を classical DP cost matrix で解く) | LB +0.10 〜 +0.20 (= Beam Search の派生) | 中 | low | 697431 |

### 5.2 戦略 doc (= `winning-strategy.dense.md`) に追記推奨セクション (= 提案のみ)

- §「**typewell-aware CV**」: pseudo-typewell duplicate 13 group を 1 fold に閉じ込める方針 (= host 698449 確認)
- §「**Online TTA**」: rerun 中の test-time adaptation、現状 host 黙認、+0.37 RMSE 実測 (= 698002)
- §「**TVT 定義の host 公式声明**」: 既存 §「物理関係」を上書きせず、追記として「typewell TVT = lateral TVT (per pair)」を host 確認の note 付きで明記 (= 698282)

### 5.3 `independent-edges.dense.md` に追記推奨候補

- **Edge P (新規)**: pseudo-typewell linkage prior (= train lateral の TVT を test inference の prior として利用) — §5.1 候補 C
- **Edge Q (新規)**: typewell-aware GroupKFold (= 公開議論にまだ出ていない) — §5.1 候補 A

### 5.4 `strategy-critique.dense.md` に追記推奨観点

- **CV-LB gap の主因仮説**: pseudo-typewell 共有による fold leak の可能性 — exp003 / exp006 の CV 推定が public LB と乖離している場合、まずここを疑う

---

## 6. 注意点 / 残課題

### 6.1 取得済みだが本 doc で深掘りしなかった thread

- **697433 quick EDA notebook**: `https://www.kaggle.com/code/songhow/well-prediction-eda-exploratory` で公開、内容は既存 `notebooks-deepdive.dense.md` 範疇 (= subagent 別担当)
- **697406 Geological Formations**: BUDA / Anacacho / Austin Chalk / Eagle Ford / Olmos の wikipedia 解説、域 = Texas Eagle Ford 周辺。`first-principles.dense.md` で既に活用済み
- **697418 Diagram of the problem**: 画像のみ (text 取得不可)、**画像内容は確認できていない** = 残課題

### 6.2 確認できなかった事項

| 項目 | 理由 | 補完手段 |
|---|---|---|
| author username (= author を取得した dict が空) | API レスポンスで author フィールドが nested object として展開されていない (= python API パース未対応) | `tools/discussion_fetch.py` を改修して `author.user.userName` を抜く処理を追加可能 (今回は時間制約で skip) |
| upvote 数 / postDate 厳密値 | 同上、API レスポンスの一部 field が parse できず | 同上、必要なら raw JSON を再パース |
| 697418 の画像内 diagram | API は image attach の URL/binary を返さない | Kaggle Web UI で手動確認、または selenium で SPA レンダリング |

### 6.3 host への未解決質問 (= 我々が動向を watch すべき)

1. **698002**: online learning / TTA は rule 適合か → 現状 host silent、もし「禁止」アナウンスが出たら戦略 5.1-B を破棄
2. **697418 reply**: TVT > Z な行の物理的整合性について — host から正式回答無し
3. **697857**: actual hidden test の Geology 列の有無 — host から正式回答無し (実質確定だが書面 confirm なし)

### 6.4 30 分以内中間 commit の状況

- 1 回目 commit: `1ecc4d9 chore(research): snapshot Kaggle discussion topics + messages for ROGII (20 topics)` (= raw データ snapshot)
- 2 回目 commit: 本 doc を Write した後に実施予定 (= 本パッチで実施)

### 6.5 後続 task (= 別 subagent または次 session)

| task | 担当推奨 | 期待成果 |
|---|---|---|
| 697418 diagram 画像取得 + 解析 | Web UI 手動 / selenium subagent | TVT > Z 矛盾の hint、host PowerPoint 補完 |
| typewell duplicate 13 group の geological cluster 検証 | EDA subagent | pseudo-typewell prior (Edge P) の effective size 推定 |
| online TTA の実装 prototype | exp007 subagent | LB +0.37 上積みの再現可否 |
| typewell-aware GroupKFold 実装 | exp 系 subagent | CV-LB gap の定量化 |

---

## 7. 出典

- 全 20 topics raw: `docs/research/raw/discussions_topics.json`
- 全 messages raw (= top-level 70 + nested replies): `docs/research/raw/discussions_messages.json`
- 取得スクリプト: `tools/discussion_fetch.py`
- 個別 thread URL は本 doc 内に inline で記載

主要 host 声明 thread (priority order):
1. https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697329 (= submission仕様、code competition の仕組み)
2. https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698449 (= pseudo-typewell の存在)
3. https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698282 (= TVT の正式定義)
4. https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697416 (= welcome / 動画 / pptx)
5. https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697400 (= dataset 修正 backend のみ)
6. https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697552 (= submission runtime 期待値)
7. https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698002 (= online TTA 実測)
