# ROGII Wellbore Geology Prediction — 優勝戦略 plan

> このファイルは `/home/yusuke_kaya/.claude/plans/rogii-wellbore-geology-modular-shannon.md` の repo 内コピー (origin)。両者は同じ内容を保つこと (片方を編集したらもう片方も追従させる)。harness 側は plan 一覧管理用、repo 側は実装中の参照用。

## Context (なぜこの plan を書くか)

- Kaggle コンペ **ROGII - Wellbore Geology Prediction** (https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction) に参戦する
- 期間: **2026-05-05 〜 2026-08-05** (88 日、今日は 2026-05-10 なので残り 86 日)
- 賞金: 1 位 **$25,000** / 2 位 $13,000 / 3 位 $7,000 / 4 位 $5,000
- ユーザー指定: **1 位 ($25K) 厳密狙い** + **計算資源は Kaggle Notebook + 自宅 GPU マシン + Colab Pro/Pro+ 併用** + **週 20h+ 本気モード**
- 評価指標: **RMSE** on hidden TVT zone
- 制約: Code Competition (Notebook 提出)、CPU/GPU いずれも ≤ 9hr runtime、**Internet disabled**、外部公開データ・pretrained model は **OK** (private dataset としてアップロード)
- リポジトリ: `ykaya-jp/ROGII` (GitHub) は作成済みだが、`/home/yusuke_kaya/projects/kaggle/ROGII` にはまだクローンしていない (Phase 0 で clone)

> このコンペで何の問題を解くのか (一行)
> 水平井 lateral 区間で 1ft 毎に **TVT (True Vertical Thickness, bit が今地層内のどこにいるかを示す垂直位置)** を回帰予測する **masked sequence regression**。
> 入力: trajectory (X, Y, Z, MD), Gamma Ray (GR), 部分マスクされた TVT_input, 縦井 reference の typewell (TVT, GR, Geology label)。

## 既知の数値ベンチマーク (Phase 1 リサーチ結果より)

- 公開 LB baseline: **12.602** (`romantamrazov/rogii-super-baseline-lb`, 推定 LightGBM 系)
- 公開 NN starter: CV **15.5** (`cdeotte/nn-starter-cv-15-5`, Chris Deotte 氏)
- 出典: Kaggle Code セクション直接確認 (Agent 1 出力)
- 1 位を取りに行くには大幅に下回る RMSE が必要。類似コンペの優勝パターンから推測して **LB 5-8 帯** を最終目標に置く

> 注意: 上記 RMSE 帯は推測値。Phase 1 EDA で評価 zone の TVT 値域 (典型的に数 ft 〜 数十 ft オーダー) を確定したうえで再校正する。

## 構造的に近い 3 つの先例 — これを真似する

| 先例 | 構造一致度 | 真似する核心技術 | 出典 |
|---|---|---|---|
| **SPWLA 2023 Depth Shift** (1 位 Dreamstar) | 最高 | Multi-scale Depth Warping (MDWD) + GRU + DTW 後処理 | `pddasig/Machine-Learning-Competition-2023` |
| **Kaggle Ventilator Pressure** (1 位 Gilles Vandewiele) | 高 | LSTM+CNN+Transformer hybrid / multi-task (target+diff+cumsum) / **15-fold seed ensemble の weighted MEDIAN** / PID 物理 post-process | `GillesVandewiele/google-brain-ventilator` |
| **FORCE 2020 Lithology** (1 位 Olawale Ibrahim) | 中 | Stratified 10-fold + Bestagini polynomial features + 単一 XGBoost 深掘り | `olawaleibrahim/2020_FORCE_Lithology_Prediction` |

技術手法カタログ全 35 件は Phase 1 リサーチ報告 (Agent 3 出力) を参照。Phase 0 完了後 `docs/research/past-comps.dense.md` と `docs/research/method-catalog.dense.md` に転記する。

## 優勝戦略の構造原理 — **構造原理が異なる 3 案を全部回し、最終 stacking で 1 位を取る**

### 案 A — GBM Stack with Rich Features (SUPER BASELINE 拡張系)

- LightGBM / XGBoost / CatBoost の 3 model を `kagglib.baselines.run_all()` で並走
- 特徴量:
  - GR rolling stats (mean / std / min / max / diff / skew, 窓幅 ±10ft / ±50ft / ±200ft)
  - Trajectory derivatives (MD/X/Y/Z の 1st/2nd derivative, dogleg severity, dip, azimuth)
  - Plane-fit baseline TVT (visible TVT_input から最小二乗) の予測値と residual
  - DTW distance to typewell (`dtaidistance` で計算)
  - k-NN 近傍 well からの TVT pattern retrieval (X/Y 座標 base)
  - Bestagini-style polynomial features
  - well/Geology embedding (CatBoost native)
- CV: **GroupKFold by well_id, 10-fold** (Phase 1 EDA の評価 zone 構造に合わせて再設計)
- Optuna で 200 trials Bayesian tuning
- 期待: 単独 LB **9-11**

### 案 B — Sequence Transformer (Cross-Attention to Typewell)

- Encoder: **PatchTST** (patch=64 or 128 ft) on horizontal stream
- Cross-Attention: query = horizontal patches, key/val = typewell patches (GR + Geology label embedding)
- Multi-task head: TVT (主) + ΔTVT (lag-1 diff) + Geology layer ID (aux classification)
- Masked TVT pretext で self-supervised pretrain (visible TVT_input を 15% random mask して reconstruct)
- 自宅 GPU + Colab Pro+ で長時間学習 → model artifact を Kaggle に **private dataset** としてアップロード → 推論専用 Notebook で 9hr に収める
- **15-seed × 5-fold = 75 model** の weighted MEDIAN ensemble (mean ではなく median, Ventilator 流)
- 期待: 単独 LB **6-9**

### 案 C — Geometry-Physics Hybrid (Plane-fit + DTW + 1D U-Net residual)

- Step 1: visible TVT_input から **最小二乗 plane fit** → linear baseline TVT (geometry projection)
- Step 2: **DTW** で typewell-horizontal GR alignment → GR-driven correction
- Step 3: **CUSUM** で fault boundary 検出 → セグメント分割
- Step 4: **1D U-Net** (PyTorch) を residual (= observed TVT - baseline TVT) 学習に当てる
- Self-supervised pretrain: random window mask → reconstruct
- 期待: 単独 LB **8-10**、解釈性最高、外れ値耐性高、fault 明示処理

### 最終 ensemble: **A + B + C を OOF stacking**

- meta-learner: Ridge と LightGBM の 2 種を比較、OOF RMSE で blend weight 最適化
- weighted MEDIAN を併用 (Ventilator 1 位流)
- Physics post-process: typewell Geology 境界 snap, dip continuity 制約, PID-style smoothing
- pseudo-labeling 1 round (high-confidence test pred を train に追加)
- 期待: **LB 5-8 (1 位射程)**

## トレードオフ表

| 軸 | 案 A (GBM) | 案 B (Transformer) | 案 C (Hybrid) | Final Stack |
|---|---|---|---|---|
| 実装難度 | Low | High | Mid-High | Mid |
| 開発工数 | 2 週 | 4 週 | 3 週 | 2 週 |
| 期待 LB | 9-11 | 6-9 | 8-10 | **5-8** |
| Kaggle 推論 runtime | <30 min | 1-2 hr | <1 hr | 数 hr |
| Overfit リスク | Low | High | Mid | Low |
| 解釈性 | 高 | 中 | 最高 | 中 |
| 1 位差別化 | 中 | 高 | 高 | 最高 |

## マイルストーン (88 日 / 週 20h+)

### Phase 0 — Day 1 (今日 = 2026-05-10): Bootstrap

- `gh repo clone ykaya-jp/ROGII /home/yusuke_kaya/projects/kaggle/ROGII`
- 既存 README.md / .gitignore は維持しつつ、`_template-timeseries/` の構造を merge (timeseries + tabular 双方の lib を併用するため `_template-tabular/` の関連 module も import)
- `kaggle competitions download -c rogii-wellbore-geology-prediction -p data/raw/ --unzip`
- `uv sync --extra dev` (kagglib editable で参照)
- `git remote add origin git@github.com:ykaya-jp/ROGII.git` 確認
- 初期 commit `chore: bootstrap from _template-timeseries`

### Phase 1 — Day 2-7 (5 日): EDA & 仕様確定

- `notebooks/00_eda.ipynb` で以下を全て確定:
  - **train/test の wells 数**、各 well の MD レンジ・lateral 長さ・1ft step か他か
  - **evaluation zone の正体** — well 中央? 末端? 長さ分布? 連続? 複数?
  - **TVT_input の visible / NaN パターン** (= 評価 zone と完全一致するか)
  - **typewell の TVT 範囲** vs horizontal の TVT 範囲のオーバーラップ
  - **test 側で `ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA` 列が与えられるか確認** (= train only と書いてあるが要実物確認)
  - **typewell は test 側でも与えられるか** (与えられないと案 B/C のアプローチ前提が崩壊)
  - GR スケール統一性 (well 間で normalize 必要か)
  - 評価 zone 周辺の TVT_input 連続性
- 報告書 `docs/research/data-spec.dense.md` (日本語) に確定情報を箇条書き
- 公開 LB SUPER BASELINE / NN starter ノートブックのソース読解 → 何を真似てはいけないか把握、`docs/research/public-baselines.dense.md` に要約
- **Phase 1 完了条件**: data-spec が確定するまで実装に入らない (= EDA 駆動)

### Phase 2 — Day 8-21 (2 週): 案 A (GBM Stack) 全力

- Feature engineering スプリント (上記 案 A の features 全て):
  - 基本 rolling stats / derivatives
  - Plane fit residual
  - DTW distance to typewell (`dtaidistance` 利用)
  - k-NN 近傍 well retrieval
  - Bestagini polynomial features
- LightGBM / XGBoost / CatBoost を `kagglib.baselines.run_all()` 経由で 10-fold (GroupKFold by well_id)
- Optuna で各 model を 200 trials Bayesian tuning
- OOF と test pred を `outputs/oof/exp{NNN}-{model}.parquet` に保存
- **第 1 submission**: LB 11 切りを目標 (= baseline 12.602 を上回る)
- **Phase 2 完了条件**: 単独 LB **9-11** + OOF が stacking に使える形式で保存済み

### Phase 3 — Day 22-42 (3 週): 案 C (Geometry-Physics Hybrid)

- Plane fit baseline 実装 (`numpy.linalg.lstsq` で平面係数推定)
- DTW alignment (`dtaidistance` で typewell GR と horizontal GR を warping)
- CUSUM fault detection (`statsmodels` or 自前実装)
- 1D U-Net (PyTorch, encoder-decoder + skip) を residual 予測に当てる
- Self-supervised pretrain on FORCE 2020 wells (random window mask → reconstruct)
- 自宅 GPU で学習、Kaggle Notebook には推論コードのみアップ + model artifact 経由
- **Phase 3 完了条件**: 単独 LB **8-10** + Phase 2 OOF と blend して LB が動くこと確認

### Phase 4 — Day 43-70 (4 週): 案 B (Sequence Transformer) 主力

- PatchTST + Cross-Attention 自前実装 (PyTorch)
- 自宅 GPU + Colab Pro+ で長時間 pretrain (FORCE 2020 / VOLVE / Brendon Hall facies dataset で pretext pretrain → ROGII データで fine-tune)
- Multi-task: TVT (主) + ΔTVT (aux) + Geology label (aux)
- Masked TVT pretext (15% random mask)
- **15-seed × 5-fold = 75 model** weighted MEDIAN ensemble
- ONNX export して Kaggle Notebook で GPU 推論、9hr 内に収まることを `time.perf_counter()` 計測で確認
- **Phase 4 完了条件**: 単独 LB **6-9** + 推論 runtime ≤ 7 hr (バッファ込み)

### Phase 5 — Day 71-83 (2 週): Stacking & Post-Process

- A + B + C の OOF を meta-learner (Ridge / LightGBM) で blend
- weighted MEDIAN ensemble (mean は使わない) を併用試行
- Physics post-process:
  - typewell Geology label 境界に TVT を snap (近傍境界より dist が小なら)
  - dip continuity 制約 (隣接 ft で TVT 変化が物理的に妥当か)
  - PID-style smoothing (high-frequency noise 除去)
- pseudo-labeling 1 round (high-confidence test pred を train に追加して案 A/C のみ再学習; 案 B は重いので skip)
- **Phase 5 完了条件**: 単独 LB **5-8** (1 位射程到達)

### Phase 6 — Day 84-88 (5 日): Final Submission Selection

- CV-LB diff 検証で robust な 2 final submissions を選ぶ:
  - **Sub 1**: CV best (overfit 最小、private LB 安全)
  - **Sub 2**: ensemble final (上限狙い、public LB best)
- Kaggle Notebook 提出後 monitor + 不調なら緊急修正
- **Phase 6 完了条件**: 2 sub を確定し submit 済み

## リスクと対策

1. **データ仕様の誤読** (例: ANCC/EGFDL 等が test で NaN だった、typewell が test で与えられない)
   - 対策: Phase 1 EDA で実 test ファイルを必ず眺めて確定。EDA report ができるまで実装着手禁止
   - 兆候: 案 B/C の前提が崩れる
   - 対応: 前提が崩れたら案を削除し、案 A の追加 feature engineering で時間を埋める

2. **9hr Notebook 推論制約** (案 B の 75 model を Kaggle 内で全部走らせると間に合わない)
   - 対策: Phase 4 で推論時間を必ず計測。間に合わない場合は seed 数を減らす / ONNX + half precision / single-fold fallback

3. **Internet disabled 制約** (外部 dataset/model の取り込み)
   - 対策: Phase 4 で model artifact upload pipeline を整備。FORCE 2020 dataset / VOLVE / Brendon Hall facies は private dataset 化

4. **公開 LB に overfit** (LB は part of test, private LB が真の評価)
   - 対策: GroupKFold by well_id で robust な CV。CV-LB diff > 0.5 ft なら overfit 警戒
   - Sub 確定時は CV best と LB best の **両方** を提出

5. **Sequence Transformer overfit** (200 wells は深層学習には少ない)
   - 対策: FORCE 2020 (98 train wells) / VOLVE で pretext pretrain。data augmentation (time shift, GR mixup, well drop) を厚く
   - 兆候: 案 B 単独 LB が想定 (6-9) より悪化、案 A < 案 B となる
   - 対応: pretrain dataset 拡大 / patch サイズ調整 / 案 B を ensemble 重み低めで使用

6. **時間配分が偏って Phase 5 で stack が間に合わない**
   - 対策: Phase 2 終了時点で **既に LB に 1 sub が乗っている状態** を作る (リスク先送り防止)
   - Phase 3-4 が遅れたら案 B/C のいずれかを削って Phase 5 を確実に確保

7. **TVT_input の使い方を誤る** (過剰に依存して mask region で破綻)
   - 対策: visible / hidden 区間を model 入力で **明示 mask channel** として渡す。training 時にも random window mask augmentation でモデルに mask region への耐性を学ばせる

## 検証方法 (機械的)

- 各 phase 完了時に `make submit M="<exp> CV=<n.nn>"` で Kaggle 提出 → public LB を `docs/dev/leaderboard.dense.md` に記録
- CV-LB diff を毎回算出 → 0.5 ft 以上ズレたら overfit 警戒、`docs/dev/cv-lb-diff.md` に記録
- 最終 2 sub は **必ず両方 commit/submit してから** sub 確定 (Kaggle 規定で 2 sub 選択可)
- Phase 4 の Sequence Transformer は推論時間を `time.perf_counter()` で計測 (9hr - 安全バッファ 2hr = 7hr 以内)

## 重要な reuse / 既存資産

- `~/projects/kaggle/_shared/kagglib/src/kagglib/` (LGB/XGB/CB baseline `baselines.py`, CV `cv.py`, stacking `stacking.py`, pseudo-label `pseudo.py`, ensemble `ensemble.py`, fe `fe.py`, tracking `tracking.py`) — 既存。**そのまま import**
- `~/projects/kaggle/new-competition.sh` (bootstrap script) — Phase 0 で `timeseries` domain で起動するが、tabular template の baseline lib も併用 (案 A の GBM stack 用)
- `~/projects/kaggle/_template-timeseries/` (sequence model 用のディレクトリ骨格)
- 過去の `~/projects/kaggle/orbit-wars/docs/` の構造 (`docs/research/`, `docs/strategy/`, `docs/dev/`) を踏襲

## 重要ファイル (将来のパス)

- `/home/yusuke_kaya/projects/kaggle/ROGII/notebooks/00_eda.ipynb` — Phase 1 EDA
- `/home/yusuke_kaya/projects/kaggle/ROGII/docs/research/data-spec.dense.md` — Phase 1 確定データ仕様 (日本語)
- `/home/yusuke_kaya/projects/kaggle/ROGII/docs/research/past-comps.dense.md` — 類似コンペ集約 (Agent 2 結果転記)
- `/home/yusuke_kaya/projects/kaggle/ROGII/docs/research/method-catalog.dense.md` — 技術手法 35 件 (Agent 3 結果転記)
- `/home/yusuke_kaya/projects/kaggle/ROGII/docs/strategy/winning-strategy.dense.md` — この plan の repo 内詳細版コピー
- `/home/yusuke_kaya/projects/kaggle/ROGII/docs/dev/leaderboard.dense.md` — exp 毎の LB 記録
- `/home/yusuke_kaya/projects/kaggle/ROGII/docs/dev/cv-lb-diff.md` — CV と LB の乖離トラッキング
- `/home/yusuke_kaya/projects/kaggle/ROGII/src/rogii/features.py` — feature engineering
- `/home/yusuke_kaya/projects/kaggle/ROGII/src/rogii/models/gbm.py` — 案 A
- `/home/yusuke_kaya/projects/kaggle/ROGII/src/rogii/models/transformer.py` — 案 B
- `/home/yusuke_kaya/projects/kaggle/ROGII/src/rogii/models/hybrid.py` — 案 C
- `/home/yusuke_kaya/projects/kaggle/ROGII/configs/exp{NNN}.yaml` — 実験設定
- `/home/yusuke_kaya/projects/kaggle/ROGII/outputs/oof/exp{NNN}-*.parquet` — OOF predictions
- `/home/yusuke_kaya/projects/kaggle/ROGII/submissions/exp{NNN}.csv` — Kaggle submission file
- `/home/yusuke_kaya/projects/kaggle/ROGII/experiments/exp{NNN}/notes.md` — 実験毎メモ (日本語)

## 1 位を取るための差別化要因 (= 我々の独自 edge)

1. **典型 baseline の上に Geometry-Physics Hybrid (案 C) を積む** — public ノートブックは案 A の延長で頭打ちになっている公算が高い。物理 prior + DTW + CUSUM fault detection は competitor の盲点
2. **Cross-Attention to typewell (案 B)** — typewell との相互参照を明示的にアーキテクチャに焼き込む。SPWLA 2023 1 位の MDWD + GRU の延長線上で、PatchTST + cross-attention 化は 2026 時点の新規組合わせ
3. **Ventilator 流の weighted MEDIAN ensemble** — mean ではなく median。15-seed × 5-fold の seed 多様性を最大限活用
4. **Pretext pretrain on FORCE 2020 + VOLVE** — ROGII 単独では 200 wells で深層学習には不足。外部 well-log dataset で pretext pretrain することで案 B の deep model を成立させる
5. **Multi-task head** — TVT 主 + ΔTVT + Geology label aux で representation を richer にする (Ventilator 1 位流)

## 開始前の必要決定 (= ユーザー承認待ち)

このファイル (= 戦略全体) を承認いただいたら、以下を順次実行:

- **Phase 0 (今日)**: repo clone + bootstrap (1-2 時間で完了)
- **Phase 1 (Day 2-7)**: EDA & data-spec.dense.md 提出 → ユーザーレビュー → Phase 2 GO 判定
- 各 phase 完了時に LB 結果を報告し、次 phase に進むか案削除/差替えするかを判断する re-plan checkpoint を置く

## 次の行動 (= ExitPlanMode 後の最初のコマンド予定)

```bash
gh repo clone ykaya-jp/ROGII /home/yusuke_kaya/projects/kaggle/ROGII
cd /home/yusuke_kaya/projects/kaggle/ROGII
# _template-timeseries の構造を最小限 merge
# data download
uv run kaggle competitions download -c rogii-wellbore-geology-prediction -p data/raw/ --unzip
uv sync --extra dev
```

ただし `gh repo clone` の認証状況、`kaggle competitions download` のルール accept 状況は実環境依存なので、Phase 0 開始時にユーザーへ確認する。
