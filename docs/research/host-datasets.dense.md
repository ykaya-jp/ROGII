# Host 公開 dataset 一覧と利用計画 (2026-05-10 検出)

> `kaggle datasets list -s rogii` で公開された Kaggle dataset を 1 回スキャン。コンペ進行中で参加者が公開した artifact / mirror / blend 用 dataset が多数。
> 過去の lesson (`~/.claude/CLAUDE.md` 2026-05-10「orbit-wars host dataset 検証 skip 失敗」) に従い、Phase 1 のうちに棚卸しする。

## A. 流用価値 ★★★ (= 最優先)

### A1. `karnakbaevarthur/rogii-code-helper-dataset` (1.36 GB, 2026-05-09 update, 51 DL)

LB 10.784 (`physics-informed-baseline`) の **完成 model artifacts 一式**:

| ファイル | サイズ | 用途 |
|---|---|---|
| `artefacts/final_lgb_lgb0.txt` / `lgb1.txt` / `lgb2.txt` | 5-7 MB ×3 | LightGBM 3 model |
| `artefacts/final_cb.cbm` | 3.3 MB | CatBoost final |
| `artefacts/final_xgb.json` | 7.6 MB | XGBoost final |
| `artefacts/ridge_meta.pkl` + `ridge_weights.json` | 456 B + 152 B | Ridge meta-learner |
| `artefacts/ensemble_weights.json` | 119 B | Stacking weights |
| `artefacts/postproc_params.json` | 70 B | Post-process params |
| `artefacts/oof_predictions.parquet` | 149 MB | OOF (5-fold) |
| `artefacts/feature_importance.parquet` | 4.4 KB | feature ranking |
| `artefacts/features.json` | 2.3 KB | feature list (再現に必須) |
| `artefacts/train_df.parquet` | 1.47 GB | 前処理済 train |
| `artefacts/test_df.parquet` | 6.8 MB | 前処理済 test |
| `catboost_info/learn_error.tsv` | 12 KB | CB 学習曲線 |

利用計画: **Phase 3 Track 3b で private dataset 化 → 推論 kernel から model load → 自前 OOF と blend**。
期待: 自前 LB 11.x × karnakbaev 10.784 → blend **LB 10.3-10.5**。
リスク: ライセンス確認必須 (karnakbaev kernel が `physics-informed-baseline` で公開している前提)。

### A2. `thermostatic/rogii-tabicl-v2-public-assets` (107 MB, 2026-05-06)

| ファイル | サイズ | 用途 |
|---|---|---|
| `tabicl-2.1.1-py3-none-any.whl` | 253 KB | TabICL package wheel |
| `tabicl-regressor-v2-20260212.ckpt` | 114 MB | TabICL pretrained regressor |

利用計画: **Internet disabled な Kaggle Notebook で TabICL を動かす唯一の手段**。
needless090 LB 10.081 stack の base model 再現に必須。
Phase 3 Track 3b で private dataset 化 → kernel に attach → `pip install /kaggle/input/.../tabicl-2.1.1-py3-none-any.whl` で導入。

### A3. `needless090/rogii-tabicl-mirror` (107 MB)

A2 の mirror (同サイズ)。冗長化用、もしくは A2 がアクセス不能なときのフォールバック。

### A4. `buchananliang/rogii-karnak-top2-public-artefacts` (6.6 MB, 2026-05-08)

karnak (≠ karnakbaev) Top2 の artifact。詳細未確認だが軽量。Phase 3 で内容調査して blend 候補に。

## B. 流用価値 ★★ (要中身調査)

### B1. `enisteper1/rogii-train-dataset` (937 MB, 2026-05-09)

train data の整理版 (推測)。元 data/raw/ と diff 取って 0 ならスルー、整形が含まれていれば使う。

### B2. `alfaxadeyembe/rogii-lgbm-model-artifacts` (1.2 MB)

LGBM model のみ。軽量だが score 不明。

### B3. `iathar/rogii-wellbore-models` (2.2 MB)

詳細不明。

### B4. `geeknik/rogii-blend-public` / `innerf1re/rogii-blend-srcs-public`

blend source (CSV?) の可能性。Phase 5 stacking の OOF source として使えるかも。

### B5. `akshankrithick/rogii-gold-top10-direction-source` (138 KB)

「Gold Top10 direction source」 = どの kernel が gold medal 圏かの方向性メモ？小さいので 1 度開く価値あり。

## C. 流用価値 ★ (背景理解 only)

- `nina2025/rogii-07` / `rogii-03`, `foysalemonshanto/rogii-dataset`, `prantikchandra/my-rogii-set`, `niazmahmud0201/rogiidataset`, `alexanderkwesi/seuns-rogii-welbourne` 等は usabilityRating 低、おそらく学習用に再アップロードした個人 mirror。スキップ。

## 取り込み手順 (Phase 3 Track 3b 用)

1. `make data-host` (新規追加予定) で A1 / A2 / A3 / A4 を `data/external/host/` に DL
   - `kaggle datasets download -d karnakbaevarthur/rogii-code-helper-dataset -p data/external/host/karnakbaev/ --unzip`
   - `kaggle datasets download -d thermostatic/rogii-tabicl-v2-public-assets -p data/external/host/tabicl/ --unzip`
   - `kaggle datasets download -d buchananliang/rogii-karnak-top2-public-artefacts -p data/external/host/karnak_top2/ --unzip`
2. ローカルで artifact を `notebooks/03_karnakbaev_replay.ipynb` で開いて feature_importance / ensemble_weights / postproc_params の中身を読み解く
3. `LICENSE` ファイルが含まれているか確認 (含まれていない場合は kernel 説明欄を確認、Apache 2.0 default 想定)
4. Kaggle 上での参照は **private dataset 化せず、`competition_sources` に dataset を追加**するだけで OK (kernel-metadata.json の `dataset_sources` に追記)
5. Kaggle kernel 内で `from pathlib import Path; KARNAKBAEV = Path("/kaggle/input/rogii-code-helper-dataset")` でパス解決

## 注意 (lesson 由来)

- `~/.claude/CLAUDE.md` 2026-05-10 lesson: 「Kaggle agent comp で host dataset 検証 skip → 17% 勝率失敗」
- ROGII は agent comp ではないが、**host 公開 artifact を Phase 1 で見落とした場合の機会損失**は同じ。今回 Phase 1.6 完了時点で確認できた = 危機回避。
- 今後 ROGII 以外の Kaggle コンペでも `kaggle datasets list -s <slug>` を Phase 1 開始の最初のコマンドにすることを `kaggle-onboard` skill が要求済み (precondition checklist に `kaggle datasets list -s <slug> executed`)。
