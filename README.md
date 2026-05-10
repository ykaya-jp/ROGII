# ROGII — Wellbore Geology Prediction

Kaggle competition: <https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction>

水平井 lateral 区間で 1ft 毎に **TVT (True Vertical Thickness, 地層内の垂直位置)** を回帰予測する masked sequence regression。
評価指標: **RMSE** on hidden TVT zone。

## クイックスタート

```bash
make install     # uv sync (kagglib editable, lightgbm/xgboost/catboost/darts/sktime/tsfresh/torch 等)
make download    # data/raw/ に DL
make eda         # notebooks/00_eda.ipynb 起動
make baseline    # 4 baseline (Ridge/LGB/XGB/CB) × 5-fold CV
make submit M="lgb baseline cv=0.000"
```

## 戦略 docs

- 優勝戦略 (主): [`docs/strategy/winning-strategy.dense.md`](docs/strategy/winning-strategy.dense.md)
- データ仕様 (Phase 0 確認分): [`docs/research/data-spec.dense.md`](docs/research/data-spec.dense.md)
- 過去類似コンペ集約: `docs/research/past-comps.dense.md` (Phase 1 で書く)
- 技術手法カタログ: `docs/research/method-catalog.dense.md` (Phase 1 で書く)
- LB tracking: `docs/dev/leaderboard.dense.md` (Phase 2 から)
- CV-LB diff tracking: `docs/dev/cv-lb-diff.md` (Phase 2 から)

## ベンチマーク (Phase 1 リサーチ時点)

| ノートブック | 種別 | スコア |
|---|---|---|
| `romantamrazov/rogii-super-baseline-lb` | LightGBM 系 baseline | LB **12.602** |
| `cdeotte/nn-starter-cv-15-5` | Neural Net starter (Chris Deotte 氏) | CV **15.5** |

## 制約 (Code Competition)

- Notebook 提出 (`submission.csv`)
- CPU/GPU いずれも ≤ 9 hr runtime
- **Internet disabled** (外部 dataset / pretrained は private dataset としてアップロード)
- 外部データ・pretrained model **OK**

## Public LB tracking

| exp | model | CV | public LB | diff | note |
|---|---|---|---|---|---|
| 001 | LGB baseline (TBD) | - | - | - | Phase 2 で記入 |

## メモ

- 重要ファイルパス・実装ロードマップ・リスクは [`docs/strategy/winning-strategy.dense.md`](docs/strategy/winning-strategy.dense.md) 参照
- Plan の harness 元は `~/.claude/plans/rogii-wellbore-geology-modular-shannon.md` (= `docs/strategy/winning-strategy.dense.md` と同期)
