# exp003 — tysig (xcorr + multi-scale SC + WLS) 実験ノート

## 目的

exp002 (CV TVT hidden RMSE = 13.82 ft) の上に、戦略 doc Phase 1.6 §B 優先順 1-3 を全部積んで **CV 12.5 ft 以下 / LB 12 切り** を狙う。

参考 plan: [`docs/strategy/winning-strategy.dense.md`](../../docs/strategy/winning-strategy.dense.md) §1.6 §B。

## 追加した特徴量と各々の根拠

### 1. `xcorr_tvt_offsets` — tasmim 流 (4 features)

各行で `last_known_TVT` 周辺の `[-100, +100] ft` の範囲を `step=2.0` で 101 オフセット sweep し、観測 GR の局所窓 (window=30) と typewell GR をオフセット位置で interpolate した値との **Pearson 相関を最大化** するオフセットを探す。

| 特徴量 | 説明 |
|---|---|
| `xcorr_delta` | 最良オフセット (signed, ±100 ft 範囲) |
| `xcorr_corr`  | 最良相関 (Pearson, [-1, 1] にクリップ) |
| `xcorr_absdiff` | 最良 offset での `|obs - tw_gr|.mean(axis=1)` (低品質行は 999 sentinel) |
| `xcorr_tvt_off` | `last_known_TVT + xcorr_delta` (推定 TVT) |

実装: `src/rogii/tysig.py:xcorr_tvt_offsets`。tasmim の per-row Python loop を **vectorised matmul** に書き換えた (~50× 高速化, 5092k 行で ~17 min)。出典: `_research_kernels/tasmim__lb-11-068.../...py:236-263`。

### 2. `multi_scale_sc` — romantamrazov 流 (12 features)

Self-NCC を **3 つの window 半径 (8, 15, 25)** で並列実行し、3 つの独立した TVT 推定を生成。各 scale で `(raw_tvt, score, d=raw_tvt - last_known_TVT)`、加えて 3-scale の **median / std 集約** も追加。

| 特徴量 | 説明 |
|---|---|
| `sc{8,15,25}_raw_tvt` | 各 scale の Self-NCC 最良 TVT |
| `sc{8,15,25}_score` | 各 scale の NCC スコア |
| `sc{8,15,25}_d` | 上記 - last_known_TVT |
| `sc_med_tvt`, `sc_std_tvt`, `sc_med_d` | 3 scale 集約 |

実装: `src/rogii/tysig.py:multi_scale_sc`。出典: `_research_kernels/romantamrazov__rogii-super-solution.../...py:188-209`。

### 3. `wls_b_well` — recent-weighted b_well (formation あたり 3 features × 6 = 18)

formation 毎の `b_well` を **recent-weighted exponential decay (`decay=0.02`)** で推定。tail (現在地に近い visible row) を重く重みづけることで、along-lateral で `b_well` が drift するケースに対応。

| 特徴量 | 説明 |
|---|---|
| `tvtFw_{F}` | `-Z + f_imp_F + b_F_wls` (新 baseline TVT) |
| `bww_{F}`   | `b_F_wls` 値 (well 単位 broadcast) |
| `tvtFw_{F}_d` | `tvtFw_F - last_known_TVT` |

formation `F` ∈ {ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA}。

実装: `src/rogii/tysig.py:wls_b_well` (既存) + `src/rogii/features.py:_per_well_features` formation block 拡張。出典: `_research_kernels/romantamrazov__rogii-super-solution.../...py:181-186`。

### 4. `gr_detrend_resid` — GR linear detrending (1 feature)

GR を MD で線形回帰し残差を取る。well 固有の trend (mud-cake / casing 影響) を除いて anomaly のみを残すことで、typewell との shape matching の特徴抽出力を上げる。実装は既に exp002 にもあったが、`feature_columns(typewell_avail=True)` 側に明示的に lift 済み。

### 5. 既存維持 (exp002 から踏襲)

- 6-formation FormationPlaneKNN imputer (K=10 centroid plane fit)
- `tvt_formula = -Z + ANCC_imputed + b_well` (median b_well, last-50 b_well)
- target = `TVT - last_known_TVT` (residual)
- Beam search 5 configs (cons / loose / vcons / sm5 / vloose)
- Self-NCC anchored tw_diff (33 features = 3 anchors × 11 offsets)
- affine GR cal, prefix RMSE, GR rolling stats, trajectory derivatives

## 実装と CV 結果

(後続セクションは train_v3 完了後に追記)

### feature 数

| バージョン | features 数 | 増加 |
|---|---|---|
| exp002 (`FEATURE_COLS`) | 76 | baseline |
| exp003 (`FEATURE_COLS_V3`) | 169 | +93 |

新 93 features の内訳:
- formation × WLS: 18 (`tvtFw_*`, `bww_*`, `tvtFw_*_d` × 6 formations)
- Multi-scale SC: 12 (3 × {raw_tvt, score, d} + 3 集約)
- xcorr (visible-prefix + tasmim offsets): 7 (`xcorr_tvt`, `xcorr_score`, `xcorr_d`, `xcorr_delta`, `xcorr_corr`, `xcorr_absdiff`, `xcorr_tvt_off`)
- Beam (5 configs × 2) + 集約 5 = 15
- tw_diff (3 anchors × 11 offsets) = 33
- その他 (a_cal, b_cal, gr_detrend, sc_trust, pfx_rmse, tw_gr_*, tw_tvt_range): 8

### CV TVT hidden RMSE

(`make build_v3 && uv run python -m rogii.train_v3` 完了後に追記)

## 失敗回避ログ

- **`xcorr_tvt_offsets` を per-row Python loop で書いてはいけない**:
  773 wells × ~6,500 行 × 101 offsets × 30-row window → 5M × 3000 ops = 15B ops。Python loop だと 27 min/train。**vectorised matmul** にしたら ~17 min。Kaggle 9hr 制約を満たすにはこのレベルの vectorise が必須。
- **WLS b_well を formation block に勝手に入れない**: exp002 の `FEATURE_COLS` に WLS が混じると、既存の `train_features.parquet` が cache 不整合で失敗する。`feature_columns(wls_avail=...)` フラグで切り分け。
- **`out` dict の key を well 毎にバラつかせない**: `add_features` の `chunks[0].keys()` を全 chunk concat に使う実装。typewell が無い well では typewell-derived key が抜けるため、結果がぐちゃぐちゃになる。**現状 typewell カバレッジ 98.3%** で 13 wells が edge case。要モニター。
- **multi-scale SC の score 出力範囲は [-1, 1]**: スモークテストで `(sc_score >= -1.0 - 1e-3).all()` を確認したが、非常に短い well では NCC 計算が degenerate になる場合がある (`std < 1e-6`)。`+1e-6` regularization で吸収。

## 残課題 / 次に試したい変種

- **Particle Filter** (romantamrazov の `run_pf_ancc` / `run_pf_z`): exp004 候補。Beam Search より滑らか / 不確実性込みで TVT 推定。500 particles × N=2 の vectorized NumPy で 5 sec/well。
- **Numba JIT** でさらに beam_search を加速 (~3-5×)。Kaggle 9hr 安全マージン拡大用。
- **OOF を ridge / LGB で stack** (exp005): exp002 + exp003 + 将来の exp004 を CV 上で blend して LB 10 帯入りを目指す。
- **Geology label boundary snap** (exp006): typewell の Geology label 境界に近い行を post-processing で snap し、layer 内の TVT discontinuity を矯正。
- **formation imputer KDTree を XY 以外も使う**: 現状 X/Y 中央値だけだが、Z の median や lateral 長で K-NN したほうが extrapolation が安定する可能性。
