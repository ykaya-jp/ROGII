# exp005 — cache + karnakbaev artifacts blend

> **目的**: LB 9.x 帯 (Gold 圏) 直接到達。Approach B = `karnakbaevarthur/rogii-code-helper-dataset` の事前訓練 LGB×3 + XGB + CatBoost artifacts をロードして infer + 自前計算 test features で predict + ridge/ensemble blend + post-process。
> **ベース branch**: `feat/phase-3-exp005-cache-blend` (from `feat/phase-2-exp003`)
> **公開元**: https://www.kaggle.com/code/karnakbaevarthur/top-2-rank-10-784-physics-informed-baseline (LB 10.784)

## 0. 状況 (2026-05-10 開始時)

- 我々の現状 LB: exp002 = **14.695** / exp003 = PENDING (期待 11-12)
- 公開 LB Gold 圏: 9.256 - 9.919 (Top 20)
- 必要改善: -5.5 ft (vs exp002)
- submit 残数: 1 日 5 回

## 1. 採った Approach

### 候補 (構造原理)

| Approach | 概要 | 期待 LB | リスク |
|---|---|---|---|
| **A: thbdh5765 cache + 自前 LGB stack** | 公開済 train features cache (= PF/Beam/SC/tvt formula 全部入り 173 cols)、test 側は自前計算、自前 LGB×3 + CB stack | 10-11 帯 | feature 計算 ロジック自前移植が必要 (= bug リスク) |
| **B: karnakbaev artifacts blend (採用)** | 事前訓練 LGB×3 + XGB + CB + Ridge meta + postproc params をロード、test features を kernel 内で自前計算 (= 公開 1978-line script そのまま使える) | **10.784 → 9-10 帯** | path 修正のみ、低リスク |
| C: TabICL 投入 | TabICL 2.1.1 wheel + ckpt を attach、base model に追加 | needless090 LB 10.081 | TabICL train + infer の確立済 path がない |

### 選定理由

**B**:
1. test features を自前計算する script (= karnakbaev 公開 kernel) が既に存在、infer mode で動作確認済 (作者 LB 10.784)
2. artifacts (LGB×3 .txt + XGB .json + CB .cbm + Ridge .pkl + postproc.json) が `karnakbaevarthur/rogii-code-helper-dataset` で公開、license `apache-2.0`
3. self-contained で 8 hr CPU 内で完走可能 (= 作者 kernel 実績)
4. 失敗してもこちらの local CV/EDA 資産にダメージなし、独立 branch で実験できる

## 2. host dataset 確認

| Slug | サイズ | 中身 | 使い方 | License |
|---|---|---|---|---|
| `karnakbaevarthur/rogii-code-helper-dataset` | 1.36 GB | LGB×3 (.txt) / XGB (.json) / CB (.cbm) / Ridge (.pkl) / features.json (158 features) / postproc_params.json / ensemble_weights.json + train_df.parquet (1.47 GB cache) + test_df.parquet (6.8 MB) + OOF.parquet (149 MB) | kernel `dataset_sources` に追加して `/kaggle/input/rogii-code-helper-dataset/artefacts/` から load | apache-2.0 |
| `thbdh5765/rogii-v1-train-cache` | 1.25 GB | train_df.pkl (=2.5 GB raw) + schema (173 cols) + preview 1000 rows. **train wells のみ**、test 側は自前計算必須 | Approach A 用 (= 自前 LGB stack)。今回は未使用 (Approach B 採用) | CC0-1.0 |

### thbdh5765 schema 主要 features (173 cols, 173 features + well + id - 174 total)

```
last_known_tvt
pf_ancc / pf_ancc_std / pf_ancc_d / pf_z / pf_z_d / pf_vs_z   # Particle Filter 2-channel (ANCC + Z)
beam_cons_d / beam_loose_d / beam_vcons_d / beam_sm5_d / beam_vloose_d / beam_mid_d / beam_stiff_d  # Beam 7 configs
beam_mean_d / beam_std_d / beam_med_d                          # Beam aggregation
sc8_d / sc8_score / sc15_d / sc15_score / sc25_d / sc25_score  # Multi-scale Self-NCC (h=8/15/25)
sc_cons_d / sc_trust / hyb_d / signal_std / signal_mean_d
tvtF_<formation>_d × 6 (ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA)   # tvt formula features
tvtFw_<formation>_d × 6                                            # weighted version
bw_<formation> × 6 / bww_<formation> × 6                           # beam-weighted typewell stats
frm_rmse_<formation> × 6 / form_mean_d / form_std_d / form_rng_d   # plane-fit RMSE
knn_d / dense_ancc / dense_std / dense_dist / dense_rmse / dense_bias
tvt_dense_d / tvt_densew_d / tvt_d50_d
pf_vs_form / pf_vs_dense / form_vs_dense / beam_vs_form / sc_vs_beam
cal_a / cal_b / pfx_rmse / known_len / eval_len
slp_all / slp_50 / slp_z / pfx_gr_slope / slp_b_d_all / slp_b_d_50
ktvt_range / ktvt_std / md_since / frac / frac2 / sqrt_frac / z / dx / dy / dz / dxy
tda-120 ... tda120 (13 anchors at TVT - last_known offsets)
tdbc-40 ... tdbc40 (11 anchors at beam_cons reference)
tdsc-30 ... tdsc30 (11 anchors at SC reference)
tdpf-30 ... tdpf30 (11 anchors at PF reference)
tw_range / tw_gr_mean / grm5/21/51/101 + grs5/21/51/101  # GR mean/std rolling
glag1/5/15/30 + glead1/5/15/30                            # GR lag/lead
target  # = tvt - last_known_tvt
```

**結論**: thbdh5765 は train 側に **PF (Particle Filter) MAP estimate** + **Beam search 7 configs** + **multi-scale Self-NCC** + **plane-fit per formation** + **dense IDW imputer** + **GR rolling stats** + **TVT-anchored offset features** すべて入っている。これは `romantamrazov/rogii-super-solution-lb-top-3` と同様の features set。

### karnakbaev features.json (158 features)

thbdh5765 とほぼ同じカテゴリだが **2-channel PF (Z + ANCC) のみ、Beam 5 configs (cons / loose / vcons / sm5 / vloose)、SC 1-window (h=15)**。やや小さいが modeling 完了済が強み。

## 3. 計画 (TODO)

- [x] branch `feat/phase-3-exp005-cache-blend` 作成
- [x] thbdh5765/karnakbaev datasets metadata + 小ファイル DL → 中身確認
- [x] karnakbaev kernel script (`physics-informed-baseline.py`, 1978 行) のロジック把握
- [ ] kernel script (`exp005_cache_blend.py`) 作成 = karnakbaev kernel base + path 修正 + (任意) ensemble weight 微調整
- [ ] kernel-metadata.json 作成 (dataset_sources に `karnakbaevarthur/rogii-code-helper-dataset`)
- [ ] ローカル smoke (= 1 well で test features build + predict 確認)
- [ ] kernel push → run → wait → submission.csv 取得
- [ ] submit + LB 確認
- [ ] 達成 LB を本ファイル §5 に記載

## 4. 失敗回避ログ

- **path 問題**: karnakbaev kernel は `ARTEFACT_DIR = /kaggle/input/datasets/karnakbaevarthur/rogii-code-helper-dataset/artefacts` だが、Kaggle dataset attach の標準 path は `/kaggle/input/<slug>/` = `/kaggle/input/rogii-code-helper-dataset/`。kernel script で両方を試して fallback する path resolver を入れる
- **`use_ridge` 値**: kernel 1931 行で `pp_params["use_ridge"]=true` を読むが 1953 行で `use_ridge = False` に強制 override → ensemble_nm (= ensemble_weights.json {lgb2=0.43, xgb=0.29, cb=0.28}) を使う。これは作者の最終形。我々もそれに従う
- **runtime cap**: 8 hr safety、test features build (~30 min for 3 wells × 14k rows) + infer (1 min) + postproc (1 min) で 30-40 分予測。十分余裕
- **submit quota**: 5/day。今日 (5/10) は exp003 で 1 回使用済 (= PENDING)。残 4 回。慎重に use
- **license**: `apache-2.0` (karnakbaev) — commit/kernel docstring に "based on karnakbaevarthur/physics-informed-baseline (apache-2.0)" 必須
- **rebuild test features only**: `MODE = "infer"` 内で `test_df = build_dataset(TEST_DIR, is_train=False)` で 必ず live test を再計算 (= cached test_df.parquet を使わない、private rerun 対応)

## 5. 結果

(TBD — submit 後に追記)

## 6. References

- karnakbaev kernel (LB 10.784, public): https://www.kaggle.com/code/karnakbaevarthur/top-2-rank-10-784-physics-informed-baseline
- karnakbaev artefacts dataset: https://www.kaggle.com/datasets/karnakbaevarthur/rogii-code-helper-dataset
- thbdh5765 train cache: https://www.kaggle.com/datasets/thbdh5765/rogii-v1-train-cache
- romantamrazov super solution (Top 3 LB ~10.1, MIT): https://www.kaggle.com/code/romantamrazov/rogii-super-solution-lb-top-3
- 我々の docs/research/host-datasets.dense.md (host dataset 一覧)
- 我々の docs/research/top3-distill.dense.md (Top 3 6 要素分析)
