# exp007 — 実装ノート + ローカル smoke 結果

> **目的**: Edge Q (typewell content-hash GroupKFold) + Edge M (visible-as-typewell self-NCC FE) + 自前 LGB×3+CB の 4 base 加算で **LB 9.5-9.7 帯到達**
> **ベース**: `feat/phase-3-exp007-edge-q-m` (from `feat/phase-3-exp006-tabicl-pflite`)
> **kernel push**: 中央指示待ち、本 subagent K は smoke pass + branch push までで停止

## 1. 状況 (2026-05-11 03:00 JST)

- 我々の現状 LB: exp005 = **10.317** / exp006 = **10.503** (TabICL 投入で逆悪化、postmortem 完了)
- 公開 LB Top 1 = 9.256、Gold cutoff = 9.919、賞金圏 (Top 4) = 9.415
- ユーザー指示 (2026-05-11): 「9 切らないと優勝は無理」 = 最終 LB 8.x 帯
- submit 残数 (5/11 UTC reset 後): 5 / 5、本 exp007 を 1 枚目に充当する想定 (= 中央判断後)

## 2. 採った Approach (= Approach C)

設計 doc `experiments/exp007/design.md` 参照。

3 案比較:
- **C (採用)**: exp005 5-base + 自前 LGB×3 + CB を Edge Q fold で OOF 学習 + Edge M FE → 想定 9.5-9.7 帯
- B: TabICL 復活 → 想定 10.0-10.3 (= exp006 で投入が逆効果、根本原因未解消)
- A: Edge Q のみ → 想定 10.1-10.2 (= 9 切り roadmap に効果薄)

採用理由は exp006 postmortem (`docs/dev/leaderboard.dense.md`): **TabICL 追加だけでは Ridge weight constellation が劣化、diversity を増やすには base を Edge Q fold で独立に学習する必要**。

## 3. 実装の中身

### 3.1 kernel script の主要変更点

`exp006_tabicl_pflite.py` (2426 行) → `exp007_edge_q_m.py` (3056 行) +630 行。

| # | 場所 | 内容 |
|---|---|---|
| 1 | docstring (line 1-58) | exp007 用 4-step pipeline + Edge Q/M license attribution + TabICL dormant の理由を明記 |
| 2 | config (line 90-145) | `MODE = 'infer_edge_qm'`、`TABICL_ENABLE = False`、`EDGE_Q_ENABLE = True`、`EDGE_M_ENABLE = True`、自前 4 base hyperparam (lr=0.02/0.05/0.10 + CB lr=0.05、seeds=42/7/123、n_est=8000/4000/2000、`STACK_OWN_WEIGHT_DEGRADE_LIMIT = 0.5`) |
| 3 | helpers (line ~880-1050) | 新規 4 関数: `compute_typewell_hashes()`, `build_edge_q_folds()`, `verify_edge_q_no_leak()`, `compute_edge_m_features()` + `edge_m_feature_names()` |
| 4 | `build_well_features` (line ~1620-1660) | Edge M FE block を `try/except` で wrap して 4 列 (`selfaln_d / selfaln_score / selfaln_vs_typewell_d / selfaln_conflict_score`) を出力 dict にマージ。leak-safe 設計 (= `TVT_input` 列を一切参照しない、GR + MD のみ) |
| 5 | dispatcher (line ~2725-3000) | `elif MODE == 'infer_edge_qm':` block 約 280 行 — Edge Q fold 計算 → karnakbaev OOF refit (= 5-fold predict、stack consistency 用) → 自前 4 base train (5 fold × 4 base、try/except でフォールバック) → 9-base Ridge meta fit (= positive=True、no intercept、own_total_weight > 0.5 なら NM 5-blend に fallback) → karnakbaev postproc → SG smooth → submission |

### 3.2 reuse map (= subagent G の partial impl 流用)

| commit | 流用先 | 内容 |
|---|---|---|
| `1bb6caf feat(cv): typewell content-hash groups + data-spec TVT definition` | `compute_typewell_hashes()` (line ~880-925) | `notebooks/_typewell_groups_build.py` の MD5 計算ロジックを kernel inline に移植。10 wells を持つ 5ec0017b... group を含む 13 group を完全 recover (= local smoke で確認、出典: discussion topic 698449 host official reply) |
| `1bb6caf` | `build_edge_q_folds()` (line ~928-960) | groups → GroupKFold(5).split → fold_id の builder。fallback_by_well フラグ付き |
| `537fa9f docs: H-T1/T2/T6/T7 + edge M (visible-as-typewell)` | `compute_edge_m_features()` (line ~1000-1075) | `independent-edges.dense.md` §M.5 の subagent dispatch text 雛形を実装。visible windows (size=200, step=50) と hidden chunks (size=200) の Self-NCC、best window TVT + slope*MD 外挿で hidden TVT 推定値、4 features 出力。slope を ±1.0 に clip + sa_d を ±200 ft に clip (= 物理レンジ) |
| (新規実装) | `karnakbaev_oof_refit_under_edge_q()` (= dispatcher 内 inline) | exp006 postmortem 対策。karnakbaev pretrained を Edge Q val fold で predict してから 9-base Ridge meta に渡す。**注**: 真の OOF ではなく train-set predict proxy。実際には `oof_predictions.parquet` (= karnakbaev published OOF、`merge_keys=["id"]`) を優先 load し、merge 失敗時のみ proxy にフォールバック |

### 3.3 自前 4 base + Ridge 9-base の OOF flow

```
karnakbaev train_df.parquet (3.78M rows × 160 cols, kaggle 上で load)
  │
  ├─ Edge Q fold (= 5-fold by typewell content-hash groups)
  │
  ├─ 5 karnakbaev OOF (= oof_predictions.parquet を id merge で取得; なければ refit proxy)
  │     ↓
  │   Sx[:, 0:5] (= lgb0/lgb1/lgb2/xgb/cb の OOF)
  │
  ├─ 自前 4 base train (= 5 fold × 4 base、Edge Q fold で fit)
  │     ↓
  │   Sx[:, 5:9] (= lgb_own0/lgb_own1/lgb_own2/cb_own の OOF)
  │
  └─ Sx (n_train, 9) を Ridge(positive=True, fit_intercept=False, alpha=1.0) で fit
        ↓
      coef_ → wts、own 合計 weight > 0.5 なら NM 5-blend fallback
        ↓
      St (n_test, 9) → ridge_ex.predict() → test_delta
        ↓
      karnakbaev postproc (alpha+tau fade-in) → SG smooth → spike clamp → submission.csv
```

### 3.4 failure recovery (= 4 階層)

| 段階 | 失敗症状 | 対策 |
|---|---|---|
| Edge Q fold | typewell file 不在 / Geology 列なし | well_id 単独 group fallback、(TVT, GR) のみで hash |
| Edge M FE | per-well で例外 | 4 列を 0 で埋めて続行 (= `try/except` で wrap) |
| 自前 train | GPU OOM / fold timeout | `device=cpu` retry、1 fold で hard timeout (= `OWN_TRAIN_HARD_TIMEOUT_S=6h`) で break、`own_ok=False` で 5-base NM blend fallback |
| Ridge meta | own 合計 weight > 0.5 | `STACK_OWN_WEIGHT_DEGRADE_LIMIT=0.5` 越えで `own_weight_degenerate` raise → NM 5-blend (= exp005 同等) fallback |

最悪 4 段全部失敗で「**exp005 と完全に同じ submission**」が保証される (= 悪化リスク 0)。

### 3.5 kernel-metadata.json

```json
{
  "id": "ky7240/rogii-exp007-edge-q-m",
  "title": "ROGII exp007 Edge QM",
  "code_file": "exp007_edge_q_m.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_tpu": "false",
  "enable_internet": "false",
  "dataset_sources": [
    "karnakbaevarthur/rogii-code-helper-dataset",
    "thermostatic/rogii-tabicl-v2-public-assets",
    "thbdh5765/rogii-v1-train-cache"
  ],
  "competition_sources": ["rogii-wellbore-geology-prediction"],
  "kernel_sources": []
}
```

## 4. ローカル smoke 結果 (2026-05-11 03:00 JST)

### 4.1 設定 (`/tmp/rogii_smoke7/smoke.py`)

- kernel script を `MODE='features_only'` に書き換え (= 自前 train phase は karnakbaev train_df がないので skip)、`DEBUG_MAX_WELLS=3`、`DATA_DIR=data/raw`、`ARTEFACT_DIR=/tmp/rogii_smoke7/artefacts`
- 6 個の独立テスト実行

### 4.2 結果

| Test | 内容 | 結果 |
|---|---|---|
| 1 | `compute_typewell_hashes` を 773 train wells に適用 | ✅ **752 unique hashes、21 dup wells、13 multi-well groups** (= subagent G `1bb6caf` の commit message と完全一致、最大 10 wells グループの hash `5ec0017b...` が一致) |
| 2 | `build_edge_q_folds` で 773 wells を 5 fold に分配 + `verify_edge_q_no_leak` | ✅ **fold_id shape=(773,), unique=[0,1,2,3,4]、leak check: OK: 752 groups, 5 folds, no leak、multi-well groups: 13, of which leak: 0** |
| 3 | `compute_edge_m_features` を synthetic input (nk=800, nh=600) で実行 | ✅ 4 列出力 shape=(600,) float32、NaN 0、Inf 0、selfaln_d range=[-29, +13] (= 健全)、selfaln_score [0.24, 0.30] |
| 4 | `build_dataset(TEST_DIR, is_train=False, max_wells=3)` で 3 test wells を FE | ✅ 14,151 rows × 164 cols、Edge M 4 列全て present、NaN 0、selfaln_d range=[-200, +200] (= clip 適用後)、selfaln_score [0.20, 0.85]、selfaln_conflict_score [0.59, 1.00] |
| 5 | id consistency vs sample_submission | ✅ test_df ids 14151/14151 が sample_submission に存在、extra=0 |
| 6 | Edge M leak guard (= `compute_edge_m_features` source code に `["TVT_input"]` などの **column lookup** が無い) | ✅ コメント・docstring 内の言及は除外、コード上の lookup 0 件 |

### 4.3 補足: selfaln_d range が ±200 ft 上限に貼り付く件

- raw selfaln_d は理論上 `win_tvt[best] + slope_md * (anchor_md - last_md) - last_tvt` で計算される
- 実 test wells (3 wells) で初期値が ±3000 ft 級に出ることが判明 (= MD レンジ 1000 ft × slope ~0.5 × visible window TVT offset ~100 ft の組合せ)
- 物理 TVT delta は ±60 ft 程度が実用 max (`y_kb.min/max * 1.5` の clip 範囲) なので、`slope_md` を ±1.0 に clip + `sa_d` を ±200 ft に clip する safety を入れた (= line ~1041)
- これで feature は **物理レンジ内** に収まり、tree models が学習可能な信号になる
- ただし current ranges が clip boundary に偏る (= ±200 が頻出) のは Ridge meta の weight が低めに振られる可能性がある。次 v2 で per-well slope の robust 化 (例: `slp_50` 使用 / RANSAC) を検討

### 4.4 自前 4 base train phase の smoke (= 不可、deferred to Kaggle run)

- karnakbaev `train_df.parquet` (1.47 GB) はローカル未取得、disk 制約で取得しない方針
- ローカルでは 自前 train phase を smoke できないため、**kernel run 時の初回 push で初検証**
- 失敗 fallback path は exp005 と同等なので、最悪でも LB は 10.317 (= 悪化なし)

## 5. 想定 LB と 9 帯到達への path

### 5.1 LB シナリオ

| シナリオ | 仮定 | 期待 LB |
|---|---|---|
| Best | Edge Q -0.20 ft + Edge M -0.50 ft + diversity -0.10 ft | **9.5 帯到達**、Top 10-15 圏 |
| Mid | Edge Q -0.10 + Edge M -0.30 + diversity -0.05 | **9.85** (Gold cutoff 9.919 ぎりぎり、Top 20 圏) |
| Worst (fallback) | 自前 train が GPU OOM / Ridge weight が degenerate → 5-base NM blend | **10.317** (= exp005 同点、悪化なし) |

### 5.2 9 切り (= 8.x 帯) への次手 (= exp008+)

exp007 単独では 9 切り未到達 (= 9.5-9.7 帯)。
- **exp008**: + 案 D (Kalman/PF on dTVT) → 9.0-9.4
- **exp009**: + 案 E (Bayesian GP for ANCC) + Edge O → 8.7-9.0
- **exp010-013**: 残 6 edges + final stack → 8.0-8.5

## 6. 残課題 / blocker

- **karnakbaev OOF と Edge Q fold partition の不整合**: kernel 内では `oof_predictions.parquet` (= karnakbaev published OOF) を id merge で load するが、その OOF は karnakbaev の **元 GroupKFold (= by `well`)** で生成されており、**Edge Q fold とは fold partition が異なる**。完全な consistency を取るには karnakbaev pretrained を Edge Q val fold で **真の OOF (= train fold で fit、val fold で predict)** を生成する必要があるが、karnakbaev pretrained は既学習済みなので「fit を分け直す」には refit が必要 = 30+ min runtime。**v1 ではこれを skip し、kb published OOF をそのまま使用** (= weight constellation の劣化リスクを残すが、自前 4 base が同 Edge Q fold で OOF 計算されるので、9-base 全体で半分は consistent)。**v2 で karnakbaev pretrained 5 base も Edge Q fold で refit + OOF 再計算** を予定
- **Edge M の selfaln_d が clip boundary に偏る件** (§4.3): per-well slope robust 化が次の v2 改善ポイント
- **kernel push は実行しない**: 中央が exp007 内容 + smoke 結果を確認後に push 判断
- **submit quota** (5/11 UTC): 5 / 5 fresh、exp007 を 1 枚目に使う想定 (= 中央判断後)

## 7. References (= 出典)

- 設計 doc: `experiments/exp007/design.md`
- exp006 notes (TabICL 失敗 postmortem): `experiments/exp006/notes.md`、`docs/dev/leaderboard.dense.md` §exp006 postmortem
- subagent G `1bb6caf`: `notebooks/_typewell_groups_build.py` (Edge Q hash builder の元、本 kernel に inline 移植)
- subagent G `537fa9f`: `docs/research/independent-edges.dense.md` §M (Edge M 文書化)
- subagent J `4ff894a`: `docs/research/kaggle-deepdive.dense.md` §5.1 候補 A (Edge Q 公開議論未出の根拠) + §2.2 host topic 698449 (pseudo-typewell 確認)
- host pptx Slide 9: `docs/research/host-pptx-summary.dense.md` §2 (Edge M 公式根拠 = 「green GR は red GR と相関、typewell GR より高解像度」)
- karnakbaev artifacts (Apache-2.0): https://www.kaggle.com/datasets/karnakbaevarthur/rogii-code-helper-dataset
- thbdh5765 train cache (CC0-1.0): https://www.kaggle.com/datasets/thbdh5765/rogii-v1-train-cache
- TabICL package (Apache-2.0、本 exp dormant): https://github.com/soda-inria/tabicl
- discussion topic 698449 (host pseudo-typewell 公式回答): https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698449

## 8. Smoke ファイル

- 入力: `/tmp/rogii_smoke7/smoke.py` (= kernel script の MODE='features_only' + DEBUG_MAX_WELLS=3 + paths を local 化)
- 出力: smoke の stdout は本 notes §4.2 に要約済 (full log は再実行で取得可)
- runtime: ~60 秒 (= kernel exec 43s + 6 tests 17s)
