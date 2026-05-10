# exp007 — Edge Q + Edge M + 自前 LGB×3+CB ブレンド (LB 9.5-9.7 帯狙い)

## 0. 状況 (2026-05-11)

- exp005 = LB **10.317** (Silver 確定、自前 best)
- exp006 = LB **10.503** (TabICL 投入で逆悪化、postmortem 完了)
- 公開 LB Top 1 = 9.256、Gold cutoff = 9.919、賞金圏 (Top 4) = 9.415
- ユーザー指示 (2026-05-11): 「9 切らないと優勝は無理」 = 最終 LB 8.x 帯到達

## 1. Approach

採用: **Approach C = Edge Q + Edge M + 自前 4 base + 9-base Ridge re-fit**

| 案 | 概要 | 期待 LB |
|---|---|---|
| **C (採用)** | exp005 5-base + 自前 LGB×3 + CB を Edge Q fold で OOF 学習 + Edge M FE | **9.5-9.7 帯** |
| B | TabICL 復活 (seed/ctx 拡大) | 10.0-10.3 |
| A | Edge Q のみ + 既存 5-base Ridge re-fit | 10.1-10.2 |

採用理由:
1. exp006 postmortem (`docs/dev/leaderboard.dense.md`): TabICL 追加だけでは Ridge weight constellation が劣化、diversity を増やすには base model を **同じ Edge Q fold で独立に学習** する必要
2. Edge Q (= typewell content-hash GroupKFold) は subagent J `kaggle-deepdive.dense.md` §5.1 候補 A、公開議論未出
3. Edge M (= visible-as-typewell self-alignment) は host pptx Slide 9 公式推奨、公開 kernel で実装例ゼロ

## 2. 必要 host dataset

| Slug | 役割 |
|---|---|
| `karnakbaevarthur/rogii-code-helper-dataset` (1.36 GB, Apache-2.0) | base 1-5 (= LGB×3 + XGB + CB) artifacts + train_df.parquet (= 自前 train の入力) |
| `thbdh5765/rogii-v1-train-cache` (1.25 GB, CC0-1.0) | 補助 (schema が exp005 test FE と完全一致しないため、本 exp では参考扱い、attach のみ) |
| `thermostatic/rogii-tabicl-v2-public-assets` (107 MB, Apache-2.0) | 本 exp 未使用 (= dormant)、attach のみ |

## 3. 構造 (= 9-base Ridge stack)

5 base (karnakbaev pretrained, Edge Q fold で OOF 再計算) + 4 自前 base (Edge Q fold で fit) = **9-base Ridge meta**。

### 3.1 自前 4 base hyperparameter

| key | model | lr | num_leaves / depth | n_est | seed |
|---|---|---|---|---|---|
| `lgb_own0` | LightGBM GPU | 0.02 | 127 | 8000 | 42 |
| `lgb_own1` | LightGBM GPU | 0.05 | 127 | 4000 | 7 |
| `lgb_own2` | LightGBM GPU | 0.10 | 127 | 2000 | 123 |
| `cb_own` | CatBoost GPU | 0.05 | depth=8 | 5000 | 42 |

### 3.2 OOF flow

1. karnakbaev `train_df.parquet` を load (= 既存 schema、3.78M rows × 160 cols)
2. **Edge Q fold 計算**: 全 train wells (773 well) の typewell file の MD5 hash → 13 group (35 wells) を duplicate group 識別 → `GroupKFold(5).split(X, y, groups=hash)` で `fold_id` 列を train_df に付与
3. **Edge M FE 列を追加**: per-well で visible 部 GR ⇔ hidden 部 GR の self-NCC、4 列 (`selfaln_d / selfaln_score / selfaln_vs_typewell_d / selfaln_conflict_score`)
4. **karnakbaev pretrained 5 base を Edge Q fold で predict** → karnakbaev OOF を Edge Q fold で再計算 (= weight constellation の consistency 確保)
5. **自前 4 base を Edge Q fold で 5-fold OOF 学習** (= train fold で fit、val fold で predict、test に対しては 5 fold mean)
6. 9-base OOF matrix を組み Ridge meta `(positive=True, fit_intercept=False, alpha=1.0)` を fit
7. test 9-base predictions に Ridge meta を適用、karnakbaev `postproc_params` (alpha + tau fade-in)、SG smooth、spike clamp

### 3.3 Edge Q (typewell content-hash GroupKFold)

`subagent G commit 1bb6caf` の `notebooks/_typewell_groups_build.py` を kernel 内 helper にインライン移植。

```python
def compute_typewell_hashes(train_dir, well_ids):
    hashes = {}
    for wid in well_ids:
        tw_path = train_dir / f"{wid}__typewell.csv"
        if not tw_path.exists():
            hashes[wid] = wid  # fallback
            continue
        df = pd.read_csv(tw_path)
        h = hashlib.md5()
        for col in ("TVT", "GR", "Geology"):
            if col not in df.columns: continue
            if df[col].dtype == "object":
                h.update(df[col].fillna("NaN").astype(str).str.cat(sep="|").encode())
            else:
                h.update(df[col].fillna(-99999.0).round(4).values.tobytes())
        hashes[wid] = h.hexdigest()
    return hashes
```

**leak 検査** (smoke 必須 assert): 同 hash の 2 wells が同 fold に在ることをループ検証。

### 3.4 Edge M (visible-as-typewell self-alignment) FE

`subagent G commit 537fa9f` で `independent-edges.dense.md` §M に文書化。実装は本 exp 初出。

per-well 計算:
1. visible 部 (= `TVT_input` not NaN) GR から K=10-30 windows を抽出 (window_size=200 samples、step=50)
2. hidden 部 (= `TVT_input` is NaN) を chunk_size=200 samples で分割、各 chunk について visible windows との **Self-NCC** を計算、best window 選出
3. best window TVT (= visible 末端 TVT) + (hidden chunk MD - best window MD) × prefix dTVT/dMD slope を hidden TVT 推定値
4. typewell match 推定 TVT (= `beam_cons_d` proxy) との差を `selfaln_vs_typewell_d`、conflict 度を sigmoid 化

出力 4 列 (float32 per row): `selfaln_d / selfaln_score / selfaln_vs_typewell_d / selfaln_conflict_score`。

**leak 検査**: train 側 Edge M FE で `TVT_input` 列を一切参照しないこと (= GR 列のみで NCC、TVT は visible 末端 + slope)。smoke で `assert "TVT_input" not in edge_m_intermediate_columns`。

## 4. kernel script 構造

`exp006_tabicl_pflite.py` (2426 行) を base に派生。

| # | 場所 | 内容 |
|---|---|---|
| 1 | docstring (line 1-50) | exp007 用 4-step pipeline + Edge Q/M license attribution + TabICL dormant の理由 |
| 2 | config | `MODE = 'infer_edge_qm'`、`TABICL_ENABLE = False`、`EDGE_Q_ENABLE = True`、`EDGE_M_ENABLE = True`、自前 base hyperparam |
| 3 | helpers | `compute_typewell_hashes()`, `build_edge_q_folds()`, `compute_edge_m_features()`, `karnakbaev_oof_refit_under_edge_q()` を新規 |
| 4 | dispatcher | `elif MODE == 'infer_edge_qm':` block 約 300 行 |
| 5 | failure recovery | (1) Edge Q phase 失敗 → naive `well` GroupKFold fallback、(2) 自前 train phase 失敗 → 5-base only fallback (= exp005 同等)、(3) Edge M FE 失敗 → Edge M 列を 0 で埋めて続行 |

### 4.1 kernel-metadata.json

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

### 4.2 runtime 見積

| 段階 | 時間 |
|---|---|
| test feature build (6 wells × 14k rows) | 30 sec |
| train_df.parquet load (1.47 GB) | 30 sec |
| Edge Q fold (= 773 typewell MD5) | 30-60 sec |
| Edge M FE for train (3.78M rows, parallel joblib) | 10-20 min |
| Edge M FE for test | 30 sec |
| karnakbaev OOF refit under Edge Q (5 fold predict) | 3-5 min |
| 自前 LGB×3 + CB train (5 fold × 4 base, GPU) | 60-90 min |
| 9-base Ridge meta fit + predict | <1 sec |
| postproc + SG smooth + submission | 5 sec |
| **合計** | **75-120 min** |
| 9 hr cap 余裕 | **7+ hr** |

## 5. failure mode 4 件 + recovery

### 5.1 Edge Q で test typewell に Geology 列なし
- 症状: test typewell は Geology 不在 (= host 公式)。Edge Q は train 側 fold builder なので test 側影響なしだが trap
- 対策: file 不在時 `wid` fallback、Geology 不在時は (TVT, GR) のみで hash

### 5.2 Edge M FE で hidden side leak
- 症状: hidden 部の `TVT_input` を参照すると leak
- 対策: GR 列のみ読む、TVT 推定は visible 末端 TVT + slope 外挿、smoke で intermediate columns に `TVT_input` 不在を assert

### 5.3 自前 LGB train が GPU OOM
- 症状: T4 16GB で 3.78M × 160 × 5 fold が borderline
- 対策: `device=cpu` fallback、`n_estimators` 5000→3000、`num_leaves` 127→63

### 5.4 Ridge weight が自前 base に集中
- 症状: 自前 LGB が overfit して Ridge weight 0.6+ に → karnakbaev 汎化能力捨てる
- 対策: Ridge weight log し、自前合計 0.5+ なら **5-base only NM blend (= exp005 同等)** に fallback

## 6. 実装手順

- [x] branch 作成 (= 1st commit、cherry-pick `kaggle-deepdive`)
- [ ] 設計 doc (本 file) → 2nd commit + push
- [ ] kernel script + metadata → 3rd commit + push
- [ ] ローカル smoke → 4th commit + push (notes.md と一緒)
- [ ] **kernel push は実行しない** (中央指示待ち)

## 7. 想定 LB

| シナリオ | 仮定 | LB |
|---|---|---|
| Best | Edge Q -0.20 + Edge M -0.50 + diversity -0.10 | **9.5 帯** |
| Mid | Edge Q -0.10 + Edge M -0.30 + diversity -0.05 | **9.85** (Gold cutoff ぎりぎり) |
| Worst (fallback) | 自前 train 失敗 → 5-base NM blend = exp005 同点 | **10.317** (悪化なし) |

## 8. 9 切りへの次手 (exp008+)

exp007 単独では 9 切り未到達 (= 9.5-9.7 帯)。
- exp008: + 案 D (Kalman/PF on dTVT) → 9.0-9.4
- exp009: + 案 E (Bayesian GP for ANCC) + Edge O → 8.7-9.0
- exp010-013: 残 6 edges + final stack → 8.0-8.5

## 9. References

- `docs/research/kaggle-deepdive.dense.md` §5.1 候補 A (Edge Q 根拠、subagent J)
- `docs/research/host-pptx-summary.dense.md` §2 Slide 9 (Edge M 公式根拠)
- `docs/research/independent-edges.dense.md` §M (Edge M 詳細、subagent G `537fa9f`)
- `docs/dev/leaderboard.dense.md` §exp006 postmortem (TabICL 失敗分析)
- subagent G `1bb6caf`: `notebooks/_typewell_groups_build.py` (Edge Q hash builder)
- subagent G `537fa9f`: Edge M 文書化
- karnakbaev artifacts (Apache-2.0): https://www.kaggle.com/datasets/karnakbaevarthur/rogii-code-helper-dataset
- thbdh5765 (CC0-1.0): https://www.kaggle.com/datasets/thbdh5765/rogii-v1-train-cache
- TabICL (Apache-2.0、本 exp dormant): https://github.com/soda-inria/tabicl
