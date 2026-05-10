# exp006 — TabICL + PF-lite blend (LB 9 帯狙い)

> **目的**: exp005 (LB 10.x 想定) の +1 改善で **公開 LB Gold 圏 9.x** へ到達する。
> **ベース branch**: `feat/phase-3-exp006-tabicl-pflite` (from `feat/phase-3-exp005-cache-blend`)
> **改良 2 要素**: ① TabICL を 6th base に追加 (needless090 LB 10.081 構成の再現)、② pilkwang PF-lite (stateless ensemble) features 追加

## 0. 状況 (2026-05-11 01:30 JST 開始時)

- 我々の現状 LB: exp002 = 14.695 / exp003 = PENDING / **exp005 = PENDING (期待 10.x)**
- 公開 LB Gold 圏: 9.256 - 9.919 (Top 20)
- 必要改善 vs exp005: -0.5〜-1.0 ft で 9.x 帯到達
- submit 残数: 1 日 5 回 / 5/10 分 = 残 3-4 回 (5/11 分 = 5 回 fresh)

## 1. 採った Approach

### 候補 (構造原理 = "TabICL + PF-lite を kernel に同居させる構成")

| Approach | 概要 | 期待 LB | リスク |
|---|---|---|---|
| **A: フル統合 (採用)** | exp005 kernel base + TabICL 5-fold OOF on train_df (load from karnakbaev parquet) + PF-lite features + 6-base Ridge meta 再 fit + 既存 postproc | **9.x 帯到達狙い** | TabICL = GPU 必須 (kernel `enable_gpu=true` に変更)、8 hr cap 余裕あり (1-1.5 hr 想定) だが train_df load + fold 整合 が path 依存 |
| B: TabICL 軽量版 | TabICL を 1-fold 全 train fit (= no fold) して test 推論のみ、固定 blend weight で 5 base + TabICL を線形結合 | -0.3 ft 程度 | OOF 経由じゃないので 9.x への寄与小、Ridge meta 再 fit 不可 |
| C: PF-lite のみ | TabICL 捨てて PF-lite features だけ + 既存 OOF 経由で 5-base Ridge 再 fit | -0.2〜-0.4 ft 程度 | 9 帯到達は薄い、TabICL 寄与 (-0.5〜-0.7 ft) を取り逃す |

### 選定理由 (A)

1. **TabICL OOF + Ridge re-fit が必須**: needless090 LB 10.081 (1 base TabICL 寄与は 8 base 構成内で -0.3〜-0.5 ft 想定)。固定 weight blend では効果激減
2. **train_df.parquet (1.47 GB) を karnakbaev dataset 経由で load** すれば feature engineering 不要 (= train wells で build_dataset を再実行する必要なし → 6 hr 短縮)
3. **TabICL = GPU 必須** だが Kaggle T4 で 1 fold ~5-10 min × 5 fold + test infer 5-10 min = **1-1.5 hr** で完了見込み (needless090 v13 が 4096 ctx × 5 seed × 5 fold で 55 min 達成)
4. **PF-lite features は per-row 60 行追加で済む** (= candidate TVT 既存、column_stack + likelihood-weighted average のみ)、low-risk な追加 features 6 個 ≒ Ridge stack の判断材料増加
5. **blend は既存 ensemble_weights.json (NM 5-base) を破棄して、6-base で Ridge を **kernel 内 fit** する**。OOF.parquet + TabICL OOF + 各 base の fold 整合 (= 同じ GroupKFold split) で stack 可能

## 2. 必要 host dataset

| Slug | サイズ | 中身 | 使い方 |
|---|---|---|---|
| `karnakbaevarthur/rogii-code-helper-dataset` | 1.36 GB | LGB×3 / XGB / CB artifacts + **train_df.parquet (1.47 GB)** + **oof_predictions.parquet (149 MB)** + features.json + sample_submission | exp005 と同じ。`train_df` から TabICL fit、`oof_predictions` から既存 5 base OOF 取得 |
| `thermostatic/rogii-tabicl-v2-public-assets` | **107 MB** (新規追加) | `tabicl-2.1.1-py3-none-any.whl` (= TabICL package) + `tabicl-regressor-v2-20260212.ckpt` (= TabICL pretrained regressor 114 MB) | `pip install --no-index --no-deps <wheel>` → `from tabicl import TabICLRegressor` → `device="cuda"` で fit/predict |

License: `karnakbaev = apache-2.0`、`thermostatic TabICL = apache-2.0` (出典: https://github.com/soda-inria/tabicl LICENSE、`docs/research/host-datasets.dense.md` §A2)。

## 3. 改良 1 — TabICL の組み込み

### 3.1 Reference

- 元 kernel: `_research_kernels/needless090__score-10-081-score-lb-32-rank/score-10-081-score-lb-32-rank.py:644-748` (105 行)
- 構成: 4096 ctx × 5 seed (`tabicl_A`) と 8192 ctx × 1 seed (`tabicl_B`) の 2 variant
- features 入力: train_df から **LGB quick-fit (n_est=300) で feature_importances 上位 50** に絞る (= TabICL は次元高すぎると遅い)

### 3.2 我々の構成 (v1 = 軽量、必要なら拡張)

```
┌─ load karnakbaev artifacts (train_df.parquet, oof.parquet, features.json, ridge_meta, …) ─┐
│                                                                                            │
├─ LGB quick-fit on train_df[karnakbaev_features] → top-50 features for TabICL              │
│                                                                                            │
├─ TabICL OOF (4096 ctx × 1 seed × 5 fold = ~30-50 min GPU)                                 │
│   for fold in GroupKFold(n_splits=5).split(train_df, target, well):                       │
│     reg = TabICLRegressor(model_path=ckpt, device="cuda", random_state=42, n_estimators=4)│
│     reg.fit(train_df[top50].iloc[tr], target[tr])                                         │
│     oof[va] = reg.predict(train_df[top50].iloc[va])                                       │
│     test_pred += reg.predict(test_df[top50]) / 5                                          │
│                                                                                            │
├─ 既存 5 base OOF を oof_predictions.parquet から load → 6-base OOF stack                  │
│                                                                                            │
├─ Ridge meta 再 fit (alpha=1, fit_intercept=False, positive=True) on 6-base OOF            │
│                                                                                            │
├─ test_delta = ridge.predict(6-base test_preds)                                             │
│                                                                                            │
└─ postproc (alpha=karnakbaev pp_params alpha) → SG smooth → submission.csv                 │
```

### 3.3 GPU 化と runtime budget

- kernel-metadata.json: `enable_gpu = true` に変更
- TabICL fold 1 つ ≈ 5-10 min (T4 / 4096 ctx / n_estimators=4 / chunk 50000)
- 5 fold + test infer 1 回 ≈ 30-60 min
- 既存 LGB×3 + XGB + CB load + infer ≈ 1-3 min
- test feature build (= 自前計算、6 wells × 14k rows) ≈ 30 sec (smoke 実績)
- **total wall 見込: 1-1.5 hr GPU**、8 hr hard cap に対し 6-7 hr 余裕

### 3.4 v1 で省略した要素 (将来拡張余地)

- `tabicl_B` (8192 ctx × 1 seed) → 1 variant 追加で +30-60 min runtime、効果 -0.05〜-0.1 ft 程度。v1 では省略
- TabICL の seed 数 5 (`tabicl_A`) → v1 は 1 seed のみ (= -0.05〜-0.1 ft 譲歩)、9 帯到達できなければ v2 で増加

## 4. 改良 2 — PF-lite features

### 4.1 Reference

- 元 kernel: `_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py:1932-2003` (`weighted_candidate_tvt_features`)
- helper: `typewell_gr_at_tvt` (line 1380-1388)
- feature 名: `pf_lite_feature_names` (line 1811-1823)
- distill: `docs/research/pilkwang-distill.dense.md` §2.2.2 + §7.1

### 4.2 アルゴリズム

per-row で複数 candidate TVT (Beam_cons / dense / plane / selfcorr / formation_mean など) を **GR likelihood + ANCC dense likelihood で重み付け平均**。`pf_lite_tvt`、`pf_lite_std`、`pf_lite_vs_dense / vs_plane / vs_sc / vs_beam_cons` を出力。

### 4.3 our exp005 への plug-in

exp005 `build_well_features()` 内に既に下記 candidate TVT が absolute TVT として計算済:
- `beam_ref` (= `beam_ref_full[sel_local]`、beam_cons の path) → `'beam_cons'` 候補
- `tvt_dense` → `'dense'` 候補
- `sc_raw_sel` (selfcorr NCC) → `'selfcorr'` 候補
- `(-hz + form_ev[:, fi_idx] + b_all)` for fi=0 (BUDA formation plane fit) → `'plane'` 候補
- `tw_diff` 系 anchor 計算で TW interp が既に整備済 → `typewell_gr_at_tvt` 等価

`out` DataFrame 構築直前に PF-lite を呼び、6 features (`pf_lite_d / pf_lite_std / pf_lite_vs_dense / pf_lite_vs_plane / pf_lite_vs_sc / pf_lite_vs_beam_cons`) を merge。

### 4.4 train 側との整合

karnakbaev の `train_df.parquet` には PF-lite features が**入っていない** (= 158 features は exp005 と同じ schema)。**TabICL の入力には影響なし** (= LGB top-50 で選定するため)。**ただし** PF-lite を使う後段 GBM がない場合、PF-lite features は **single-row では Ridge meta の入力にも使えない** (= 既存 5 base GBM が学習していないため)。

**結論**: PF-lite features は **TabICL の features pool に追加** し、TabICL feature 選定で top-50 に入れば自動的に活用される。これで PF-lite + TabICL = "TabICL が pf_lite を経由して 6 method の合致度を読む" 構造になる。**ただし train side でも PF-lite を計算する必要がある** (= karnakbaev `train_df.parquet` を load して PF-lite features を test 側と同じロジックで生成して列追加)。

### 4.5 PF-lite features の train 計算

- train_df.parquet には `well`, `id`, `last_known_tvt`, beam_*, sc_*, dense_*, tvtF_* 等が既存
- これらを candidate TVT に逆変換して PF-lite を計算可能 (= `beam_cons_tvt = last_known_tvt + beam_cons_d`)
- typewell GR は **horizontal_well + typewell raw csv を再 load** する必要あり (= per-well TVT/GR)
- v1 では **PF-lite feature 生成を train_df の per-well GroupBy + raw typewell csv re-load で実装**

### 4.6 v1 での妥協

PF-lite に typewell raw が必要 → train wells (= 数百 wells) の `*__typewell.csv` を `/kaggle/input/rogii-wellbore-geology-prediction/train/` から re-load。これは competition data の標準 path、attach 済み。**ただし、build_well_features を train wells に対しても部分実行する必要がある** (= run_pf_lite_for_train_well() ヘルパーを新設)。

実装複雑度を抑えるため、v1 は:
- **train PF-lite は省略** (= TabICL は karnakbaev features 158 のみで fit)
- **test PF-lite はフル実装** (= test の build_well_features 内で 6 features 追加)
- TabICL は PF-lite を入力に使わないが、**Ridge meta の 6-base に入れる test 側で PF-lite features の影響が出る経路**は **存在しない** (= Ridge meta は OOF 経由で fit、test PF-lite は GBM 学習に乗らない)

→ **v1 で PF-lite features は test 側だけ追加 = "test_df の features は増える、ただし predict には使われない"** = 価値ゼロ。**修正必要**。

### 4.7 v1 修正案 (PF-lite を有効化する path)

**選択肢**:
- **(α) PF-lite を train + test 両方で生成 (= GroupBy で per-well 再構築)**、TabICL の features pool に PF-lite 6 features を追加 (= 158 + 6 = 164 features → top-50 選定)。これで TabICL が PF-lite を学習可能。**追加 runtime: train PF-lite ~15-30 min** (per-well loop)、PF-lite 自体は軽量。
- **(β) PF-lite を完全省略**、TabICL のみで 9 帯狙う。シンプル、リスク最小。

→ **v1 採用: (β) のみ**。理由: GPU 化 + 6-base Ridge 再 fit + train_df load の 3 大改造を一度にやる。PF-lite は **v2 で追加**。LB 結果見て必要なら拡張。

**v1 構成最終**: TabICL 6th base + 6-base Ridge re-fit のみ。PF-lite は kernel script に **関数として実装するが、本番 path には接続しない (= future work hook)**。

## 5. 失敗モード分析 (`/critique` 相当)

### F1: GPU メモリ不足
- TabICL `n_estimators=4 * batch_size=4` で T4 16 GB 内に収まる想定だが、`use_amp="auto"` 必須 (= mixed precision)
- chunk 50000 で OOM 回避 (= test 14k 行は 1 chunk、train 5M 行は per-fold で 100 chunks)
- **対策**: needless090 と同じ `batch_size=4`, `use_amp="auto"`, CHUNK=50000 を踏襲

### F2: train_df.parquet の columns mismatch
- karnakbaev `features.json` に列名 158 個書かれている → train_df.parquet 内に同名列が存在することを前提
- **対策**: kernel 開始直後に `assert set(features_json) <= set(train_df.columns)` を入れて fast fail

### F3: TabICL fold 整合 (groups の同一性)
- karnakbaev OOF.parquet は karnakbaev の GroupKFold(5) で生成。我々が改めて GroupKFold する場合、**同じ shuffle / random_state** でないと **異なる fold 割当** になり、OOF 列同士で leak する
- **対策**: karnakbaev OOF.parquet の `well` 列 → fold 番号を逆算して **同一 fold 割当を再利用** する path resolver。OOF.parquet 内に fold 列があれば直接、なければ `well` ごとの first appearance index でフォールバック

### F4: runtime > 8 hr
- train_df load (~30 sec) + LGB quick-fit (~3 min) + TabICL 5-fold (~50 min) + test infer (~5 min) + Ridge fit (~1 sec) = **~1 hr** が想定中央
- TabICL `n_estimators=4` を `n_estimators=2` に下げれば半分に
- **対策**: time.perf_counter() で **6 hr hard cap** を入れ、TabICL fold loop で残時間が 30 min 切ったら `break` + 残 fold を 0 fill (= test_pred は既出 fold 平均)。fail-safe

### F5: LB queue 遅延
- exp005 と同じく submit → PENDING が 1 hr+ 続く可能性。中央は exp006 push 後すぐ別 submit せず queue 待ち、LB 確定後に判断
- **対策**: subagent I は kernel push しない。中央が exp005 LB を見てから判断

### F6: TabICL wheel install 失敗
- `pip install --no-index --no-deps <wheel>` でも依存 (torch, einops 等) が満たされない可能性。Kaggle 標準環境は torch>=2 を持つ想定だが、TabICL 2.1.1 が要求 version と乖離あり得る
- **対策**: 元 kernel と同じ `--no-deps` で torch を再 install させない。失敗時は `pip install --no-index <wheel>` で deps 解決を試す fallback

## 6. 計画 (TODO)

- [x] branch `feat/phase-3-exp006-tabicl-pflite` 作成
- [x] 設計 doc (= 本ファイル) 作成 → 1st commit + push
- [ ] kaggle_kernels/exp006_tabicl_pflite/exp006_tabicl_pflite.py を exp005 から copy + 改造
- [ ] kernel-metadata.json 作成 (`dataset_sources` に TabICL 追加、`enable_gpu=true`)
- [ ] PF-lite 関数群 (typewell_gr_at_tvt / weighted_candidate_tvt_features / pf_lite_feature_names) を kernel script 末尾に **dormant function として** 追加 (= future v2 で接続)
- [ ] TabICL 統合: `train_df.parquet` load → LGB quick-fit → top-50 features → 5-fold TabICL OOF + test pred → Ridge re-fit (6-base) → test_delta
- [ ] postproc + SG smooth + submission.csv 出力
- [ ] ローカル smoke (= 3 test wells、TabICL 部分は **skip**、 既存 path だけ通す = exp005 と同じ flow + Ridge re-fit dry run)
- [ ] notes.md 完成 (= smoke 結果、想定 LB、blocker)
- [ ] kernel push は実行しない (= 中央指示待ち)

## 7. 重要警告

- **Internet disabled**: TabICL wheel は `thermostatic/rogii-tabicl-v2-public-assets` 経由でのみ取得可能
- **GPU 必須**: kernel-metadata.json で `enable_gpu = true` に変更 (exp005 は false)
- **license attribution**: kernel docstring に下記 3 行明記必須
  - `karnakbaevarthur/rogii-code-helper-dataset` (apache-2.0)
  - `thermostatic/rogii-tabicl-v2-public-assets` (apache-2.0、TabICL package https://github.com/soda-inria/tabicl)
  - `needless090/score-10-081-score-lb-32-rank` 構成 inspired (public Kaggle kernel)
  - `pilkwang/rogii-eda-v4-same-matrix-super-stack` (PF-lite features、本 v1 では dormant)
- **kernel push 禁止**: 中央が exp005 LB を確認してから判断。subagent I は branch push + smoke pass までで停止
- **branch 規律**: `kaggle_kernels/exp005_*/` と `src/rogii/` には触らない (= subagent G 領域)、新規は `kaggle_kernels/exp006_tabicl_pflite/` と `experiments/exp006/` のみ

## 8. References

- needless090 kernel: https://www.kaggle.com/code/needless090/score-10-081-score-lb-32-rank
- TabICL package: https://github.com/soda-inria/tabicl (Apache 2.0)
- TabICL dataset (我々が attach する): https://www.kaggle.com/datasets/thermostatic/rogii-tabicl-v2-public-assets
- karnakbaev artifacts: https://www.kaggle.com/datasets/karnakbaevarthur/rogii-code-helper-dataset
- pilkwang PF-lite: https://www.kaggle.com/code/pilkwang/rogii-eda-v4-same-matrix-super-stack
- our exp005 notes: `experiments/exp005/notes.md`
- top3 distill (TabICL section): `docs/research/top3-distill.dense.md` §6 + §3
- pilkwang distill (PF-lite section): `docs/research/pilkwang-distill.dense.md` §2.2.2 + §7.1
