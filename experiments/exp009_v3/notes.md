# exp009 v3 — 開発ノート (Phase 5 ROI 3 layer + fold-misalign 解消)

## 0. ブランチ + コミット履歴

`exp008_v3/notes.md` §0 と同一 (= 同 branch `feat/phase-5-roi3-v3`)。
exp009 v3 の kernel 修正は commit `72db496 feat(exp009-v3): inject 4 改修`。

## 1. 4 改修の inject 位置 (= exp009_case_e_edge_o.py 修正後 line 番号)

| 改修 | 内容 | 修正 line |
|---|---|---|
| 1 | LGB `objective="huber"` + `alpha=0.9` | `LGB_BASE` dict, 359-360 |
| 1 | CB `loss_function="Huber:delta=0.5"` | `CB_PARAMS` dict, 393 |
| 2 | hetero sample_weight 構築 (per-well-stats.parquet) | own train Step E.1, 3839-3895 |
| 2 | `lgb.LGBMRegressor.fit(... sample_weight=w_tr, eval_sample_weight=[w_va])` | own train, 3953-3971 |
| 2 | `Pool(... weight=w_tr)` (CB) | own train, 4019-4021, 4028-4030 |
| 3 | LGB schedule + seed loop + MEDIAN | own train, 3919-3993 |
| 3 | CB seed loop + MEDIAN | own train, 3996-4046 |
| 4 | path b blend = kb simple-avg + own Ridge + 1D grid | Step F, 4051-4128 |
| 4 | legacy n-base Ridge fallback | Step F, 4131-4174 |

## 2. ローカル smoke 結果

exp008 v3 と同じ `smoke_phase5_v3.py` を共用 (= 4 改修は両 kernel で同一 logic)。
4/4 PASS。詳細は `exp008_v3/notes.md` §2 参照。

exp009 固有の検証は **kernel push 後の Kaggle 実 run でしか取れない**:
- GP fit (= sklearn GaussianProcessRegressor, 200 inducing point, n_restarts=3) は train FE rebuild に依存
- Edge O dir-aware Beam features は per-well 計算が必要

## 3. Multi-seed MEDIAN 戦略 + 9 hr cap 検証

- 構成: 同上 (LGB×1 + CB×1) × 3 seed × 5 fold = 30 run
- runtime 想定 (exp009 固有):
  - test FE build: 60 min (Kalman + GP + Edge O)
  - train FE rebuild: 60 min (Edge M + Kalman + GP + Edge O over TRAIN_DIR)
  - kb 5 base test predict: 5 min
  - 自前 train Multi-seed MEDIAN: 5 hr (= 2.5-5 hr 上限)
  - Ridge / path b blend: 5 min
  - post-proc + submission: 10 min
  - **合計: 約 8 hr = 9 hr cap 内 (60 min buffer)**
- cap 超過時の fallback:
  - CB seed 数を 3 → 2 (= 6 → 5 model × 5 fold = 25 run)
  - それでも超過したら GP の `n_restarts=3` → 1 (= GP 精度低下と引き換え)

## 4. 想定 LB (= 4 改修累積)

| stage | exp009 v2 base | + Huber | + hetero | + MEDIAN | + path b | 累積 mid |
|---|---|---|---|---|---|---|
| CV | 9.1 (推定) | -0.30 | -0.20 | -0.30 | -0.30 | **8.0** |
| LB | 未確定 | -0.30 | -0.20 | -0.30 | -0.30 | **8.2-8.5** |

= **8 台達成射程内 (= ユーザー要求満足)**

## 5. 残課題

`exp008_v3/notes.md` §5 と同一。

## 6. kernel push 直前チェックリスト (= 中央向け)

`exp008_v3/notes.md` §6 と同一 + 追加:

- [ ] kernel-metadata.json の slug が `rogii-exp009-gp-edgeo` のまま (= v2 push と同じで v3 push)
- [ ] GP + Edge O が attach データセット内に揃っているか確認
- [ ] GPU runtime で Huber gradient computation が動作するか smoke (= exp009 は feature dim 196 で計算重い)
