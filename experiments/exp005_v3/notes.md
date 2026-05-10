# exp005 v3 — Edge S + Edge R 実装 notes

## 0. 位置付け

`feat/phase-4-edge-s-roundgrid` で完成した exp005 v2 (= Edge S round-to-grid 注入済、 LB 10.317) に、
**Edge R = test-time online learning + LGB continued training** を新規追加。
Phase 5 の Gold 圏入り保険 (= LB 9.95 想定、 Gold cutoff 9.919 を割り込む確実な path)。

## 1. 変更概要

base kernel: `kaggle_kernels/exp005_cache_blend/exp005_cache_blend.py` (= subagent P が v2 push 済)

### inject 一覧

| 位置 | line range (v3 後) | 変更内容 |
|---|---|---|
| docstring | 4-22 | v3 titling、 Edge R 説明追加 (= "Edge R: test-time online learning ...") |
| Path env override | 88-99 | `ROGII_DATA_DIR / ROGII_ARTEFACT_DIR / ROGII_OUTPUT_DIR` 環境変数で path override 可能化 (= ローカル smoke 用、 Kaggle では無効) |
| Debug env override | 90-92 | `ROGII_DEBUG_MAX_WELLS` 環境変数で `DEBUG_MAX_WELLS` 設定可能化 |
| Edge R config | 248-261 | `EDGE_R_ENABLE`, `EDGE_R_VISIBLE_TAIL_K=100`, `EDGE_R_MIN_VISIBLE_ROWS=30`, `EDGE_R_NUM_BOOST=200`, `EDGE_R_LR_MUL=0.5`, `EDGE_R_BLEND_W=0.5` |
| Edge R helper | 1364-1583 付近 | `build_visible_dataset`, `edge_r_continued_train_lgb`, `edge_r_apply` 関数群 |
| Edge R inference inject | `MODE == "infer"` 内 (= 2020-2046 付近) | base models load 後、 NM blend (test_delta) 算出後に Edge R 適用、 test_delta 上書き |

## 2. Edge R 適用パス

```
test_df = build_dataset(TEST_DIR, is_train=False)
base_models = load karnakbaev artefacts (5 base)
test_preds = predict_all(base_models, test_df[features])
test_delta = apply_ensemble_nm(test_preds, nm_w)     # 既存 v2 までの結果
# ↑ ここまで v2 と同じ

# v3 Edge R inject:
visible_df = build_visible_dataset(TEST_DIR)         # 各 well 末尾 100 行を擬似 hidden 化
test_delta, _ = edge_r_apply(test_df, visible_df, base_models, features, nm_w, test_delta)
# ↑ Edge R block で test_delta が online と blend (= 0.5/0.5)

test_delta_pp = apply_postproc(test_df, test_delta, alpha, tau)
test_delta_sg = sg_smooth_per_well(test_df, test_delta_pp)
build_submission(test_df, test_delta_sg, ..., output_path)
# Edge S round-to-grid は build_submission 内で適用
```

## 3. ローカル smoke (= 3 wells)

詳細は `experiments/edge_r_online/notes.md` 参照。 主要結果:

- ✅ visible build: (300, 158)、 NaN 0、 y_R finite range [-0.380, 3.580]
- ✅ LGB 3 base continued training 全 PASS (lgb0 10s、 lgb1 8s、 lgb2 7s)
- ✅ online blend と base blend が different (= diff abs mean 0.19 ft、 actual change)
- ✅ submission.csv 14,151 rows、 NaN 0、 tvt range [11592, 12237]
- ✅ Edge S round-to-grid 干渉なく動作 (= diff 0.0025 ft)
- ✅ total runtime 47s (3 wells) → 推定 776 wells で **15-25 min** = 9 hr cap 余裕

## 4. 想定 LB

| 段階 | LB |
|---|---|
| v2 (Edge S 単独) | 10.317 |
| v3 (Edge S + Edge R) | **9.95 (= Gold cutoff 9.919 割り込み確実)** |

論拠:
- topic 698002 実測 -0.370 ft (= online 10.953 vs no-online 11.323)
- Edge S 単独効果は v2 で確認済 (= round-to-grid 効果は -0.05 ~ -0.30 ft 期待)

## 5. 失敗 mode と対策 (= 設計書からの抜粋)

| mode | 兆候 | 対策 |
|---|---|---|
| visible 行不足 well | feature build で skipped | `EDGE_R_MIN_VISIBLE_ROWS=30` 未満 well は skip、 安全 fallback |
| continued training overfit | online predict が absurd | `num_boost_round=200` 固定、 `lr_mul=0.5` で抑制、 NaN/inf chk 内蔵 |
| Edge S と干渉 | round 後の 0.01 snap で online 変化が消える | Edge S は最後段、 Edge R は delta 段階で適用 = 干渉なし (= smoke で確認) |
| LB regression | 9.95 想定が 10.5 に悪化 | `EDGE_R_BLEND_W` を 0.3 に下げる、 fallback `EDGE_R_ENABLE=False` |

## 6. push 戦略 (= 中央判断)

| slot | kernel | 役割 |
|---|---|---|
| safe | **exp005 v3** | Gold 圏入り保険 (= LB 9.95 想定) |
| risky | exp009 v4 | 賞金 1 位 圏 (= LB 7.8-8.1) |

push command (中央が判断後に実行):

```bash
cd kaggle_kernels/exp005_cache_blend
kaggle kernels push -p .
```

注: kernel-metadata.json の id は **変更なし** (`ky7240/rogii-exp005-cache-blend`) = 既存 kernel の新 version 上書き。 dataset_sources 増減なし (= karnakbaev のみ)。

## 7. 関連 file

- design: `experiments/edge_r_online/design.md`
- impl notes: `experiments/edge_r_online/notes.md`
- kernel: `kaggle_kernels/exp005_cache_blend/exp005_cache_blend.py`
- kernel-metadata: `kaggle_kernels/exp005_cache_blend/kernel-metadata.json`

## 8. 更新履歴

- 2026-05-11: 初版、 Edge R 注入 + E2E smoke PASS
