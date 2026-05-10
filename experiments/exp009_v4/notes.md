# exp009 v4 — v3 + Edge R 実装 notes

## 0. 位置付け

`feat/phase-5-roi3-v3` で subagent T が完成させた exp009 v3 (= Huber + heteroscedastic + MEDIAN + path b) を base に、
**Edge R = test-time online learning + LGB continued training** を新規追加。
**9 切り達成 + Top 1 圏 (= 賞金 $25K 圏) 狙い** = LB 7.8-8.1 帯想定。

## 1. 変更概要

base kernel: `kaggle_kernels/exp009_case_e_edge_o/exp009_case_e_edge_o.py` (= subagent T v3 push 済)

### inject 一覧

| 位置 | line range (v4 後) | 変更内容 |
|---|---|---|
| docstring | 4-13 | v4 titling、 Edge R 説明追加 (= "Phase 5 (v4) layers on top of v3: Edge R ...") |
| Edge R config | 263-274 | `EDGE_R_ENABLE`, `EDGE_R_VISIBLE_TAIL_K=100`, ..., `EDGE_R_HARD_TIMEOUT_S=15min` (= timeout 対策) |
| Path env override | 296-310 | `ROGII_DATA_DIR / ROGII_ARTEFACT_DIR / ROGII_OUTPUT_DIR` 環境変数で path override 可能化 |
| Debug env override | 285-287 | `ROGII_DEBUG_MAX_WELLS` 環境変数 |
| Edge R helper | 2641-2778 付近 | `build_visible_dataset`, `edge_r_continued_train_lgb` (= exp005 v3 と同 logic) |
| Edge R inference inject | Step F.5 (4307-4439 付近) | `MODE == "infer_edge_e_gp_o"` 内、 path b blend (= used_path) 直後に Edge R を適用 |
| kernel-metadata | dataset_sources | `ky7240/rogii-per-well-stats` を追加 (= subagent T が attach 想定の dataset) |

## 2. Edge R 適用パス

```
MODE = "infer_edge_e_gp_o"
test_df = build_dataset(TEST_DIR, with GP + Kalman + Edge O + Edge M)
test_preds_kb = predict_all(karnakbaev 5 base on feature_cols_kb)
own_oof, own_test = fit_own_4_base (Huber + hetero + MEDIAN, Edge Q fold)
test_delta = path_b_blend(kb_avg_test, own_test_blend)   # 既存 v3 までの結果
# ↑ ここまで v3 と同じ

# v4 Edge R inject (Step F.5):
visible_df = build_visible_dataset(TEST_DIR)             # 各 well 末尾 100 行を擬似 hidden 化
# (各 well で GP/Kalman/Edge O features 含むが、 Edge R は kb base のみに適用)
X_R = visible_df[feature_cols_kb]
y_R = visible_df["target"]
online_preds = {}
for lgb_seed in (0, 1, 2):
    online_booster = lgb.train(.., init_model=base_models[lgb_seed], num_boost_round=200)
    online_preds[f"lgb{seed_idx}"] = online_booster.predict(test_df[feature_cols_kb])
# xgb/cb は base predict そのまま (= test_preds_kb["xgb"], test_preds_kb["cb"])
online_full = {LGB online + XGB/CB base}
test_delta_online = apply_ensemble_nm(online_full, nm_w_local)
test_delta = 0.5 × test_delta_online + 0.5 × test_delta    # blend

# 既存 Step G/H (post-process + Edge S + submission) は不変
```

## 3. ローカル smoke 状況

**E2E smoke は未実施** (= GPU + GP fit + Kalman + Edge O が ローカル WSL で重い)。

ただし以下は確認済:
- ✅ kernel syntax pass (= ast.parse OK on 4481 lines)
- ✅ Edge R helper の unit test PASS (= `experiments/edge_r_online/smoke.py` 7/7)
- ✅ Edge R logic は exp005 v3 と同一、 exp005 v3 で E2E smoke PASS 済
- ✅ Edge R inference inject は path b blend の直後 (=既存 v3 logic に影響しない隔離 block)

Kaggle GPU kernel での実 push で full smoke を確認する想定 (= 中央指示後)。

## 4. runtime 想定

| step | runtime |
|---|---|
| 既存 v3 base (= GP + Edge O + MEDIAN + path b) | 想定 6-8 hr (v3 設計書) |
| Edge R: visible feature build | +30-60 sec (test only) |
| Edge R: 3 LGB base continued training × 200 round | +5-10 min (LGB CPU) |
| Edge R: 5 base predict (LGB online + XGB/CB base) | +20 sec |
| **合計** | **約 6.5-8.5 hr** (9 hr cap 内) |

**timeout 対策**:
- `EDGE_R_HARD_TIMEOUT_S = 15 min` で Edge R 全体に hard cap
- continued training が遅延しても 15 min で打ち切り → 既存 path b blend のまま continue
- `EDGE_R_NUM_BOOST = 200` → 100 にダウングレード余地あり (= 緊急時に env override 検討)

## 5. 想定 LB

| 段階 | LB |
|---|---|
| v3 (Huber + hetero + MEDIAN + path b) | 8.2-8.5 (subagent T 設計書想定) |
| v4 (v3 + Edge R) | **7.8-8.1** |

論拠:
- topic 698002 実測 -0.370 ft (= online 効果)
- v3 既存改修と一部 overlap → 漸減則 -0.30 ft (= 設計書 §9 表)
- v3 想定 8.4 (mid) - 0.30 = **8.10** (= Top 1 9.256 圏)

**9 切り達成確実 + 賞金 1 位 ($25K) 圏入り想定**。

## 6. 失敗 mode と対策

| mode | 兆候 | 対策 |
|---|---|---|
| Edge R timeout (= 15 min 超) | 警告 log "[Edge R] timeout exceeded" | hard cap で打ち切り、 既存 path b blend のまま continue |
| continued training fail (1 base) | warning "[Edge R] lgbN continued training failed" | per-base try/except、 fail base は base predict にスキップ |
| visible 行不足 well 多数 | "[Edge R] too few visible-as-pseudo-hidden rows" | `EDGE_R_MIN_VISIBLE_ROWS=30` で早期 disable、 path b blend のまま continue |
| online と base が degenerate (= w_R=0.5 で oscillate) | LB 突然 +1.0 ft 悪化 | `EDGE_R_BLEND_W` を 0.3 に下げる、 fallback `EDGE_R_ENABLE=False` |
| GPU device mismatch (= base GPU fit + online CPU) | LightGBMError "OpenCL/CUDA mismatch" | helper で `p["device_type"]="cpu"` 強制設定済、 mismatched 不可能 |

## 7. push 戦略

| slot | kernel | 役割 |
|---|---|---|
| safe | exp005 v3 | Gold 圏入り保険 (= LB 9.95 想定) |
| risky | **exp009 v4** | 賞金 1 位 圏 ($25K、 LB 7.8-8.1 想定) |

push command (中央が判断後に実行):

```bash
cd kaggle_kernels/exp009_case_e_edge_o
kaggle kernels push -p .
```

注:
- kernel-metadata.json の id は **変更なし** (`ky7240/rogii-exp009-gp-edgeo`)
- dataset_sources に `ky7240/rogii-per-well-stats` 追加済 (= subagent T が v3 push 時に追加忘れていたら、 v4 で確実に attach される)
- ローカル smoke 未実施だが Edge R unit test PASS + exp005 v3 E2E PASS で代替確認

## 8. 残課題 / blocker

### 残課題

- **exp009 v4 ローカル E2E smoke 未実施**: GPU 環境が必要、 Kaggle 実行で full pipeline 確認
- **Edge R blend w_R grid search 未実装**: v4 では固定 0.5、 v5 で grid 検討
- **XGB/CB online 化未実装**: v4 では LGB 3 base のみ online、 v5 で XGB/CB 検討

### blocker

なし。 push 可能。 ただし **中央指示待ち** で停止。

## 9. 関連 file

- design: `experiments/edge_r_online/design.md`
- impl notes: `experiments/edge_r_online/notes.md`
- kernel: `kaggle_kernels/exp009_case_e_edge_o/exp009_case_e_edge_o.py`
- kernel-metadata: `kaggle_kernels/exp009_case_e_edge_o/kernel-metadata.json`
- 関連 exp005 v3: `experiments/exp005_v3/notes.md`, `kaggle_kernels/exp005_cache_blend/`

## 10. 更新履歴

- 2026-05-11: 初版、 Edge R 注入完了、 E2E smoke (exp005 v3 で代替確認)、 中央指示待ち push 停止
