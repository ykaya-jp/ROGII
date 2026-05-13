# thbdh v10 NB の構造調査 (= ravaghi の super-set + augmentation + TabICL)

> 起点: 2026-05-13 Phase 10、 exp017 LB scoring 中の 待ち時間活用
> Source: `.work/public-nb-fork/thbdh_v10_as_is.ipynb` = `thbdh5765/rogii-v10-fresh-artifact-infer` (= 16 votes、 5/12 03:09 lastRun)
> Status: fork attempt 完了 (= ky7240/rogii-exp018-thbdh-v10-fork)、 TabICL CUDA 必須で **CPU 不可 ERROR** (= 92 sec で fail)、 諦め

---

## 1. structure summary

- 1 cell に 109989 chars (= 全 logic を single cell に詰めた inference-only NB)
- **52 関数定義**
- inference-only mode (`ROGII_INFERENCE_ONLY=1`)
- artifact cache 中心 (= pretrained model + cached features を Kaggle dataset から load)
- Numba JIT 採用

## 2. paradigm 構成 (= keyword 出現頻度から推定)

```
keyword         count   解釈
TabICL          15+42×  Tabular ICL transformer = main base、 GPU 必須
GP              16×     Gaussian Process (= 我々 案 E Sparse GP と同 paradigm)
PF              60×     Particle Filter (= dual Z + ANCC、 ravaghi 同)
beam            21×     beam_search + run_all_beams
md_since        9×      apply_pp envelope (= 同 formula)
last_known_tvt  9×      anchor base value
aug_k           9×      data augmentation (= Phase B path F、 5M rows expansion)
CatBoost         6×     補助 base
lightgbm         1×     軽微利用 (= ravaghi が LGB 主軸とは対照的)
knn              3×     per-row kNN inference (= Phase B path G の partial)
savgol           1×     post-smoothing
```

## 3. 主要関数 (= 52 関数のうち substantive)

```
build / feature:
  wls_b_well                      (= 我々 features.py:wls_b_well と同名)
  multi_scale_sc                   (= 我々 multi_scale_ncc と類似 paradigm)
  gr_envelope, gr_energy           (= GR signal feature)
  beam_search, run_all_beams      (= beam search ensemble)
  affine_cal, robust_slope         (= robust 統計)

dual PF:
  _pf_z_loop, _pf_ancc_loop        (= Numba JIT 内部)
  run_pf_z, run_pf_ancc            (= 公開 API、 ravaghi 同名 + 同 paradigm)

TabICL + GP:
  (TabICL は scikit-learn-style regressor、 GP は Matern 系 Sparse GP)
  apply_exact_train_coordinate_blend  (= train coordinate を test に blend、 独自 paradigm)

augmentation:
  (= aug_k 9 出現、 visible_ratio random masking)

util:
  env_flag, env_int, env_float
  _cache_candidates、 find_artifact_dir
```

## 4. ravaghi vs thbdh v10 比較

```
component               ravaghi (LB 9.43)        thbdh v10 (LB 不明)         我々 ours
─────────────────────────────────────────────────────────────────────────────────────
LGB / CatBoost          ✅ 3×3 = 6 base          ✅ small role               ✅ 6 base
TabICL                  ❌                       ✅ main base (= GPU 必須)   ✅ exp006 試行済
GP / Sparse GP          ❌                       ✅ 利用                     ✅ exp009 v2 で 案 E inject 済
dual PF Z + ANCC        ✅                       ✅                          ✅ exp016 inject 済
beam search             ✅ Numba                 ✅ Numba                    ✅ 旧 pandas (= exp016 inject 未)
seg_b_well              ✅ cell 6                ✅ wls_b_well 同等          ✅ src/rogii (= exp016 未 inject)
multi_scale_ncc         ✅ cell 6                ✅ multi_scale_sc           ✅ src/rogii (= exp016 未 inject)
apply_pp envelope       ✅ md_since/tau          ✅ md_since/tau             ✅ src/rogii/postproc_optuna.py
3-axis Optuna grid      ✅ 500-trial TPE         ✅ likely                   ✅ exp016 2530-cell exhaustive
data augmentation       ❌                       ✅ aug_k 採用                ❌ 案 F sketch のみ
per-row kNN             ❌                       ✅ partial                  ❌ 案 G sketch のみ
train coordinate blend  ❌                       ✅ 独自 paradigm            ❌
Climber blend           ✅ ravaghi Climber       ✅ Hill Climb 派生 推定     ❌→ inline self-Climber inject 済
```

## 5. thbdh v10 が LB 何 ft かの推定

- thbdh 過去 v4 (= aeroridge) は LB 9.916 (= rank 36/779、 docs/research/2026-05-11-public-source-audit.dense.md § 1.5)
- v10 はそれより 6 version 後、 augmentation + TabICL + dual PF Z 等の改善が加わっていれば LB 9.5-9.7 帯?
- 直接 LB 値は kernel page を見る必要 (= 我々 fork は 失敗、 thbdh original を 観察するしかない)
- 推定: **LB 9.50 〜 9.75 帯** (= ravaghi 9.43 と aeroridge 9.916 の中間、 ただし TabICL 含むので上振れ可能)

## 6. exp018 (= thbdh v10 fork) は GPU quota reset 後に再 push 候補

- GPU quota reset = weekly、 Kaggle の reset 日次は user 不明 (= 確認必要)
- reset 後に exp018 を **enable_gpu=true** + TabICL 動作 で再 push、 LB 取得

ただし優先度:
- exp020 (= A3/A4/A5 inject、 親 doc § 3) の方が LB lift 期待大 (= ravaghi 9.43 path)
- exp018 thbdh v10 fork は **paradigm diversity の reference point** として位置付け、 priority 中

## 7. 我々 Phase B/C 設計への informing

- thbdh v10 が augmentation + TabICL + GP を **production 動作** で実装 = 我々 Phase B (= path F augmentation) + Phase C (= TabICL) の **生きた reference**
- 特に `aug_k` の具体 column construction、 TabICL の context size 設定等を thbdh v10 から学べる
- 親 doc `docs/research/2026-05-11-public-source-audit.dense.md § 1.5` の aeroridge schema 解析 を v10 で update する価値あり

## 8. 関連 doc

- 親 (= ravaghi 構造解析): `2026-05-13-ravaghi-cell6-structural-mapping.dense.md`
- thbdh v10 author: `thbdh5765/rogii-v10-fresh-artifact-infer`、 また v11 (= 3 votes、 5/12 09:55) が公開
- 過去 audit: `docs/research/2026-05-11-public-source-audit.dense.md § 1.5` (= aeroridge schema 解析)
- 我々 Phase B sketch: `docs/research/2026-05-11-winning-path-F-augmentation-sketch.dense.md`
- 我々 Phase C: `docs/research/2026-05-11-winning-path-A-nn-sketch.dense.md` (= NN seq 別 paradigm)
