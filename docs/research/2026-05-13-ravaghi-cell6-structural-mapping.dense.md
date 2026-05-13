# ravaghi cell 6 = 我々 Phase A 全 modules の super-set、 構造 mapping 完了

> 起点: 2026-05-13 Phase 9 deep dive、 ravaghi NB cell 6 (= 27844 chars) の解析で **Phase A 5 module 完全 mapping** 発見
> 動機: exp017 LB scoring 待ち中に「hc は何の組合せで LB 9.43 か」 を構造 reverse engineer
> 親 doc: `2026-05-13-hc-correction-signature-selective-blockwise.dense.md` (= 改善 path 確定)

---

## 1. 結論

ravaghi/wellbore-geology-prediction-hill-climbing **cell 6** に以下 18 関数が定義されており、 我々 Phase A 全 module と **構造 1:1 対応**:

```
ravaghi cell 6 func              我々 implementation                            Phase A AC
─────────────────────────────────────────────────────────────────────────────────────
_beam_jit                        src/rogii/beam_pf_numba.py:_beam_jit_core      A3 Numba beam
_pf_ancc                         src/rogii/beam_pf_numba.py:run_pf_ancc         A3 ANCC PF
_pf_z                            src/rogii/beam_pf_numba.py (= raunakdey07 port) A3.1 dual PF Z (= exp016 inject 済)
beam_search                      src/rogii/beam_pf_numba.py:beam_search_numba   A3
run_pf_ancc                      同上                                            A3
run_pf_z                         同上                                            A3.1
seg_b_well       ★               src/rogii/segment_features.py:seg_b_well       A4 (= 5 tests pass、 exp016 未 inject)
multi_scale_ncc  ★               src/rogii/segment_features.py:multi_scale_ncc  A5 (= 5 tests pass、 exp016 未 inject)
build_well                       (= 各 well 単位の feature build)                (= 我々 features.py 内 build_features_v3)
build_dataset                    (= 全 wells aggregation)                        同上
... (= 残 helper funcs)
```

→ **Hill Climb LB 9.43 = A2 Climber + A3 Numba beam ±2 + A3.1 dual PF Z + A4 seg_b_well + A5 multi_scale_ncc + A6 apply_pp 3-axis Optuna**

---

## 2. ours exp016 vs hc の構造比較

```
component                  ours exp009 v2 (LB 9.738)   ours exp016 (RUNNING)         hc fork (= 期待 LB 9.43)
A2 Climber                 ❌ Ridge(positive=True)      ✅ inline self-Climber         ✅ ravaghi Climber
A3 Numba beam ±2           ❌ pandas beam_search        ❌ exp016 未 inject (= 旧 beam) ✅ _beam_jit
A3.1 dual PF Z             ❌                           ✅ run_pf_z inject 済           ✅ _pf_z
A4 seg_b_well              ❌                           ❌ exp016 未 inject             ✅ cell 6 内
A5 multi_scale_ncc         ❌                           ❌ exp016 未 inject             ✅ cell 6 内
A6 apply_pp 3-axis grid    ❌ 2-axis grid              ✅ 2530-cell exhaustive        ✅ Optuna 500-trial TPE
```

→ **exp016 (= RUNNING) = A2 + A3.1 + A6 のみ統合**、 A3 / A4 / A5 は **未 inject**。 つまり Phase A.5 (= 別 kernel exp020) で A3/A4/A5 を追加 inject すれば構造的完全同等。

---

## 3. 期待 LB lift 試算 (= 各 module の単独効果)

仮定: 各 module は独立 lift、 累積する (= 親 docs § 7 「Hill Climb 強さ分析」 と整合)

```
component             単独 lift (ft)       cumulative LB     status
基準 = exp009 v2                            9.738              ✅
A2 Climber           -0.05 〜 -0.15         9.59-9.69          exp016 inject 済
A3.1 dual PF Z       -0.05 〜 -0.15         9.44-9.64          exp016 inject 済
A6 apply_pp 3-axis    -0.10 〜 -0.25        9.19-9.54          exp016 inject 済
A3 Numba beam ±2      -0.05 〜 -0.15        9.04-9.49          exp020 候補
A4 seg_b_well         -0.05 〜 -0.10        8.94-9.44          exp020 候補
A5 multi_scale_ncc    -0.05 〜 -0.10        8.84-9.39          exp020 候補
```

- **exp016 期待 LB 9.19-9.54 帯** (= A2+A3.1+A6)
- **exp020 期待 LB 8.84-9.39 帯** (= + A3/A4/A5)
- 上限 8.84 まで詰めれば **rank top 10 圏** (= LB 9.43 現在 8 位)、 **gold 9.6 帯クリア**

→ **段階的 LB 8 切りも視界に入る**

---

## 4. exp020 implementation 概要 (= 後続 kernel)

`kaggle_kernels/exp020_phase_a_plus/exp020_phase_a_plus.py` 想定:
1. exp016 (= ky7240/rogii-exp015-cpu-sparse-gp version 2) の py を base copy
2. cell 6 から `seg_b_well`、 `multi_scale_ncc` 関数 (= 我々 src/rogii/segment_features.py からも copy 可) を inline
3. `build_features` 内で seg_b_well / multi_scale_ncc を呼び features 拡張
4. A3 Numba beam ±2 を既存 `beam_search` と並走 (= ablation 可)
5. Climber blend で「ours 9 base + seg_b_well 1 base + multi_scale_ncc 1 base」 = 11 base
6. apply_pp は exp016 と同じ 3-axis exhaustive grid

サイズ: exp016 (= 4862 行) + 約 500 行追加 = 5400 行想定

---

## 5. exp017 fork submit の意義 (= 構造比較 reference point)

exp017 = ravaghi NB のまま + inline Climber = **「Hill Climb 完全 copy + 我々 Climber 互換性確認」**

LB 結果次第で:
- **exp017 LB ≈ 9.43**: 我々 Climber が ravaghi PyPI Climber と数値等価、 exp020 設計の base 確定
- **exp017 LB > 9.43 (= 例 9.55)**: 我々 Climber が劣る、 mode `continuous` でなく `discrete` 等の調整必要
- **exp017 LB < 9.43**: ravaghi original より良い、 raunakdey07 風 normalize_weights=True 効果

→ exp017 PENDING 中、 LB 確定で 重要 calibration

---

## 6. 関連 doc

- 親 (= 改善 path): `2026-05-13-hc-correction-signature-selective-blockwise.dense.md`
- ravaghi cell 22 (= apply_pp): 親 doc § 4 で formula 解析、 我々 `src/rogii/postproc_optuna.py:42-45` と完全一致確認
- 我々 Phase A criteria: `.criteria/kaggle-rogii-phase-a-2026-05-12.yaml`
- 既存 src 実装: `src/rogii/{hill_climb,beam_pf_numba,segment_features,postproc_optuna}.py` (= 36 tests pass)
