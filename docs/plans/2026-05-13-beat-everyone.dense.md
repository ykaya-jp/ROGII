# ROGII 「公開 NB に負ける真因 + 全員に勝つ path」 plan

> 作成: 2026-05-13 user 指示「なぜ公開 NB に負けるか + 全員に勝つには、 徹底リサーチして plan を立てろ」
> Source: 3 Explore agents の dense research (= 8 既存 docs + 11 件 GM-level deep dive doc + LB 779 team の paradigm 推測)
> 配置: `~/.claude/plans/wobbly-moseying-petal.md`、 実装着手時 `~/projects/kaggle/ROGII/docs/plans/2026-05-13-beat-everyone.dense.md` に promote 想定

---

## Context

### 現状 (= 2026-05-13 早朝、 UTC 22:57)
- **LB rank**: ours exp009 v2 LB **9.738** (= rank ~17/779、 5/11 SCORED)
- **LB 1 位**: Virtute **8.966** (= 5/12 10:21Z、 9 切り)
- **公開 NB 最強**: ravaghi/wellbore-geology-prediction-hill-climbing **LB 9.43** (= eiheychan fork で 8 位入りした 9.430 = 我々が再現できない)
- **Gold cutoff**: ~rank 40 = LB 9.6 帯
- **Deadline**: 2026-08-05 = **残 84 day**

### 直近 14h 学習 (= 2026-05-12 14:00Z 〜 2026-05-13 06:00Z)
1. **exp017 hillclimb fork v3 LB = 10.030** (期待 9.43、 -0.60 ft 大破綻)
   - OOF 10.380 (= ravaghi 10.394 より 0.014 良) vs LB 10.030 = **完全 decoupling**
   - 真因: PF/Numba non-determinism + CPU mode floating-point + 環境差
   - **結論: 公開 NB fork は確実 defense ラインではない** (= 仮説 棄却)
2. **exp015 (= 内部 exp016) API "RUNNING" stale cache** = 8h+ 走っているように見えて実は止まっていた、 lastRunTime + output 空 で発覚
   - exp016-v2 = `ky7240/rogii-exp016-phase-a-integrated-v2` 再 push 済 (= 22:46Z)、 実走確認待ち
3. **GM-level deep dive 11 件 + analyses §14 + exp020 blueprint** 産出済 (= ~2000 行)

### Why this plan now
user 質問: 「**なぜ普通に公開 NB に負けるのか、 どうすれば全員に勝てるか**」。 これは「軽さ-driven でなく数理本質で攻める」 (= ~/projects/kaggle/CLAUDE.md §11 commit) の core 質問。 plan は **両方の答えを構造的に固定** し、 残 84 day で gold (= 80% 確信)・優勝 (= 30% 確信) の 2 path を提示する。

---

## §1. 「公開 NB に負ける」 真因 (= ranked、 root cause)

### R1. Phase A 5 module の **未完全 inject** (= 確度 大、 修正 medium)
- ravaghi cell 6 (= 27K 行 setup) は **A2 Climber + A3 Numba beam ±2 + A3.1 dual PF Z + A4 seg_b_well + A5 multi_scale_ncc + A6 apply_pp envelope** の **6 module 全部含む**
- 我々 exp016 内部統合 = **A2 + A3.1 + A6 のみ inject** (= line 155-309)、 **A3 Numba beam / A4 seg_b_well / A5 multi_scale_ncc 未 inject**
- A4 / A5 は実装済 (= src/rogii/segment_features.py、 10 tests pass)、 inject するだけ
- 出典: `docs/research/2026-05-13-ravaghi-cell6-structural-mapping.dense.md §2`
- **期待 lift**: -0.15 〜 -0.35 ft

### R2. Ensemble weight の **paradigm gap** (= 確度 大、 修正 small、 既統合)
- ours exp009 v2 = `Ridge(positive=True)` 9 base → overcounted base に対し subtract 不可
- ravaghi LB 9.43 = Climber (= Caruana 2004 generalized、 negative weight 許容、 continuous Gaussian σ=0.02 + patience 1500、 raunakdey07 派生)
- 状態: src/rogii/hill_climb.py で **自前再実装済 (= 16 tests pass)**、 exp016 inline inject 済
- ただし: exp017 で LB 10.030 = **inline Climber 動作するが LB 期待を裏切る** = parameter tuning or PF 非決定性で OOF と LB decouple
- 出典: `docs/research/2026-05-13-hypothesis-calibration-slope-cap-trap.dense.md §1-2`、 `docs/research/2026-05-12-submission-analyses.md §14`
- **期待 lift (= exp016 base に対し)**: -0.05 〜 -0.15 ft

### R3. Post-proc 探索 **granularity gap** (= 確度 大、 修正 small、 既統合)
- ours exp009 v2 = 2-axis grid (alpha × tau)
- ravaghi LB 9.43 = Optuna TPE 500-trial 3-axis (alpha × tau × w_pf)
- ours exp016 = **3-axis exhaustive 2530-cell grid** (= hc 同等以上)
- formula 自体は **両者完全一致**: `d *= 1 - exp(-md_since / tau)` envelope (= `src/rogii/postproc_optuna.py:42-45` と `ravaghi cell 22` 1:1 一致)
- 出典: `docs/research/2026-05-13-ravaghi-cell6-structural-mapping.dense.md §4`
- **期待 lift (= exp016 vs exp009 v2)**: -0.10 〜 -0.25 ft

### R4. **CV-LB decoupling** = OOF が test set を represent しない (= 確度 大、 修正 大)
- exp017 で OOF 10.380 > ravaghi 10.394 (= 我々 better) なのに LB 10.030 << ravaghi 9.43 (= 我々 worse)
- 真因仮説:
  - (a) PF Z dual + Numba JIT の **parallelism non-determinism** = 再 run で結果ばらつき
  - (b) CPU mode LGB の **floating-point precision** ≠ GPU mode (= 公開 fork submitter は GPU)
  - (c) test 3 wells の hidden 真値 distribution が train 770 wells の hidden distribution と systematic に異なる
- 出典: `docs/research/2026-05-12-submission-analyses.md §14 教訓 1-4`
- **修正 path**: Phase B B3 (= stratified Edge Q + adversarial validation) で **fold strategy を test-proxy 化**
- **期待 lift (= calibration 確立後)**: -0.05 〜 -0.10 ft

### R5. **Test 3 wells 特化 paradigm 不在** (= 確度 中、 修正 大)
- Test set = `000d7d20` (visible 27%) + `00bbac68` (visible 20%) + `00e12e8b` (visible 33%) の **3 wells のみ**、 14151 hidden rows
- 00e12e8b で ours mean_abs **1.40 ft**、 max 4.26 ft、 1280 連続 rows +3.30 ft 一様 bias
- 我々 LGB は train 770 wells で fit、 test 3 wells に対し over-generalized
- 改善 path:
  - per-well visible_ratio adaptation (= Phase B-1 augmentation)
  - per-row kNN inference (= Phase B-2 paradigm 6、 enisteper LB 9.960 evidence)
  - Per-Well MoE (= Phase C-2、 paradigm 3、 全 NB 未公開機会)
- 出典: `docs/research/2026-05-13-test-set-divergence-analysis.dense.md`、 `docs/research/2026-05-13-slope-extrapolation-decisive-analysis.dense.md`
- **期待 lift**: -0.20 〜 -0.50 ft (= Phase B + C 累積)

### R6. **公開 NB 全 miss の paradigm 機会喪失** (= 確度 中、 修正 大、 独占機会)
- 公開済 paradigm: HC (= ravaghi) / aeroridge augmentation (= thbdh v4 LB 9.916) / TabICL+aug (= thbdh v11) / per-row kNN (= enisteper LB 9.960)
- **公開 NB 全 miss**:
  - Paradigm 3 (Per-Well MoE、 3-expert visible_ratio × b_cluster gate) — **00bbac68 evidence あり**
  - Paradigm 5 (NN seq Mamba state-space + Multi-task aux head) — takaito LSTM tutorial のみ
- これらは我々の **独占機会**
- 出典: Agent 2 research, `docs/research/2026-05-11-public-source-audit.dense.md §1-7`
- **期待 lift**: -0.30 〜 -0.60 ft (= Phase C)

---

## §2. 「全員に勝つ」 (= LB 1 位 ≤ 8.966) 数理的必要条件

### NC1. self-compiled paradigm ≥ 75% (= 公開 NB artifact 直 ingest ≤ 25%)
- 必要根拠: ARChitects 2024 「static transduction で LB 天井 11 帯、 transduction + induction mix が必須」
- 検査: Ridge meta blend で `own_base_weights.sum() / total_weights.sum() ≥ 0.75`
- 状態: exp009 v2 = 9 自前 base + Sparse GP + Edge Q/R/S/O = self-compiled 80%+、 ✅ 達成圏

### NC2. paradigm 多様性 ≥ 3 種 mix (= ARChitects 2024 strict 法則)
- transduction (= 公開 base 改良) + induction (= 自前 solver) + force/hand-craft の 3 paradigm 以上
- 状態: 現状 transduction (= LGB+CB blend) + induction (= Sparse GP) のみ、 force/hand-craft (= per-well rule、 MoE) 不足
- Phase B (augmentation + kNN) + Phase C (NN seq + MoE) で 4-5 paradigm 達成

### NC3. CV-LB correlation σ(ratio drift) < 0.01
- 各 submit 後の `LB / est` 比 σ を 5 submit ごとに再算出
- σ > 0.02 = overfit or rule 変更、 即 CV strategy 再検証
- 状態: 現状 exp017 で OOF 10.380 → LB 10.030 で decoupling 検出、 ⚠️ **未達成、 critical bottleneck**

### NC4. LB ≤ 8.966 累積 lift = -0.772 ft (= 9.738 base から)
- 構成: Phase A.5 (-0.30〜-0.50) + Phase B (-0.30〜-0.50) + Phase C (-0.20〜-0.50) = **-0.80〜-1.50 ft 累積**
- 下限シナリオで LB 8.94、 上限で LB 8.24 = 1 位射程

### NC5. shake-up 耐性: private LB ≥ 0.5x public LB
- public 30% / private 70% split
- fold 0 (= adversarial val で test-similar 140 wells) と LB |差| ≤ 0.2 ft で proxy
- final 7 day 前に safe + risky 2 submit freeze、 safe = LB ≤ 9.0、 risky = LB ≤ 8.5

---

## §3. 5 Phase 実装 roadmap (= 残 84 day)

### Phase A.5 = exp020 (= Week 1-2、 7-14 day)
**目標**: Phase A 完全構成、 LB **9.4 〜 9.5 帯** (= ravaghi 9.43 同等射程)

**実装**:
1. **exp016-v2 完走 + submit 確認** (= 残 1-6h、 LB 出る前提 < 9.7 帯)
2. exp020 kernel 作成 (= `kaggle_kernels/exp020_phase_a_plus/exp020_phase_a_plus.py`、 blueprint = `docs/dev/2026-05-13-exp020-phase-a-plus-blueprint.dense.md`)
   - exp016-v2 base + **A3 Numba beam ±2** inject (= `src/rogii/beam_pf_numba.py:_beam_jit_core` + `beam_search_numba`、 12 tests pass)
   - + **A4 seg_b_well** inject (= `src/rogii/segment_features.py:seg_b_well`、 5 tests pass)
   - + **A5 multi_scale_ncc** inject (= `src/rogii/segment_features.py:multi_scale_ncc`、 5 tests pass)
3. ablation v1〜v4 (= A3 / A3+A4 / A3+A4+A5 / full Phase A) を 4 段階 submit (= 24h quota 配慮で 2 day 分けて)
4. 各 submit 後 30 min 以内に `docs/research/2026-05-12-submission-analyses.md` に §15-§18 append

**critical files**:
- `kaggle_kernels/exp016_phase_a_integrated/exp016_phase_a_integrated.py` (= base copy 元、 4862 行)
- `src/rogii/{beam_pf_numba, segment_features, postproc_optuna, hill_climb}.py` (= 既存実装、 計 48 tests pass)
- `kaggle_kernels/exp020_phase_a_plus/` (= 新規作成)
- `.criteria/kaggle-rogii-phase-a-plus-2026-05-13.yaml` (= 新規 plan criteria、 `/plan` 起票必要)

### Phase B-1 = exp021 (= Week 2-3、 augmentation paradigm 5)
**目標**: visible_ratio robust LGB、 LB **9.1 〜 9.3 帯**

**実装**:
1. `src/rogii/augment.py` 新規 (= `docs/research/2026-05-11-winning-path-F-augmentation-sketch.dense.md §3.1` の `augment_well_visible_ratio` 実装)
2. `src/rogii/features.py` 拡張 (= aug_k feature column 対応)
3. `src/rogii/cv.py` 拡張 (= 4-key stratified Edge Q + aug_k stratify、 winning-path-F §3.3)
4. exp021 kernel (= exp020 base + augmentation re-train、 train data 5-6× 増、 Colab Pro+ 30-60h run)

**critical files**:
- `src/rogii/augment.py` (= 新規、 100 行想定)
- `tests/test_augment.py` (= 新規 TDD、 RED → GREEN)
- `kaggle_kernels/exp021_phase_b_aug/`

### Phase B-2 = exp022 (= Week 3、 per-row kNN paradigm 6)
**目標**: enisteper LB 9.960 paradigm 6 取り込み、 LB **8.9 〜 9.1 帯**

**実装**:
1. `src/rogii/knn_row.py` 新規 (= `docs/research/2026-05-11-winning-path-G-perrow-knn-sketch.dense.md §2` の `compute_knn_row_features` 実装)
2. **leak guard 3 層 (CRITICAL)**:
   - (a) 自 well 除外
   - (b) target leak 防止 (= TVT_input は signature に含めない)
   - (c) fold-aware (= val side wells 除外)
3. FAISS index for 3.78M rows × 121 cols (= Local GPU 10-20 min)
4. exp022 kernel (= exp021 base + knn_row features 6 cols add、 9-base + 6 paradigm-6 cols で blend)

**critical files**:
- `src/rogii/knn_row.py` (= 新規、 150 行想定)
- `tests/test_knn_row.py` (= 新規 TDD + leak guard explicit test)

### Phase B-3 = CV-LB calibration 強化 (= 並走、 NC3 必達)
**目標**: σ(ratio drift) < 0.01 (= 5 submit × 2 cycle で確証)

**実装**:
1. `docs/research/2026-05-13-local-vs-lb-correlation.md` 新規 (= 各 submit ごとに append、 ratio + σ_trend)
2. C2.v2 stratified Edge Q 全 base 再 OOF 生成
3. adversarial validation で test-similar 140 wells 特定 → fold 0 として固定
4. 形名参同: `/plan` で `.criteria/kaggle-rogii-cv-lb-calib-2026-05-13.yaml` 起票

### Phase C-1 = exp023 (= Week 4-6、 NN seq paradigm 5、 高 compute)
**目標**: paradigm 5 (= 公開 NB 全 miss、 独占) 取り込み、 LB **8.5 〜 8.9 帯** ★Win 圏

**実装**:
1. `src/rogii/nn_seq.py` 新規 (= Mamba state-space O(n)、 5-fold × 5-seed = 25 ensemble、 multi-task aux head 6 formation)
2. Colab Pro+ A100 で run (= 50-100 A100-hr 想定、 50h/月 quota の半分使用)
3. exp023 kernel (= exp022 base + NN seq features 拡張)

**critical files**:
- `src/rogii/nn_seq.py` (= 新規、 400 行想定)
- `src/rogii/moe.py` (= 新規、 Per-Well MoE paradigm 3、 Phase C-2)

**risk mitigation (= B1 bottleneck)**:
- 25 ensemble → 9 ensemble (= 3-fold × 3-seed) で compute 削減、 LB lift 期待 -0.15〜-0.40 ft (= 25 の 60% 効果)
- もし 9 ensemble で LB 9.0 切れたら、 25 ensemble に拡大して LB 8.5 切り狙う

### Phase D = final submission freeze (= deadline 7 day 前 = 2026-07-30)
**目標**: safe + risky 2 submit 確定

**実装**:
1. safe slot = Phase B 完了版 (= LB ≤ 9.0、 shake-up 耐性高、 Sparse GP M=500 + 案 Z Physics solver) = **rule 耐性確保**
2. risky slot = Phase C 完了版 (= LB ≤ 8.5、 NN seq + MoE、 private collapse risk あり)
3. C1 fold 0 OOF と LB |差| ≤ 0.2 ft の最終確認

---

## §4. Critical files 全リスト

### 既存修正
- `kaggle_kernels/exp016_phase_a_integrated/exp016_phase_a_integrated.py` (= exp020 base copy 元)
- `kaggle_kernels/exp016_phase_a_integrated/kernel-metadata.json` (= id ky7240/rogii-exp016-phase-a-integrated-v2 にて push 済)
- `src/rogii/features.py` (= aug_k 列対応、 inclination feature multi-window 拡張)
- `src/rogii/cv.py` (= 4-key stratified Edge Q + aug_k stratify)

### 新規作成
- `src/rogii/augment.py` (= Phase B-1)
- `src/rogii/knn_row.py` (= Phase B-2)
- `src/rogii/nn_seq.py` (= Phase C-1)
- `src/rogii/moe.py` (= Phase C-2)
- `src/rogii/gp_sparse.py` (= 案 Z 並走 Sparse GP M=500、 既存 exp009 v2 M=200 拡張)
- `kaggle_kernels/exp020_phase_a_plus/exp020_phase_a_plus.py` (= Phase A.5)
- `kaggle_kernels/exp021_phase_b_aug/exp021_phase_b_aug.py` (= Phase B-1)
- `kaggle_kernels/exp022_phase_b_knn/exp022_phase_b_knn.py` (= Phase B-2)
- `kaggle_kernels/exp023_phase_c_nn_seq/exp023_phase_c_nn_seq.py` (= Phase C-1)
- `tests/test_{augment,knn_row,nn_seq,moe,gp_sparse}.py` (= TDD)
- `.criteria/kaggle-rogii-{phase-a-plus,phase-b-aug,phase-b-knn,phase-c-nn,cv-lb-calib}-2026-05-13.yaml` (= 形名参同)
- `docs/research/2026-05-13-local-vs-lb-correlation.md` (= 各 submit ratio 監視)

### Reuse 既存 (= 触らない、 inject only)
- `src/rogii/{hill_climb, beam_pf_numba, segment_features, postproc_optuna}.py` (= 48 tests pass、 inject 用)

---

## §5. 期待 LB 帯 + 確信度

```
段階                目標 LB    確信度   実装難易度  compute
exp016-v2 完走      9.5-9.7    70%      ✅ done    1-6h Kaggle CPU
Phase A.5 exp020    9.4-9.5    80%      Easy       Kaggle CPU 3-6h × 4 ablation
Phase B-1 exp021    9.1-9.3    60%      Medium     Colab Pro+ 30-60h
Phase B-2 exp022    8.9-9.1    50%      Medium-high  Local GPU 24h + FAISS
Phase B-3 calib     -          90%      Easy       Local CPU 10h
Phase C-1 exp023    8.5-8.9    35%      High       Colab A100 50-100h
Phase C-2 exp023+   8.3-8.7    25%      High       Local + Colab
─────────────────────────────────────────
Gold (≤9.5)         確信度 80% (= Phase A.5 完了で射程)
Win (≤8.5)          確信度 30% (= Phase C 完走必須、 compute risk + private collapse risk)
LB 1 位 (≤8.966)    確信度 50% (= Win 確信度より高い、 1 位 8.966 は Phase B 完走で射程)
```

---

## §6. Verification (= 形名参同 + GM §8 #9 ルール)

### 各 phase ごと
1. **Plan**: `/plan kaggle-rogii-<phase-id>-2026-05-13` で `.criteria/<id>.yaml` 起票 (= acceptance_criteria 機械的判定可能形式)
2. **Implement**: TDD (RED → GREEN → REFACTOR)、 全 unit test pass
3. **Submit**: `competition_submit_cli` 経由 (= 我々 root cause path、 `kaggle competitions submit` CLI bug 回避)
4. **Verify**:
   - LB scoring 後 30 min 以内 `docs/research/2026-05-12-submission-analyses.md` に §N append (= GM §8 #9 schema 厳守)
   - `/verify kaggle-rogii-<phase-id>-2026-05-13` で reviewer subagent 独立判定
   - CV-LB Spearman 計算、 σ_trend 更新
   - C1 fold 0 OOF と LB |差| 算出 (= shake-up proxy)

### 5 submit ごと (= GM §8 #10 周期分析)
1. trend coef 統計分析 (= 過去 5 件 coef σ、 安定 / drift 判定)
2. source pool ROI ranking (= 各 paradigm の累積 LB lift 寄与)
3. 未活用 source 抽出 (= 公開 NB の未取込 paradigm audit)

### final submit (= deadline 7 day 前)
1. safe + risky 2 submit を freeze
2. C1 fold 0 OOF と LB |差| ≤ 0.2 ft 最終確認
3. private collapse risk 評価 (= bootstrap 95% CI lower ≥ 0.5)

---

## §7. Risk + Mitigation

| Risk | 深刻度 | Mitigation |
|---|---|---|
| **B1**: Colab A100-hr 制約 (= Phase C で 50h quota vs 100h 想定) | 大 | NN seq 25 → 9 ensemble、 lift 40% 削減許容 (= -0.40 → -0.24 ft) |
| **B2**: Phase A.5 で LB ≤ 9.5 切れない (= R1 + R2 inject 効果不足) | 大 | A3/A4/A5 単独 ablation で effect 切り分け、 削除候補 paradigm を identify |
| **B3**: CV-LB σ drift > 0.02 (= NC3 違反) | 大 | 即 stop & investigate、 stratified Edge Q strategy 再検証 |
| **B4**: per-row kNN leak guard 3 層が CV-LB gap 拡大 (= 親 doc C4 critique) | 中 | 1 層ずつ ablation で実効性検証、 必要 minimum layers のみ採用 |
| **B5**: risky slot LB ≤ 8.5 未確認で deadline | 中 | safe + risky を「確実 9.0 + upside 8.5」 → 「確実 9.3 + upside 9.0」 にdowngrade、 無理せず |
| **B6**: PF / Numba non-determinism で **exp020 でも LB drift** (= R4 再発) | 中 | seed 固定 + 単 thread mode verify、 exp020 を 2 回 run で結果比較 |

---

## §8. User 判断確定事項 (= 2026-05-13 plan 承認時)

1. **着手順序**: **順次** (= Phase A.5 → B-1 → B-2 → C-1)。 user 指示「推奨に従う、 まっすぐ優勝に向かってくれ」 → effect isolate 優先、 ablation 確実、 14 day で Phase A.5 + B 完了、 LB 9.0 帯到達目標
2. **公開 NB fork priority** = **高、 Colab / Local GPU で並走**:
   - user 指示「Google Colab とか使えばいいのでは、 Local でも GPU あるし」
   - GPU quota reset (= weekly) を待たず、 **Phase A.5 と並走で** ravaghi LB 9.43 fork を Colab Pro+ A100 で run、 submission.csv 生成 → Kaggle dataset upload → 我々 CPU kernel から submit
   - もし LB 9.43 再現できれば **defense ライン確実獲得**、 再現できなければ exp017 の真因が environment 差でなく PF non-determinism 確定 (= 重要 calibration finding)
   - 実装: `kaggle_kernels/exp019_ravaghi_colab_run/` + Colab notebook `notebooks/2026-05-13-ravaghi-colab-rerun.ipynb`
3. **NN seq compute** (= Phase C-1): 25 ensemble (= 50-100 A100-hr) を default、 Colab Pro+ 月予算 50h と Local GPU 168h/week で 賄える、 不足時 9 ensemble に削減
4. **Phase B-2 per-row kNN leak guard 3 層**: 全 3 層を default 採用、 1 層ずつ ablation で実効性検証、 必要 minimum のみ採用 (= CV-LB stability 優先)
5. **Phase C-3 Physics constraint loss**: skip (= marginal lift ≤ 0.05 ft、 §11 「軽さ-driven 回避」 と整合)

---

## §9. 関連 doc

### 既存 GM-level deep dives (= 11 件、 ~2000 行)
- `docs/research/2026-05-13-test-set-divergence-analysis.dense.md`
- `docs/research/2026-05-13-prediction-microstructure-deepdive.dense.md`
- `docs/research/2026-05-13-slope-extrapolation-decisive-analysis.dense.md`
- `docs/research/2026-05-13-naive-slope-extrapolation-is-harmful.dense.md`
- `docs/research/2026-05-13-physical-geometry-inclination-prior.dense.md`
- `docs/research/2026-05-13-full-773-wells-confirmed-physics.dense.md`
- `docs/research/2026-05-13-hypothesis-calibration-slope-cap-trap.dense.md`
- `docs/research/2026-05-13-hc-correction-signature-selective-blockwise.dense.md`
- `docs/research/2026-05-13-ravaghi-cell6-structural-mapping.dense.md`
- `docs/research/2026-05-13-thbdh-v10-structural-survey.dense.md`
- `docs/research/2026-05-12-hill-climb-strength-analysis.dense.md`

### Strategic + planning
- `docs/dev/2026-05-12-plan-gold-to-winning.dense.md` (= 既存 plan §3 案 D/X/Y/Z/W)
- `docs/dev/2026-05-13-exp020-phase-a-plus-blueprint.dense.md` (= 待ち時間で作った)
- `docs/research/2026-05-11-winning-path-A-nn-sketch.dense.md` (= NN seq sketch)
- `docs/research/2026-05-11-winning-path-D-moe-sketch.dense.md` (= MoE sketch)
- `docs/research/2026-05-11-winning-path-F-augmentation-sketch.dense.md` (= aug sketch)
- `docs/research/2026-05-11-winning-path-G-perrow-knn-sketch.dense.md` (= kNN sketch)

### Submission tracking
- `docs/research/2026-05-12-submission-analyses.md` (= §1-§14、 exp017 LB 10.030 + 教訓 4 件)
- `docs/dev/leaderboard.dense.md`
- `~/projects/kaggle/CLAUDE.md` (= GM-level workflow、 §11 commit)
- `~/.claude/CLAUDE.md` (= 主道フレームワーク、 lessons.md)

---

## §10. 即着手 action (= plan 承認後の最初の 24h)

### Track 1 (= 自前 path、 直線で Phase A.5 着手)
1. **exp016-v2 完走確認** (= UTC 23:00 〜 翌 04:00 想定、 6h)
   - lastRunTime + output で **実走 verify** (= API stale cache 警戒)
   - COMPLETE 後 即 `competition_submit_cli` で submit
   - LB scoring 30-60 min
2. **submission-analyses.md §15 append** (= exp016-v2 LB 出る前後 30 min 以内)
3. **Phase A.5 = exp020 着手** (= exp016-v2 LB ≤ 9.7 帯確認後 即)
   - `/plan kaggle-rogii-phase-a-plus-2026-05-13` で `.criteria/` 起票
   - `kaggle_kernels/exp020_phase_a_plus/` 作成 (= exp016-v2 copy + A3+A4+A5 inject)
4. **CV-LB calibration doc 起動** (= `docs/research/2026-05-13-local-vs-lb-correlation.md` 新規、 各 submit append routine 開始)

### Track 2 (= 並走、 Colab GPU で ravaghi fork 再現確認、 user 承認 §8.2)
1. **Colab notebook 作成** = `notebooks/2026-05-13-ravaghi-colab-rerun.ipynb`
   - ravaghi NB を Colab Pro+ A100 で完全 run (= cache load + LGB/CB + Climber + Optuna)
   - submission.csv 生成
2. **Kaggle dataset upload** = `ky7240/rogii-ravaghi-colab-output` (= submission.csv のみ)
3. **Kaggle CPU kernel 作成** = `kaggle_kernels/exp019_ravaghi_colab_run/`
   - dataset から submission.csv 読み込み → そのまま output
   - kernel-metadata に dataset_sources 追加
4. **Submit + LB 確認**
   - LB 9.43 再現 → **defense ライン確実獲得 + exp020 並走で安心 base**
   - LB > 9.5 (= 大破綻) → exp017 と同じ真因確認、 PF non-determinism 確定、 postmortem 起動
5. **Local GPU 環境調整** (= 並走、 user 承認時間あれば)
   - `data/processed/` 含む Kaggle env reproduce on local A100
   - 並列 ablation 用 framework 構築 (= Phase C compute 確保)

これで残 84 day で **Gold 80% / Win 30% / LB 1 位 50%** 確信のロードマップ確定。 Track 1 (= 主道、 Phase A.5 → B → C) + Track 2 (= 公開 NB fork 再現 = defense reference) の **二並走で安全 + upside 両取り**。
