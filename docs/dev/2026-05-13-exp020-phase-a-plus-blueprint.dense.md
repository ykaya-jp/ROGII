# exp020 = Phase A.5 (= A3 Numba beam ±2 + A4 seg_b_well + A5 multi_scale_ncc inject) blueprint

> 起点: 2026-05-13 exp017 LB 10.030 (= 公開 fork が CPU 環境で再現不可) を受けて、 **自前 ours 系で Phase A 完全構成を実現** する path
> 親 doc: `2026-05-13-ravaghi-cell6-structural-mapping.dense.md` (= 構造 1:1 対応確認)
> 前提: exp016 COMPLETE 後着手、 exp016 LB を出発点として A3+A4+A5 単独効果を 1 kernel で測定

---

## 1. 目的 + 期待 LB lift

### 目的
exp016 (= A2 + A3.1 + A6 のみ統合) に **A3 + A4 + A5 を追加 inject** し、 Phase A 全 5 module を完全実装。

### 期待 LB lift (= 親 doc § 3 試算から)
```
exp016 (= 着手前提)        :  9.2 〜 9.5 帯 (= 期待中央)
+ A3 Numba beam ±2         : -0.05 〜 -0.15
+ A4 seg_b_well            : -0.05 〜 -0.10
+ A5 multi_scale_ncc       : -0.05 〜 -0.10
合計 (= exp020)            :  8.9 〜 9.4 帯
```

→ exp020 で **Gold cutoff 9.6 帯確実クリア**、 上振れで **LB 9.0 切り射程**

---

## 2. 設計上の前提

### 2.1 exp016 base
- exp016 = `kaggle_kernels/exp016_phase_a_integrated/exp016_phase_a_integrated.py` (= 4862 行)
- 既存統合済: inline Climber (= line 155-284)、 `_apply_pp_3axis` (= line 293)、 `_optimize_postproc_grid_2530` (= line 309)、 `run_pf_ancc` (= line 1224)、 `run_pf_z` (= line 1163)
- 未統合: 旧 `beam_search` (= line 888、 pandas-based ±1) はそのまま残し、 並走で Numba 版を追加

### 2.2 module location
- A3 Numba beam: `src/rogii/beam_pf_numba.py:beam_search_numba` (= 9 tests pass)
- A4 seg_b_well: `src/rogii/segment_features.py:seg_b_well` (= 5 tests pass)
- A5 multi_scale_ncc: `src/rogii/segment_features.py:multi_scale_ncc` (= 5 tests pass)

### 2.3 Kaggle 環境前提
- CPU mode (= GPU quota reset 待ちでも push 可)、 enable_internet=false
- Numba JIT は CPU でも動作、 cell の最初に warm-up step 推奨
- dataset_sources 既存継承 (= exp016 と同じ 4 件)

---

## 3. 実装手順 (= step-by-step)

### Step 1: kaggle_kernels/exp020_phase_a_plus/ ディレクトリ作成
```bash
mkdir -p kaggle_kernels/exp020_phase_a_plus
cp kaggle_kernels/exp016_phase_a_integrated/exp016_phase_a_integrated.py \
   kaggle_kernels/exp020_phase_a_plus/exp020_phase_a_plus.py
```

### Step 2: kernel-metadata.json 作成
```json
{
  "id": "ky7240/rogii-exp020-phase-a-plus",
  "title": "ROGII exp020 phase A plus",
  "code_file": "exp020_phase_a_plus.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "false",
  "enable_tpu": "false",
  "enable_internet": "false",
  "dataset_sources": [
    "karnakbaevarthur/rogii-code-helper-dataset",
    "thermostatic/rogii-tabicl-v2-public-assets",
    "thbdh5765/rogii-v1-train-cache",
    "ky7240/rogii-per-well-stats"
  ],
  "competition_sources": ["rogii-wellbore-geology-prediction"],
  "kernel_sources": []
}
```

### Step 3: A3 Numba beam ±2 inject
exp016 内の旧 `beam_search` (= line 888-940) **直後** に、 `src/rogii/beam_pf_numba.py` の `_beam_jit_core` + `beam_search_numba` をそのまま inline。 既存 `beam_search` も残し、 並走可能化:

```python
# === A3 Numba beam ±2 (= self-implemented, see src/rogii/beam_pf_numba.py) ===
from numba import njit

@njit(cache=True, fastmath=True)
def _beam_jit_core(hgr_full, tw_tvt, tw_gr, last_known_tvt, bs, mc, es, r, delta_range=2):
    # ... (= 200 行程度、 src/rogii/beam_pf_numba.py 参照)
    pass

def beam_search_numba(hgr_full, tw_tvt, tw_gr, last_known_tvt, bs, mc, es, r, tag, delta_range=2):
    # wrapper, calls _beam_jit_core
    pass
```

旧 `beam_search` と並走、 ablation 可能:
```python
# main pipeline 内
beam_offs_legacy = compute_beam_features(...)  # 旧 ±1
beam_offs_numba  = compute_beam_features_numba(...)  # 新 ±2 (= A3)
```

### Step 4: A4 seg_b_well inject
`src/rogii/segment_features.py:seg_b_well` を inline:

```python
# === A4 seg_b_well (= per formation × phase 4 phase) ===
def seg_b_well(hw_features, formations=FORMATIONS):
    """Compute b_well per formation × phase (= early/mid/late/wls/full).
    Returns 4 new feature columns per formation.
    """
    out = {}
    for f in formations:
        for phase in ['early', 'mid', 'late', 'wls']:
            col = f'b_well_{f}_{phase}'
            out[col] = compute_b_well_segment(hw_features, f, phase)
    return out
```

build_features 内で seg_b_well の出力を **新 24 columns (= 6 formation × 4 phase)** として追加。

### Step 5: A5 multi_scale_ncc inject
`src/rogii/segment_features.py:multi_scale_ncc` を inline:

```python
# === A5 multi_scale_ncc (= softmax NCC、 hws=(8,15,25)) ===
def multi_scale_ncc(hw, tw_tvt, tw_gr, hws=(8, 15, 25)):
    """Multi-scale NCC with softmax ensemble.
    Returns 3*N_anchor features.
    """
    pass
```

旧 self_corr_tvt と並走可能、 ablation。

### Step 6: Climber blend を 9 base → 11 base に拡張
新 base:
- base 10: ours 9-base + A4 seg_b_well features を加えた LGB
- base 11: ours 9-base + A5 multi_scale_ncc features を加えた LGB

または、 Climber に新 OOF 列を 2 個追加して 11 列 OOF matrix で fit:

```python
# 既存 oof_preds (= 9 base) + seg_b_well predictions + multi_scale_ncc predictions
oof_preds_extended = np.hstack([oof_preds_9base, oof_seg_b_well, oof_multi_ncc])  # (n, 11)
climber.fit(oof_preds_extended, y)
```

### Step 7: apply_pp 3-axis grid は exp016 と同じ (= 2530-cell exhaustive、 line 309)
変更不要、 ただし Climber blend output が変わるので `model_delta` 引数の値が変化、 grid best param が違う可能性

### Step 8: Kaggle に push
```bash
.venv/bin/kaggle kernels push -p kaggle_kernels/exp020_phase_a_plus/
```

slug `ky7240/rogii-exp020-phase-a-plus` 新規作成 (= 1 回目 push で half-created 回避のため、 v1 から始める)

---

## 4. ablation 設計 (= GM §11 datapoint 価値)

exp020 を 1 kernel で run しつつ、 内部 flag で **どの module が active か** を制御:

```python
# 環境変数または定数で ablation 制御
USE_A3_NUMBA = True
USE_A4_SEG_BWELL = True
USE_A5_MULTI_NCC = True

# 4 runs ablation:
# - exp020 v1: USE_A3=False, USE_A4=False, USE_A5=False  → exp016 同等 base reproduce
# - exp020 v2: USE_A3=True,  USE_A4=False, USE_A5=False  → A3 単独効果
# - exp020 v3: USE_A3=True,  USE_A4=True,  USE_A5=False  → A3+A4
# - exp020 v4: USE_A3=True,  USE_A4=True,  USE_A5=True   → full Phase A.5
```

4 submit で各 module の単独 lift を測定。 ただし 24h quota は 5 件 / day、 4 件 ablation + exp017 + exp016 = 6 件 → 24h で 1 件 over → **段階的 submit (= v1+v2 を 1 日目、 v3+v4 を 2 日目)** で組む。

または、 4 runs を **kernel 内部で並列** に呼び、 各 submission.csv を別 file 名 で保存、 1 push で 4 file output → 1 submit で 1 file 選んで投稿、 残 3 file は次 submit で。

---

## 5. 検証順序 (= exp016 結果 dependency)

### exp016 COMPLETE 後の判断分岐:

#### Case A: exp016 LB ≤ 9.5 (= Phase A 目標達成)
- exp020 着手、 ablation v1〜v4 順次 submit
- 目標: LB 9.0 帯到達

#### Case B: exp016 LB > 9.5 but < 9.74 (= exp009 v2 同等以下、 marginal)
- exp020 着手、 ただし「A3/A4/A5 inject が effective か疑問」 → ablation v2 (= A3 単独) を最初に submit、 effect 確認
- 効果なければ exp020 path 中止、 Phase B (= augmentation + per-row kNN) に pivot

#### Case C: exp016 LB ≥ 9.74 (= 大破綻、 exp017 と同パターン)
- exp020 着手前に **postmortem** (= 我々 inline Climber / Numba PF の non-determinism 真因調査)
- `/postmortem` skill 起動、 lesson 化
- inline Climber を ravaghi PyPI Climber (= internet enable + pip install) と比較する simple kernel を別途 push

---

## 6. submission-analyses entry skeleton (= exp020 submit 後)

```markdown
## §15 v_(n) → v_(n+1) 分解 (= exp020 phase A plus ablation v4 = full)

### submission_id: <TBD>
- timestamp (UTC): <yyyy-mm-dd HH:MM>
- build_commit: <git short SHA>
- mode: exp020 = exp016 base + A3 Numba beam ±2 + A4 seg_b_well + A5 multi_scale_ncc
- base submission: <exp016 submission_id>
- source_count: 3 (= A3 + A4 + A5)
- per-task source diff:
  - A3 Numba beam ±2: <effect ft>
  - A4 seg_b_well: <effect ft>
  - A5 multi_scale_ncc: <effect ft>
- effect isolate: <ablation v2/v3/v4 比較>
- 仮説帰納: <expected vs actual>
- roadmap refine: <Phase B/C 着手判断>
- shake-up risk: <C1 fold 0 OOF vs LB |差|>
```

---

## 7. 教訓継承 (= exp017 LB 10.030 失敗から、 lessons.md 候補)

- **PF Z dual + Numba JIT は CPU で deterministic 化必要**、 exp020 では seed 固定 + 単 thread mode で再現性 verify
- **OOF と LB の decoupling 防止**: CV strategy を Phase B B3 (= stratified Edge Q + adversarial val) で test set proxy 化
- **inline 化 module の numerical 等価性 test 必須**: src/rogii/{hill_climb, beam_pf_numba, segment_features}.py の 36 tests に **PyPI 等価性 test** 追加

---

## 8. timeline 試算

- exp020 implementation: 60-90 min (= exp016 base copy + 500 行追加)
- Kaggle push: 5 min
- kernel run (= CPU): 3-6h (= exp016 と同等)
- LB scoring: 30-60 min
- 合計 1 ablation: ~7-10h
- 4 ablation: ~30-40h (= 段階的 2-3 日)

→ deadline 8/5 まで残 84 day、 exp020 + Phase B + C で 70 day 想定、 余裕あり

---

## 9. 関連 doc

- 親 (= 構造 mapping): `docs/research/2026-05-13-ravaghi-cell6-structural-mapping.dense.md`
- 親 (= 教訓): `docs/research/2026-05-12-submission-analyses.md` §14
- exp016 source: `kaggle_kernels/exp016_phase_a_integrated/exp016_phase_a_integrated.py`
- src 実装: `src/rogii/beam_pf_numba.py`、 `src/rogii/segment_features.py`、 `tests/test_*.py` (= 19 tests pass for A3+A4+A5)
