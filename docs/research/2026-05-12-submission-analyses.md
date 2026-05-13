# ROGII Submission 分解 doc (= GM §8 形名参同、 全 submit を「次に活かす」)

> 開始: 2026-05-12
> 用途: 毎 submit 後 30 分以内に append、 per-task source diff + effect isolate + 仮説帰納 + roadmap refine
> ルール: ~/.claude/CLAUDE.md "Plan Mode 出力ルール (Kaggle 関連、 CRITICAL)" = 本文日本語、 コード識別子・URL・file:line・数式・数値は英数字
> 出典: ~/projects/kaggle/CLAUDE.md §8.1 / §11.1 + kaggle-grandmaster-mindset skill §8

---

## ベースライン (= 2026-05-11 まで全 SCORED submit、 9 件)

| # | submission_id | timestamp (UTC) | mode | LB public | LB-LB_(n-1) | note |
|---|---|---|---|---|---|---|
| 1 | 52515207 | 2026-05-10 13:25 | exp002 LGB residual baseline | 14.695 | — | tvt_formula + 6-formation FormationPlaneKNN |
| 2 | 52519856 | 2026-05-10 15:58 | exp003 LGB tysig (自前 features 単独) | 17.510 | +2.815 ⚠️ | blind submit、 CV なし、 features 過剰 overfit |
| 3 | 52520314 | 2026-05-10 16:14 | exp005 cache blend (karnakbaev pretrained 5-base) | 10.317 | -7.193 ✅✅ | base = karnakbaev artifact |
| 4 | 52521223 | 2026-05-10 16:51 | exp006 + TabICL 6th base | 10.503 | +0.186 ⚠️ | TabICL CUDA error、 fallback で Ridge weight 劣化 |
| 5 | 52526612 | 2026-05-10 22:10 | exp007 + Edge Q + Edge M + 自前 4 base | 10.677 | +0.360 ⚠️ | fold-misalign leak、 karnakbaev OOF fold ≠ Edge Q fold |
| 6 | 52528539 | 2026-05-11 00:13 | exp005 v2 + Edge S | 10.203 | -0.114 ✅ | dTVT 0.01 grid snap 効果分離測定成功 |
| 7 | 52528863 | 2026-05-11 00:32 | exp005 v3 + Edge S + Edge R online TTA | 10.387 | +0.184 ⚠️ | Edge R refuted (= base 依存性、 karnakbaev では逆効果) |
| 8 | 52532035 | 2026-05-11 03:15 | exp008 v2 + 案 D Kalman/PF on dTVT | **9.957** | -0.246 ✅✅ | **9 切り達成**、 AR(1) state-space で dTVT auto-correlation chronicle |
| 9 | 52534803 | 2026-05-11 05:37 | exp009 v2 + 案 E Sparse GP M=200 + Edge O | **9.738** | -0.219 ✅✅ | **現 best、 rank 17/779**、 ANCC posterior + variance で uncertainty capture |
| 10 | 52539046 | 2026-05-11 08:31 | exp008 v3 (Huber + hetero + MEDIAN + path b) | 10.253 | +0.515 ⚠️ | exp008 v2 (9.957) 比 +0.296 悪化、 改修群を全 reject |
| 11 | 52559431 | 2026-05-12 00:15 | exp010 fold reform (= stratified Edge Q + adversarial drop) | 10.227 | +0.489 ⚠️ | **fold 構造変更で base stack 連鎖崩壊**、 §11 詳細分解。 exp016 では fold 保持確定 |
| 12 | 52559810 | 2026-05-12 00:33 | exp013 winning path B (= D1+D4+D9 inject) | 10.029 | +0.291 ⚠️ | **lookup loaded 0 wells で実質空 inject + featurization noise 化**、 §12 詳細分解。 lookup build なしでは退避 |

### CV-LB ratio trend

current best: LB 9.738 / rank 17/779 (= 2026-05-11 05:37 時点)、 残り deadline 85 day (= 2026-08-05)
public LB top 1: Virtute 9.256 (= 2026-05-11 時点)

---

## §N v_(n-1) → v_n 分解 schema (= 全 submit 必須記入)

各 submit について以下を append:

### submission_id: <ID>
- **timestamp (UTC)**: <yyyy-mm-dd HH:MM>
- **build_commit**: <git short SHA>
- **mode**: <exp 名 + 簡潔説明>
- **base submission**: <v_(n-1) submission_id>
- **source_count**: <変更要素の数>
- **force_count**: <hand-craft / hard-coded 要素の数>
- **est_total_score**: <OOF or CV ベース予測値、 ft>
- **public_score (LB)**: <Kaggle 公開値、 ft>
- **lb_minus_est**: <LB - est、 calibration drift>
- **lb_div_est**: <LB / est、 ratio>
- **per-task source diff**: <変更要素 1 つずつの効果 list、 単独 lift 実測値あれば併記>
- **effect isolate**: <各変更 source が真効果 / noise / 重複 explanation のいずれか>
- **仮説帰納**: <予測 vs actual 乖離の解釈、 「なぜ予想と違ったか」>
- **roadmap refine**: <次 submit で何を変えるか、 ablation 候補>
- **shake-up risk 評価**: <C1 fold 0 OOF と LB の |差|>

---

## Phase A 実装中 (= 2026-05-12 開始)

### AC-A1 PENDING exp 状態 (= 2026-05-12 08:30 UTC 確認)

| exp | kernel slug | kernel status | LB | 備考 |
|---|---|---|---|---|
| exp010 fold reform | ky7240/rogii-exp010-foldreform-v1 | **KernelWorkerStatus.COMPLETE** | TBD | submission.csv 取得済 (/tmp/rogii-kernel-output/exp010/、 318K)、 ただし `kaggle competitions submit -f` が 400 Bad Request で失敗 (= ROGII は Code Competition、 kernel UI からの "Submit to Competition" が必要) |
| exp013 winning-path-B on v2 | ky7240/rogii-exp013-winning-b-on-v2 | **KernelWorkerStatus.COMPLETE** | TBD | 同上 (/tmp/rogii-kernel-output/exp013/、 317K)、 注意: log に "[winpath B] lookup loaded: 0 wells" = winning-path-B lookup が空 (= D1+D4+D9 inject 効果 questionable) |
| exp015 Sparse GP M=500 | ky7240/rogii-exp015-cpu-sparse-gp | **KernelWorkerStatus.ERROR** | — | run 失敗、 修正 + re-run 必要 |

### kernel run log から判明 (= LB 出る前の参考値)

- **exp010 OOF**: path b grid search で w_kb*=0.75、 OOF RMSE = **10.4580** (= kb_only 10.4806、 own_only 10.7233)、 9-base + Edge R + Edge S 適用済
- **exp013 OOF**: 同 OOF RMSE = **10.4538** (= kb_only 10.4806、 own_only 10.7009)、 ただし winpath B lookup 空、 9-base + Edge R + Edge S 適用済

**重要**: OOF 10.45 は train wells で計算 (= visible-as-pseudo-hidden、 path b)、 LB と数値オーダー異なる。 exp009 v2 LB 9.738 の OOF も 10.x 帯想定なので、 OOF 10.45 が「悪い」 とは言い切れない。 真の評価は LB 出てから。

### user action 必須 (= kernel UI から submit)

ROGII は **Code Competition format**、 CLI 経由の `kaggle competitions submit` は 400 拒否。 ユーザーは Kaggle UI で kernel page にアクセスし、 "Submit to Competition" を実行する必要:

1. https://www.kaggle.com/code/ky7240/rogii-exp010-foldreform-v1 → "Submit to Competition" click
2. https://www.kaggle.com/code/ky7240/rogii-exp013-winning-b-on-v2 → 同上
3. exp015 は ERROR、 修正後 re-run + submit

submit 完了後、 `kaggle competitions submissions rogii-wellbore-geology-prediction` で LB 取得可。 30 分以内に本 doc の §1 表に append + 各 exp の v_(n-1) → v_n 分解 entry を append。

---

---

## §11 v_9 → v_10 分解 (= exp010 fold reform、 LB 10.227)

### submission_id: 52559431
- **timestamp (UTC)**: 2026-05-12 00:15:10
- **build_commit**: 3777691 (= feat/phase-a-gold-confirm-2026-05-12 branch、 ただし kernel は別 source)
- **mode**: exp010 fold reform on exp009 v2 base = stratified Edge Q + adversarial drop
- **base submission**: 52534803 (= exp009 v2 LB 9.738)
- **source_count**: 2 (= stratified Edge Q fold + adversarial drop)
- **force_count**: 0
- **est_total_score**: (OOF path b RMSE = 10.4580、 LB-est gap +0.231 想定)
- **public_score (LB)**: **10.227** ⚠️
- **lb_minus_est**: -0.231 ft (= 10.227 - 10.458)
- **lb_div_est**: 0.978
- **vs v_(n-1) = exp009 v2 LB 9.738**: **+0.489 ft 悪化** ⚠️
- **per-task source diff**:
  - source 1: stratified Edge Q (= visible_ratio × tw_gr_resid_std × b_ANCC_med の 3-key stratify) → **-0.5 ft 期待だったが +0.489 ft 悪化** = 仮説 H-A1 誤り
  - source 2: adversarial drop (= train-test mismatch wells の sample_weight ダウン) → 効果分離不可
- **effect isolate**:
  - **stratified Edge Q fold の単独効果**: H10 postmortem では σ_fold 0.526 → 0.303 と書かれていたが、 **σ_fold 小さくしても LB 改善せず、 むしろ悪化**。 これは fold 平準化が **base stack の chemistry を壊した** ことを示唆
  - **base stack 全体の Co-evolve 不在**: karnakbaev 5-base の OOF は元 fold で fit 済、 我々が Edge Q stratified で再 generate すると mismatch (= exp007 fold-misalign LB 10.677 と同 pattern)
- **仮説帰納**:
  - exp007 (= 自前 4-base + Edge Q fold) で LB 10.677 → 自前 base が Edge Q fold で性能不足だった
  - exp009 v2 (= karnakbaev fold 維持 + 案 D + 案 E) で LB 9.738 → fold 触らないと改善
  - exp010 (= Edge Q stratified) で LB 10.227 → fold 触ると base stack 連鎖崩壊
  - **教訓**: **fold 構造変更は exp009 v2 base に対して逆効果**、 Phase A 統合 kernel exp016 では fold を保持
- **roadmap refine**:
  - Phase A 統合 kernel **exp016 では exp009 v2 fold を完全に保持**
  - fold reform は Phase B 以降で **fold + base + features を Co-evolve させる再設計** が必須 (= 単純な fold 変更ではない)
  - 案 D Kalman、 案 E Sparse GP、 Edge S/O はすべて exp009 v2 fold で fit 済 → これらを base に **module-level inject** で Phase A を進める
- **shake-up risk 評価**: C1 fold 0 (= 140 wells LB-proxy) OOF 未計算、 後で算出

---

## §12 v_9 → v_10b 分解 (= exp013 winning path B、 LB 10.029)

### submission_id: 52559810
- **timestamp (UTC)**: 2026-05-12 00:33:16
- **build_commit**: 85ac8c7 (= feat/cv-strategies-final-2026-05-11 branch、 別の source)
- **mode**: exp013 winning path B on v2 = exp009 v2 base + D1+D4+D9 feature inject
- **base submission**: 52534803 (= exp009 v2 LB 9.738)
- **source_count**: 3 (= D1 + D4 + D9 inject)
- **force_count**: 0
- **est_total_score**: (OOF path b RMSE = 10.4538)
- **public_score (LB)**: **10.029** ⚠️
- **lb_minus_est**: -0.425 ft
- **lb_div_est**: 0.959
- **vs v_(n-1) = exp009 v2 LB 9.738**: **+0.291 ft 悪化** ⚠️
- **per-task source diff**:
  - **CRITICAL**: kernel log で **"[winpath B] lookup loaded: 0 wells"** が **3 回**繰り返し → D1+D4+D9 feature inject の **lookup table が空** で、 実質的に何も inject されていない
  - inject されたものは 0 features、 ただし featurization pipeline が拡張版 (= D1+D4+D9 column 化) で実行され、 既存 features の column 順序 / dtype 変更で **base model 推論が劣化**
- **effect isolate**:
  - winning path B (= karnakbaev-top2 distill から D1+D4+D9 feature inject) の効果は **0 (= lookup 空)**
  - LB 10.029 は base 9.738 から +0.291 悪化 = 拡張 featurization の **noise 化**
- **仮説帰納**:
  - 「D1+D4+D9 inject で -0.05〜-0.15 ft 改善」 仮説 H-A2 は **lookup table build を skip した結果完全に誤り**
  - lookup table は karnakbaev OOF 系列から build する必要、 単純に kernel run しても 0 wells lookup
  - 拡張 featurization が **空 inject でも noise 化する** = column-wise structure change が base model にとって OOD
- **roadmap refine**:
  - winning path B (= D1+D4+D9 inject) は **lookup table を別途 build** してから再試行
  - lookup build script は `winpath B` 系の kernel で実装、 karnakbaev OOF → top2 distill → D1/D4/D9 feature 化
  - **当面 Phase A 統合 kernel から除外**、 Phase B 以降で再設計
- **shake-up risk 評価**: 同上未計算

---

## §13 重大発見 (= 2 件失敗から導く Phase A 統合 kernel 設計指針)

両 exp が exp009 v2 base に対して悪化した事実から、 Phase A 統合 kernel **exp016 の設計指針** を確定:

1. **exp009 v2 fold を完全保持** (= karnakbaev 元 fold)、 Edge Q stratified / adversarial drop は **inject しない**
2. **base model (= karnakbaev 5-base + 自前 4-base) も保持**、 これらの OOF / test_preds を **そのまま使う**
3. **改修は module-level inject のみ**:
   - A3 Numba JIT beam ±2 → 既存 `beam_search`, `beam_search_dir` を Numba 化 + ±2 拡張、 既存と並走 (= ablation 可能)
   - A4 segment b_well → 既存 `b_well` 計算に early/mid/late/wls phase 列を **追加** (= 既存削らない)
   - A5 softmax NCC → 既存 `self_corr_tvt` を multi-scale softmax 拡張に置換 or 並走
   - A6 Optuna 3-axis postproc → 既存 alpha × tau grid を Optuna 500-trial に置換、 ただし alpha/tau range は既存 grid の最良点を center に
   - A2 Climber → 既存 Ridge meta を Climber に置換 (= allow_negative_weights=True、 precision 0.001)
4. **feature inject (= D1+D4+D9 系) は当面除外** (= lookup table empty で逆効果確定)
5. **ablation 必須**: 各 module 単独で 1 submit して LB lift 実測、 5 module で 5 submit + 統合 1 submit = 6 submit budget (= 残 24h で 5 件 quota 厳守)

→ exp016 のフロー: 
- Step 1: exp009 v2 完全 copy (= 既存 LB 9.738 を再現確認)
- Step 2: A3 (Numba beam ±2) だけ inject → submit → ablation 実測
- Step 3: A4 + A5 add → submit
- Step 4: A6 (Optuna) add → submit
- Step 5: A2 (Climber) add → submit (= 統合完了)

---

## Hill Climb (LB 9.43) 強さの本質分析 (= 2026-05-12 user 追加指示、 別 doc)

詳細: `docs/research/2026-05-12-hill-climb-strength-analysis.dense.md` を参照

### 要約
- Hill Climb LB 9.43 の **MAIN 差分 = `hill_climbing.Climber`** (= Caruana 2004 系 greedy ensemble selection、 negative weight 許容、 precision 0.001) + **Optuna 500-trial postproc** (= alpha × tau × w_pf 3-axis TPESampler)
- user は **3 軸で既に優位** (= base diversity 9 vs 6、 Sparse GP M=200、 Edge O/S)
- **2 軸で劣勢** (= ensemble weight、 postproc search 精度) → Phase A2 + A6 で取り込み必須
- Phase A 完了で LB **8.83 〜 9.39** (= 上振れで 9 切り、 下振れで Hill Climb 凌駕)
- 我々が独占できる **7 軸の優位** (= aug、 per-row kNN、 NN seq、 MoE、 GP M=500、 Multi-task、 Edge O/S) → 優勝路で構造的有利

### 検証中の仮説 (= roadmap refine が pending)

- **H-A1**: stratified Edge Q fold (= exp010) で fold-misalign が解消され、 exp007 (LB 10.677) より +0.5 ft 改善
- **H-A2**: D1+D4+D9 feature inject (= exp013) は exp009 v2 base に対して -0.05〜-0.15 ft 改善 (= winning-path-B sketch evidence)
- **H-A3**: Sparse GP M=200→500 (= exp015) は M=200 lift -0.219 ft (実測) の 1.2-1.5x = -0.26〜-0.33 ft 期待
- **H-A4** (= Phase A 独立): Numba Beam ±2 + dense PF は LGB+XGB LB 9.83 の単独 lift -0.05〜-0.10 ft (= LGB+XGB は同時投入で 0.83 ft 累積、 単独不明)
- **H-A5**: segment b_well per formation × phase は data-driven boundary で -0.05〜-0.15 ft
- **H-A6**: score-weighted softmax NCC は -0.05〜-0.10 ft (= Geostat NCC + LGB+XGB 共通技)
- **H-A7**: alpha × tau × w_pf 3-axis + SG smoothing は -0.05〜-0.15 ft
- **H-A8**: Hill Climb weight optimization 自前 (= A3-A6 後の 9-base に exhaustive sweep) で -0.05〜-0.15 ft (= Hill Climb LB 9.43 の core paradigm)

期待累積 (H-A4 から H-A8): -0.30 〜 -0.80 ft = LB **9.0-9.4 帯** (= Gold cutoff 接触圏)

---

## §14 v_9 → v_11 分解 (= exp017 hillclimb fork v3、 LB **10.030**、 想定外 大破綻)

### submission_id: 52586245
- **timestamp (UTC)**: 2026-05-12 16:14:18
- **build_commit**: (= fork、 ravaghi/wellbore-geology-prediction-hill-climbing + 我々 inline Climber)
- **mode**: ravaghi NB を fork、 PyPI `hill-climbing` を **我々 src/rogii/hill_climb.py inline** に置換、 CPU mode で run
- **base submission**: 52534803 (= exp009 v2、 LB 9.738)
- **source_count**: 1 (= fork 全体を 1 unit、 PyPI Climber → inline Climber 置換のみ)
- **force_count**: 0
- **est_total_score (OOF)**: 10.380 (= Optuna Trial 448 best、 `alpha=1.0, tau=75, w_pf=0.12`)
- **public_score (LB)**: **10.030**
- **lb_minus_est**: -0.350 (= LB が OOF より 0.35 ft 良い、 ただし通常逆転)
- **lb_div_est**: 0.966
- **per-task source diff**:
  - PyPI Climber → inline Climber: numerical 等価性確認なし (= 後段でも未検証)
  - CPU mode (= GPU quota 切れ): LGB/CB は cache load 経路、 ravaghi original の cache 利用、 ただし PF (Numba) は **再 run** で seed 効かない non-determinism risk
  - `from numba import njit` 内 multi-thread = seed 固定でも結果ばらつき
- **effect isolate**: **OOF 10.380 (= 我々 inline Climber + Optuna) は ravaghi original の OOF 10.394 より 0.014 ft good**。 にもかかわらず LB **10.030 vs 期待 9.43 = -0.60 ft 悪化**。 OOF と LB の **decoupling**
- **仮説帰納**:
  - H1 (= 確度 大): **PF / Numba parallelism non-determinism** が test set で異なる結果を生み、 LB drift
  - H2 (= 確度 中): **CPU mode LGB の floating-point precision** が GPU mode と differ、 model output micro-diff が Climber blend で amplified
  - H3 (= 確度 小): **我々 inline Climber に hidden bug** (= ただし OOF で good 出ているので、 OOF-LB decoupling が真因の可能性大)
  - H4 (= 構造的): **OOF (= train 770 wells hidden rows) と LB (= test 3 wells hidden rows) は完全に異なる集団**、 ravaghi LB 9.43 は **test 3 wells に偶然 lucky な** kernel run で取れた値の可能性
- **roadmap refine**:
  - **exp016 (= 自前統合、 RUNNING) は ravaghi NB と別 base (= 我々 9-base + Sparse GP + Edge Q/R/S/O) なので、 同 LB drift には直接さらされない、 ただし PF Z dual の Numba non-determinism は同じ risk**
  - exp016 完走後の LB を観察、 **CV-LB gap (= AC-A-cv-lb-spearman) を critical 検査** する
  - 将来: **public NB fork は確実 defense ラインでない、 自前構成の方が CV-LB stable** という教訓
- **shake-up risk 評価**: exp017 で OOF 10.380 → LB 10.030 = **gap +0.35 ft (= 通常 -0.5 ft 程度なので異常に小)、 test set の hidden 真値が train 分布と異なる可能性、 shake-up risk 大**

### 教訓 (= GM §11 「優勝本質性」 commit、 lessons.md 候補)

1. **公開 NB fork ≠ LB 確保**: ravaghi NB LB 9.43 は author の original run、 我々 fork で **PF non-determinism / CPU mode / library version 等の差** で LB drift 可能性大、 fork で 9.4 帯を「確実 defense ライン」 と読み込んだのは過信
2. **OOF と LB の decoupling 検証必須**: OOF 改善が LB 改善に reflect されない、 = CV strategy が test set を represent していない指標、 Phase B B3 (= CV-LB correlation 強化) 必須
3. **inline 化時の numerical 等価性 test 必要**: 我々 inline Climber が PyPI と数値等価か **直接 unit test で確認**、 これを skip した我々の test 不足
4. **PF Z dual + Numba JIT の seed 固定 verification 必要**: 多 thread での re-run で seed 効かない可能性、 deterministic 化を確認

---

## 関連 doc

- plan: /home/yusuke_kaya/.claude/plans/floating-cuddling-haven.md (+ repo copy: docs/dev/2026-05-12-plan-gold-to-winning.dense.md)
- criteria: .criteria/kaggle-rogii-phase-a-2026-05-12.yaml
- LB tracking: docs/dev/leaderboard.dense.md
- submission postmortems: docs/dev/submission-postmortems.dense.md
- past CV strategies: .criteria/kaggle-rogii-cv-strategies-2026-05-11.yaml
