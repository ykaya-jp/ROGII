# ROGII Wellbore Geology Prediction — 金メダル → 優勝戦略 plan

> 作成日: 2026-05-12 (= 残り 1 ヶ月+)
> 起源: user 指示「公開 NB に負けてる、 取り入れる / 捨てる / 凌駕する戦略を考察、 まず金メダル sub → 優勝」
> 適用範囲: `/home/yusuke_kaya/projects/kaggle/ROGII/`
> 出力ルール: `~/.claude/CLAUDE.md` Plan Mode Kaggle ルール (= 本文日本語、 コード識別子・URL・file:line は英数字)

---

## §0. Context (= この plan が必要な理由)

- **現状**: `exp009 v2` が LB **9.738** / rank **17/779** (Silver 高位) で停滞、 公開 NB Hill Climb (Roman Tarasov、 LB **9.43**) に **-0.29 ft 抜かれた**、 LGB+XGB (LB **9.83**) はギリギリ僅差
- **既知の base**: karnakbaev 5-base + 自前 4-base + 案 D (Kalman/PF on dTVT) -0.219 ft + 案 E (Sparse GP M=200) -0.219 ft + Edge S (-0.114 ft) + Edge O + Ridge meta positive (`docs/dev/leaderboard.dense.md:18`)
- **PENDING**: exp010 RUNNING 19h+ (fold reform、 期待 LB 9.5-9.7) / exp013 (D1+D4+D9 inject) / exp015 (Sparse GP M=500)
- **目標**: ① 金メダル (= rank ~10、 LB **9.0-9.4** 帯) → ② 優勝 (= rank 1、 LB **7.5-8.5** 帯)
- **resource**: Kaggle Notebook + Local GPU + Colab Pro+ A100 の **3 並列**
- **commit**: `~/projects/kaggle/CLAUDE.md §11` 「優勝本質性 criterion 一生 commit」 (= 軽さでなく数理本質、 self-compiled 75%+、 single paradigm 禁止)

---

## §1. 現状把握 (= Phase 1 Explore 結果サマリ)

### 1.1 公開 NB 5 件の真の脅威評価

| NB | LB | vs 9.738 | 真の差分 (= 核心技術) | 採用判断 |
|---|---|---|---|---|
| Hill Climb (Tarasov) | 9.43 | **-0.29 ft 上回る** | `hill_climbing.Climber` で **exhaustive ensemble weight optimization** + `ravaghi/wellbore-geology-prediction-artifacts` 依存 | ⭐ 核心 paradigm は自前再実装 (= artifact 直 ingest は §11 違反) |
| LGB+XGB | 9.83 | +0.09 ft 接戦 | Numba JIT **Beam ±2** + dense O(1) PF + **segment b_well** per formation × phase + **softmax NCC** + alpha×tau×w_pf **3-axis** + Savitzky-Golay | ⭐⭐ 複数要素を ablation 付きで段階移植 |
| AeroRidge (Pilkwang) | 10.152 | +0.41 ft (下回る) | LGB×3 + CB×3 + **TabICL×2** = 8 base、 dual GPU、 pretrained checkpoint 依存 | ❌ user 既に上、 TabICL CUDA error 経験あり (exp006 LB 10.503) |
| h-blend v2 | ~9.956 | +0.22 ft (下回る) | 4 sub blend (Nina2025 9.956 dominant)、 transduction、 external dataset 依存 | ❌ §11 違反、 reproducibility 0 |
| Geostat NCC | 9.946 | +0.21 ft (下回る) | 3-fold (不安定)、 **Softmax NCC**、 scipy.optimize.minimize | ⚠️ Softmax NCC のみ抽出、 残りは skip |

### 1.2 全 5 NB + user が共通して miss している盲点 (= 優勝の余地)

| ID | 盲点 | evidence |
|---|---|---|
| H.1 | **per-row kNN inference** (paradigm 6) | enisteper LB 9.960 が `knn_row_ANCC / b_well / dist` 121 cols 実装 (`6fa34b8` commit、 `docs/research/2026-05-11-winning-path-G-perrow-knn-sketch.dense.md`) |
| H.2 | **data augmentation by visible_ratio randomization** (paradigm 5) | aeroridge LB 9.916 が aug_k schema (= 5M rows = 773 wells × ~6500 rows × 多 visible_ratio) で証拠 (`docs/research/2026-05-11-winning-path-F-augmentation-sketch.dense.md`) |
| H.3 | **NN sequence modeling (Mamba/Longformer)** | 誰も投入していない、 takaito/rogii-training-lstm-tutorial-notebook が LSTM tutorial で paradigm valid 実証 |
| H.4 | **asymmetric formation imputation** (= public LB boost + private collapse risk) | pilkwang のみ部分実装、 `docs/research/pilkwang-distill.dense.md` |
| H.5 | **cross-well geological-distance transfer** | 全員 euclidean X/Y のみ、 層序 distance metric 未使用 |
| H.6 | **physics constraint loss** (= soft penalty) | `TVT = -Z + ANCC + b_well` (Pearson -1.0、 residual std 0.007 ft) は feature 化済だが loss penalty 化未実装 (`docs/research/problem-essence.dense.md`) |
| H.7 | **multi-task across 6 formations** | 全員 ANCC のみ最適化、 ASTNU/ASTNL/他 4 を補助 target 化していない |

---

## §2. ドラフト戦略 critique (= Plan agent 反証思考、 重大 5 件)

| # | 指摘 | 出典 | 重大度 |
|---|---|---|---|
| C1 | **paradigm lift の線形加算仮定が崩れる** (= 案 D Kalman/PF と案 E Sparse GP が両方とも b_well posterior subspace を取り合う、 ensemble OOF correlation 第一主成分 variance ratio で実測必須) | `docs/research/strategy-critique.dense.md §1` | 高 |
| C2 | **Hill Climb weight opt 単独の lift 過大評価** (= 我々 base は既に LB 9.7 帯、 weight opt 単独は実測 -0.05〜-0.15 ft が limit) | Hill Climb NB 構造 + LB 数値推論 | 中 |
| C3 | **Numba Beam ±2 + N=600 PF の lift estimate 根拠なし** (= LGB+XGB は ±2 + softmax NCC + segment b_well 同時投入で 0.83 ft 累積、 単独 lift 不明、 ablation 必須) | LGB+XGB NB title "v5: targeting ~9.0-9.5 (from 10.464)" + cell 0 OOF 9.67 | 中 |
| C4 | **per-row kNN の leak guard 3 層実装 cost が過小評価** (= fold-aware + well-aware + row-distance threshold の混在で CV-LB gap 拡大 risk) | `docs/research/2026-05-11-winning-path-G-perrow-knn-sketch.dense.md §2.1` | 高 |
| C5 | **physics constraint loss は既に feature 化で satisfied** (= soft penalty 化 marginal lift ≤ 0.05 ft) | `docs/research/problem-essence.dense.md` `tvt_formula` feature 既存 | 中 |
| O1 | **exp010 完走未確定が Phase A blocker** | `git log` 直近 commit | 運用 |
| Sc1 | **Phase C で compute 予算 break** (= NN seq 25 train × 4-8h = 100-200 A100-hr 想定 vs Colab Pro+ 50 A100-hr/月) | Colab Pro+ pricing | scale |
| Sc2 | **submit quota 150 件 上限 vs 想定 130 件 + ablation 余裕 20 件のみ** | quota 5/day × 30 day | scale |

**結論**: ドラフトの「Phase A 即 lift + B paradigm 5/6 + C NN+MoE」 順序自体は valid、 ただし **paradigm 加算は線形でない** ことを前提に **ablation を毎 phase 厳格化**、 **leak guard を critical path 化**、 **compute 予算は Phase C で再評価** 必須。

---

## §3. 構造原理が異なる 4 代替案 (= GM mindset §11 中立指示、 推奨せず軸提示)

### 案 (D) main agent ドラフト = 既存 paradigm 2/3/5/6 を時系列で積む
- **構造原理**: Phase A (post-proc + 公開 NB 移植) → B (paradigm 5/6 = aug + kNN) → C (paradigm 2 induction = NN + MoE + physics) を順次
- **着想根拠**: `docs/research/2026-05-11-winning-path-{A,D,F,G}-sketch.dense.md` 4 sketch 既起票
- **期待 LB**: 累積 -0.50 〜 -1.20 ft (= 9.738 → 8.5-9.2 帯)
- **compute**: 200-300 hr (= phase 横断)
- **risk**: paradigm 重複 (C1)、 leak guard cost (C4)

### 案 (X) LLM-driven feature synthesis + program synthesis (= paradigm ④ 真剣ルート)
- **構造原理**: GPT-5.4 / Claude Opus に既存 features + 物理関係を渡し、 **新 derivation rule を Python code で program synthesis**、 sandbox 実行 → LGB stack に inject
- **着想根拠**: ARChitects 2024 paper "transduction + induction の combination" + `docs/research/independent-edges.dense.md §1.7` 「9 候補 edge 中 0 が公開 top に存在しない」 = LLM 新 edge 提案余地大
- **期待 LB**: -0.10 〜 -0.30 ft (= 9.738 → 9.4-9.6 帯)
- **compute**: 20 hr (= LLM inference + sandbox + LGB retrain)
- **risk**: hallucinate (= 物理的に意味なし feature)、 leak guard 未統合

### 案 (Y) 公開 NB 全集約 stack + Hill Climb weight optimization (= transduction 純粋)
- **構造原理**: Hill Climb + LGB+XGB + AeroRidge + h-blend v2 + Geostat NCC + needless090 + pilkwang の **7 NB OOF + test_pred** を集約、 我々 base 9 を加えて **12-15 base super-stack**、 Hill Climb 自前 weight opt
- **着想根拠**: `_research_kernels/needless090__score-10-081-score-lb-32-rank/` で top10 既に多 base stack 実証
- **期待 LB**: -0.15 〜 -0.30 ft (= 9.738 → 9.4-9.6 帯)
- **compute**: 8-15 hr (= Local CPU)、 GPU 不要、 **最安**
- **risk**: ⚠️ `~/projects/kaggle/CLAUDE.md §11.1` 違反 (= 数理本質性 ❌)、 shake-up 同時 collapse (= 全 base 同 public LB overfit pattern)、 fold mismatch で blend 不安定

### 案 (Z) Physics-first analytic solver (= induction 真剣ルート)
- **構造原理**: `TVT(s) = -Z(s) + ANCC(s) + b_well(s) + ε(s)` (residual std 0.008 ft) を **解析的逆問題化**、 ANCC を Gaussian Process posterior + plane-fit prior、 b_well を Kalman/PF state-space、 既存 LGB を **回帰 head として捨てて** posterior mean 直接 submit
- **着想根拠**: `docs/research/first-principles.dense.md §1.5` Bayesian posterior 定式化 + 案 E Sparse GP M=200 が -0.219 ft 実測 (= 同 paradigm 延長は数理的 valid)
- **期待 LB**: **-0.30 〜 -0.60 ft** (= 9.738 → 9.1-9.4 帯)
- **compute**: 25 hr (= Local GPU + Kaggle GPU、 GP M=500 = 5-15h + multi-formation 並列 = 15-25h)
- **risk**: GP M=500 OOM (= Kaggle 16GB)、 LGB non-linear correction が失われる、 Beam alignment が物理 model に乗らない

### 案 (W) Sequence-to-sequence NN end-to-end を主軸に昇格 (= deep paradigm 真剣ルート)
- **構造原理**: ドラフトの Phase C1 を **戦略主軸に**、 全 wells (length 5000-8000) を **Mamba (O(n) state-space)** または **Longformer (O(n × window) sparse attention)** で seq-to-seq、 multi-task aux head で 6 formation 同時 predict、 5-fold × 5-seed = 25 ensemble、 既存 stack に Ridge blend
- **着想根拠**: `docs/research/2026-05-11-winning-path-A-nn-sketch.dense.md §1.3`、 takaito LSTM tutorial、 cdeotte NN starter CV 15.5、 ARChitects induction 側
- **期待 LB**: -0.30 〜 -0.60 ft (= 9.738 → 9.1-9.4 帯)
- **compute**: **100-200 A100-hr** (= Colab Pro+ 単独で予算不足、 Local + Kaggle GPU 並列必須)
- **risk**: 200 wells で deep model overfit (`strategy-critique.dense.md §1` 観察 2)、 Kaggle 9h runtime 超過 risk

---

## §4. トレードオフ表 + 選択軸

### 4.1 5 案 × 6 評価軸

| 評価軸 | 案 D (ドラフト) | 案 X (LLM-driven) | 案 Y (公開 NB stack) | 案 Z (Physics solver) | 案 W (NN seq 主軸) |
|---|---|---|---|---|---|
| 実現可能性 | 高 | 中 (LLM filter 必要) | 高 | 中 (OOM risk) | 中-低 (compute+overfit) |
| 新規性 | 中 | 高 | 低 | **最高** | 高 |
| 技術 risk | 中 | 高 (hallucinate) | 中-高 (shake collapse) | 高 (OOM) | **高** (overfit+runtime) |
| compute コスト | 200-300 hr | 20 hr | **8-15 hr** | 25 hr | **100-200 hr** |
| shake-up 耐性 | 中 | 中-低 | **最低** | 高 (物理 invariant) | 中 |
| 数理本質性 (= §11) | 中 | 中 | **低** (§11 違反) | **最高** | 高 |

### 4.2 選択軸 3 つ (= user が判断するための frame)

- **軸 1: 「金メダル確実優先」 vs 「優勝路 upside 優先」**
  - 確実 (= rank ~10、 LB 9.0-9.4 帯到達確率 70%): **案 Y + 案 D Phase A** (= shake risk 受容)
  - upside (= rank 1、 LB 7.5-8.5 帯到達確率 30%): **案 Z + 案 W 併走** (= 下振れ Silver 落ちあり)

- **軸 2: 「compute 最小」 vs 「数理本質性最大」**
  - 予算余裕なし: **案 Y → 案 Z** 順 (= 案 W 撤退)
  - §11 commit 厳守: **案 Z 単独** (= 案 W 次点、 案 Y 不適格)

- **軸 3: 「学習価値 (= 次 comp 転用)」 vs 「即効性」**
  - 長期 skill: **案 W (Mamba/Transformer)** = NN seq transferable
  - 残 1 ヶ月で結果優先: **案 D + 案 Z** = 既存 evidence 延長で fast iteration

---

## §5. 推奨実装シナリオ (= 案 D ドラフトを基軸 + 案 Z 並走、 §11 commit 反映)

> **判断軸**: user が「Recommended に従う」 + 残り 1 ヶ月 + 3 並列 compute と回答済 + `~/projects/kaggle/CLAUDE.md §11.1` 「優勝本質性」 一生 commit
> → 案 D (ドラフト) を **Phase A/B で実行**、 同時に案 Z (Physics solver) を **Local GPU で並走**、 Phase C で案 Z 成果を Ridge blend に統合、 案 W (NN seq) は Phase C で **段階導入** (= compute 予算と相談)、 案 Y (transduction 純粋) は **safe slot 用** に限定採用
> → user が別軸を選ぶ場合は §3 案 X/W/Y を主軸に組み替え可

### 5.1 Phase A: 金メダル確保 (= Week 1、 LB 9.5-9.4 帯到達)

| task | 内容 | resource | expected lift | dependency |
|---|---|---|---|---|
| A1 | PENDING exp010/013/015 結果回収 + LB / OOF / CV-LB gap を `docs/research/2026-05-12-submission-analyses.md` に append (= GM §8 形名参同) | Kaggle | — | (none) |
| A3 | **Numba JIT Beam ±2 + dense O(1) PF grid** を `src/rogii/beam_pf_numba.py` で実装 (= LGB+XGB LB 9.83 移植、 ablation で単独 lift 実測必須) | Local | -0.05〜-0.10 ft | (none) |
| A4 | **Segment b_well per formation × phase** (= early/mid/late/wls、 data-driven boundary、 hard-coded 禁止) | Local | -0.05〜-0.15 ft | A3 |
| A5 | **Score-weighted Softmax NCC** (= Geostat NCC + LGB+XGB 共通技、 既存 Self-NCC に softmax normalize 層追加) | Local | -0.05〜-0.10 ft | A3 |
| A6 | **Alpha × tau × w_pf 3-axis post-proc grid** + Savitzky-Golay (window=17, poly=3) | Kaggle | -0.05〜-0.15 ft | A4 + A5 |
| A2 | **Hill Climb weight optimization 自前実装** (= `src/rogii/hill_climb.py` で exhaustive weight sweep、 Roman Tarasov の `hill_climbing.Climber` を OOF base に対して再現) | Local | -0.05〜-0.15 ft | A3-A6 完了後 (= weight space 広い方が lift 大) |

**累積期待**: -0.30 〜 -0.80 ft → LB **9.0-9.4 帯** = Gold cutoff 接触圏

### 5.2 Phase B: Gold 確実 + 公開 NB 凌駕 (= Week 2、 LB 8.5-9.2 帯)

| task | 内容 | resource | expected lift | dependency |
|---|---|---|---|---|
| B1 | **winning-path F: data augmentation** (= `src/rogii/augment.py`、 visible_ratio ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30} で 6 段 re-mask、 aeroridge schema 凌駕、 boundary spike 抑制) | Colab Pro+ (= 5-base × 5-fold × 6 aug_k = 150 fits) | -0.10〜-0.30 ft | A1 完了 (= base 固定) |
| B2 | **winning-path G: per-row kNN inference** (= `src/rogii/knn_row.py`、 FAISS index for 3.78M rows × 121 cols、 self-well leak 防止 3 層: fold-aware + well-aware + row-distance threshold) | Local | -0.10〜-0.30 ft | A1 完了 |
| B3 | **CV-LB correlation 強化** (= C2.v2 stratified Edge Q fold で OOF 全 base 再生成、 σ_fold drift 監視) | Local | calibration | A1 完了 |
| Z1 (並走) | **案 Z Sparse GP M=500 multi-formation** (= `src/rogii/gp_sparse.py`、 ANCC posterior + variance を 6 formations 並列 fit) | Local + Kaggle | -0.30〜-0.60 ft (= 案 Z 期待全体) | (none) |

**累積期待**: -0.50 〜 -1.20 ft → LB **8.5-9.2 帯** = **Gold 確実 (rank 5-10)**、 enisteper LB 9.960 + aeroridge LB 9.916 **両方上回る**

### 5.3 Phase C: 優勝路 (= Week 3-4、 LB 7.6-8.9 帯)

| task | 内容 | resource | expected lift | dependency |
|---|---|---|---|---|
| C1 | **winning-path A: NN seq (Mamba)** (= `src/rogii/nn_seq.py`、 5-fold × 5-seed = 25 ensemble、 multi-task aux head で 6 formation 同時 predict) | Colab Pro+ A100 (= 50-100 A100-hr) | -0.20〜-0.50 ft | B 完了 |
| C2 | **winning-path D: Per-Well MoE** (= `src/rogii/moe.py`、 visible_ratio × b_cluster gate、 3-expert、 00bbac68 well で per-row delta 改善 evidence) | Local | -0.10〜-0.30 ft | B 完了、 C1 と並列可 |
| C3 | **Physics constraint loss** (= H.6、 LGB custom objective で `tvt_formula` residual penalty) ※ marginal lift ≤ 0.05 ft で skip 候補 (= C5 critique 反映) | Local | -0.00〜-0.05 ft | B 完了 |
| C4 | **Multi-task aux head** (= H.7、 C1 NN に統合、 ANCC + ASTNU + ASTNL + 他 3 formation を補助 target) | Colab Pro+ | -0.05〜-0.10 ft | C1 に組み込み |
| C5 | **Ridge meta blend** (= NN + MoE + Physics solver + 既存 stack を統合、 positive 制約 + Hill Climb weight opt) | Local | -0.05〜-0.10 ft | C1 + C2 完了 |

**累積期待**: -0.80 〜 -2.10 ft → LB **7.6-8.9 帯** = **優勝圏 (rank 1-3)**

### 5.4 Phase D: 終盤 (= deadline 7 day 前 freeze)

| task | 内容 | resource |
|---|---|---|
| D1 | **Adversarial validation で test-similar wells 特定** (= C1 fold 0 = LB-proxy 140 wells を主要 calibration set 化) | Local |
| D2 | **safe + risky 2 submit freeze** ↓ | — |
| D2.safe | Phase A 完了版 + 案 Z (Physics solver) Ridge blend = **LB 9.0-9.4 帯**、 shake-up 耐性高、 §11 OK | — |
| D2.risky | Phase C 完了版 + pilkwang asymmetric formation imputation = **LB 8.5-9.0 帯**、 private collapse risk あり | — |

---

## §6. 依存関係 DAG + 3 並列 compute 割当

### 6.1 DAG

```
Phase A:
  A1 (PENDING 回収) ──┐
                      ├──→ A4 (segment b_well) ──┐
  A3 (Numba Beam ±2) ─┤                          ├──→ A6 (3-axis postproc) ──→ A2 (Hill Climb weight opt) ──→ Phase A submit (期待 LB 9.0-9.4)
                      └──→ A5 (softmax NCC) ─────┘

Phase B (= A1 完了で着手可、 A2 と並列):
  B1 (aug aug_k 6 段) ─┐
                       ├──→ B3 (OOF C2.v2 再生成) ──→ Phase B submit (期待 LB 8.5-9.2)
  B2 (per-row kNN) ────┘
  Z1 (Sparse GP M=500) [並走、 Phase C で Ridge blend 統合]

Phase C (= B 完了で着手):
  C1 (NN seq Mamba + C4 multi-task aux) ──┐
                                           ├──→ C5 (Ridge meta blend) ──→ Phase C submit (期待 LB 7.6-8.9)
  C2 (Per-Well MoE 3-expert) ─────────────┤
  C3 (Physics loss) [optional、 skip 候補] ┘

Phase D:
  D1 (Adversarial val) ──→ D2.safe + D2.risky を 7day 前 freeze
```

### 6.2 3 並列 compute 割当

| worker | 担当 task | 根拠 |
|---|---|---|
| **Kaggle Notebook** (= 9h/run、 30h/week GPU quota) | Phase A submit kernel 全件 (= exp010/013/015 + A3-A6 統合 kernel)、 Phase B augmentation re-train kernel、 Phase Z1 multi-formation GP の半数 | submit kernel は internet 不要、 最終 submission の host environment 一致 |
| **Local GPU** (= 24h × 7 = 168h/week) | A2 Hill Climb + OOF aggregate、 B2 FAISS index (5M+ rows)、 B3 OOF C2.v2 再生成、 C2 MoE、 Z1 GP M=500 fit (= 5-15h)、 全 ablation 測定 | dataset 操作 + FAISS index 構築は internet 利用可な local が圧倒的 efficient、 GP M=500 単機 GPU feasible |
| **Colab Pro+ A100** (= 50 A100-hr/月) | C1 NN seq (Mamba 5-fold × 5-seed = 25 runs × 2h = 50h) | A100 high-memory が 5000-row well 学習に必須、 月予算 50hr で 25 train 届く |

**compute 予算注意**: §2 Sc1 で指摘の通り、 Phase C で予算 break risk。 NN seq は **3-fold × 3-seed = 9 train** に減らす option を Phase C 入口で再評価 (= 50 A100-hr 内収束)。

---

## §7. critical files

### 7.1 既存修正
- `/home/yusuke_kaya/projects/kaggle/ROGII/src/rogii/cv.py` (= A2 で Hill Climb 用 OOF aggregate helper 追加)
- `/home/yusuke_kaya/projects/kaggle/ROGII/src/rogii/features.py` (= B1 augmentation の aug_k 列対応)
- `/home/yusuke_kaya/projects/kaggle/ROGII/src/rogii/submit.py` (= weight stack 拡張)
- `/home/yusuke_kaya/projects/kaggle/ROGII/scripts/aggregate_oof.py` (= Hill Climb weight column 追加)
- `/home/yusuke_kaya/projects/kaggle/ROGII/docs/strategy/winning-strategy.dense.md` (= 4 phase + safe/risky 反映)
- `/home/yusuke_kaya/projects/kaggle/ROGII/docs/dev/leaderboard.dense.md` (= submit 毎 update)
- `/home/yusuke_kaya/projects/kaggle/ROGII/docs/dev/submission-postmortems.dense.md` (= submit 毎 postmortem)

### 7.2 新規 create
- `/home/yusuke_kaya/projects/kaggle/ROGII/src/rogii/hill_climb.py` (= A2)
- `/home/yusuke_kaya/projects/kaggle/ROGII/src/rogii/beam_pf_numba.py` (= A3)
- `/home/yusuke_kaya/projects/kaggle/ROGII/src/rogii/augment.py` (= B1)
- `/home/yusuke_kaya/projects/kaggle/ROGII/src/rogii/knn_row.py` (= B2)
- `/home/yusuke_kaya/projects/kaggle/ROGII/src/rogii/gp_sparse.py` (= 案 Z 並走)
- `/home/yusuke_kaya/projects/kaggle/ROGII/src/rogii/nn_seq.py` (= C1)
- `/home/yusuke_kaya/projects/kaggle/ROGII/src/rogii/moe.py` (= C2)
- `/home/yusuke_kaya/projects/kaggle/ROGII/src/rogii/physics_loss.py` (= C3、 optional)
- `/home/yusuke_kaya/projects/kaggle/ROGII/.criteria/kaggle-rogii-phase-a-2026-05-12.yaml`
- `/home/yusuke_kaya/projects/kaggle/ROGII/.criteria/kaggle-rogii-phase-b-2026-05-19.yaml`
- `/home/yusuke_kaya/projects/kaggle/ROGII/.criteria/kaggle-rogii-phase-c-2026-05-26.yaml`
- `/home/yusuke_kaya/projects/kaggle/ROGII/.criteria/kaggle-rogii-phase-d-2026-06-02.yaml`
- `/home/yusuke_kaya/projects/kaggle/ROGII/docs/dev/2026-05-12-plan-gold-to-winning.dense.md` (= 本 plan の repo 内 copy + 進捗 log)
- `/home/yusuke_kaya/projects/kaggle/ROGII/docs/research/2026-05-12-submission-analyses.md` (= GM §8 全 submit 分解 doc)

---

## §8. Verification (= 形名参同基準、 各 phase の機械的判定条件)

### 8.1 Phase 完了基準

| Phase | LB 必要値 | CV-LB gap | ablation | shake-up |
|---|---|---|---|---|
| A 完了 | LB ≤ **9.5** ft | \|CV - LB\| ≤ 0.3 | A2-A6 各単独 lift ablation (= 1 件ずつ AB 比較) | C1 fold 0 OOF と LB \|差\| ≤ 0.2 |
| B 完了 | LB ≤ **9.3** ft (= Hill Climb 9.43 凌駕) | \|CV - LB\| ≤ 0.3 | B1 augmentation w/o + B2 kNN w/o ablation | adversarial AUC ≤ 0.7 |
| C 完了 | LB ≤ **8.5** ft (= 優勝圏入口) | \|CV - LB\| ≤ 0.3 | C1 NN w/o stack + C2 MoE w/o stack ablation | bootstrap 95% CI lower ≥ 0.5 |
| D 完了 | safe LB ≤ 9.0 + risky LB ≤ 8.5 | freeze 完了 | — | safe = adversarial AUC ≤ 0.6 |

### 8.2 「金メダル達成」 形名参同基準
- final 2 submission の public LB **9.4 以下** AND
- CV-LB Spearman **≥ 0.7** AND
- C1 fold 0 OOF と LB \|差\| **≤ 0.2** AND
- `docs/strategy/<comp>-final-submission-strategy.md` で「safe slot LB X.XX、 risky slot LB Y.YY、 7 day 前 freeze 完了」 明文化

### 8.3 「優勝達成」 形名参同基準
- final 2 submission の public LB **8.5 以下** + private LB **8.5 以下** AND
- NN seq + MoE + Physics solver の **3 paradigm が独立に CV 0.5 ft 以下の lift** 実証 AND
- ensemble の OOF correlation 第一主成分 variance ratio **≤ 0.6** (= paradigm 独立性確保) AND
- 1 ヶ月後の final result で rank 1 確認

### 8.4 毎週末 (= 日曜 23:59) checkpoint
`/home/yusuke_kaya/projects/kaggle/ROGII/docs/dev/HANDOFF-<date>.md` で必須 update:
1. LB 推移 + CV-LB ratio σ 再計算 (= `~/projects/kaggle/CLAUDE.md §8.1` #9)
2. paradigm coverage (= ②/③/④/⑤/⑥/NN seq の充足)
3. compute 残予算 (= Kaggle quota / Colab unit / Local 占有)
4. submit quota (= 残 day × 5 vs 残 ablation)
5. shake-up risk (= C1 fold 0 OOF と LB の \|差\|)

### 8.5 毎 submit 後 30 分以内 (= GM §8 形名参同)
`docs/research/2026-05-12-submission-analyses.md` に append:
- per-task source diff (= 全件、 score delta 上下 10 件)
- effect isolate (= 各変更 source の真効果)
- 仮説帰納 (= 予測 vs actual 乖離)
- roadmap refine

---

## §9. 実装規律 (= GM mindset 原則準拠)

各 exp で必ず以下を回す:

1. **`.criteria/exp0XX.yaml`** を起票 (= 形名参同 Plan、 `/plan` skill)
2. **submit 後 30 分以内** に `docs/research/2026-05-12-submission-analyses.md` に append (= GM §8)
3. **CV-LB correlation** を毎回 update (= GM §1.1 Bestfitting 「Good CV is half of success」)
4. **per-task source diff + 仮説帰納 + roadmap refine** (= GM §8 必須)
5. 完了宣言前に **`/verify <exp-id>`** で reviewer agent 独立判定

並列実行:
- 並列 worker (= Kaggle / Local / Colab) には `~/.claude/rules/agents.md` 「git worktree + tmux + AgentTeams」 原則準拠
- 中央 coordinator が merge / submit / deploy を担当、 worker は branch push までで停止

レビュー:
- 主要 PR は `~/.claude/rules/codex-integration.md` Codex を一次レビュアー、 Claude `code-reviewer` agent は補助
