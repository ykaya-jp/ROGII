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

## 関連 doc

- plan: /home/yusuke_kaya/.claude/plans/floating-cuddling-haven.md (+ repo copy: docs/dev/2026-05-12-plan-gold-to-winning.dense.md)
- criteria: .criteria/kaggle-rogii-phase-a-2026-05-12.yaml
- LB tracking: docs/dev/leaderboard.dense.md
- submission postmortems: docs/dev/submission-postmortems.dense.md
- past CV strategies: .criteria/kaggle-rogii-cv-strategies-2026-05-11.yaml
