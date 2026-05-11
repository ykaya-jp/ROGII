# ROGII Wellbore Geology Prediction project-local instructions

> Kaggle competition: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction
> Deadline: 2026-08-05 23:59 UTC (= 残り約 85 day、 as of 2026-05-12)
> User Kaggle ID: ky7240、 Entered: True、 Team count: 831

## CRITICAL: Session start protocol (= `~/projects/kaggle/CLAUDE.md §0.1` 準拠)

1. 本ファイル read
2. **Latest HANDOFF: `docs/dev/HANDOFF-2026-05-12.md`** ★ ← 必ず first read (session end 時に作成、 まだ無ければ skip)
3. `git log --oneline -30` + `.venv/bin/kaggle competitions submissions rogii-wellbore-geology-prediction` で最新確認
4. 該当 handoff doc の "Next session 即 action" section を 1 つずつ実行

## Project context

- **タスク**: Wellbore Geology Prediction (= horizontal well の hidden TVT region を予測する geosteering 問題)
- **evaluation**: RMSE in feet (= lower is better)、 public 30% / private 70%
- **現状**: exp009 v2 LB **9.738** (= rank **17/779**、 5/11 05:37 UTC SCORED)
- **公開 NB 最強**: Hill Climbing (Roman Tarasov、 LB 9.43、 user 抜きトップ)
- **target**:
  1. 金メダル (= rank ~10、 LB **≤ 9.4**)
  2. 優勝 (= rank 1、 LB **≤ 8.5**)

## アクティブな plan (= 2026-05-12 開始)

- **strategy plan**: `/home/yusuke_kaya/.claude/plans/floating-cuddling-haven.md`
  - repo copy: `docs/dev/2026-05-12-plan-gold-to-winning.dense.md`
- **active criteria**: `.criteria/kaggle-rogii-phase-a-2026-05-12.yaml` (= Phase A 金メダル確保)
- **submission 分解 doc** (= GM §8 必須): `docs/research/2026-05-12-submission-analyses.md`

## 4 Phase 概要

| Phase | 期間 | 目標 LB | 主要 task |
|---|---|---|---|
| A | Week 1 | ≤ 9.5 (Gold cutoff 接触) | PENDING 回収 + Numba Beam ±2 + segment b_well + Softmax NCC + 3-axis postproc + Hill Climb weight opt |
| B | Week 2 | ≤ 9.3 (Hill Climb 9.43 凌駕) | data augmentation (paradigm 5) + per-row kNN (paradigm 6) + Sparse GP M=500 並走 |
| C | Week 3-4 | ≤ 8.5 (優勝圏) | NN seq (Mamba) + Per-Well MoE + Multi-task aux head + Ridge meta blend |
| D | 7 day 前 freeze | safe ≤ 9.0 + risky ≤ 8.5 | Adversarial val + 2 submit dual |

## 3 並列 compute 割当

- **Kaggle Notebook** (= 9h/run、 30h/week GPU quota): Phase A submit kernel 全件 + Phase B augment re-train + Phase Z1 multi-formation GP の半数
- **Local GPU** (= 24h × 7 = 168h/week): A2 Hill Climb + B2 FAISS index + B3 OOF C2.v2 再生成 + C2 MoE + Z1 GP M=500 + 全 ablation 測定
- **Colab Pro+ A100** (= 50 A100-hr/月): C1 NN seq (Mamba 5-fold × 5-seed = 25 runs × 2h = 50h)

## 形名参同サイクル (= 各 exp で必ず)

1. `.criteria/<exp-id>.yaml` 起票 (= `/plan` skill)
2. 実装
3. Kaggle submit
4. 30 分以内に `docs/research/2026-05-12-submission-analyses.md` に append (= GM §8 schema: per-task source diff + effect isolate + 仮説帰納 + roadmap refine)
5. 完了宣言前に `/verify <exp-id>` で reviewer subagent 独立判定

## Session end protocol (= `~/projects/kaggle/CLAUDE.md §0.2` 準拠)

context clear する前に必ず:

1. `docs/dev/HANDOFF-<today>.md` 新規作成 (= 既存 doc は historical で残す)
2. 本ファイル "Latest HANDOFF" pointer を新 doc 名に update
3. `git add docs/dev/HANDOFF-<today>.md CLAUDE.md && git commit -m "docs(handoff): <date> session continuation pointer" && git push`
4. user に「session clear OK」 提示

## 関連 doc

- 親: `~/projects/kaggle/CLAUDE.md`
- 親親: `~/.claude/CLAUDE.md`
- Kaggle research workflow: `~/.claude/rules/kaggle-research-workflow.md`
- GM mindset skill: `~/.claude/skills/kaggle-grandmaster-mindset/SKILL.md`
- agents rule: `~/.claude/rules/agents.md` (= 並列開発 + AgentTeams)
- codex review integration: `~/.claude/rules/codex-integration.md`
