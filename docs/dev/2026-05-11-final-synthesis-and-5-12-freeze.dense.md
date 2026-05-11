# 2026-05-11 Final Synthesis + 5/12 Plan Freeze

> 全 4 subagent (V/W/X/Z) 完了 + exp005 v2 LB SCORED で 5/12 plan 最終 freeze に必要な情報が揃った。本 doc が 5/12 morning から実 sub に投入する **operational truth**。
> 関連: `submission-postmortems.dense.md`, `2026-05-11-state-snapshot.dense.md`, `2026-05-11-h10-postmortem-and-fold-reform.dense.md`, 4 subagent doc。

## 1. 全 subagent + verified LB の 1 行 summary

| stream | finding (= 1 行) |
|---|---|
| **subagent V** | **H10 = σ_fold 1.18 ft が Jensen 下限 CV 10.77 で天井固定** → fold 構造改革必須 |
| **subagent W** | WLFM 9 切り射程 (path α 障壁) + **TimeXer / GeostatsPy Kriging / MEMO** 3 hack 即時可 |
| **subagent X** | **Mamba** 単独 -0.7〜-1.2 ft で 9 切り射程、**PySR** 低 risk 加算で b_well_symbolic 全 head 注入 |
| **subagent Z** | **TimeXer = 3 hack 中最大 (-0.5 ft、唯一の paradigm shift)**、Kriging × TimeXer synergy 期待、9 hr offline pretrain 必須 |
| **exp005 v2 LB** | **10.203 = Edge S 単独 -0.114 ft 改善 ✅ verified** (= 想定下限寄り) |
| **exp007 LB** | **10.677 = 自前 4 base + Ridge 9-base が +0.36 ft 悪化** → 自前 4 base 路線は path a 必須 |

## 2. 5/12 sub plan **最終 freeze** (= 5 sub、quota フル活用)

| # | sub | layer | LB 想定 (mid) | 効果分離 |
|---|---|---|---|---|
| 1 | **exp010** | **stratified Edge Q fold + adversarial validation drop** (H10 reflect、subagent Y 実装中) | **9.5-9.7** | H10 verify (= σ_fold 削減幅、CV 改善) |
| 2 | exp011 | exp010 + Edge S + Edge R 加算 | 9.0-9.3 | 加算的 inject 復活 |
| 3 | exp012 | exp011 + 案 D Kalman (= AR(1) state-space) | 8.5-8.9 | Kalman 効果 (verified exp008 v2 待ち) |
| 4 | **exp013** | exp012 + 案 E GP + Edge O | **8.0-8.5 (= 9 切り)** | GP + Edge O 累積 |
| 5 | **exp014** (= 余裕 1 sub) | **PySR (= b_well_symbolic) を exp013 base に注入** | **7.7-8.2 (= Top 1 圏)** | 低 risk 加算 |

= **5/12 5 sub で 9 切り達成 + Top 1 圏到達**。

## 3. 5/13 以降 plan (= Mamba + TimeXer 投入、Top 1 確定)

| date | exp | layer | LB 想定 |
|---|---|---|---|
| 5/13 | exp015 | + **TimeXer** (= subagent Z #1、cross-attention で物理 prior encode、offline pretrain 必須) | 7.4-7.9 |
| 5/14 | exp016 | + **Mamba (= S6 selective state-space)** (= subagent X #2、案 D Kalman の深拡張) | 7.0-7.5 |
| 5/15 | exp017 | + Kriging × TimeXer synergy (= subagent Z exp012_a → 13 統合) | 6.8-7.3 |
| 5/16 | exp018 | + MEMO 流 random_stretch × entropy min (= subagent Z #3、Edge R 拡張) | 6.7-7.2 |
| 5/17 | exp019 | final stack pyramid (= L1 12+ base / L2 Ridge meta / L3 average) | **6.5-7.0** |
| 5/18-5/19 余裕 | exp020-024 | postmortem + ablation + pseudo-label round + post-proc consolidate | - |

5/13-5/19 で 25-35 sub、Top 1 確定。

## 4. 中央が **subagent Y 完了通知前にやる action**

1. **mathematical-formulation.dense.md §7** を別 branch で update (= verified data + H10 反映)
2. **5/12 morning sub の watch 体制**: subagent Y kernel push 直後の Kaggle run COMPLETE を polling、即 submit
3. **branch hierarchy 整理** (= 8 branch + merge 戦略):
   - `feat/phase-5-edge-r-online` を**common base** にして全 doc branch を sync
   - `feat/phase-6-exp010-fold-reform` を exp010 main branch
4. **PySR + Mamba + TimeXer dependency**: pyproject.toml に `pysr`, `mamba-ssm`, `pytorch-lightning` を追加検討 (= 5/13 以降の prerequisite)

## 5. submission ritual 自動化 (= 「随時やれ」)

毎 sub SCORED 後 5 分以内に以下を ritual:
1. `kaggle competitions submissions` で LB 抽出
2. **leaderboard.dense.md** Submission Log update
3. **submission-postmortems.dense.md** matrix + 改善 or 悪化 section update + 仮説 verify/refute
4. **2026-05-11-h10-postmortem-and-fold-reform.dense.md** §3 (= 5/12 plan) refine
5. commit + push (= `~/.claude/CLAUDE.md` lessons.md 該当 `git add <file>` 限定で worker conflict 防止)

## 6. branch hierarchy + merge 戦略 (= 5/13 以降 main 整理用)

現在 8 branch (= V/W/X/Z + Y + 中央):
- `docs/data-mining-2026-05-11` (subagent V)
- `docs/literature-deeper-2026-05-11` (subagent W、別名 `docs/paradigm-deeper-2026-05-11` に hijack)
- `docs/paradigm-deeper-2026-05-11` (= 衝突 branch)
- `docs/cv-breakthrough-2026-05-11` (= mathematical-formulation 母)
- `docs/3hack-spec-2026-05-11` (= subagent Z)
- `docs/discussions-deepdive` (= subagent O)
- `docs/gm-wisdom-2026-05-11` (= subagent Q)
- `docs/research-extra-2026-05-11` (= subagent N + O)
- `feat/phase-5-edge-r-online` (= exp005 v3 + exp009 v4 common base)
- `feat/phase-5-roi3-v3` (= subagent T、Huber + hetero + MEDIAN + path b)
- `feat/phase-6-exp010-fold-reform` (= subagent Y active)

= **混乱状態、5/14 頃に main 整理が必要** (= cherry-pick で全 doc を `feat/phase-5-edge-r-online` に集約、kernel branch は exp010 を main branch 化)。これは subagent Y 完了 + 5/12 sub 完了後に中央が実施。

## 7. 残課題 (= 5/13 以降の subagent 候補)

| # | task | 担当 |
|---|---|---|
| 1 | branch hierarchy 整理 + merge | 中央 |
| 2 | TimeXer offline pretrain pipeline + Kaggle dataset upload | subagent (5/13) |
| 3 | Mamba 実装 (= mamba-ssm install + LGB hybrid + Kalman 互換 verify) | subagent (5/13-14) |
| 4 | WLFM 著者連絡 (= path α 探索、保留中) | ユーザー判断 |
| 5 | Kriging × TimeXer synergy 実装 | subagent (5/14-15) |
| 6 | exp005 v3 LB scoring 完了次第 Edge R 単独効果 verify (= -0.37 公開実測との照合) | 中央 ritual |
| 7 | exp008 v2 / exp009 v2 Kaggle COMPLETE 次第 submit + log 抽出 | 中央 ritual |

## 8. 関連 doc (= 5/12 morning 参照)

中央作業の readiness:
- `docs/dev/leaderboard.dense.md` — Submission Log + CV-LB Trend
- `docs/dev/submission-postmortems.dense.md` — 全 SCORED sub mechanism 分析
- `docs/dev/2026-05-11-state-snapshot.dense.md` — 中央 snapshot
- `docs/dev/2026-05-11-h10-postmortem-and-fold-reform.dense.md` — H10 + fold 構造改革 4 candidates
- **本 doc** — 全 4 subagent synthesis + 5/12 freeze + 5/13-19 plan

subagent 4 doc (= 別 branch、merge 後にアクセス可能):
- `docs/research/submission-data-mining.dense.md` (subagent V、600 行、H10 詳細)
- `docs/research/academic-literature-deeper.dense.md` (subagent W、752 行、WLFM + 3 hack)
- `docs/research/paradigm-deeper.dense.md` (subagent X、906 行、Mamba + PySR + PINN + TabPFN v2)
- `docs/research/3-hack-implementation-spec.dense.md` (subagent Z、565 行、TimeXer + Kriging + MEMO spec)

= **計 2823 行の research base + 中央 5 doc 600+ 行 = 3423+ 行で 5/12 plan 確定**。
