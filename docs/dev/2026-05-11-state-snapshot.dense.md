# 2026-05-11 02:30 UTC State Snapshot (= 優勝寄与最大化 priority)

> ユーザー指摘 (2026-05-11 02:00 UTC) 「軽量かどうかでやることを選ぶのではなく、優勝により近づけるかどうかで今後のやることもちゃんと選んでね」
> 本 doc: 中央 idle 0 + 優勝寄与最大化 の standing order を実行した中央 snapshot。subagent V/W/X 並列 dispatch 中の active branch conflict 回避のため、`docs/dev/` 配下に新規作成。
> 関連 doc: `submission-postmortems.dense.md`, `leaderboard.dense.md`, `mathematical-formulation.dense.md` (= `docs/cv-breakthrough-2026-05-11` branch)

## 1. 真の sub state (= 2026-05-11 02:30 UTC)

| sub | LB | direction | 数理 verify |
|---|---|---|---|
| exp002 (baseline) | 14.695 | - | - |
| exp003 (= 自前 tysig 単独) | 17.510 | ⚠️⚠️ 大失敗 +2.815 | 自前複雑 features 単独 deprecate 決定 |
| exp005 v1 (karnakbaev blend) | **10.317** | ✅ Silver -4.378 | 母法決定 |
| exp006 (+ TabICL) | 10.503 | ⚠️ 微悪化 +0.186 | TabICL CUDA error fallback |
| exp007 (+ Edge Q+M+自前 4 base+Ridge 9-base) | **10.677** | ⚠️ 悪化 +0.360 | **fold-misalign 影響量 +0.29 (想定 +0.12 の 2.4 倍 severe)** |
| **exp005 v2 (+Edge S 単独)** | **10.203** | ✅ -0.114 | **Edge S 単独効果 verified (想定下限寄り)** |
| **exp005 v3 (+Edge S +Edge R)** | _PENDING_ | (予測 9.65-9.95) | Edge R 単独効果分離待ち、Gold cutoff 9.919 ぎりぎり |
| exp008 v2 (案 D Kalman + Edge S) | _Kaggle 上 RUNNING_ | (予測 10.0-10.2) | 案 D Kalman 単独効果分離 |
| exp009 v2 (案 E GP + Edge O + Edge S) | _Kaggle 上 RUNNING_ | (予測 9.5-9.9) | 案 E + Edge O 累積効果 |

= **6 SCORED + 1 PENDING + 2 RUNNING = 5/5 used (today UTC 5/11 quota 2/5、残 3)**

## 2. verified data から学んだ 9 切り roadmap 再校正

### 2.1 layer 別 verified vs 予測

| layer | 予測 LB delta | 実測 LB delta | accuracy |
|---|---|---|---|
| Edge S 単独 (= exp005 v2 - v1) | -0.05〜-0.30 | **-0.114** | ✅ 範囲内、中域寄り |
| 自前 4 base + Ridge 9-base (= exp007 - exp005) | (改善想定) | **+0.360 (悪化)** | ⚠️ 完全 miss |
| TabICL (= exp006 - exp005) | (改善想定) | **+0.186 (悪化)** | ⚠️ 完全 miss (CUDA error fallback) |
| Edge R 単独 (= exp005 v3 - v2 予定) | (-0.37 公開実測) | PENDING | verify 待ち |
| 案 D Kalman 単独 (= exp008 v2 - v1 予定) | (-0.30〜-0.60) | RUNNING | verify 待ち |
| 案 E + Edge O 累積 (= exp009 v2 - v1 予定) | (-0.40〜-0.80) | RUNNING | verify 待ち |

### 2.2 教訓 (= 5/12 以降の意思決定 root)

1. **基底 stack 改造 (= 自前 4 base 追加 / TabICL 追加) は失敗 path** = +0.18〜+0.36 ft 悪化、verified 2 件
2. **加算的 inject (= Edge S 単独) は想定通り改善** = -0.114 ft、verified 1 件
3. → **5/12 戦略 = 加算的 inject を最優先、基底 stack 改造は path a (= karnakbaev 真の OOF 再生成) 達成後のみ**

### 2.3 重要な layer 効果の sign 確定状況

| layer | sign 確定 | 数値 |
|---|---|---|
| Edge S | ✅ + (improve) | -0.114 ft |
| Edge R | (公開実測 +0.37、verify 待ち) | exp005 v3 LB - 10.203 で計算可 |
| 案 D Kalman | 不明、RUNNING | exp008 v2 LB - 10.317 で計算可 |
| 案 E GP + Edge O | 不明、RUNNING | exp009 v2 LB - 10.317 で計算可 |
| 自前 4 base (Ridge) | ⚠️ - (worsen) | +0.36 ft |
| TabICL | ⚠️ - (worsen with fallback) | +0.186 ft |
| Edge Q + Edge M (= exp007 layer 分離不能) | 不明、自前 4 base 込みで悪化 | 分離不能 |

## 3. 残 sub plan (= 今日 3 + 明日 5 = 8 sub で 9 切り達成判定)

### 3.1 今日 (UTC 5/11) 残 3 sub

| # | sub | base | 効果分離 | 想定 LB |
|---|---|---|---|---|
| 3/5 | exp008 v2 (Kaggle 上 RUNNING) | exp005 + Kalman + Edge S | 案 D Kalman 効果 | **10.0-10.2** |
| 4/5 | exp009 v2 (Kaggle 上 RUNNING) | exp005 + GP + Edge O + Edge S | 案 E + Edge O + Edge S 累積効果 | **9.5-9.9** |
| 5/5 | exp009 v4 (= v3 + Edge R、または exp008 v3 / exp009 v3) | 自前 4 base + path b + Edge R | risk-return 賭けの 1 手 | **7.8-8.1 (Top 1 圏)** または **悪化** |

5/5 の選択軸:
- **safe**: exp008 v2 v3 (= Huber + hetero + MEDIAN + path b、ただし自前 4 base 路線継承で悪化リスク中)
- **risk**: exp009 v4 (= 上記 + Edge R 重ね合わせ、Top 1 圏 or 悪化)
- **新候補**: exp005 v3 と exp008 v2 / exp009 v2 LB 出てから判断 (= 自前 4 base 路線が verify 済悪化と判明したら exp009 v4 push せず、karnakbaev base + Edge S + Edge R + 案 D / 案 E のみで合成した **新 exp010** を緊急設計)

### 3.2 明日 (UTC 5/12) 5 sub plan

verified data + subagent V/W/X 完了 doc を踏まえて再構築。候補:

| 候補 sub | layer | 想定 LB |
|---|---|---|
| exp010 = exp005 v3 + 案 D Kalman feature | exp005 + Edge S + Edge R + Kalman (= 自前 base なし、加算的 inject 純粋路線) | 9.3-9.7 |
| exp011 = exp010 + 案 E GP + Edge O | 上記 + GP + Edge O | 8.8-9.3 |
| exp012 = exp011 + Edge N (offset well retrieve) | 上記 + Edge N | 8.5-9.0 |
| exp013 = exp012 + 案 G MoE or final stack consolidate | (subagent V/W/X 結果次第) | 8.2-8.7 |
| 余裕 1 sub | postmortem 後の re-attempt or 新 paradigm (= Mamba / TabPFN v2 / PINN) | 不確実 |

= **加算的 inject 純粋路線で 8.x 帯到達**、自前 4 base 路線は **path a 達成まで凍結**。

## 4. subagent V/W/X 並列 dispatch 状況

| subagent | scope | branch | 出力 |
|---|---|---|---|
| **V** | 6 SCORED sub の log mathematical-level deep mining、H 仮説 5-10 件 | `docs/data-mining-2026-05-11` | `submission-data-mining.dense.md` |
| **W** | 学術文献 deeper (WLFM, ICML 2024, UQ, SEG/SPE, Pyrcz, TTT) | `docs/literature-deeper-2026-05-11` | `academic-literature-deeper.dense.md` |
| **X** | 優勝寄与高 4 paradigm deeper (Mamba/S4, TabPFN v2, PINN, PySR) | `docs/paradigm-deeper-2026-05-11` | `paradigm-deeper.dense.md` |

= **並列度 3、kill リスク中間 commit で manage**、完了次第 synthesis で 5/12 plan 最終 refine。

## 5. 中央 standing order (= 「優勝寄与最大化、軽量基準禁止」)

ユーザー指示 (2026-05-11 02:00 UTC) で確定:

1. **LB scoring 待ち / kernel run 待ち = idle time 禁止**: 必ず subagent dispatch or 中央 doc 作成
2. **タスク選定基準 = 優勝寄与最大化** (= 軽量/重い基準でなく)
3. **branch conflict は中間 commit + 別 doc 作成で manage**: 重い作業も skip しない
4. **並列度 3-4 を許容**: kill リスクは 30 分中間 commit で minimize
5. **postmortem 毎 sub SCORED 後 5 分以内 update**: §1 の verified data を即追加

## 6. 直近の natural break point

| 通知 | 期待時刻 | 中央 action |
|---|---|---|
| exp005 v3 LB scoring 完了 | 数十分以内 | Edge R 単独効果 verify + postmortem update + 5/12 plan refine |
| exp008 v2 Kaggle COMPLETE | 30-60 min | submit (= 3/5) + 案 D Kalman 効果分離測定 plan |
| exp009 v2 Kaggle COMPLETE | 30-60 min | submit (= 4/5) + 案 E 効果分離測定 plan |
| subagent V/W/X 完了 | 30-90 min | synthesis + 5/12 plan 最終 refine |

これらが全部揃った時点で **5/11 23:00 UTC 頃の最終 sub plan freeze** + 5/12 morning sub queue 確定。

## 7. 関連 doc

- `docs/dev/leaderboard.dense.md` — Submission Log + CV-LB Trend
- `docs/dev/submission-postmortems.dense.md` — 全 SCORED sub の mechanism 分析
- `docs/research/mathematical-formulation.dense.md` (= 別 branch) — 数理 framework
- `docs/research/first-principles.dense.md` — generative model + 計測値
- `docs/research/gm-wisdom.dense.md` — GM 叡智
- `docs/research/cv-breakthrough.dense.md` — 12 paradigm shift
- `docs/research/submission-data-mining.dense.md` (= subagent V 進行中)
- `docs/research/academic-literature-deeper.dense.md` (= subagent W 進行中)
- `docs/research/paradigm-deeper.dense.md` (= subagent X 進行中)
