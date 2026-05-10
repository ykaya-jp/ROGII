# 戦略 doc 案 A/B/C 批判的再評価 (2026-05-10)

> 母法: `docs/research/first-principles.dense.md` + `docs/research/independent-edges.dense.md` + `docs/research/top3-distill.dense.md`
> 対象: `docs/strategy/winning-strategy.dense.md` の **案 A (GBM Stack with Rich Features)**、**案 B (Sequence Transformer Cross-Attn)**、**案 C (Geometry-Physics Hybrid)**
> 推奨は出さない、**改修候補 4 つ + 選択軸** のみ提示。

## 1. 各案が生成過程モデルの何を捕捉しているか

`first-principles.dense.md` §1.4 の確率モデル:
$$
\text{TVT}(s) = -Z(s) + \text{ANCC}(s) + b_{\text{well}}(s) + \varepsilon(s)
$$
Bayesian posterior:
$$
p(\text{TVT}_h \mid \cdot) = \int p(\text{TVT}_h \mid Z_h, \widehat{\text{ANCC}}_h, \widehat{b}_h) \cdot p(\widehat{\text{ANCC}}_h, \widehat{b}_h \mid \cdot) \, d\widehat{\text{ANCC}} \, d\widehat{b}
$$

各案の捕捉率を 5 段階 (0-4) で評価:

| 構成要素 | 案 A (GBM) | 案 B (Transformer) | 案 C (Hybrid) | 公開 Top 3 (R) |
|---|---|---|---|---|
| Z (trajectory) | 4 | 4 | 4 | 4 |
| ANCC point estimate | 3 (= plane-fit imputer 移植時) | 2 | 4 (plane-fit core) | 4 |
| **ANCC posterior variance** | 0 | 0 | 0 | **0** ← 公開 top も捕捉せず |
| b_well constant | 4 | 3 | 3 | 4 |
| **b_well drift (MD-linear)** | 0 | 0 | 0 | **0** ← 同上 |
| **dTVT AR(1) state** | 1 (= dTVT lag features) | 2 (= sequence input) | 1 | **1** ← 同上 |
| typewell alignment (Beam/PF) | 3 (= features 化) | 3 (= cross-attn) | 3 | 4 (Beam×7+PF) |
| **soft alignment (differentiable)** | 0 | 1 (= cross-attn は近い) | 0 | **0** |
| Multi-task aux head | 0 | 4 | 1 | 0 |
| Per-well adaptive | 0 | 1 | 0 | 0 |
| Posterior sampling (Diffusion) | 0 | 0 | 0 | 0 |
| **末端 systematic bias** (MD 9000+) | 0 | 0 | 0 | **0** ← 全員捕捉せず |

**観察 1**: 案 A/B/C どれも **ANCC posterior variance / b_well drift / 末端 bias / soft alignment** を捕捉していない。これらは独自 edge 候補 D-I の出現位置と一致 (`independent-edges.dense.md` §1.7)。

**観察 2**: 案 B (Transformer) は multi-task と soft alignment で他 2 案より「構造的に強い paradigm」だが、**200 wells で deep model overfit**のリスクが大きい。

**観察 3**: **公開 Top 3 と案 A の差は小さい** (= ANCC point estimate と b_well constant までで RMSE 10.0 帯到達)。残りの 0 評価項目が 1 位への delta。

## 2. Irreducible Error 到達可能性

`first-principles.dense.md` §2.6: LB 8 帯到達には **MD 4000 ft 末端 std を 12 ft 以下** に圧縮必要。

| 案 | 末端 std 圧縮の見込み | 根拠 |
|---|---|---|
| 案 A | 14-15 ft (= 公開 baseline 12.6 から微改善) | i.i.d. point estimate + features rich 化のみ |
| 案 B | 11-13 ft (= deep model + cross-attn) | sequence prior + multi-task で末端弱化を緩和 |
| 案 C | 12-14 ft | plane-fit + DTW + 1D U-Net residual で hybrid |
| **案 A + D + E** | **10-12 ft** | state-space で AR(1) prior + GP で ANCC posterior |
| **案 A + D + E + F** | **8-10 ft** | + soft alignment + cross-attn pretrain |
| **案 A + D + E + F + I** | **7-9 ft** | + diffusion sampling で uncertainty |

**観察**: **案 A 単独 + 移植 6 要素では 10.0 帯まで**。**LB 8 帯は独自 edge D + E が必須**、6 帯は F + I まで全部入れれば理論的に届く。

## 3. 独自性評価 (= 公開 Top との差分)

| 案 | 公開 Top 3 / Top 32 との paradigm 差 | 評価 |
|---|---|---|
| 案 A | ほぼ同一 (= LGB stack + Beam + PF + plane-fit) | 低 |
| 案 B | seq Transformer は public top にない (TabICL は別 paradigm) | 中-高 |
| 案 C | plane-fit + DTW + 1D U-Net = top に部分一致、1D U-Net 残差は新規 | 中 |
| 案 D | state-space on dTVT は **全公開 top にない** | **高** |
| 案 E | GP imputation は **全公開 top にない** | **高** |
| 案 F | soft-DTW + cross-attn は **全公開 top にない** | **高** |
| 案 G | per-well MoE は **全公開 top にない** | 中-高 |
| 案 H | MD-linear b_well は **全公開 top にない** (= R は constant + recent-weighted まで) | 中 |
| 案 I | conditional diffusion は **全公開 top にない** | **高** |

## 4. 改修候補 4 つ (= 推奨せず軸のみ)

### 4.1 改修 a — 案 A を keep + 独自 edge D/H を加算 (= 安全策)

- 工数最小 (= 残 86 日のうち最大 4 日で投入完了)
- LB 期待 9-10 帯 (= 公開 top と同程度 + minor improve)
- リスク低 (= 既存 paradigm の延長)
- **独自性弱**: 1 位は取れない可能性大

### 4.2 改修 b — 案 A + D + E + G で **multi-edge stack** (= 中道策)

- 工数 5-7 日
- LB 期待 8-9 帯
- リスク中 (= GP scalability, MoE cluster choice)
- **独自性中**: 公開 top にない 3 paradigm を加算

### 4.3 改修 c — 案 B を案 F (Soft-DTW + Cross-Attn) に **置換** + A の feature stack も並走

- 工数 7-10 日 (= deep model 学習)
- LB 期待 7-9 帯
- リスク中-高 (= 200 wells overfit, pretext pretrain 必須)
- **独自性高**: deep paradigm が 1 位射程

### 4.4 改修 d — 全部破棄して 独自 edge **D + E + F + I の純構成**

- 工数 12-15 日 (= 全部新規実装)
- LB 期待 6-8 帯 (= 理論的に最も低い)
- リスク高 (= 全部 deep / Bayesian、同時 debug 困難)
- **独自性最大**: 1 位に最も近いが失敗時のリカバリ困難

## 5. 比較表 (= 4 改修候補 + 現行 plan)

| 観点 | 現行 (A+B+C) | 改修 a (A+D/H) | 改修 b (A+D+E+G) | 改修 c (A+F+置換) | 改修 d (D+E+F+I) |
|---|---|---|---|---|---|
| 工数 | 88 日 (= 元 plan) | +4 日 | +5-7 日 | +7-10 日 | +12-15 日 |
| LB 期待 | 5-8 (元 plan の希望) | 9-10 | 8-9 | 7-9 | **6-8** |
| 独自性 | 中 | 低 | 中 | 高 | **最高** |
| リカバリ | 中 (= 案 ABC の 3 道) | 高 (= A 戻し) | 中 | 低 | **最低** |
| Phase 4-5 timing | gradient | minor delay | moderate | major | major |
| 残 86 日適合 | 適合 | **超適合** | 適合 | tight | 危ない |
| 1 位確実性 | 中 | 低 | 中 | 中-高 | **高だが博打** |

## 6. 選択軸 (= 中央が判断すべき)

判断 1: **コンペ残期間との trade-off**
- 残 86 日 vs 改修 d (15 日 + Phase 3-5 = 70+ 日) → 工数余裕は微妙
- 改修 a (4 日追加) なら schedule に余裕、改修 c なら Phase 4-5 を圧縮

判断 2: **risk-return balance**
- 1 位確実性高 = 改修 c or d (高リスク・高 return)
- 1 位射程はあるが安全 = 改修 b (中)
- 入賞で十分 = 改修 a (top 5-10 狙い)

判断 3: **独自性 vs 移植**
- 独自 edge 多く投入 (= 改修 d) → コンペ後の write-up 価値高、技術的にも面白い
- 移植中心 (= 改修 a) → 公開 top の延長で確実だが gold 圏は微妙

判断 4: **チーム構成 (= 1 人 vs subagent 並列)**
- 1 人作業: 改修 b までが現実、改修 c/d は subagent 並列が必須
- subagent 並列 = kill リスクあり (= 今回 4 並列が全 kill された経験) → 並列度は控えめ + 各 task の中間 commit を頻繁化

## 7. 戦略 doc 修正提案 (= 推奨せず編集案のみ)

### 7.1 §「優勝戦略の構造原理」の section update

現行: 案 A (GBM Stack) / 案 B (Sequence Transformer) / 案 C (Geometry-Physics Hybrid) の 3 案。

更新候補:
- **§ "案 A 拡張":** A に Phase 1.6 §B の core 6 要素 + 独自 edge D + H を加算 (改修 a/b の構成)
- **§ "案 D — State-Space Kalman/PF":** 新規 section (independent-edges §1)
- **§ "案 E — Bayesian GP Imputation":** 新規 (independent-edges §2)
- **§ "案 F — Soft-DTW + Cross-Attn":** 案 B を置換 (independent-edges §3)
- **§ "案 I — Conditional Diffusion":** 新規 (independent-edges §6, optional)

### 7.2 §「マイルストーン」の Phase 配分 update

現行 Phase 2-6 の 88 日想定 → 残 86 日 + 改修選択で:

| 改修選択 | Phase 2 (LB 12) | Phase 3 (LB 10) | Phase 4 (LB 9) | Phase 5 (LB 8) | Phase 6 (final) |
|---|---|---|---|---|---|
| 改修 a | 9 日 (= 現行) | 14 日 (移植 6 + D + H) | - | 14 日 (stack + post) | 8 日 |
| 改修 b | 9 日 | 14 日 (移植 + D + E) | 14 日 (G/MoE) | 14 日 (stack) | 8 日 |
| 改修 c | 9 日 | 14 日 (移植) | 21 日 (案 F deep) | 14 日 (stack) | 8 日 |
| 改修 d | 9 日 | 14 日 (D + E) | 28 日 (F + I) | 14 日 (stack) | 8 日 |

### 7.3 §「リスクと対策」の追加

- **リスク 8 (新規)**: Subagent kill 多発 (= 今回 4 並列が全停止した経験) → 並列度を 2-3 に抑える、各 task の中間 commit を頻繁化、kill 後の復旧 SOP を策定
- **リスク 9 (新規)**: 末端 systematic bias (`extrap mean = -16 at MD 9000+`) → Phase 5 post-proc で stage-aware regression / 案 J の検討

## 8. 関連ファイル

- `docs/strategy/winning-strategy.dense.md` (= 編集対象)
- `docs/research/first-principles.dense.md` (= 母法)
- `docs/research/independent-edges.dense.md` (= 独自 edge 候補集)
- `docs/research/top3-distill.dense.md` (= 公開 top の解体)
- `docs/dev/schedule-2026-05-10.dense.md` (= 残 86 日スケジュール)
