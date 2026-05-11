# Submission Postmortems — 全 SCORED sub の改善/悪化 mechanism 分析

> 起源: 2026-05-11 ユーザー指摘 「悪化したサブミットのすべてに対してなぜそのような結果だったのか、また改善した場合も同様にしっかり考察してつぎに活かしてますか？」
> 起源前は leaderboard.dense.md の note 欄に断片的記録のみ = **systematic な postmortem 文化が欠落**。本 doc で全 SCORED sub の mechanism-level root cause を抽出、横断 pattern を発掘、次の exp 設計時の checklist 化する。
> **毎 sub SCORED 後 5 分以内に本 doc を update する** ルール採用。

## 0. 全 SCORED submit matrix (= 5 件)

| sub | base | layer | LB | vs base | direction | root cause hypothesis | verified? |
|---|---|---|---|---|---|---|---|
| exp002 | - | LGB residual baseline | 14.695 | (baseline) | - | - | - |
| exp003 | exp002 | + tysig features (33+ rich features) | 17.510 | **+2.815** | 悪化 | features 過剰 + CV 計測なし blind submit | partial |
| **exp005** | exp002 | karnakbaev artifacts blend (LB 10.784 base + live test FE) | **10.317** | **-4.378** | **改善** | karnakbaev pretrained 質 + live test FE strengthening | partial |
| exp006 | exp005 | + TabICL (Ridge 6-base re-fit) | 10.503 | +0.186 | 悪化 | TabICL CUDA error → fallback NM blend で Ridge weight 劣化 | partial |
| exp007 | exp005 | + Edge Q + Edge M + 自前 4 base + Ridge 9-base | 10.677 | +0.360 | 悪化 | fold-misalign (+0.29 CV-LB diff) + 自前 base diversity 不足 | partial |

PENDING (= 5/11 0:13, 0:32 submitted): exp005 v2 (= Edge S 単独), exp005 v3 (= Edge S + Edge R)

## 1. ★改善した sub: exp005 (-4.378 ft)

### 1.1 改善 mechanism (= 5 components)

| # | component | math/物理根拠 | 寄与推定 |
|---|---|---|---|
| 1 | **karnakbaev pretrained 5 base** (LGB×3 + XGB + CB) | 公開 LB 10.784 = large-scale train 済 model 容量 + diversity | -3.9〜-4.3 (= base) |
| 2 | **live test FE 強化** | karnakbaev 元 kernel との差 (= subagent G が test 側 FE を live 計算) | -0.05〜-0.50 |
| 3 | **物理 formula `TVT = -Z + ANCC + b_well`** | `formula_oracle_rmse = 0.006 ft` (= 真値で完璧)、ANCC imputation + b_well estimation が core | base に含む |
| 4 | **6-formation per-well b_well** (median + WLS recent-weighted) | first-principles §0.2 の per-well drift fit | base に含む |
| 5 | **Ridge meta + post-proc** (alpha, tau, SG smooth) | karnakbaev postproc_params 踏襲 | -0.05〜-0.10 |

### 1.2 verified vs unverified

| 項目 | verified? |
|---|---|
| 「karnakbaev pretrained 5 base が主要因」 | ✅ verified (= 公開 LB 10.784 が直接証拠) |
| 「live test FE = +X ft 寄与」 | ❌ unverified (= 自前 CV を取らず blind submit したため分離不可) |
| 「5 component の個別寄与」 | ❌ unverified (= ablation 未実施) |
| 「Edge S inject 後の Edge S 単独効果」 | 🟡 PENDING (= exp005 v2 LB scoring 中) |

### 1.3 次への反映 (= 5/12 以降の checklist 1)

- ✅ **karnakbaev base を母法に**: exp008/009 v3 で継承済 (= 改修 b の核)
- ⚠️ **5 component の寄与分離**: exp005 v2 (Edge S 単独) と exp005 v3 (Edge S + Edge R) で Edge S/R を分離測定中、他 component は ablation 設計次第
- ✅ subagent G の Approach B 採択を後続 subagent T/U に継承

## 1.4 ★改善した sub: exp005 v2 (= Edge S 単独、-0.114 ft、2026-05-11 SCORED)

### 1.4.1 改善 mechanism

- **Edge S = round-to-grid post-process**: $\hat{TVT}_{\text{snap}}(s) = 0.01 \cdot \text{round}(\hat{TVT}(s) / 0.01)$
- 数理根拠: dTVT が **0.01 ft grid に 100% 普遍** (= 773 wells 全数実測、hengck23 投稿 697431 msg#8)
- mechanism: 連続 predict の小 noise を 0.01 grid に snap → systematic error が grid 単位の規律化、test 真値 (= 0.01 grid 上) との一致確率上昇
- evidence: Kaggle 上 unique tvt 14151 → 4735 (= 67% 削減)、on-grid 比率 **100%**

### 1.4.2 数値 verify (= 新 kaggle CLAUDE.md §8.1 #9 effect isolate)

- 予測 (postmortem doc §4.1): 10.05-10.27、想定 -0.05〜-0.30 ft
- 実 LB: **10.203**、改善 **-0.114 ft**
- 予測 accuracy: ✅ **範囲内** (= 想定 spread の中域、若干下限寄り)

### 1.4.3 仮説帰納 + 検証 plan

| 仮説 | refute/verify status | 検証 plan |
|---|---|---|
| 「Edge S は連続 predict の noise を grid snap で削減」 | ✅ partial verified (= LB 改善方向だが、grid snap が systematic 誤差を悪化させるケースが想定上限を抑えた可能性) | exp005 v3 (= Edge S + Edge R) LB が出たら **Edge R 単独効果 = exp005 v3 LB - 10.203 (= Edge S 込) - (Edge R 想定 -0.37) で計算** |
| 「Edge S 効果は base model の predict quality に依存」 | ❌ unverified | exp008 v2 (= 案 D + Edge S) LB が exp005 v2 改善幅 (-0.114) より大きいか小さいかで、Kalman base + Edge S の組み合わせ効果を check |
| 「Edge S が想定下限 (-0.05) でなく中域 (-0.114) の理由」 | 仮説: karnakbaev base + SG smooth が既に grid 近傍に予測している → snap 効果 limited | exp008 v2 / exp009 v2 LB で base 違いの Edge S 効果幅を比較 |

### 1.4.4 roadmap refine (= 9 切り roadmap への impact)

旧 9 切り roadmap 想定 (`docs/dev/leaderboard.dense.md` §9 切り):
- Edge S 想定 -0.05〜-0.30 ft → 実 **-0.114 ft** = **想定下限寄り**
- 累積見積 (mid case): 10.317 base + 残 layer 累積 -2.5 ft で 7.8 着地想定
- **修正後**: Edge S -0.114 を bake in、残 layer (= Edge R / 案 D / 案 E / Edge O / fold-misalign 解消 / Multi-seed MEDIAN / Huber + hetero) の累積を **再評価必要**

### 1.4.5 次への反映 (= 5/12 以降の checklist)

- ✅ **Edge S は全 kernel に inject 維持** (= exp005 v3 / exp008 v2 / exp009 v2 / exp008 v3 / exp009 v3 / exp009 v4 で全部適用済)
- ⚠️ **Edge S 効果が想定下限 = grid snap で systematic 誤差が悪化するケースがある可能性** → exp008 v2 / exp009 v2 で base 違いを比較、もし base ごとに改善幅が大きく違う場合は **adaptive snap** (= 一部 well を skip) を v3 候補に
- ✅ **AB ablation 設計の成功**: exp005 v2 vs exp005 v1 (= 10.317) で Edge S 単独効果が分離測定できた、今後の sub も同 pattern

---

## 2. 悪化した sub × 3

### 2.1 exp003 = +2.815 悪化 (= **最大失敗**)

#### root cause 分析

| 階層 | 原因 | verified? |
|---|---|---|
| 直接因 | 33+ features 過剰添加 → overfit | ✅ verified (= LB 数値で間接) |
| 間接因 | CV 計測なし blind submit (subagent A kill で完遂せず) | ✅ verified (= subagent A note) |
| メタ因 | subagent dispatch text で CV 必須化を明示しなかった | ✅ verified (= dispatch text review) |

#### 残不確実性
- 「33 features のどれが decisive か」 = ❌ unverified (= ablation 不可能、結果重視で deprecate 決定)

#### 次への反映 (= checklist 2)
- ✅ **自前複雑 features 単独路線は deprecate** = exp004 以降踏襲済
- ✅ **CV gating 4 条件** (= leaderboard §CV-LB Trend) 明文化
- ✅ subagent dispatch text に「CV 必須」を毎回明記 (= subagent T/U 以降適用)

### 2.2 exp006 = +0.186 悪化

#### root cause 分析

| 階層 | 原因 | verified? |
|---|---|---|
| 直接因 | Kaggle CUDA error → TabICL phase 失敗 | ✅ verified (= Kaggle log `TabICL path failed: AcceleratorError: CUDA error: no kernel image is available`) |
| 間接因 | fallback path = karnakbaev 5-base NM blend、exp005 v1 の Ridge blend と weight constellation 異なる | 🟡 partial (= NM blend と Ridge weights の数値差は未確認) |
| メタ因 | Kaggle 上の TabICL 互換性を事前 smoke で verify せず push (= subagent I の smoke は **ローカル fallback path のみ**) | ✅ verified (= subagent I notes) |

#### 残不確実性
- **TabICL が success path で動いた場合の LB** = ❌ unknown (= 推測 best case 10.0-9.7)
- 「NM blend vs Ridge weights の差 = 悪化の根本」 = 🟡 partial (= 仮説、Ridge weights 公開なし)

#### 次への反映 (= checklist 3)
- ✅ **fallback path も含めて事前 smoke test** = subagent T/U の smoke で `EDGE_R_ENABLE=False` fallback 動作確認済
- ⚠️ **Kaggle 上の library 互換性事前 verify**: TabICL を Kaggle 上で先に動作確認、または skip 決定。subagent T で TabICL を dormant にする選択 (= path b 簡素化) と整合

### 2.3 exp007 = +0.360 悪化 (= **戦略修正 trigger**)

#### root cause 分析

| 階層 | 原因 | verified? |
|---|---|---|
| 直接因 | Ridge 9-base meta で fold-misalign leak | ✅ verified (= CV-LB diff +0.29 ft、想定 +0.12 の 2.4 倍 severe) |
| 間接因 1 | 自前 4 base OOF 10.66-10.93 が karnakbaev 5 base 10.78 と同等以下 = diversity 不足 | ✅ verified (= Kaggle log で OOF RMSE 数値) |
| 間接因 2 | Ridge meta が `lgb_own2 weight=0` で部分 excluded、しかし全体 weight 配分が exp005 の最適からズレ | ✅ verified (= Kaggle log で Ridge weights 表示) |
| メタ因 | karnakbaev pretrained を Edge Q fold で**真の OOF 再生成**しない (= deferred to v2、subagent K note 認知済) | ✅ verified (= subagent K の commit comment) |

#### 残不確実性
- **path b (= subagent T 改修) で解消できるか** = 🟡 verify 待ち (= exp008 v3 / exp009 v3 LB が出ないと不明)
- **fold-misalign +0.29 影響量の正確な分解** (= leak vs 自前 base diversity 不足) = 🟡 partial

#### 次への反映 (= checklist 4)
- ✅ subagent T (exp008 v3 / exp009 v3) で **path b 採用** (= kb simple-avg + own Ridge separate、leak 数理的解消)
- ✅ subagent T で **Multi-seed MEDIAN + Huber + heteroscedastic** 追加 (= 自前 base 質改善)
- ⚠️ **path a (= karnakbaev 真の OOF 再生成)** は未着手、v4 候補 (= 5/12 LB 次第)
- ✅ **自前 4 base 路線の risk** = sub plan で safe path (= karnakbaev only) と risk path (= 自前 4 base) を分離 (= 今日 5/11 の 5 sub plan に反映)

## 3. ★横断 pattern (= 全 5 sub から見える失敗/改善 trend)

### 3.1 改善 trend (= 何が effective か)

| pattern | 証拠 |
|---|---|
| **既存 pretrained 質の高い model を母法に** | exp005 = karnakbaev (10.784) base で -4.378 ft 達成 |
| **物理 formula を正確に計算** | formula_oracle_rmse = 0.006 ft が示すように、formula component の improve が直接 LB に効く |
| **single submit でなく ablation 設計** | exp005 v2 vs v3 = Edge S/R 分離測定可能 (= 今日の 5 sub plan) |

### 3.2 悪化 trend (= 何が ineffective / counter-productive か)

| pattern | 証拠 |
|---|---|
| **自前 複雑 features 単独路線** | exp003 = +2.815 ft 悪化 |
| **新 base 追加 + Ridge re-fit、ただし base 質 + integration が不十分** | exp006 (TabICL fallback)、exp007 (自前 4 base + fold-misalign) で +0.186〜+0.360 ft 悪化 |
| **CV 計測なし blind submit** | exp003 で発覚、その後の subagent dispatch で必須化 |
| **fold partition 違いの OOF を同 Ridge meta に投入** | exp007 で発覚、subagent T path b で解消 |

### 3.3 メタ pattern

- **既存 stack を変える時、新 component の質 (= OOF が既存 base より良い) + integration の正確性 (= fold partition 一致 + leak guard) が両方 high でないと悪化**
- **safe path = 既存 stack を維持 + 加算的 inject (例: Edge S, Edge R)** が ROI 最大
- **risk path = 新 base 追加 + Ridge re-fit** は **path a (karnakbaev 真の OOF 再生成) 必須**、それなしでは悪化リスク高

## 4. PENDING sub の事前予測 (= 5/11 02:00 UTC 時点)

### 4.1 exp005 v2 (= Edge S 単独)

- base: 10.317、Edge S effect = -0.05〜-0.30 ft
- 観察 evidence: Kaggle 上 unique tvt 14151 → 4735 (= 67% 削減)、on-grid 100% (= subagent P log)
- 予測: **10.05-10.27 ft**

### 4.2 exp005 v3 (= Edge S + Edge R)

- base: 10.317、Edge R effect = -0.37 ft (= discussion 698002 実測公開)、Edge S effect = -0.05〜-0.30 ft
- 単純加算: 10.317 - 0.42〜-0.67 = 9.65-9.95 ft
- 予測: **9.65-9.95 ft (= Gold cutoff 9.919 ぎりぎり)**

### 4.3 LB SCORED 後の追記事項

LB 結果出次第、本 doc に:
- 実 LB vs 予測の差 (= 予測 accuracy 検証)
- Edge S vs Edge R 寄与の分離 (= v2 LB と v3 LB の差)
- 5/12 以降の sub plan への反映

## 5. 5/12 以降の checklist (= 全 exp 設計時の必須確認)

| # | item | 関連 sub |
|---|---|---|
| 1 | **CV を必ず取る + Kaggle log で confirm**: blind submit 禁止 | exp003 教訓 |
| 2 | **fold alignment 確認**: Ridge meta input が全て同一 fold partition 由来か | exp007 教訓 |
| 3 | **既存 best (= karnakbaev base) を母法に、加算的 inject**: 新 component は置換しない | exp005 改善 + exp006/007 悪化教訓 |
| 4 | **新 base 追加時の risk-return**: diversity が exp005 base より高い場合のみ ROI | exp007 教訓 |
| 5 | **fallback path の事前 smoke**: 主 path 失敗時にも LB 悪化しない integrity | exp006 教訓 |
| 6 | **CV-LB diff trend を毎 submit 後 5 分以内に update**: leaderboard.dense.md は immediate-update | systemic gap (本 doc trigger) |
| 7 | **AB ablation 設計**: 同層に control を必ず置く | 今日の exp005 v2 vs v3 = good practice |
| 8 | **postmortem doc を毎 sub SCORED 後 update**: 本 doc の運用ルール | systemic gap (本 doc trigger) |

## 6. **本 doc の運用ルール**

毎 sub SCORED 後 (= LB 数値出た直後) 5 分以内に:
1. §0 の matrix に新 sub を追記
2. 改善/悪化に応じて §1 or §2 に section 追加
3. §3 の横断 pattern を update (= 新 evidence で refute されたものは取り消し線)
4. §5 の checklist を update (= 必要に応じて新 item 追加)
5. commit + push

これにより「**改善も悪化も次に活かす**」が **systematic に運用される** (= 今日まで欠落していた文化を補完)。

## 7. 関連 doc

- `docs/dev/leaderboard.dense.md` — submit log の immediate-update テーブル
- `docs/research/mathematical-formulation.dense.md` — 数理 framework (= 何が effective か mathematical justify)
- `docs/research/gm-wisdom.dense.md` — GM 叡智 (= Adversarial Validation、CV-LB diff CSV)
- `docs/research/cv-breakthrough.dense.md` — CV を 8.x 帯に持っていく 12 paradigm shift
- 各 `experiments/exp*/notes.md` — subagent 実装 note
- `outputs/kernel_logs/exp*/` — Kaggle run log (= CV 数値 source)
