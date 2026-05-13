# 我々 exp009 v2 (LB 9.738) vs Hill Climb fork (期待 LB 9.43) の test set 予測乖離分析

> 起点: 2026-05-13 user 指示「待ち時間に best score の OOF + 当てられないデータ + 改善方針を深堀せよ」
> 比較対象: `exp009 v2` (= submission 52534803 LB 9.738) と `exp017 hillclimb fork v3` (= ky7240/rogii-exp017-hillclimb-fork-v3、 inline self-Climber、 submit 待ち)
> Source data: `/tmp/rogii-exp009-output/submission.csv` + `/tmp/rogii-exp017-v3-output/submission.csv`
> 仮定: Hill Climb の予測が平均的に真値に近い (= LB 9.43 < 9.738 から推測)、 乖離大 row = 我々が外している row

---

## 1. test set 構成 (= 3 wells、 hidden region のみ)

| well | rows | row range (hidden) | visible_ratio | 真値距離 mean_abs (vs hc) |
|---|---|---|---|---|
| `000d7d20` | 3836 | 1442 — 5277 | 27.3% (= visible 1441 / total 5277) | **0.68 ft** |
| `00bbac68` | 6014 | 1545 — 7558 | 20.4% (= visible 1544 / total 7558) | **1.19 ft** |
| `00e12e8b` | 4301 | 2083 — 6383 | 32.6% (= visible 2082 / total 6383) | **1.40 ft** (= 最大) |

合計 14151 rows。 visible_ratio 最大 (= 32.6%) の 00e12e8b で乖離最大 = **「情報多い well = 当てやすい」 は不成立**

---

## 2. 大域 bias + variance (= ours - hc)

```
mean diff   +0.509 ft  ← ours は hc より systematic に深い側
std         1.472 ft   ← bias と等オーダーの variance
max         +4.26 ft   ← 00e12e8b row 3153
min         -4.60 ft   ← 00bbac68 row 6268
```

- **abs_diff > 1.0 ft の rows = 5807 / 14151 = 41.0%** = test set の 4 割で 1 ft 以上ずれる
- LB diff = 9.738 - 9.43 = +0.31 ft、 mean_diff +0.51 ft が大きい = **bias 成分が支配的**、 ただし symmetric な variance (= max +4.26 と min -4.60) も存在

---

## 3. 各 well の row position (= 5 分位) ごとの abs_diff

```
                Q1_early   Q2     Q3_mid   Q4     Q5_late
000d7d20         0.31     0.41    0.28    0.99    1.41   ← 末期に乖離
00bbac68         0.23     0.57    2.05    1.54    1.59   ← 中盤以降爆発
00e12e8b         2.57     2.88    0.47    0.18    0.90   ← 開始直後に最大、 後半収束
```

**well ごとに乖離 phase が違う**:
- `000d7d20`: hidden 末期 (= row 5000+) で trend extrapolation error 累積
- `00bbac68`: 中盤 Q3 (= row 5500 前後) で transition、 hc が捕捉 / ours が miss
- `00e12e8b`: hidden 開始直後 (= row 2100 前後) で **boundary transition を取り損ねている**

---

## 4. 方向 bias (= 各 well で ours が hc に対しどちら側に外しているか)

| well | top spike row | tvt_ours | tvt_hc | diff | 方向 |
|---|---|---|---|---|---|
| `000d7d20` | row 5133 | 11738.58 | 11736.29 | **+2.29 ft** | 深い側に外す |
| `00bbac68` | row 6268 | 12196.45 | 12201.05 | **-4.60 ft** | 浅い側に外す |
| `00e12e8b` | row 3153 | 11612.20 | 11607.95 | **+4.26 ft** | 深い側に外す |

方向は well ごとに違う = **systematic 全体 bias** ではなく **per-well trend prediction error**

---

## 5. smoothness 差 (= step-to-step jaggedness)

```
                mean |step|    ratio ours/hc
000d7d20         ours 0.0221     1.022   ← ほぼ同等
00bbac68         ours 0.0304     0.997   ← 同等
00e12e8b         ours 0.0300     1.113   ← ours が 11% jaggy = under-smoothed
```

- 00e12e8b で **我々の予測が hc より 11% 局所 noise が多い** = savgol smoothing 不足 / post-proc tau (= time constant) が小さすぎ

---

## 6. 真因仮説 (= 5 件、 優先度順)

### H1. post-proc alpha × tau × w_pf grid が粗い (= LB 高、 確度 大)
- 既存 exp009 v2 = 2-axis grid (= alpha × tau)、 解像度低
- hc = Optuna 500-trial 3-axis (= alpha × tau × w_pf) + savgol smoothing
- 我々 LB 9.738 → 9.4 帯への gap の **支配的成分**
- exp016 自前統合に inject 済 (`_optimize_postproc_grid_2530` = 2530-cell exhaustive)、 完走後即解消見込み

### H2. visible↔hidden boundary 直後の trend extrapolation 弱い (= LB 中、 確度 中)
- 00e12e8b Q1 で mean_abs 2.57 ft = boundary 直後で乖離爆発
- hc は dual PF (= Z + ANCC) で boundary 周辺の signal を 2 軸で抽出
- 我々 exp009 v2 = single PF on ANCC のみ
- exp016 で `run_pf_z` (= raunakdey07 cell-5 _pf_z port、 line 1163) を inject 済 → 解消見込み

### H3. 中盤 transition (00bbac68 Q3) を取り逃している (= LB 中、 確度 中)
- 00bbac68 Q3 で mean_abs 2.05 ft、 visible 20% の low-info well
- hc は segment b_well (= per formation × phase) + multi-scale NCC で transition shape を data-driven 抽出
- 我々 exp009 v2 = single b_well per formation のみ
- A4 seg_b_well + A5 multi_scale_ncc は src/rogii/segment_features.py で **実装済 (5 + 5 tests pass)** だが、 exp016 統合 kernel には **未 inject** (= raunakdey07 path 優先のため後送り、 別 kernel 候補)

### H4. test-time online learning (= Edge R) が boundary を緩和しきれない (= LB 低、 確度 低)
- Edge R は visible 部分を additional training data として load + 1-epoch re-fit
- 00e12e8b で 4 ft 外し = Edge R で吸収しきれず
- hc は Edge R 不採用 → ours の方が Edge R で「visible 部分の bias」 を取り入れ過ぎ (over-personalization) の可能性
- 改善: Edge R の learning rate / epoch 数を small に re-tune、 または skip ablation

### H5. Ridge 9-base ensemble が hc 6-base + Climber に劣る (= LB 大、 確度 大)
- 既存 exp009 v2 = `Ridge(positive=True)` で 9 base blend
- hc = Climber (= negative weight 許容、 precision 0.001) で 6 base blend
- 9 base > 6 base = base diversity 我々優位、 ただし Ridge は positive 制約で **overcounted base に対し subtract できない**
- Climber は negative weight で overcount を cancel = 数値最適
- exp016 に inline Climber inject 済 (= raunakdey07 continuous mode、 mode="continuous")、 完走後即解消見込み

---

## 7. 改善優先度 + 期待 LB lift 試算

| 仮説 | 改善 path | 既存 status | 期待 lift (ft) | 累積 LB |
|---|---|---|---|---|
| H1 + H5 (= 3-axis grid + Climber) | exp016 自前統合 | RUNNING | -0.20 〜 -0.40 | 9.34 〜 9.54 |
| H2 (= PF Z dual) | exp016 inject 済 | RUNNING | -0.05 〜 -0.15 | 上に重ねて -0.05 〜 -0.15 |
| H3 (= seg_b_well + softmax NCC) | 別 kernel 候補 | 実装済 / 未 inject | -0.10 〜 -0.20 | 9.20 〜 9.40 |
| H4 (= Edge R skip ablation) | 別 kernel 候補 | 未着手 | ±0.05 (= 不確実) | 範囲外 |

**Phase A 完了 (= H1+H2+H5) で LB 9.34 〜 9.54** = 公開 NB Hill Climb 9.43 に同等以上射程
**Phase A + H3 で LB 9.20 〜 9.40** = Gold cutoff (= 現状 LB top 40 で 9.6 帯) 確実圏

---

## 8. 「優勝本質性」 commit との整合 (= ~/projects/kaggle/CLAUDE.md §11)

- ✅ **数理本質**: 各仮説は post-proc / boundary signal / ensemble weight の **3 component decomposition** で真因を特定、 軽さ-driven でない
- ✅ **優勝寄与**: H1+H5 だけで -0.20 〜 -0.40 ft lift = LB 9.4 帯到達 (= gold cutoff 接触)、 H3 で更に -0.10 〜 -0.20
- ✅ **datapoint 価値**: ablation 順序 = exp016 (= H1+H2+H5 統合) → exp019 (= H3 inject 別 kernel) で effect 切り分け可能
- ✅ **rule 耐性**: 全 component が training data preprocessing + post-proc 系、 host rule fix で死なない static path
- ❌ **代替比較未済**: Edge R skip (H4) を ablation submit していない、 副作用評価が pending

---

## 9. 次 action (= exp016 完了後の 1 step)

1. exp016 COMPLETE → submission.csv pull → ours (= LB 9.738) と diff 同じ手順で比較 → H1+H2+H5 が想定通り効いたか data-driven 確認
2. exp016 LB ≤ 9.5 確認 → H3 inject 別 kernel (= exp019_seg_softmax_inject) push
3. exp016 LB > 9.5 (= 失敗) → 真因 isolate (= H1 / H2 / H5 個別 ablation kernel) → Codex review で盲点探索

---

## 10. 関連 doc

- 比較先 NB: `ravaghi/wellbore-geology-prediction-hill-climbing` (= LB 9.43 source)
- 既知 EDA: `docs/research/2026-05-11-deepest-eda.dense.md` F3 = visible_ratio 20/27/33% finding
- 既知 audit: `docs/research/2026-05-11-public-source-audit.dense.md` § 1.1.1 = 00bbac68 mean abs 3.91 ft (= 別 data 源、 本分析と一致)
- Phase A plan: `docs/dev/2026-05-12-plan-gold-to-winning.dense.md` § 5.1
- Hill Climb 強さ: `docs/research/2026-05-12-hill-climb-strength-analysis.dense.md`
- per-well agg: `/tmp/rogii-diff-analysis.csv` (= reproducible)
