# 物理 prior: well の inclination が完全水平 (= ≥ 89°) なら hidden TVT slope は flat、 これを feature 化

> 起点: 2026-05-13 GM deep dive Phase 5、 「slope continuation harmful」 法則の物理的 root cause 解明
> Source: `data/raw/train/<well>__horizontal_well.csv` (= 773 wells) から 100 wells sample、 X/Y/Z/MD geometry 加味
> 親 doc: `2026-05-13-naive-slope-extrapolation-is-harmful.dense.md` (= statistical layer)、 本 doc は **physical geometry layer**

---

## 1. 結論 (= 一発)

ROGII では well の **visible_tail inclination ≥ 89° (= 完全水平 bucket)** に属する場合、 hidden actual TVT slope は **mean -0.0009、 abs 0.007 ft/row** で **flat に極めて近い**。 全 test wells (= 000d7d20、 00bbac68、 00e12e8b) はこの bucket。 → **inclination を per-row feature 化** すれば LGB が自動で「水平 well なら slope 抑制」 を学習可能。

これは exp016 (= post-proc + Climber + dual PF Z) の slope drift 対策と **直交独立な improvement axis**。

---

## 2. inclination 統計 (= train 100 wells)

```
                  visible_tail incl   hidden incl   incl 変化
mean              87.83°             88.04°        +0.21°
std                1.35°              0.65°
min               80.63°             86.31°
P25               87.00°
P50               87.96°
P75               88.77°
max               89.93°             89.51°
```

- 全 wells が **80° 以上 = near-horizontal**、 90% が 87-89° 帯
- visible → hidden で平均 0.21° 上昇 (= 更に水平化)、 一部 wells で逆方向あり
- 80° 未満の well は train 内に存在しない (= 100 sample 観察)

## 3. inclination bucket 別 slope (= train 100 wells)

```
bucket          n   mean slope_tail   mean hid_actual_slope   abs slope_tail   abs hid_actual_slope
75-85°          2   +0.116            +0.0145                  0.116           0.018         ← outlier
85-89°         80   -0.0049           +0.0018                  0.018           0.010
89-90° (= 完全水平) 18   +0.0014           -0.0009                  0.022           0.007         ★ 最も flat
```

- **89-90° bucket の hidden actual slope = mean -0.0009、 abs 0.007 ft/row** = **flat に極めて近い**
- 85-89° bucket は abs 0.010、 やや slope あり
- 75-85° bucket は outlier (2 wells のみ)、 slope 大

## 4. slope_tail と hid actual slope の相関 (= predictability)

```
pair                              correlation
(incl_tail, slope_tail)            -0.20   ← incl 大 → slope_tail 小 (期待通り)
(incl_tail, hid_actual_slope)      -0.24   ← incl 大 → hidden slope も小
(incl_hid, hid_actual_slope)       +0.10   ← 弱
(slope_tail, hid_actual_slope)     +0.24   ← slope_tail だけでは 6% R² (= 94% は他 signal)
```

→ **slope_tail を hidden に linear 継続するモデルは原理的に explanatory 6%、 残 94% を別 feature で補填する必要**
→ **incl_tail は slope_tail と同等の predictor (-0.24 vs +0.24)**、 つまり **incl は補完 signal**

## 5. test 3 wells の inclination と slope (= 直接測定)

```
well        visible_tail incl   slope_tail   hidden incl (前 1000 row)
000d7d20     89.82°              +0.0163     88.61°
00bbac68     89.01°              +0.0131     88.57°
00e12e8b     89.82°              +0.0416     87.91°  ★ 水平 well 中で outlier slope
```

### 解釈:
- **3 well すべて visible_tail incl ≥ 89°** = train 完全水平 bucket
- 同 bucket train 統計 (= mean hid_actual_slope -0.0009 ft/row) から、 **3 well すべて hidden 真 slope ≈ 0** が物理的予測
- **00e12e8b の slope_tail +0.042 は 水平 bucket の slope_tail 平均 +0.001 から外れた outlier** (= 水平 well なのに上向き curve continue)
- ours が +0.042 を 30% 継続して +0.012 ft/row を hidden に与えた = train 統計 -0.0009 から **大乖離**
- hc は 19% 継続して +0.008、 まだ +0.009 ft/row over だが ours より 30% 改善

## 6. 新仮説 H_geo (= 直接的 feature engineering 提案)

### H_geo.1: per-row inclination を LGB feature 化
- 全 row で X、 Y、 Z は test 含めて public (= 7 列の 1 つ)
- per-row local inclination = `arctan2( sqrt(dX² + dY²), abs(dZ) )` (= window 5-50 rows)
- LGB が「incl 大 → slope 抑制」 を自動学習
- 実装: src/rogii/features.py に `compute_inclination_features(window=20)` 新規追加
- 期待 lift: -0.05 〜 -0.15 ft (= train 100 wells 統計から)

### H_geo.2: visible_tail inclination による per-well global rule
- visible_tail incl ≥ 89° の wells で post-proc 内に **slope cap = 0.005 ft/row** を hard constraint
- 実装: post-proc inside `if visible_tail_incl > 89: slope = min(slope, 0.005)`
- 期待 lift: 主に 00e12e8b で 1280 rows の +3.3 ft bias を 1.0 ft 帯に縮小 = -0.10 〜 -0.20 ft (test 全体)

### H_geo.3: hidden 区間内の **inclination 推移** に応じた dynamic slope decay
- 全 row inclination が時間と共に変化 → ある row で incl が +0.5° 上昇 = 更に水平化、 そこで slope 抑制
- 実装: per-row feature + Climber blend
- 期待 lift: 既存 feature と相補、 -0.05 〜 -0.10 ft

### H_geo.4: inclination 異常 well の自動検出 + segment fallback
- visible_tail incl < 87° (= まだ curve section の可能性) の wells で slope tail を **継続** する判断
- train 100 wells で incl 75-85° bucket (= 2 wells) の hid_actual_slope abs 0.018 = 全 mean より大
- 実装: per-well slope adaptation by inclination
- 期待 lift: train wells で生じる稀ケースを fix、 test wells に直接効果なし (= 全 test が ≥ 89°)

## 7. exp016 / Phase A との関係

- exp016 自前統合 = post-proc 3-axis grid (= alpha × tau × w_pf) + dual PF Z + inline Climber
- これらは **slope 抑制を間接的に達成** (= alpha 大で smoothing 強化、 PF Z で boundary 物理 prior)
- **直接的 inclination feature は未統合**
- → H_geo.1 + H_geo.2 を **exp020 (= 別 kernel) で inject** が strict 独立 axis

実装順序:
1. exp016 完走 + LB 確認 → どこまで slope drift が解消したか測定
2. exp019 (= seg_b_well + softmax NCC) → trend shift event 対策
3. exp020 (= H_geo.1 + H_geo.2 inject) → inclination prior 統合

各 step で LB lift を isolate (= GM §11 datapoint 価値)

## 8. caveat + open question

- 100 wells sample bias: 89-90° bucket が 18/100 で statistically thin、 全 773 wells で再確認推奨 (= Q11)
- inclination は test data の X/Y/Z から計算 = test public info、 leak なし
- 注意: hidden region X/Y/Z は **既知** (= ROGII は trajectory 公開、 TVT のみ hidden)、 つまり H_geo.1 は test-time inference で fully accessible
- Q12: train wells で「inclination shift がある区間 (= visible incl 88° → hidden incl 89°)」 wells で TVT slope shift も同時発生か検証 → physical causality 確認

## 9. 「優勝本質性」 §11 整合

- ✅ **数理本質**: 3D geometry → 1D inclination → TVT slope の 3 段 derivation、 軽さ-driven でない
- ✅ **優勝寄与**: -0.05 〜 -0.20 ft 単独、 exp016 と組み合わせで -0.20 〜 -0.50 ft 累積 = LB 9.2-9.5 帯
- ✅ **代替比較**: ravaghi NB / thbdh v10 NB は inclination feature を **明示利用していない** (= 我々独自軸)
- ✅ **rule 耐性**: pure feature engineering、 host rule fix で死なない
- ✅ **datapoint 価値**: exp020 単独で H_geo 効果測定可能

## 10. 関連 doc

- 親 (= statistical): `2026-05-13-naive-slope-extrapolation-is-harmful.dense.md`
- 親 (= surface to slope): 同シリーズ 3 件
- ROGII data structure: `data/raw/train/<well>__horizontal_well.csv` の X、 Y、 Z column (= 3D geometry)
- 実装候補: `src/rogii/features.py` に `compute_inclination_features` 新規関数追加
