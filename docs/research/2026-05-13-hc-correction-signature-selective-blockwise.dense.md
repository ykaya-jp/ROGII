# hc の ours-correction signature: **per-row selective + block-wise + non-linear**

> 起点: 2026-05-13 GM deep dive Phase 8、 親 doc § 7 Q18 を直接 simulation で answer
> 動機: 改善 path の選択 (= hard cap vs soft constraint vs selective fix) を hc の動作 reverse engineer で確定
> 親 doc: `2026-05-13-hypothesis-calibration-slope-cap-trap.dense.md` (= 仮説 calibration)

---

## 1. 結論 (= 一発)

hc (= ravaghi Hill Climb LB 9.43) の ours (= LGB+ensemble LB 9.738) に対する correction は **constant ではなく per-row selective block-wise**:

- **signal/noise ratio 4.9 〜 9.5** (= 局所 systematic な correction、 noise でない)
- **zero-crossings 0.7-1.3%** (= 連続 row で同 direction = block-wise)
- **P90/P50 ratio 2.8-4.3** (= heavy-tail、 特定 row のみ大幅 correction)
- **linear regression R² 0.02-0.25** (= constant linear trend で全く説明できない non-linear)

つまり hc は **「特定 condition の連続 row に対し block-wise correction、 他 row はほぼ素通し」**。 改善 path 候補:
- ❌ hard cap (= 全 row strict rule、 親 doc § 4 で sim 棄却)
- ❌ constant linear shift (= R² 低、 説明できない)
- ✅ **selective conditional fix** (= per-row で condition 検出 → block-wise apply)

---

## 2. correction signature 詳細 (= test 3 wells)

### 2.1 大域 stats
```
well        global mean diff    global std    rolling-50 mean|diff|   rolling-50 std    signal/noise
000d7d20    +0.285             0.819         0.667                    0.137             4.86
00bbac68    +0.261             1.570         1.167                    0.220             5.31
00e12e8b    +1.056             1.630         1.386                    0.146             9.49 ★
```

- 全 well で signal/noise > 4 = **systematic correction が noise の 5-9 倍強**
- 00e12e8b で signal/noise 9.49 = ours と hc が systematic に異なる pattern を持つ

### 2.2 zero-crossings (= signed diff の direction change)
```
well        zero-crossings   ratio
000d7d20     28              0.7%
00bbac68     81              1.3%
00e12e8b     36              0.8%
```

- 全 well で zero-crossing < 1.3% = **連続 100 row 単位で同 sign の correction**
- これは「連続区間で hc が ours より一様に高い or 低い」
- 親 doc § 4 hot-zone clustering (= 1280 連続 row +3.30 ft on 00e12e8b) と整合

### 2.3 percentile distribution (= |diff| の分布)
```
well        P10      P25     P50     P75     P90     P99    P90/P50
000d7d20    0.105    0.227   0.500   1.119   1.468   2.077  2.93
00bbac68    0.081    0.258   0.894   1.988   2.521   4.079  2.82
00e12e8b    0.107    0.266   0.828   2.851   3.549   3.993  4.29 ★
```

- 00e12e8b の P90/P50 ratio **4.29** = **特定 row のみ heavy tail** = selective correction
- median は 0.5-0.9 ft、 90 percentile は 1.5-3.5 ft = 上位 10% で disproportionately 大

### 2.4 linear trend regression (= constant fix 検定)
```
well        slope            intercept    R²       解釈
000d7d20    +0.000372        -0.966       0.254    弱 trend (= row 進むにつれ diff +)
00bbac68    -0.000138        +0.888       0.023    no linear trend
00e12e8b    -0.000628        +3.715       0.229    弱 trend (= row 進むにつれ diff -)
```

- 全 well で R² < 0.26 = **constant linear shift で 26% も説明できない** = correction は non-linear
- 親 doc § 4-5 の 10 分位 bucket pattern (= bin ごとに sign 反転) が R² 低の元凶

---

## 3. correction の "形" を 4 type で分類

### Type A. constant shift (= 全 row 同じ方向、 同じ magnitude)
- 統計指標: R² > 0.7、 zero-crossings ≈ 0
- 我々の test wells: **当てはまらない** (R² < 0.26)

### Type B. linear trend (= 行進行で degree 変化)
- 統計指標: R² > 0.5、 slope > 0.001
- 我々の test wells: **当てはまらない**

### Type C. block-wise constant (= 連続区間で同方向、 区間境界で反転)
- 統計指標: signal/noise > 3、 zero-crossings 数 % 程度
- 我々の test wells: **完全一致** (000d7d20 signal/noise 4.86、 00bbac68 5.31、 00e12e8b 9.49、 zero-crossings 0.7-1.3%)

### Type D. random noise (= 全 row 独立、 noise dominated)
- 統計指標: signal/noise < 1、 zero-crossings ~50%
- 我々の test wells: **当てはまらない**

→ **hc の correction = Type C = block-wise selective constant**

---

## 4. 改善 path 確定 (= 親 doc Path 1' refinement)

### Path 1' (= conditional selective cap、 親 doc 提案) を block-wise design に refine

```python
def hc_style_correction(pred, mask_hidden, incl, slope_tail, well_id):
    """
    Apply block-wise selective correction in hidden region.
    Triggers: visible_tail slope outlier (= |slope_tail| > 0.030 ft/row, train P85)
              AND inclination ≥ 89° (= 完全水平)
    Block: 1000 row segments、 each block gets a single correction magnitude
    """
    if not (abs(slope_tail) > 0.030 and incl >= 89):
        return pred  # no correction needed
    n = len(pred)
    block_size = 1000
    for block_start in range(0, n, block_size):
        block_end = min(block_start + block_size, n)
        if not mask_hidden[block_start:block_end].any():
            continue
        # estimate block's predicted slope
        block_pred = pred[block_start:block_end]
        block_slope = np.gradient(block_pred).mean()
        # if predicted block slope > 5x train P50 hidden actual slope (= 0.005)、 suppress
        if abs(block_slope) > 0.025:  # 5x threshold
            anchor_b = pred[block_start - 1] if block_start > 0 else pred[0]
            target_slope = 0.005 * np.sign(block_slope)
            for i in range(block_start, block_end):
                pred[i] = anchor_b + target_slope * (i - block_start + 1)
            pred[block_start:block_end] = np.linspace(
                pred[block_start - 1] if block_start > 0 else anchor_b,
                anchor_b + target_slope * (block_end - block_start),
                block_end - block_start
            )
    return pred
```

### Path 2 (= Climber blend、 親 doc 提案) で hc style correction を **別 base** として加える

- 既存 9 base + **1 新 base = "ours with hc_style_correction"**
- Climber が「どこで hc style を採用するか」 を per-blend weight で決定
- 推定 weight 0.10-0.30 で `correction base`、 残り `ours base` 系
- これは exp020 (= 別 kernel) で実装可能

### 期待 LB lift 試算

- Path 1' direct apply (= post-proc 内 inject、 ours の 00e12e8b 1280 rows を一斉 correction)
  - RMSE(ours-hc) 1.56 → 0.50 ft 圏 (= correction 70% 取り込み想定)
  - Pythagorean LB ≈ sqrt(9.43² + 0.5²) = 9.44 ft = **hc に肉薄**
  - ただし correction が **完璧一致** 仮定、 現実は 50% 程度 → LB 9.55 〜 9.65
- Path 2 Climber blend (= 既存 base + correction base)
  - Climber が optimal weight で blend、 partial correction が selective に効く
  - LB 9.55 〜 9.70 想定 (= 9 base + 1 base に対し Climber が weight 0.15 圏で blend)

---

## 5. 検証 plan (= exp016 結果との照合)

exp016 完走後:
1. exp016 submission.csv vs hc submission で同じ correction signature を計算
2. signal/noise が ours (= 4.86-9.49) より下がれば、 exp016 は selective correction を内部実装している証拠
3. signal/noise が ours と同等で残るなら、 Path 1' / 2 が必要

→ exp016 = post-proc 3-axis + Climber + dual PF Z は **internal で block-wise selective correction を生成する** ことが期待される (= hc と同 design)、 直接確認可能

---

## 6. open question (= 次 doc 候補)

- Q21. block size の最適化 (= 我々の test では 1280 rows、 train wells で平均 block 長は?)
- Q22. correction の trigger 条件の精密 ablation (= incl threshold 89° vs 88°、 slope_tail threshold 0.030 vs 0.020)
- Q23. ravaghi NB の cell 5-25 を全 read、 hc の selective correction が **どの cell に実装されているか** 特定
- Q24. thbdh v10 NB の correction signature を比較 (= 別 paradigm が同じ block-wise correction を持つか)
- Q25. 我々 features.py の "edge S" (= round-to-grid) が selective correction の前段か後段か

---

## 7. 関連 doc

- 親 (= 仮説 calibration、 hard cap 棄却): `2026-05-13-hypothesis-calibration-slope-cap-trap.dense.md`
- 親 (= 法則 confirmed): `2026-05-13-full-773-wells-confirmed-physics.dense.md`
- 親 (= geometry): `2026-05-13-physical-geometry-inclination-prior.dense.md`
- 親 (= surface to slope): `2026-05-13-test-set-divergence-analysis.dense.md` 〜 `2026-05-13-slope-extrapolation-decisive-analysis.dense.md`
- 教訓: **「correction の形」 を統計指標 (= signal/noise、 zero-crossings、 P90/P50、 R²) で分類してから path を決める**、 hard cap でなく block-wise selective が正解

---

## 8. 本 doc series 全 8 件 アグリゲート

1. `2026-05-13-test-set-divergence-analysis.dense.md` — surface 層、 test 3 wells で乖離発見
2. `2026-05-13-prediction-microstructure-deepdive.dense.md` — derivative + bin + hot zone
3. `2026-05-13-slope-extrapolation-decisive-analysis.dense.md` — anchor 棄却、 slope drift 確定
4. `2026-05-13-naive-slope-extrapolation-is-harmful.dense.md` — train 100 wells で flat>linear
5. `2026-05-13-physical-geometry-inclination-prior.dense.md` — inclination geometry
6. `2026-05-13-full-773-wells-confirmed-physics.dense.md` — 全 773 wells confirmed、 features.py 内既存
7. `2026-05-13-hypothesis-calibration-slope-cap-trap.dense.md` — hard cap 棄却、 false-positive 回避
8. **本 doc** — hc correction = block-wise selective、 改善 path 確定

合計: ~1600 行の dense analysis、 全 reproducible (data + script in /tmp)。
