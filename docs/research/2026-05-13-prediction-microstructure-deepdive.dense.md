# exp009 v2 vs Hill Climb fork の prediction micro-structure 深掘り

> 起点: 2026-05-13 user 指示「GM レベルの深掘り、 待ち時間で徹底的に EDA + エラー分析、 全部 doc に残せ」
> Source: `/tmp/rogii-exp009-output/submission.csv` (= ours LB 9.738) + `/tmp/rogii-exp017-v3-output/submission.csv` (= hc submit pending LB 期待 9.43)
> 親 doc: `2026-05-13-test-set-divergence-analysis.dense.md` (= 表面層)、 本 doc は **micro-structure 層**

---

## 1. 1st derivative (= dTVT/drow) 分析

```
                 ours d1                     hc d1                    diff (ours-hc)
                 mean      std     min/max   mean      std     min/max  mean      std   p95
000d7d20  -0.00235  0.0325  -0.26/+0.16  -0.00269  0.0327  -0.22/+0.16  +0.00033  0.025  0.050
00bbac68  -0.00089  0.0443  -0.34/+0.24  -0.00134  0.0447  -0.28/+0.24  +0.00045  0.043  0.093
00e12e8b  -0.00186  0.0409  -0.18/+0.23  -0.00201  0.0361  -0.14/+0.16  +0.00015  0.024  0.048
```

- **slope mean 全 well で ours ≈ hc** (= 平均 trend は同等)
- **slope std 00e12e8b で ours 0.041 > hc 0.036** = ours が **slope 振れが 13% 大**
- **slope diff の p95**: 00bbac68 で 0.093 ft/row = 100 rows で 9.3 ft 累積差 = explains test 実測 max 4.6 ft

## 2. 2nd derivative (= curvature) 分析

```
            ours std    hc std     ratio
000d7d20    0.00786     0.00814    0.966   ← 同等
00bbac68    0.01221     0.01220    1.001   ← 同等
00e12e8b    0.01079     0.00920    1.174   ← ours +17% over-curve
```

**00e12e8b で ours は curvature が 17% 過剰** = trend 内で over-shoot oscillation
- 物理的に TVT は smooth、 短期 curvature が大きいのは noise 混入 signal
- hc の savgol smoothing + 3-axis postproc が curvature を抑制

## 3. slope sign disagreement (= 勾配方向不一致 row 比率)

```
000d7d20    965/3836  = 25.2%
00bbac68   1869/6014  = 31.1%   ← 最大不一致
00e12e8b    871/4301  = 20.3%
```

- 30% の row で「ours は up、 hc は down」 (or 逆) = **trend を fundamental に異なる方向に外している**
- これは noise 不一致でなく、 **trend interpretation 自体が異なる**
- 直すには: ours の post-proc smoothing 強化 + 1st derivative の sign 平滑

## 4. 10 分位 (= bin) per-well 詳細パターン

### 000d7d20 (= visible_ratio 27%、 mean_abs 0.68 ft)
```
bin   n    mean_abs  signed_diff
B0  384    0.27     -0.25     ← 早期 hidden、 浅い側に bias
B1  384    0.34     -0.22
B2  383    0.38     +0.36     ← B2/B3 で **方向反転** (= 浅→深)
B3  384    0.44     +0.43
B4-B5     stable
B6  383    1.11     -1.05     ← B6 で再反転 (= 深→浅)
B7  384    0.88     +0.57     ← B7 で再反転
B8  383    1.20     +1.20     ← 末期 systematic 深い
B9  384    1.63     +1.63     ← 末期 max 2.13 ft
```

**4 回の方向反転** = trend が ours と hc で異なる anchor 構造

### 00bbac68 (= visible_ratio 20%、 mean_abs 1.19 ft、 hot 37%)
```
bin   n    mean_abs  signed_diff
B0  602    0.25     -0.24     ← 早期 浅い側
B1  601    0.21     -0.09
B2  601    0.32     -0.20
B3  602    0.82     +0.81     ← B3 で反転 (= 浅→深)
B4  601    2.09     +2.09     ← **+2.09 ft 一様 bias 開始**
B5  601    2.01     +2.01
B6  602    1.42     +1.35
B7  601    1.65     **-1.36**  ← **B7 で大反転 (深→浅) 1280→1280 ft 跳ぶ**
B8  601    1.96     -1.76
B9  602    1.22     -0.01     ← 末期は再収束
```

**bin B4-B6 で +2 ft 深い bias / B7-B8 で -1.7 ft 浅い bias** = **3 phase trend shift を完全に取り逃がしている**

### 00e12e8b (= visible_ratio 33%、 mean_abs 1.40 ft、 hot 35%)
```
bin   n    mean_abs  signed_diff
B0  431    1.60     +1.59     ← 開始直後で **+1.6 ft 一様 bias**
B1  430    3.55     +3.55     ← **+3.55 ft = 巨大 systematic miss**
B2  430    3.56     +3.56
B3  430    2.19     +0.90
B4  430    0.38     -0.37     ← B4 から急収束
B5-B7     stable
B8  430    0.64     +0.64
B9  430    1.15     +1.15     ← 末期で再上昇
```

**B0-B2 で +3.5 ft 一様 bias** = boundary 直後 30% の hidden を完全に外している = anchor base value error

## 5. hot zone clustering (= abs_diff > 1.5 ft 連続 rows)

```
well        hot rows   ratio   hot runs   largest run
000d7d20      350/3836  9.1%      12       rows 5037-5275  (n=239,  +1.85 ft、 末期)
00bbac68     2231/6014 37.1%      15       rows 3946-4596  (n=651,  +2.08 ft、 B4)
                                          rows 4823-5462  (n=640,  +2.18 ft、 B5)
                                          rows 6155-6618  (n=464,  **-3.41 ft**、 B7)
00e12e8b     1524/4301 35.4%       6      rows 2301-3580  (n=**1280**, **+3.30 ft**) ★
```

★ **00e12e8b で 1280 連続 rows = 全 4301 rows の 30% を平均 +3.30 ft 一様に外している**
- これは noise でなく **systematic anchor offset**
- 1 個の error が 1280 rows に impact = inference architecture-level の bug 級

---

## 6. 新仮説 (= 上記 micro-structure から)

### H6. visible↔hidden boundary の **anchor base value** が違う (= 確度 大、 最重要)
- 00e12e8b で B0-B2 一様 +3.5 ft bias = anchor が **3.5 ft 深い側にロック**
- ours = `last_known_tvt + target` 系で hidden の base を構築、 visible 末端 TVT を anchor
- hc = dual PF Z + ANCC で **boundary 周辺の 2 軸 signal** から anchor を recompute
- 改善: **dual PF Z (= `run_pf_z`、 raunakdey07 line 1163)** を inject → exp016 に **既存統合済**
- 確認 path: exp016 完走後 vs hc で 00e12e8b B0-B2 の bias が解消するか測定

### H7. trend shift 検出力 (= 00bbac68 で 3 phase 反転を取り逃し、 確度 大)
- 00bbac68 で bin B3 / B7 / B9 で方向反転 = **wells 内で複数 transition** 発生
- ours = single LGB pred + Ridge blend、 trend shift を smoothing で吸収
- hc = Climber が per-blend で transition shape を最適化、 segment-level の精度
- 改善: **segment b_well (= per formation × phase 4 phase)** で transition shape を data-driven 抽出
- 状態: A4 seg_b_well 実装済 (`src/rogii/segment_features.py:seg_b_well`、 5 tests pass)、 ただし **exp016 未 inject**
- next step: exp019 (= H3+seg_b_well inject) を Phase A 完了後 push

### H8. slope sign disagreement 25-31% = trend interpretation error (= 確度 中)
- ours と hc で 30% の rows で「up vs down」 反転
- これは smoothing 違いでなく、 underlying trend model の **paradigm 差** (= GBM trend vs Climber-blend trend)
- 改善: Climber inline (= H5、 既存統合済) で paradigm 一致化、 + savgol post-smooth

### H9. 00e12e8b で curvature +17% over-shoot (= 確度 中、 短期 noise 混入)
- ours の 2nd derivative std が hc より 17% 大
- 短期 noise を smoothing しきれていない
- 改善: Optuna postproc の **tau (= time constant)** 拡大、 savgol window 拡大
- 状態: H1 の 3-axis 2530-cell grid (= alpha × tau × w_pf) で自動探索、 exp016 既存統合済

---

## 7. micro-structure 分析の **直接的次 action** (= 提案)

優先度順:

1. **exp016 完了後即実行**: 同じ derivative + bin + hot-zone 分析を exp016 submission に対して実施 → 各仮説の検証
2. **exp019 (= seg_b_well + softmax NCC inject) 検討**: 00bbac68 の trend shift 3 phase を捕捉するために A4 + A5 を別 kernel inject
3. **00e12e8b の hot run 1280 rows (= 2301-3580) を **physical model** で再解析**: TVT 物理的に許容される range + GR signal pattern と照合 (= 別 doc 候補)
4. **本 doc の bin pattern 表 を Phase B B3 (= CV-LB correlation 強化) で fold separation の input に使う**

---

## 8. 残り疑問 (= 次 doc 候補)

- Q1. 00e12e8b の +3.5 ft 一様 bias = 真の anchor 値は ours か hc か? (= test ground truth 不可、 train wells で類似 pattern 検出可能)
- Q2. 00bbac68 で hidden 中盤 ~50% 地点で trend shift = この row position で wells 共通の formation transition が起きるか? (= train data の formation marker 分析)
- Q3. hot run 1280 rows = ours の prediction が **完全に flat** な可能性 (= 続報 doc で stationarity test)
- Q4. ours と hc の **frequency domain** 比較 (= FFT で どの周波数成分で diverge するか)

これらは別 doc で順次深掘り。

---

## 9. 関連 doc

- 親 (= surface 層): `2026-05-13-test-set-divergence-analysis.dense.md`
- 比較先: `ravaghi/wellbore-geology-prediction-hill-climbing`
- exp016 設計: `docs/dev/2026-05-12-plan-gold-to-winning.dense.md` § 5.1 + Hill Climb 強さ分析 doc
- 続報 候補 (= Q1-Q4 別 doc): `2026-05-13-anchor-base-value-physical-analysis.md`、 `2026-05-13-frequency-domain-comparison.md`
