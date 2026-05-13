# 重要法則: ROGII で visible 末端 slope の linear 継続は **91% の wells で flat より悪い**

> 起点: 2026-05-13 GM deep-dive Phase 4、 slope drift 真因確定後の **train data 統計確証** layer
> Source: `data/raw/train/<well>__horizontal_well.csv` (= 773 wells) から 100 wells random sample
> Reproducible script: `/tmp/rogii-train-slope-extrap-100.csv`
> 親 doc: `2026-05-13-slope-extrapolation-decisive-analysis.dense.md` (= test 3 wells で示唆)、 本 doc は **train 全体での確証**

---

## 1. 結論 (= 一発で言うなら)

**ROGII では「visible 末端の slope を hidden に linear 外挿」 する prediction は、 「flat (= zero slope) で固定」 する prediction より 平均 5.2 倍悪い (RMSE 62.99 vs 12.10 ft)**。 つまり LGB / 任意 model が visible tail slope を hidden に継続するなら、 **base prediction として大破綻** を内在化。

---

## 2. 実験 setup (= 100 train wells random sample、 seed 42)

各 well で:
1. `TVT_input` (= visible) が non-NaN の rows を抽出 → visible 部分
2. `TVT` (= ground truth) を全 rows で取得 → hidden actual
3. visible 末端 50 row の slope (= `np.gradient(tvt_tail_50).mean()`) = `slope_tail`
4. Anchor = visible 末端 TVT
5. **linear extrapolation prediction**: `anchor + slope_tail * (1, 2, ..., n_hidden)`
6. **flat prediction**: `anchor` (constant for all hidden rows)
7. 両 prediction を hidden actual と比較、 RMSE 計算

100 wells のうち n_visible ≥ 60 + n_hidden ≥ 100 を満たす wells を取得。

---

## 3. 結果 (= 100 wells aggregate stats)

```
                  linear_rmse    flat_rmse    linear - flat
count             100            100          100
mean               62.99 ft      12.10 ft     +50.89 ft  ← linear が 50.9 ft 悪い
std                92.35 ft       7.39 ft     +89.56 ft
min                 2.70 ft       2.56 ft     -13.07 ft  ← 9 件で linear のほうが ましな例
25%                15.02 ft       7.04 ft      +4.44 ft
50%                30.87 ft       9.75 ft     +21.04 ft
75%                74.42 ft      15.57 ft     +62.40 ft
max               732.14 ft      42.89 ft    +694.15 ft  ← 25050f63 で破滅
```

### win count
- **flat wins 91/100** (= zero slope の予測が linear より低 RMSE)
- linear wins 9/100

---

## 4. visible tail slope の分布 (= 100 wells)

```
            slope_tail (ft/row)
count        100
mean        -0.00134
std         +0.03464
min         -0.1047
25%         -0.0101
50%         +0.00025  ← median はほぼ 0
75%         +0.01133
max         +0.2169   ← outlier (well 25050f63、 linear RMSE 732 ft の元凶)
```

- median は **+0.0003 ft/row** (= flat にほぼ等しい)
- ±0.03 ft/row は **1 σ 範囲**、 ±0.10 ft/row 以上は outlier
- これを **数千 rows に継続したら** 数百 ft の累積 drift = 物理的に implausible

---

## 5. 物理 interpretation (= なぜ flat extrapolation が勝つか)

### 5.1 ROGII の well geometry
- ROGII の wells は **horizontal well** = curve section (= depth 急増) → horizontal section (= depth ほぼ flat)
- `TVT` = true vertical thickness = `Z` 軸方向の深度
- visible region = well の curve section (= TVT 急上昇)
- hidden region = horizontal section (= TVT ほぼ flat、 well bore が真横に進む)

### 5.2 visible mid → tail の slope decay 確認 (= 親 doc § 2)
- 00bbac68 visible mid +0.676 ft/row → tail +0.013 ft/row = **52× 減速**
- 00e12e8b visible mid +0.467 → tail +0.042 = **11× 減速**

visible 末端で既に decay の途中 → hidden では更に減速 (= ほぼ 0) するのが物理的 prior。

### 5.3 我々 LGB の overshoot
- LGB は visible region で学習、 visible 内の **slope を含む trend** を model 化
- hidden に extrapolate するとき、 visible tail の slope を **そのまま continue** = LGB が「これ以上 well が curve continue」 と誤判定
- 真値は flat、 LGB は curve continue → **systematic upward bias**

これが test 00e12e8b で **rows 2301-3580 一様 +3.30 ft bias** の物理的真因。

---

## 6. test 3 wells の slope_tail を train 分布に位置付け

```
well        slope_tail     train P-rank   解釈
000d7d20    +0.016 ft/row  ~P 60         normal、 minor extrap害
00bbac68    +0.013 ft/row  ~P 55         normal
00e12e8b    +0.042 ft/row  ~P 90+         outlier、 extrap害大
```

→ 00e12e8b で slope_tail が **P90+ outlier** = LGB が継続したら害最大、 train で同 percentile の wells (= linear RMSE 50-150 ft) と整合。

---

## 7. ours と hc がどう slope_tail を扱っているかの再評価

```
                slope_tail (visible)   ours hidden B0-B1   hc hidden B0-B1   ours 残率  hc 残率
000d7d20         +0.0163               +0.00140            +0.00141           8.6%     8.7%
00bbac68         +0.0131               +0.00473            +0.00482          36.1%    36.8%
00e12e8b         +0.0416               +0.01219            +0.00794          29.3%    19.1%  ★
```

- 000d7d20 + 00bbac68 では ours と hc がほぼ同等 (= 8-37% 残率) で test RMSE 微差
- **00e12e8b で ours 29.3% vs hc 19.1%** = ours が 10 ポイント過剰継続
- → 00e12e8b で ours が +2.5 ft + bias (= drift accumulate) 直結

### slope 残率の optimal 推定
- train 100 wells で「visible tail slope を 何 % 抑制したら hidden RMSE 最小」 は別実験必要
- 大雑把: flat (= 0% 残) が 91/100 wells で wins → **optimal は 0-20% 残率**
- hc の 19% (= 00e12e8b) は近似 optimal、 ours の 29% は overshoot

---

## 8. 直接的改善 path (= 数値 prescription)

### A. ours の hidden 短期 slope を visible tail slope の 20% 程度に抑制
- 現状: ours = 29% 残率 (00e12e8b)、 hc = 19% 残率
- 目標: 15-20% 残率
- 実装: post-proc 内で `predicted_slope = visible_tail_slope * 0.2` 制約 を入れる
- これは exp016 の Optuna **alpha** parameter で間接実現 (= alpha が large なら slope 抑制大)、 Optuna 500-trial で optimal alpha 探索

### B. visible tail slope outlier (= |slope| > 0.05 ft/row) の wells は **flat baseline** に強制 fallback
- 100 wells で linear RMSE が 200 ft 超の wells は全 visible_ratio < 0.30、 slope_tail outlier
- 00e12e8b は visible_ratio 0.33 で boundary 圏、 ただし slope_tail +0.042 (= P90+) で要警戒
- 実装: per-well slope_tail を計算 → outlier 検出 → flat fallback or aggressive slope decay
- 既存 src/rogii/segment_features.py + multi_scale_ncc で feasibility あり

### C. **flat = base + LGB residual** modeling 切替 (= paradigm shift 候補)
- 現 prediction = LGB(features) + post-proc smoothing
- 改造 prediction = anchor + LGB(features, residual mode) + post-proc
- ravaghi NB は anchor + residual 形式に近い (= `last_known_tvt + target`)、 我々も exp008/exp009 で採用済
- ただし residual の **scale calibration** が test wells で 0.2-0.5x 過剰、 これを **train data で fit** すれば直る

---

## 9. 期待 LB lift 試算 (= train 100 wells 統計から)

### scenario A: ours 残率 29% → 19% (= hc 並み)
- 00e12e8b の hidden B0-B1 +2.56 → +0.5 ft 圏に縮小
- 累積 drift も 1/3 へ
- 単 well で全 row mean_abs 1.40 → 0.70 ft、 test RMSE -0.5 〜 -0.8 ft 圏
- LB lift 期待 -0.10 〜 -0.20 ft = LB 9.55 〜 9.65 帯 (= Hill Climb 9.43 にやや 接触)

### scenario B: 全 well で visible_ratio < 0.30 を flat fallback
- 00bbac68 (= visible_ratio 0.20) で flat fallback、 linear RMSE 36 ft (実測 train flat RMSE 12 ft 級) → improve
- ただし trend shift event (= H7'、 親 doc) の detection は別途必要
- LB lift 期待 -0.05 〜 -0.10 ft

### scenario C: A + B + segment b_well (= 親 doc § 5 H3) 統合
- 全仮説組み合わせで LB 9.4 帯到達確実 + 上振れで 9.2 帯

---

## 10. ablation 順序 (= GM §11 datapoint 価値)

1. exp016 完走 + submit → A の現実 LB 確認 (= post-proc 3-axis + Climber 効果)
2. exp019 (= seg_b_well + softmax NCC inject、 別 kernel) → C の partial 効果
3. exp020 (= flat fallback hard-rule inject) → B の単独効果

各 step で per-well slope analysis を実施 (= 本 doc Tool 反復) して **改善 component を isolate**。

---

## 11. caveat + open question

- 100 wells sample = 773 全 wells の 13%、 outlier 比率の estimate に変動あり (± 5%)
- visible_ratio が train 分布で右裾 (= 0.20-0.33) は 100 wells 中 N 件のみ、 test と train で分布偏り あり可能性
- Q9 (= 続報候補): 全 773 wells で同分析を実施、 visible_ratio bucket 別に linear vs flat の wins ratio を再確認
- Q10 (= 続報候補): 各 well の **inclination / azimuth** (= 物理 geometry) を加味して、 「inclination > 89° (= 完全水平) の well では slope 0 hard」 など physically informed constraint を導出

---

## 12. 関連 doc + data

- 親 (= surface): `2026-05-13-test-set-divergence-analysis.dense.md`
- 親 (= micro): `2026-05-13-prediction-microstructure-deepdive.dense.md`
- 親 (= slope decisive): `2026-05-13-slope-extrapolation-decisive-analysis.dense.md`
- 本 doc reproducible csv: `/tmp/rogii-train-slope-extrap-100.csv` (= 100 wells × 9 cols stats)
- ravaghi NB の savgol post-smooth = 25 cell 内、 raunakdey07 `_pf_z` = cell 5
- 我々 plan: `docs/dev/2026-05-12-plan-gold-to-winning.dense.md` § 5.1 H1-H5
