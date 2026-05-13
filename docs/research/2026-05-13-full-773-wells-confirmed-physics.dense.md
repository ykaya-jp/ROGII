# 全 773 train wells で「slope continuation harmful」 法則 + inclination prior 確証

> 起点: 2026-05-13 GM deep dive Phase 6、 100 sample → 全 773 wells で再確認 + 既存 inclination feature の **未活用** 発見
> Source: 全 773 train wells `data/raw/train/*__horizontal_well.csv`、 reproducible `/tmp/rogii-train-full-sweep.csv`
> 親 doc: `2026-05-13-naive-slope-extrapolation-is-harmful.dense.md` (100 sample)、 `2026-05-13-physical-geometry-inclination-prior.dense.md` (geometry layer)

---

## 1. 結論 (= 一発)

1. **flat extrapolation は 773 wells の 86.8% で linear extrapolation を outperform** (= 100 sample の 91% と一致、 統計確証)
2. mean RMSE flat **12.81 ft** vs linear **59.99 ft**、 linear は平均 **47 ft 悪い**
3. **既存 src/rogii/features.py:136 に `inclination` feature が実装済 + LGB input に含めている**にもかかわらず、 test 00e12e8b で +3.3 ft drift = **LGB が inclination の causal effect (= incl 大 → slope 抑制) を学習しきれていない**
4. → **post-proc 段階で inclination prior を hard rule として適用する path** が未開拓

---

## 2. 全 773 wells aggregate stats (= sample size 確証)

### 2.1 flat vs linear win-rate
```
flat wins:    671/773 = 86.8%   ← (100 sample で 91%、 統計一致)
linear wins:  100/773 = 12.9%
tie:            2/773 =  0.3%
mean rmse_flat:    12.81 ft
mean rmse_linear:  59.99 ft
mean diff (lin-flat):  +47.17 ft   ← 平均 47 ft 悪化
```

### 2.2 inclination bucket 別 (= 7 bucket、 完全粒度)
```
incl_bucket   n    abs slope_tail   abs hid_actual_slope   rmse_flat   rmse_linear
<60°          1    0.569            0.062                   65.36 ft    1387.91 ft   ← 1 outlier
60-75°        1    0.321            0.016                   47.04        1205.53      ← 1 outlier
75-85°        8    0.083            0.017                   19.37         259.30      ← curve 段階
85-87°      155    0.028            0.010                   13.37          80.51
87-88°      222    0.015            0.009                   12.15          47.14      ← largest bucket
88-89°      227    0.013            0.010                   12.56          38.66
89-90° ★    159    0.022            0.009                   12.67          62.77      ← test wells はここ
```

- **89-90° bucket** = 159 wells (= 全 wells の 20.6%)、 test 3 wells すべてここ
- 89-90° bucket でも flat RMSE 12.67 < linear RMSE 62.77 = **完全水平でも slope 継続は害**
- 89-90° で abs_slope_tail 0.022 (= 85-89° bucket より大) は **outlier slope tail が混在**、 これらが linear で破綻

### 2.3 visible_ratio bucket 別
```
vr_bucket   n     rmse_flat    rmse_linear   abs_hid_actual_slope
<15%         9    21.13 ft     177.42 ft     0.015 ft/row  ← 壊滅領域
15-20%      76    12.66          98.22       0.010
20-25%     241    13.73          70.52       0.010         ← 00bbac68 がここ (= 20.4%)
25-30%     254    12.78          51.65       0.009         ← 000d7d20 (= 27.3%)
30-40%     169    11.48          39.49       0.009         ← 00e12e8b (= 32.6%)
>40%        24    10.78          21.56       0.008
```

- visible_ratio < 15% で linear RMSE **177 ft** = 構造的に linear extrap 不可能
- 顕著: **visible_ratio が大きいほど flat / linear どちらも RMSE 縮小**、 これは hidden region が短いため
- 試合は flat 圧勝で安定

### 2.4 相関 (= 全 773 wells)
```
corr(slope_tail, hid_slope_actual)  = +0.3109   ← 10% R² 説明力
corr(incl_tail, slope_tail)         = -0.4387   ← 19% R² 説明力 (= incl で slope_tail を予測可)
corr(incl_tail, hid_slope_actual)   = -0.1588   ← 2.5% R² 説明力 (= incl 単独で hidden slope 予測弱い)
```

→ **inclination は slope_tail の良い predictor、 hidden slope の直接 predictor は弱い**
→ **二段 model 案**: incl で slope_tail を normalize、 normalized slope を hidden に継続 (= 別 doc 候補)

---

## 3. 重大発見: 既存 features.py の inclination が **LGB に届いていない** signal

### 3.1 既存実装の確認
```python
# src/rogii/features.py:136
out["inclination"] = np.arctan2(np.nan_to_num(dz), np.maximum(lateral_v, 1e-9)).astype(np.float32)

# src/rogii/features.py:360 (= LGB feature list)
"inclination",
```

→ inclination は **per-row feature として LGB input に含まれている**

### 3.2 にもかかわらず test 00e12e8b で +3.3 ft drift 残存
- 00e12e8b incl_tail 89.82° = 完全水平
- ours は +0.012 ft/row 維持 (= 親 doc § 7)
- train 同 bucket mean abs_hid_slope = 0.009 ft/row、 ours は 33% over
- **LGB は inclination feature を持つが、 hidden 区間での causal effect を model 化できていない**

### 3.3 真因仮説 (= LGB の限界)
- LGB は **non-extrapolating model** (= train data の range 内で interpolate)、 hidden の特殊条件 = ある程度 extrapolate
- visible feature 範囲は curve 段階 (= incl 80-89° 移行帯)、 hidden は flat (= incl 89-90° 純粋)
- → LGB が hidden 区間の inclination を hidden TVT に対し直接 fit するには **train data 内に similar (incl, TVT) sample が必要**、 sparse な可能性
- 改善: train data から **incl 89-90° + TVT flat** な subsample を選んで重み付け、 もしくは **post-proc 段階で hard rule 適用**

---

## 4. 改善 path (= 3 軸提案)

### Path 1: **post-proc inclination hard rule** (= 最 simple、 最 effective)
- 実装: per-well で visible_tail incl 計算 → incl ≥ 89° なら **predict slope を 0.005 ft/row 以下に cap**
- 既存 src/rogii/postproc_optuna.py か、 ravaghi の `_apply_pp_3axis` 内に inject
- 1 ファイル 20 行追加 で完了、 exp020 候補

```python
def cap_slope_by_inclination(pred, mask_hidden, incl_tail):
    if incl_tail >= 89:
        slope_cap = 0.005
        for i in range(1, len(pred)):
            if mask_hidden[i]:
                actual_slope = pred[i] - pred[i-1]
                if abs(actual_slope) > slope_cap:
                    pred[i] = pred[i-1] + np.sign(actual_slope) * slope_cap
    return pred
```

期待 lift: 00e12e8b で +3.3 ft drift → +1.0 ft 圏、 test 全体 RMSE -0.10 〜 -0.20 ft

### Path 2: LGB sample weight by inclination-similarity (= 中程度 cost)
- visible row だが (incl, MD) 空間で hidden 区間に近い rows を **higher sample weight** で fit
- ravaghi の **heteroscedastic sample_weight** (= cell 11 内?) を inclination-based に拡張
- 期待 lift: -0.05 〜 -0.10 ft、 path 1 と相補

### Path 3: feature 拡張 (= multi-window inclination)
- 現状: inclination per-row、 window 不明 (= features.py 詳しく確認必要)
- 拡張: incl_5_row、 incl_20_row、 incl_50_row、 incl_visible_tail で multi-scale
- LGB が「短期 vs 長期 inclination」 を別 feature として活用可能
- 期待 lift: -0.03 〜 -0.08 ft、 path 1 + 2 と相補

---

## 5. exp016 完走後の検証 plan

exp016 = post-proc 3-axis + Climber + dual PF Z = **slope 抑制を間接的に達成**、 Path 1 の前段。

完走後、 exp016 submission.csv で:
1. 00e12e8b の hidden B0-B1 mean_signed_err を計測 (= 親 doc では ours で +2.56 ft)
2. もし +1 ft 以下 → Path 1 は marginal、 exp016 で十分
3. もし +1 ft 以上残存 → Path 1 (exp020) 着手必須

これは **datapoint 価値最大化** (= §11 #5) と整合: 1 step 1 仮説、 isolate 可能。

---

## 6. caveat + open question

- Q13 (= 続報): inclination の **window size** (= 5、 20、 50、 100 rows) の最適化 → multi-window feature
- Q14: hidden region 内で incl が変化する場合の dynamic slope cap (= per-row rule)
- Q15: 100 wells で test wells と同 incl_tail 89-90° の wells の **flat baseline RMSE 統計** で test wells flat upper-bound 算出
- Q16: ravaghi NB の cell 内に inclination usage あるか確認 (= grep 必要)

---

## 7. data files

- `/tmp/rogii-train-full-sweep.csv`: 773 wells × {well, vis_ratio, incl_tail, slope_tail, hid_slope_actual, rmse_linear, rmse_flat}
- `/tmp/rogii-train-slope-extrap-100.csv`: 100 sample subset (= 親 doc reproducible)

---

## 8. 関連 doc

- 親 (= 100 sample): `2026-05-13-naive-slope-extrapolation-is-harmful.dense.md`
- 親 (= geometry layer): `2026-05-13-physical-geometry-inclination-prior.dense.md`
- 親 (= slope decisive): `2026-05-13-slope-extrapolation-decisive-analysis.dense.md`
- 親 (= micro): `2026-05-13-prediction-microstructure-deepdive.dense.md`
- 親 (= surface): `2026-05-13-test-set-divergence-analysis.dense.md`
- 既存 inclination 実装: `src/rogii/features.py:136`
- LGB feature list: `src/rogii/features.py:360`
