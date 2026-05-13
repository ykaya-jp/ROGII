# 真因確定: ours の prediction error は **slope extrapolation drift** + **trend shift miss**

> 起点: 2026-05-13 GM-level deep dive、 anchor 仮説 (H6) 棄却後の第 3 段分析
> 親 docs: `2026-05-13-test-set-divergence-analysis.dense.md` (= surface)、 `2026-05-13-prediction-microstructure-deepdive.dense.md` (= micro)
> 本 doc は **slope-level の決定分析**、 改善 path に直結

---

## 1. 棄却された仮説と新仮説

### H6 棄却: anchor base value 違い (= 表面層 doc の最重要仮説)
- anchor (= visible 末端 TVT) と hidden 最初の予測の diff:
  - 000d7d20: anchor 11747.37、 ours -0.0002 / hc -0.0041 → **0.004 ft 差**
  - 00bbac68: anchor 12223.54、 ours +0.0024 / hc +0.0082 → **0.006 ft 差**
  - 00e12e8b: anchor 11604.82、 ours +0.0133 / hc +0.0093 → **0.004 ft 差**
- → 全 well で anchor 直後は **0.006 ft 以下の差** で完全一致

### H6' (slope extrapolation drift) 採用
- anchor 一致なのに 1000 row 後に diff +3.8 ft 出現 = anchor でなく **slope** が原因
- 小さい slope error (= 0.004 ft/row) が **累積** して大きい cumulative drift

### H7' (trend shift miss、 = 00bbac68 特有 pattern) 採用
- 00bbac68 で slope は全 segment 微差 (< 0.006 ft/row) なのに B4-B5 で **avg_signed_err +2.05 ft**
- = slope drift 累積でなく、 ある row で **level jump** (= trend shift event) を取り逃がしている
- ours は smooth、 hc は jump 検出

---

## 2. visible region の slope decay pattern (= physical interpretation)

```
well       visible mid slope    visible tail slope    decay ratio (tail/mid)
000d7d20    +0.163 ft/row        +0.016 ft/row         9.7× 減速
00bbac68    +0.676 ft/row        +0.013 ft/row         52×  減速  ← 極端
00e12e8b    +0.467 ft/row        +0.042 ft/row         11×  減速
```

物理的: well が掘り進むにつれて TVT (= vertical thickness) 変化が緩やかになる傾向 (= horizontal well で MD 増 ≠ TVT 増)。 visible tail で既に **flat 領域** に入っている。

→ hidden region 全体は **flat extension が支配的**、 ours と hc どちらが「visible tail の slope を hidden に継続するか」 が key

---

## 3. hidden 5 segment ごとの slope vs avg signed error (= 完全マップ)

### 000d7d20 (= mean_abs 0.68 ft、 最も良好な well)
```
segment              ours slope    hc slope     slope_diff   avg_signed_err
B0-B1 (0%-20%)       +0.00140     +0.00141      ε             -0.236
B2-B3 (20%-40%)      -0.00518     -0.00494      -0.00024      +0.399
B4-B5 (40%-60%)      -0.00006     -0.00114      +0.00109      +0.086
B6-B7 (60%-80%)      -0.01004     -0.01078      +0.00074      -0.240
B8-B9 (80%-100%)     +0.00213     +0.00202      +0.00010      +1.414  ← 末期で 急増
```

最後 B8-B9 で +1.41 ft = slope diff が小さい (= +0.0001) のに **累積で +1.4 ft** = **B6-B7 までの累積 reset 不成立** ⇒ 末期 anchor recalibration が必要

### 00bbac68 (= mean_abs 1.19 ft、 hot 37%) **trend shift miss pattern**
```
segment              ours slope    hc slope     slope_diff   avg_signed_err
B0-B1 (0%-20%)       +0.00473     +0.00482      -0.00008     -0.160
B2-B3 (20%-40%)      +0.00578     +0.00439      +0.00139     +0.308
B4-B5 (40%-60%)      -0.00212     -0.00305      +0.00093     +2.045  ← **slope 微差、 err 大**
B6-B7 (60%-80%)      -0.03082     -0.02541      -0.00540     -0.005   ← err 一旦 reset
B8-B9 (80%-100%)     +0.01802     +0.01259      +0.00543     -0.882
```

**B4-B5 で slope diff たった +0.0009 ft/row なのに err +2.05 ft** = slope ではなく **level jump** が原因 (= ある row 周辺で hc が下方修正、 ours が直進)

### 00e12e8b (= mean_abs 1.40 ft、 hot 35%) **slope drift pattern**
```
segment              ours slope    hc slope     slope_diff   avg_signed_err
B0-B1 (0%-20%)       +0.01219     +0.00794      +0.00425     +2.564  ← slope diff も err も大
B2-B3 (20%-40%)      -0.01850     -0.01270      -0.00580     +2.240
B4-B5 (40%-60%)      +0.00061     -0.00067      +0.00128     -0.465  ← B4 で reverse
B6-B7 (60%-80%)      -0.00244     -0.00302      +0.00058     +0.046
B8-B9 (80%-100%)     -0.00114     -0.00157      +0.00043     +0.895
```

**B0-B1 で slope diff +0.0042 ft/row** = 1000 row 累積 +4.2 ft、 これが instant max 4.26 ft (= row 3153) と整合

---

## 4. cumulative drift profile (= ours - hc 累積)

### 00e12e8b (= 最 dramatic case)
```
row 0     instant +0.004 ft   (= anchor 一致)
row 50    mean   -0.059 ft   instant -0.12 ft   (= 50 row でまだ 小)
row 100   mean   +0.103 ft   instant +0.89 ft   (= 100 row で大幅 jump)
row 200   mean   +0.560 ft   instant +1.02 ft
row 500   mean   +1.864 ft   instant +3.78 ft   ← drift acceleration
row 1000  mean   +2.688 ft   instant +3.84 ft   ← peak instant 4 ft 級
row 1500  mean   +2.907 ft   instant +1.30 ft   ← 急減 = trend shift event
row 2000  mean   +2.012 ft   instant -0.31 ft
row 3000  mean   +1.244 ft   instant -0.86 ft
row 4300  mean   +1.056 ft   instant +0.64 ft
```

**4 phase profile**:
- Phase 1 (row 0-50): aligned (diff < 0.1 ft)
- Phase 2 (row 50-1500): slope drift accelerate to +3.8 ft instant
- Phase 3 (row 1500-3000): trend shift event、 oscillation
- Phase 4 (row 3000+): partial convergence、 mean ~+1 ft

---

## 5. 改善 path (= 直接的 + 期待 lift 試算)

### A. 短期 slope (= row 0-200) のデータ駆動補正 ★最重要
- **問題**: 00e12e8b B0-B1 で ours が hc より +0.004 ft/row 過剰
- **原因仮説**: ours は visible tail の last 50 row slope を **そのまま線形外挿**、 hc は dual PF Z (= raunakdey07 `run_pf_z` line 1163) で **物理 prior** 加味して slope を抑制
- **path**: exp016 自前統合に dual PF Z **既存統合済** → 完走後 即検証
- **期待 lift**: row 0-200 で diff peak +3.8 → +1 ft 圏 = test 全体 RMSE -0.10 〜 -0.20 ft

### B. 中期 trend extrapolation (= row 200-1500) の **tau** 拡大
- **問題**: slope error が drift 累積 → mean +2.9 ft (00e12e8b)
- **原因仮説**: ours post-proc tau (= time constant) が短く、 短期 slope を遠方まで継続させる
- **path**: Optuna 500-trial で alpha × tau × w_pf 3-axis (= exp016 既存統合済) で **tau optimal** 探索
- **期待 lift**: drift profile を anchor-bound → mean +0.3 ft 圏 = test RMSE -0.05 〜 -0.15 ft

### C. trend shift event (= 00bbac68 B4-B5) の段階検出
- **問題**: ours は smooth 継続、 hc は B4-B5 で +2 ft jump detect
- **原因仮説**: hc は Climber blend + segment b_well + multi-scale NCC で「ある row 周辺で formation transition」 を検出
- **path**: A4 seg_b_well + A5 multi_scale_ncc (= src/rogii/segment_features.py、 10 tests pass) を **別 kernel (exp019)** で inject
- **期待 lift**: 00bbac68 mean_abs 1.19 → 0.50 ft 圏 = test RMSE -0.10 〜 -0.15 ft
- **CAVEAT**: A4 + A5 は exp016 統合に未含、 Phase A 完了後の Phase A.5 として実装

### D. global savgol post-smoothing window 拡大
- **問題**: 00e12e8b で curvature 17% over-shoot (= 別 doc 確認済)
- **path**: post-proc 内の savgol window を 11 → 31 級に拡大 (= raunakdey07 cell-N)
- **期待 lift**: -0.05 〜 -0.10 ft

---

## 6. exp016 が「効くか」 の事前判定 framework

exp016 完走 + submit 後、 同じ slope analysis を **exp016 vs hc** で実行:
- **success 条件**:
  - 00e12e8b B0-B1 slope diff が +0.0042 → < +0.0010 ft/row に縮小
  - cumulative drift profile で row 500-1500 mean が +2 ft → < +0.5 ft に縮小
  - 全 well の slope sign disagreement (= 親 doc § 3) が 25-31% → < 15% に縮小
- **fail 条件** (= postmortem 対象):
  - 00bbac68 B4-B5 の +2.05 ft trend shift miss が残存 (= H7' = A4/A5 inject 必須)
  - 00e12e8b B0-B1 slope drift が解消されない (= dual PF Z が効いていない)

---

## 7. 「優勝本質性」 §11 と整合確認

- ✅ **数理本質**: anchor / slope / trend-shift の 3 component decomposition で誤差を **数量的に decomposeした**、 軽さ-driven でない
- ✅ **優勝寄与**: A + B (= exp016) で -0.15〜-0.35 ft、 C (= exp019) で更に -0.10〜-0.15 ft = LB 9.2 帯射程
- ✅ **rule 耐性**: 全 path は post-proc / feature engineering、 host fix で死なない
- ✅ **datapoint 価値**: exp016 結果で A/B 仮説、 exp019 で C 仮説、 isolate 可能
- ❌ **physical model 未着手**: MD vs TVT の幾何構造、 inclination / azimuth 加味は別 doc 候補

---

## 8. 残 deep dive 候補 (= 次 doc)

- Q5. ours の prediction `tvt_ours[row]` を「per-row formula」 = `last_known_tvt + LGB_residual[row] + Edge_R_correction[row]` に分解 → どの component が slope drift 主因か
- Q6. visible region の **GR signal** の Fourier spectrum と hidden 予測 spectrum の比較
- Q7. train 770+ wells で「visible tail slope >> hidden actual slope」 を満たす ratio
- Q8. 00e12e8b の formation marker (= MD-based) と hidden bin 区切りの correlation

---

## 9. 関連 doc

- 親 (= surface): `2026-05-13-test-set-divergence-analysis.dense.md`
- 親 (= micro): `2026-05-13-prediction-microstructure-deepdive.dense.md`
- Phase A plan: `docs/dev/2026-05-12-plan-gold-to-winning.dense.md` § 5.1
- Hill Climb 強さ: `docs/research/2026-05-12-hill-climb-strength-analysis.dense.md`
- ravaghi NB: cell 5 = `_pf_z` (= dual PF Z)、 cell 14 = `_apply_pp_3axis` (= 3-axis postproc)
