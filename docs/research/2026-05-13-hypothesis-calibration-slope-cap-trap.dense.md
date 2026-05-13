# 仮説 calibration: 「visible tail slope は害」 → slope cap 盲目適用は ours を悪化 (= LB 9.74 → 12.5)

> 起点: 2026-05-13 GM deep dive Phase 7、 親 doc の「slope cap」 改善 path を **直接 simulation で反証**
> 教訓: train wells で「flat > linear」 法則は **naive linear vs flat** の比較、 LGB+ensemble (= ours) は その between space を埋めている
> 親 doc: `2026-05-13-naive-slope-extrapolation-is-harmful.dense.md`、 `2026-05-13-full-773-wells-confirmed-physics.dense.md`、 `2026-05-13-physical-geometry-inclination-prior.dense.md`

---

## 1. 結論 (= 一発、 仮説修正)

**「visible tail slope の linear continuation は flat より 91% の wells で害」 法則は正しいが、 これを ours の prediction に slope cap 0.005 ft/row を hard rule で適用すると、 LB が 9.74 → 12.5 (= 悪化 +2.8 ft)** に転落する。

理由: ours = LGB+ensemble は naive linear ではなく、 GR signal + 多 feature から smarter slope を出す。 flat より大幅 good (= mean abs 1.12 ft vs 7.32 ft to hc proxy)。 hard cap は LGB の良 slope も削る。

→ **改善 path = 全 row hard cap でなく、 selective conditional cap** (= 例: incl > 89° AND |slope| > 0.05 ft/row なら cap)

---

## 2. 直接 simulation 結果 (= test 3 wells)

### 2.1 各 prediction の hc-proxy RMSE
```
prediction          mean abs (to hc)   p95     max     RMSE
ours (= exp009 v2)   1.118 ft           3.55    4.60    1.558 ft
flat (= anchor 固定) 7.323 ft           16.92   23.44   8.927 ft
slope-cap-mix (★)    -                  -       -       8.252 ft
```

★ slope-cap-mix = 「ours の slope が |slope| > 0.005 ft/row なら anchor からの slope を 0.005 にcap」

→ flat / slope-cap-mix どちらも ours より圧倒的悪い (= hc proxy で 5-7× 悪化)

### 2.2 well 別 win-rate (= flat が ours より hc に近い row の比率)
```
000d7d20:  169/3836 = 4.4%
00bbac68:    3/6014 = 0.0%   ← 完全敗
00e12e8b:  661/4301 = 15.4%  ← 最も flat が勝つ ratio 高
```

→ どの well でも flat が ours より良い row は < 16%、 つまり ours は **84%+ の row で flat より good**

### 2.3 Pythagorean LB 推定 (= ours / flat / mix の LB を計算)
```
known:    hc LB = 9.43,  ours LB = 9.738
formula:  rmse_x ≈ sqrt(rmse_hc² + rmse(x - hc)²)  (= uncorrelated upper bound)

ours LB ≈ sqrt(9.43² + 1.558²) = 9.56 ft  (実測 9.738、 差 0.18 = errors partially correlated)
flat LB ≈ sqrt(9.43² + 8.927²) = 12.99 ft  (= ours より +3.3 ft 悪化)
mix LB  ≈ sqrt(9.43² + 8.252²) = 12.53 ft  (= ours より +2.8 ft 悪化)
```

---

## 3. なぜ「flat wins 91%」 法則と矛盾するか (= conceptual resolution)

### 3.1 train 100 sample での比較
- **比較対象**: naive linear extrapolation (= 単純に anchor + slope_tail × t) vs flat (= anchor 固定)
- **flat wins 91%**: 91% の wells で naive linear が flat より悪い
- 実用: **誰も naive linear は使わない**、 LGB / ensemble / Climber 等で smarter prediction

### 3.2 ours = LGB+ensemble の正体
- ours は GR signal + ANCC + 多 feature を input、 hidden の TVT を per-row predict
- 単純な slope continuation でなく、 row-level feature based **non-linear prediction**
- 大事: ours の per-row predicted slope は、 visible tail slope と LGB residual の **混合**、 LGB が「ある row で slope を抑制」 も内部表現
- → ours は naive linear と flat の **between space** に位置

### 3.3 LGB と flat の差 (= 5× ratio)
- flat vs hc: mean abs 7.32 ft
- ours vs hc: mean abs 1.12 ft
- LGB はナイーブな anchor 固定よりずっと smart、 残り 1.12 ft 級の精度問題が test set divergence

---

## 4. 真の改善 path (= 修正版)

### Path 1' (修正版): conditional slope cap、 = inclination + slope_tail outlier 検出
- **発火条件**: per-well で visible_tail incl ≥ 89° **AND** slope_tail > P85 percentile (= 0.04 ft/row 圏)
- **適用範囲**: 該当 well の **hidden 区間全体に slope cap = visible_tail slope の 20%**
- 00e12e8b 該当 (= incl 89.82°、 slope_tail +0.042 ft/row = train P90+ outlier、 ours 残率 29%)
- ours の 00e12e8b 予測を slope_cap 0.008 ft/row (= +0.042 の 20%) に制約
- 期待: 00e12e8b mean_abs_diff 1.40 → 0.80 ft 圏、 test 全体 RMSE -0.05 〜 -0.10 ft = LB 9.65 〜 9.70

### Path 2 (= 推奨): Climber blend で slope-suppressed base を別 base として加える
- 既存 9 base に **「slope cap 0.005 適用版 ours」 を 10 番目 base** として加える
- Climber が automatic に「どの row で slope cap を活かすか」 を per-row blend weight 決定
- これは naive hard cap でなく **soft constraint via ensembling**
- 期待 lift: -0.05 〜 -0.15 ft、 ours と direct 並走で datapoint 価値高い

### Path 3 (= 中長期): LGB 自体に **inclination * slope** interaction feature 追加
- features.py:136 の `inclination` は per-row、 ただし LGB が「incl 大 → predicted slope 抑制」 の causal 学習弱い
- 明示的 interaction feature `inclination * predicted_slope_residual` を加えれば、 LGB が自動 cap 学習
- 期待 lift: -0.03 〜 -0.08 ft、 Path 1' / 2 と相補

---

## 5. 教訓 (= GM-level、 §11 「優勝本質性」 と統合)

### 5.1 統計法則の **scope** に注意
- 「naive linear vs flat」 の比較で flat が勝つことが、 「ours vs flat」 で flat が勝つを意味しない
- 法則の **base scenarios** を必ず確認、 そのまま改善 path に飛び込まない

### 5.2 simulation-first principle
- 改善 path の期待 LB は **直接 hc-proxy simulation で確認可能** (= LB 直接測れなくても)
- doc 内に simulation 数値を必ず先に出す
- Path A doc 化 → simulation → 結果見て Path A' に修正、 という iteration

### 5.3 hard rule vs soft constraint
- hard rule (= slope cap 0.005 全 row 強制) は LGB の正確 prediction も削る = 害
- soft constraint (= Climber blend で base として加える) は LGB が「該当 row だけ」 利用 = 改善

これは exp016 設計 (= 3-axis post-proc + Climber + dual PF Z) が **既に soft constraint approach** を実装しているのと整合 = exp016 結果待ち。

---

## 6. exp016 が出る前段で「Path 1' selective cap」 を試すべきか

判断:
- exp016 が今 RUNNING、 数時間後完了予定
- exp016 で 00e12e8b drift がどこまで解消するか観察、 残る場合だけ Path 1' / 2 検討
- 並行で **Path 2 (= Climber に slope-suppressed base を追加)** を Phase B/C の Combined kernel として準備可能

→ 推奨: exp016 完走を待つ、 その間に Path 2 を **設計レベル** で詰める (= 別 doc 候補)

---

## 7. open question (= 次 doc 候補)

- Q17: 00e12e8b 4301 rows 中、 ours の hidden slope が train P95 を超える row 数 (= conditional cap の発火範囲想定)
- Q18: hc が ours より良い理由が「全 row で slope 抑制」 か「特定 row のみで slope 抑制」 か、 per-row 解析
- Q19: Climber blend で 「ours base」 + 「ours-flat-clip base」 を加えると weight 比率いくつになるか simulation
- Q20: ravaghi NB の cell 5-11 で inclination usage を再確認 (= grep 既未済)

---

## 8. 関連 doc

- 親 (= 法則 root): `2026-05-13-naive-slope-extrapolation-is-harmful.dense.md`
- 親 (= 法則 confirmed): `2026-05-13-full-773-wells-confirmed-physics.dense.md`
- 親 (= geometry): `2026-05-13-physical-geometry-inclination-prior.dense.md`
- 教訓: hard rule 適用前は必ず **direct simulation で hc-proxy RMSE 算出**
- exp016 設計: `docs/dev/2026-05-12-plan-gold-to-winning.dense.md` § 5.1
- reproducible script: 本 doc 内の python code、 inputs = `/tmp/rogii-exp009-output/submission.csv` + `/tmp/rogii-exp017-v3-output/submission.csv` + `data/raw/test/<w>__horizontal_well.csv`
