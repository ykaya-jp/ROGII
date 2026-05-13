# 超 critical: romantamrazov SUPER LB TOP 3 (= vote 1 位、 143) audit — 我々と ravaghi 両方が miss する 5 feature 発見

> 起点: 2026-05-13 user 質問「なぜ公開 NB に負けるか?」 → 公開 kernels listing で vote 1 位 (= romantamrazov SUPER 143 vote、 ravaghi 111 超え) が我々完全 unaudited と判明
> Source: `.work/romantamrazov-super/rogii-super-solution-lb-top-3.ipynb` (40 KB、 11 cells = 10 code + 1 md)
> author: Roman Tamrazov、 公開 5/8、 title 「Sub-9 Solution v2 (Tuned LGB+CB)」、 markdown「From 10.184 → target < 9.0」

---

## 1. 構成 サマリ

### Base models (= 4 base):
- LGB×3 (= lr {0.025, 0.020, 0.030}、 seed {42, 7, 123}、 num_leaves=255、 reg_lambda=3.0)
- CB×1 (= lr=0.025、 depth=7、 8000 iter、 GPU)
- Ridge stack (= 4 base、 positive=True) + simple avg、 「whichever is better」 で選択

### Post-proc:
- alpha × tau × w_pf grid (= 9 × 6 × 3 = **162 cell exhaustive**、 ravaghi Optuna 500-trial より粗)
- savgol smooth (= window 17、 polyorder 3、 per-well)

### Particle Filter:
- Dual PF (= ANCC + Z 軸両方、 N=500 each)
- `run_pf_z` (= raunakdey07 同 paradigm)

---

## 2. **5 つの我々未取込 feature** (= LB 改善の直接 path)

### F1. PF 4th feature family (= PF-ANCC anchored、 11 offsets) ★
```python
PF_OFFS = np.array([-30,-15,-8,-4,-2,0,2,4,8,15,30], np.float32)  # 4th family
```
- 既存 ANCH_OFFS / BEAM_OFFS / SC_OFFS の 3 family に加え **PF-ANCC 中心の 4 family 目**
- 各 offset で signal (= TVT、 b_well 等) を sample、 11 列 × 数 signal = 数十 features 増
- 我々 features.py で `ANCH_OFFS` 等 3 family のみ、 PF 4th family 不在
- **期待 lift: -0.02〜-0.05 ft** (= LGB が PF-ANCC neighborhood の TVT 値を直接 feature 化)

### F2. Per-formation known-zone RMSE (= per-well で trustability feature) ★
```python
# Per-formation TVT + WLS b_well + known-zone RMSE
for fi2, fn in enumerate(FORMATIONS):  # 6 formation
    b_v = ktvt + z_kn - form_kn[:, fi2]
    # ... known-zone での RMSE 算出
    form_rmse[fn] = float(rmse_per_formation)
```
- visible region で「どの formation の plane が最もよく fit するか」 を per-well RMSE で測定
- → hidden region で「fit 良い formation を強く信用」、 「fit 悪い formation は weight 下げる」
- 我々: 全 formation 同等 weight、 selectivity なし
- **期待 lift: -0.02〜-0.05 ft** (= per-well decision tree splitting、 LGB が自動学習)

### F3. Inter-signal std (= master uncertainty feature) ★
```python
# 全 signal source (= beam, NCC, PF-ANCC, PF-Z, formation plane) を集約
# 各 row で std 算出 → 「signal 間で合意あり」 vs 「signal 間で disagree」 を 1 feature
inter_signal_std = np.std([beam_pred, ncc_pred, pf_ancc_pred, pf_z_pred, ...], axis=0)
```
- 高 std = signal disagree = 予測不確実、 低 std = signal agree = 予測確実
- LGB が「不確実 row では smoothing 強化、 確実 row では fine-grained predict」 を自動学習
- 我々: 個別 signal はあるが、 inter-signal disagreement summary なし
- **期待 lift: -0.02〜-0.05 ft**

### F4. GR envelope + energy (= rolling max + RMS) ★
```python
gr_envelope_max = pd.Series(gr).rolling(W, center=True).max()
gr_envelope_min = pd.Series(gr).rolling(W, center=True).min()
gr_energy_rms   = np.sqrt((pd.Series(gr) ** 2).rolling(W).mean())
```
- 標準 rolling mean / std に加え **rolling max (= envelope) + RMS (= energy)** で local GR pattern 特性
- GR の「short-term spike」 や「sustained high」 を区別
- 我々: rolling mean / std のみ、 max / RMS 不在
- **期待 lift: -0.02〜-0.05 ft**

### F5. GR detrend residual (= linear detrend で local anomaly) ★
```python
def gr_detrend_resid(gr_arr, md_arr):
    slope = robust_slope(md_arr, gr_arr)
    return (gr_arr - slope * md_arr).astype(np.float32)
```
- Global GR trend を MD で linear fit、 residual を local anomaly feature 化
- well 内 global trend (= 例: 深くなるほど GR 増) を除いた **純 local 変動** を捕捉
- 我々: GR raw + rolling 系のみ、 detrend residual 不在
- **期待 lift: -0.02〜-0.05 ft**

---

## 3. 我々の構造的劣後 (= 親 doc § 5 を更新)

```
                    romantamrazov SUPER (143 vote)   ravaghi HC (111)   ours exp009 v2  ours exp016 計画
base                4 (LGB×3 + CB×1)                  6 (LGB×3 + CB×3)    9 ✅           9 ✅ (+ Climber)
Multi-scale NCC     ✅                                  ✅                   ❌ (impl 未 inject) A5 (exp020)
Dual PF Z+ANCC      ✅                                  ❌                   ❌            A3.1 (exp016 inject 済)
WLS b_well          ✅                                  ✅                   ✅            ✅
F1. PF 4th family   ✅ ★                                ❌                   ❌            ❌ (新)
F2. Per-form RMSE   ✅ ★                                ❌                   ❌            ❌ (新)
F3. Inter-signal σ  ✅ ★                                ❌                   ❌            ❌ (新)
F4. GR envelope+E   ✅ ★                                ❌                   ❌            ❌ (新)
F5. GR detrend      ✅ ★                                partial              ❌            ❌ (新)
ensemble            Ridge 4 stack                       Climber 6           Ridge 9       Climber 9
post-proc           grid 162                            Optuna 500 3-axis   2-axis grid   2530 cell grid
```

我々の **構造的問題**:
- ❌ 5 feature (F1-F5) 全部 miss
- ❌ Multi-scale NCC / Dual PF Z は実装済だが exp009 v2 に inject 未
- ✅ base 数 9 で diversity 優位、 ただし feature 不在で **base diversity 効果が頭打ち**

---

## 4. 直接的改善 path (= exp020 計画 拡張)

### 既存 plan §3 Phase A.5 (= exp020 A3+A4+A5 inject) に追加:
- **A8: PF 4th feature family** (= `src/rogii/features.py` に PF_OFFS family 追加、 11 offset × signals)
- **A9: Per-formation known-zone RMSE** (= `features.py` build_well 内、 6 formation × RMSE = 6 列)
- **A10: Inter-signal std** (= signal 集約 → std 計算 = 1 列)
- **A11: GR envelope + energy** (= rolling max + RMS = 2-3 列)
- **A12: GR detrend residual** (= linear detrend → residual = 1 列)

合計 **+11 列追加 features** (= 我々 165 col → 176 col)

### 期待 LB lift (= F1-F5 累積):
- 単独 -0.02〜-0.05 ft × 5 = **-0.10〜-0.25 ft 累積**
- 既存 Phase A.5 (= A3+A4+A5 + Climber) の -0.20〜-0.50 ft と相補
- exp020 拡張版 で LB **9.1-9.3 帯到達** (= 9.738 から -0.40〜-0.60 ft、 ours base から 9.5 帯射程)

---

## 5. user 質問「なぜ公開 NB に負けるか」 への直接答え

### 真因 (= top-down decomposition):
1. **5 つの feature を完全 miss** (= F1-F5、 公開 NB が vote 1 位として共有しているのに) ← **最大要因**
2. **Multi-scale NCC + Dual PF Z を実装済だが inject 未** (= 公開 NB が全部使う、 我々の 2 軸劣後)
3. **post-proc grid 粗**: 2-axis vs 3-axis 500-trial (= ravaghi)、 vs 162-cell (= romantamrazov)
4. **ensemble Ridge vs Climber**: 線形 positive 制約で overcounted base subtract 不可

### user 観察「公開 NB で UI 0.94 帯」 の正体
- LB top **9.4** 帯 = `eiheychan 9.430` (= ravaghi fork)、 公開 NB を fork でも GPU で 9.43 取れる
- LB top **9.301-9.392** 帯 = Silogram / lingyu07 / Mr.キノコ / Chris Deotte = **公開 NB + 自前 tuning**
- LB top **1 位 8.966 Virtute** = paradigm 1 (HC) + paradigm 3 (MoE) or 1+5 (NN) hybrid 推測 (= 親 doc novel-paradigms-10-candidates §S1-A2)

→ **我々の現状 9.738 = rank ~30-40 圏外、 公開 NB fork でも 9.43 = rank 9 圏 = 我々が 5 feature 取込めば自動的に rank 10 圏到達**

---

## 6. 「死ぬほど EDA / error analysis」 への接続

user 指示の死ぬほど EDA は本 doc の 5 feature 個別の検証で具体化可能:

### 各 feature の **train data 上の predictive power** 測定 plan:
1. F1 (PF 4th family): train 770 wells で PF-ANCC anchored offset feature を計算、 OOF residual との correlation 測定。 corr > 0.10 で採用判定
2. F2 (Per-formation RMSE): 各 well で 6 formation × known-zone RMSE 算出、 LGB feature importance で ranking
3. F3 (Inter-signal std): 全 signal 集約 std を計算、 OOF residual との correlation
4. F4 (GR envelope+energy): rolling 系 features の追加 importance
5. F5 (GR detrend residual): linear detrend で local anomaly が予測残差と correlation あるか

各 feature の predictive power が「期待 lift -0.02〜-0.05 ft」 を実 OOF で confirm or refute。

### 残 EDA 候補 (= 別 doc):
- 損失関数 sensitivity (= RMSE vs Huber、 sample_weight heterogeneous)
- 前処理 (= imputer FormationPlaneKNN + DenseANCCImputer の比較 + tuning)
- 特徴量 importance ranking 全 features 集約 (= 我々 165 col のうち低重要度を削除して collinearity 減らす)
- per-formation error 分解 (= 6 formation × train 770 wells で RMSE matrix)
- residual の per-row position 分布 (= visible↔hidden boundary 周辺で error pattern)

---

## 7. 関連 doc + 次 action

### Immediate (= exp020 計画 拡張):
- `docs/dev/2026-05-13-exp020-phase-a-plus-blueprint.dense.md` に **A8-A12 feature inject** 追加
- `src/rogii/features.py` build_well 内に 5 feature 追加 (= 200-300 行追加)
- TDD: `tests/test_super_features.py` 新規

### Master plan 更新候補:
- `docs/plans/2026-05-13-beat-everyone.dense.md` §3 Phase A.5 に「+ romantamrazov 5 feature」 を追記

### 関連 docs:
- ravaghi 解析: `docs/research/2026-05-13-ravaghi-cell6-structural-mapping.dense.md`
- thbdh v10 解析: `docs/research/2026-05-13-thbdh-v10-structural-survey.dense.md`
- 新 paradigm 10 件: `docs/research/2026-05-13-novel-paradigms-10-candidates.dense.md`
- 親 plan: `docs/plans/2026-05-13-beat-everyone.dense.md`
