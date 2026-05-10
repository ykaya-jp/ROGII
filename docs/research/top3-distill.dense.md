# ROGII Top3 解 distill (2026-05-10)

> **目的**: 公開 Top kernel 2 件を全行精読し、勝つために必要な 6 要素 (Beam Search / Numba PF / FormationPlaneKNN / Self-NCC / WLS b_well / Stacking) の核心を抽出して、我々の `src/rogii/` への移植 map を設計する。
>
> **方針**: コード長文丸写しは避け、各要素 30 行以内の核心抜粋 + `file:line` 引用 + 物理的解釈の解説に留める。著作権配慮 (Kaggle 公開 kernel は競技規約上 share 前提だが、丸写しは敬意を欠く)。
>
> **参照**: `docs/strategy/winning-strategy.dense.md` Phase 1.6 §B、`docs/research/public-notebook-analysis.dense.md`

---

## 出典

| ノートブック | 著者 | LB | 主アプローチ | コード行数 | 出典 URL |
|---|---|---|---|---|---|
| `rogii-super-solution-lb-top-3.py` (= **R**) | romantamrazov | **~10.1 (Top 3)** | Numba JIT Beam ×7 + 2 PF (N=500) + 6-form plane-fit + multi-scale NCC (3 windows) + WLS b_well + GR detrend + LGB×3 + CB + Ridge stack | 774 | https://www.kaggle.com/code/romantamrazov/rogii-super-solution-lb-top-3 |
| `score-10-081-score-lb-32-rank.py` (= **N**) | needless090 | **10.081 (32 位)** | Beam ×5 (NumPy) + 2 PF (N=300) + 6-form plane-fit + Self-NCC (1 window) + LGB×3 + CB×3 + TabICL ×2 (4096/8192 ctx) + Ridge stack | 799 | https://www.kaggle.com/code/needless090/score-10-081-score-lb-32-rank |
| `physics-informed-baseline.py` (= **K**) | karnakbaev | 10.784 | R + N の早期 hybrid (Beam ×5, PF, Plane-fit, Self-NCC, Affine GR cal) + XGB 追加で 5 base + Ridge | 1,978 | https://www.kaggle.com/code/karnakbaevarthur/top-2-rank-10-784-physics-informed-baseline |
| `lb-11-068-rogii-wellbore-geology-prediction.py` (= **T**) | tasmim | 11.068 | Beam ×2 + Plane-fit + **xcorr_tvt_features** (固有 edge) + GBM | 663 | https://www.kaggle.com/code/tasmim/lb-11-068-rogii-wellbore-geology-prediction |

注: R の機械的 LB 比較は明示されず "~10.1" 表記。N の 10.081 は LB に明記された確定値。**著者本人による R を意図的に拡張したのが N (TabICL 追加で stacking 強化)** という構図に見える (BEAMS list, ANCC_*, PF_* hyperparams が R と完全一致 → R が原型)。

ライセンス: 両 kernel とも `kernel-metadata.json` に license tag なし。Kaggle 公開 kernel は default で **Apache 2.0** に該当 (Kaggle Terms 8.B)。再配布する際は出典 URL + 著者名を残す。

---

## 1. Beam Search 詳細

### 1.1 R 流 (LB ~10.1, **Numba JIT, 7 configs**) — 最強版

**configs** ([R: file `rogii-super-solution-lb-top-3.py:43-51`](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py)):

```python
BEAMS=[
    (10,20.0,144.0,2,"cons"),     # 標準
    (10, 8.0, 64.0,2,"loose"),    # 動きやすい
    ( 8,35.0,220.0,1,"vcons"),    # 超保守
    (10,14.0, 90.0,5,"sm5"),      # 強平滑
    (20,4.0,  36.0,3,"vloose"),   # 大ビーム×ゆるい
    (12,12.0,100.0,3,"mid"),
    (15,25.0,180.0,2,"stiff"),
]
```

タプル意味: `(beam_size, move_cost, emit_scale, smooth_radius, tag)`。

**cost function** ([R:122](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py)):

```python
tot = cost + (gv - tw_gr[ni])**2 / es + mc * abs(d)
```

- `(gv - tw_gr[ni])**2 / es`: emit cost (= 観測 GR と typewell GR の二乗誤差を `emit_scale` で割る)
- `mc * |d|`: move cost (=典型では 1 step あたり TVT index を ±2 動かすペナルティ。`d ∈ {-2,-1,0,1,2}`)

**delta 範囲** ([R:119](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py)): `for d in range(-2,3)` ⇒ **±2 step / 1 ft**。N と R で大きく違うのはこの幅 (N は ±1)。

**pruning**: 各 step で beam_size 個まで low-cost を保持。同一 `tw_tvt index` (= 同一 next 状態) に到達する複数 path が居たら最安だけ残す ([R:124-128](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py))。最後に backtracking で path を復元 ([R:142-144](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py))。

**JIT**: `@njit(cache=True)` で AOT cache → 2 回目以降の re-execute がほぼゼロ起動。warm-up 1 回 ([R:162](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py))。

**前処理**: GR を typewell mean で fillna → smooth_radius>0 なら `rolling(2r+1, center=True).mean()` で滑らか化 ([R:152-154](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py))。

**消費 features**: 7 path × `last_tvt 差分` + mean / std / median (= 7+3 = 10 dims) を後段 GBM に渡す ([R:564-567](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py))。さらに beam_ref から ±[40,20,10,5,3,0,3,5,10,20,40] offset で `tw_diff` 11 family ([R:621](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py))。

### 1.2 N 流 (LB 10.081, **NumPy 5 configs**)

**configs** ([N: `score-10-081-score-lb-32-rank.py:60-66`](../../_research_kernels/needless090__score-10-081-score-lb-32-rank/score-10-081-score-lb-32-rank.py)): 上記 R の最初の 5 つと完全一致。R の `mid` / `stiff` を捨てた版。

**cost function** ([N:149-150](../../_research_kernels/needless090__score-10-081-score-lb-32-rank/score-10-081-score-lb-32-rank.py)): R と同型。ただし delta 範囲は **±1** (`np.array([-1,0,1])`)。

**pruning** ([N:152-156](../../_research_kernels/needless090__score-10-081-score-lb-32-rank/score-10-081-score-lb-32-rank.py)): `argsort(fc)` 全体ソート → seen set で同一 idx 重複除去 → beam_size 個取る。NumPy 純 vectorize。Numba 不要だが per-step の `argsort` で `O(beam·3·log(beam·3))` がボトルネック。

**速度比較 (推定)**: R の Numba JIT は warm-up 後 N の NumPy 比 **2-5× 高速** (per-step inner loop が JIT 化されるため)。1 well あたり beam=10 × n_steps=数千で N は数 sec / R は <1 sec オーダー。773 wells × 7 configs (R) vs 5 configs (N) で総時間は同等になる程度。

### 1.3 共通エッセンス

- **Diversity 最重要**: 異なる `(bs, mc, es, r)` で 5-7 path 生成 → mean/std/median を feature 化 → GBM が "どの beam を信頼するか" 自動学習。**単一 beam 信号ではなく consensus** が肝。
- **Hidden zone を typewell index で grid search** している。observed GR (horizontal) と typewell GR の Gaussian likelihood で path 探索 ⇒ "horizontal の bit が typewell の TVT 軸上のどの位置にいるか" を全 step で逐次推定。
- **start_tvt = last_known TVT** で初期化 ([R:157](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py))。これが residual target と完全に整合 (= residual=0 が "そのまま延長" に対応)。
- **smooth_radius が違う path を混ぜる** ことで HF noise / 大局トレンド両方を carry。`sm5` は強平滑なので long horizon で安定、`vcons/cons` は HF feature 残し。
- **beam_ref = (cons + sm5) / 2** ([R:450](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py)) で代表 path を作り、これを `tw_diff` の anchor として使う。

**物理的解釈**: 水平井 bit が **どの depth (TVT) を進んでいるか** は GR pattern matching で復元できる (= 手作業 geosteering の "stretch & squeeze" を Viterbi 風 DP で自動化したもの)。move_cost = TVT 速度の正則化、emit_scale = GR noise の信頼度。複数 (bs, mc, es) を併走するのは **ノイズと信号のトレードオフ点を 1 つに固定しない** ため。

---

## 2. Numba JIT Particle Filter

### 2.1 R/N 共通: 2 種の PF を併走

両者とも **`run_pf_z` (Z-velocity PF)** と **`run_pf_ancc` (ANCC random walk PF)** を独立に走らせ、両 path を feature にする。両 PF は同一 hyperparam ([R:53-60](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py) ≡ [N:69-75](../../_research_kernels/needless090__score-10-081-score-lb-32-rank/score-10-081-score-lb-32-rank.py))。違いは N=300 (N 流) vs N=500 (R 流) のみ。

注意: R の PF は **Numba JIT ではない** (純 NumPy vectorized)。Numba 化されているのは **Beam Search のみ**。タイトルに "Numba" を冠する R kernel でも PF は per-step `for` loop in Python だが N=500 × 数千 step が NumPy broadcast で許容速度。

### 2.2 ANCC PF (= 主 estimator)

**state**: `(pos, rate)` ([R:237-239](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py))
- `pos` = TVT + Z (= ANCC 推定値、formula `TVT = -Z + ANCC + b_well` の右辺前 2 項)
- `rate` = `d(TVT+Z)/dMD` (= ANCC が MD 進行に応じてどれだけ動くかの slope)

**初期化** ([R:237-239](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py)):
- `pos ~ N(last_tvt + last_z, ANCC_IS=0.3)`
- `rate ~ N(median_slope, ANCC_IR=0.01)` (median は known zone 末尾 30 pts から)

**transition** ([R:245-247](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py)):
```python
rate = ANCC_ALPHA * rate + N(0, ANCC_RN)        # alpha=0.998 で AR(1)
pos += rate * dm + N(0, ANCC_PN)                # process noise
tvt_e = clip(pos - z_v[i], tmin-50, tmax+50)
```

**likelihood** ([R:249-251](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py)):
```python
eg = np.interp(tvt_e, tw_tvt, tw_gr)             # particle ごとに typewell GR 値を補間
lk = exp(-0.5 * ((gr_v[i] - eg) / gs)**2)        # Gaussian on GR
```
`gs = clip(std(observed_GR - interp(TVT_input)), 10, 60)` (= known zone での GR-typewell ズレから calibrated, [R:225-226](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py))。

**resampling** ([R:253-256](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py)): systematic resampling (1/N step + uniform jitter で stratified)。`Neff < 0.5*N` で発火。**roughening**: `pos += N(0, ANCC_RP=0.1); rate += N(0, ANCC_RR=0.001)` で sample impoverishment 防止。

**output** ([R:257-258](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py)): per-step weighted mean `tv` と weighted std `std_o` (= 不確実性の出力)。両方 features に入る。

### 2.3 Z-velocity PF

**state**: `(pos, vel)` (= TVT, dTVT/dMD) ([N:206-207](../../_research_kernels/needless090__score-10-081-score-lb-32-rank/score-10-081-score-lb-32-rank.py))。

**plus**: `vel` の prior に Z-velocity との線形回帰 `vt = beta*vz + icpt` を掛ける ([N:185-188](../../_research_kernels/needless090__score-10-081-score-lb-32-rank/score-10-081-score-lb-32-rank.py))。known zone の `(dz/dmd, dtvt/dmd)` で OLS fitting → 物理的に "TVT 速度は Z 速度から決まる" 関係を学習。`beta` の典型値は **-1.0 近辺** (= Z が下がれば TVT も下がる、ただし well dip 角に依る)。

**dual likelihood** ([N:218-224](../../_research_kernels/needless090__score-10-081-score-lb-32-rank/score-10-081-score-lb-32-rank.py)): 観測 GR 1 点 + smooth GR 1 点で 2 種 likelihood を `(1-PF_GR_WT)*lp + PF_GR_WT*ls` で blend (PF_GR_WT=0.3)。HF noise 弱 + LF trend 強の効果。

### 2.4 計算量

- 1 well: N=500 particles × hidden_steps (典型 数百 ft) × O(1) per particle = 数十万 op
- 773 wells × 4 thread 並列 (joblib `prefer='threads'`) で **typically 5-10 min on Kaggle T4 ×2 (1 PF type)**。両 PF + 7 beams + features で 30-40 min。

### 2.5 共通エッセンス + 物理的解釈

PF は **Markov 仮定 + Gaussian observation** の sequential Bayes 推定。Beam Search は最尤 path 1 本を返すが、PF は **不確実性 (std)** を返すので "どこで信頼度が落ちるか" を GBM に教えられる。**ANCC PF と Z-PF を両方走らせる** のは独立 hypothesis (random walk vs Z-coupled) で robust → cross-check feature `pf_vs_z` ([R:562](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py)) も生成。

---

## 3. FormationPlaneKNN (6-formation plane-fit imputer)

### 3.1 共通実装 (R ≡ N、コード完全一致)

**index 構築** ([R:319-336](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py)):

各 well から `[X, Y]` の median (= centroid) と 6 formation の median を取り、centroid を `cKDTree` に格納。distance scale = `xy.std(0)` で X/Y を正規化。

```python
xy = self.df[['x','y']].to_numpy()
self.scale = np.where(xy.std(0)<1e-3, 1., xy.std(0))
self.tree = cKDTree(xy/self.scale)
```

**距離は 2D (X/Y のみ)。Z や MD は使わない**。理由: formation TVT は地理 (X/Y) で滑らかに変化する surface 想定 → centroid 座標で十分。

### 3.2 距離・k

- **k = `PLANE_K = 10`** ([R:40](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py))
- 取得: `nf = min(k+5, len(df))` で 15 候補取って、**self_wid を除外** (train 時 leak 防止) してから top-k ([R:341](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py))
- weight: `1 / (dist + 1e-3)` (= IDW)

### 3.3 Plane fit (= IDW Weighted Least Squares)

**3×3 normal equation を query ごとに直接 solve** ([R:347-353](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py)):

```python
A = [[Σ w·x²,   Σ w·xy,  Σ w·x],
     [Σ w·xy,   Σ w·y²,  Σ w·y],
     [Σ w·x,    Σ w·y,   Σ w  ]]      # 3x3
rhs = [Σ w·x·f, Σ w·y·f, Σ w·f]^T     # 3x6 (6 formations)
A[i,i] += 1e-9                          # ridge for stability
coef = np.linalg.solve(A, rhs)          # (3, 6)
pred = X·coef[0] + Y·coef[1] + coef[2]  # plane(x,y) per formation
```

= **IDW weighted plane** (formation を平面で局所近似)。各 formation で独立な平面係数 (a, b, c) を学習。

### 3.4 Imputer 精度確認方法 (= **重要**)

両 kernel とも **per-formation b_well RMSE** ([R:495](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py), [N:454](../../_research_kernels/needless090__score-10-081-score-lb-32-rank/score-10-081-score-lb-32-rank.py)) を known zone で計算して features に入れている:

```python
# 各 formation で b_v = ktvt + kz - form_kn (known zone での b_well = TVT + Z - ANCC)
form_rmse[fn] = sqrt(mean((ktvt - (-z_kn + form_kn[:,fi2] + b_all))**2))
```

= **known zone で plane-fit 出力を使って TVT 復元したときの residual RMSE**。`tvt_formula = -Z + ANCC + b_well` の formula 自体の精度を well 単位で測定 → GBM に "この well では plane-fit が信頼できるか" を教える。

### 3.5 DenseANCCImputer (= ANCC 専用 dense IDW)

ANCC 1 列だけは **plane fit ではなく per-well 60 サンプル を全部 cKDTree に放り込んで k=20 IDW** ([R:364-394](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py))。理由: ANCC は test でも mask されており、**地理的に細かい変動** がある (formation 平均では捉えきれない) ため fine-grained imputer を別建て。

### 3.6 物理的解釈

地下の formation top は **well 群の (X,Y) 平面上で滑らかな surface** をなす (地質学的に局所平面近似が妥当)。test well の (X,Y) で周囲 10 wells の formation centroid を平面回帰 → ANCC を imputed → `TVT_pred = -Z + ANCC_imp + b_well` で TVT を復元できる。**B-2 公開分析の 'plane-fit + b_well' formula はこの imputer がコア**。

---

## 4. Self-NCC (Normalized Cross-Correlation, self)

### 4.1 N 流 (1 window=15)

**signature** ([N:120](../../_research_kernels/needless090__score-10-081-score-lb-32-rank/score-10-081-score-lb-32-rank.py)):
```python
def self_corr_tvt(kgr, ktvt, hgr, hw=15, stride=3):
```

- `kgr / ktvt` = known zone の GR / TVT_input (length nk)
- `hgr` = hidden zone GR (length nh)
- `hw=15` ⇒ window=`2*15+1=31` ft
- `stride=3` で known zone を template に切り出し

**手順** ([N:124-136](../../_research_kernels/needless090__score-10-081-score-lb-32-rank/score-10-081-score-lb-32-rank.py)):
1. 両系列に rolling mean (window=5) で平滑化
2. known zone から stride=3 で M 個の window (size=31) を切り出し → `C` (M, 31)
3. C を z-normalize → `Cn`
4. hidden GR を ±15 ft padding (`mode='edge'`) して各 i 中心の 31 ft window → `H` (nh, 31)
5. H を z-normalize → `Hn`
6. **NCC matrix = `Hn @ Cn.T / win`** ⇒ (nh, M)
7. 各 hidden 行で `argmax` を取り、その template の **center TVT (`ktvt[ctrs]`)** を返す
8. 副作用 score = ncc.max(1) も返す (信頼度)

**output dim**: `(tvt_estimate ndarray (nh,), score ndarray (nh,))`

### 4.2 R 流 (multi-scale, 3 windows=8/15/25)

**実装** ([R:188-209](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py)): N 流 self_corr_tvt の **window size 3 つ (hw=8, 15, 25 = win=17/31/51 ft)** を独立計算 → 3 signal を返す。**stride=3 共通**。

```python
sc_res = multi_scale_sc(kgr, ktvt, hgr, hws=(8,15,25), stride=3)
sc8, sc8s = sc_res[0]; sc15, sc15s = sc_res[1]; sc25, sc25s = sc_res[2]
sc_cons = (sc8 + sc15 + sc25) / 3.
```

**lag range**: 明示的な lag search はせず、**known zone 全体** から template を切り出して NCC で「どの template が一番似ているか」を探す ⇒ 暗黙的に lag は known zone の長さ全体。

### 4.3 hyb_ref (= Beam と Self-NCC の信頼度別 blend)

両 kernel 共通 ([R:457-458](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py), [N:416-417](../../_research_kernels/needless090__score-10-081-score-lb-32-rank/score-10-081-score-lb-32-rank.py)):

```python
sc_trust = float(np.clip(len(kn) / 200., 0., 0.6))
hyb_ref = (1 - sc_trust) * beam_ref + sc_trust * sc15
```

= "known zone が長いほど Self-NCC を信頼" (上限 0.6)。known が短い → Beam (typewell-driven) を主、known が長い → Self-NCC (visible 自身の pattern) を主。

### 4.4 物理的解釈

**Slide 9 insight (公開 discussion で引用)**: 「hidden zone は同じ well の visible (= known) zone と GR pattern が類似している (同一 lateral の堆積環境のため)」。したがって typewell より **同一 well の prefix GR を template にした方が良い predictor になる** ことがある。Self-NCC は "horizontal の hidden 部分は visible 部分のどこの繰り返しか" を pattern matching で探す。multi-scale (3 windows) は **HF (17 ft) / mid (31 ft) / LF (51 ft)** を独立に取って GBM に渡す → 解像度トレードオフを 1 個に固定しない。

---

## 5. WLS b_well (Weighted Least Squares)

### 5.1 R が独自に追加 (N にはなし、これが R の edge の 1 つ)

**実装** ([R:181-186](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py)):

```python
def wls_b_well(ktvt, kz, form_col, decay=0.02):
    """Recent-weighted b_well: tail points matter more."""
    n = len(ktvt)
    if n < 3:
        return float(np.median(ktvt + kz - form_col))
    w = np.exp(decay * np.arange(n)); w /= w.sum()
    return float(np.dot(w, ktvt + kz - form_col))
```

**weight definition**: `w_i = exp(0.02 * i)` (i = 0 が最古、i = n-1 が最新)。指数で **末尾 (= bit 進行先頭の直前) を重く**。例: n=100 で `w[99] / w[0] = exp(0.02*99) ≈ 7.2 倍`。

**b_well 推定式**: `b = Σ w_i * (TVT_i + Z_i - formation_i)`
= 「formula `TVT = -Z + ANCC + b_well` の `b_well` を、known zone の (TVT_input, Z, plane-fit imputed formation) から逆算した weighted mean」

### 5.2 N との対比

N は **median ベース** ([N:441-445](../../_research_kernels/needless090__score-10-081-score-lb-32-rank/score-10-081-score-lb-32-rank.py)):
```python
b_all = float(np.median(b_v))                                     # 全 known
b_50 = float(np.median(b_v[-50:])) if len(b_v) >= 5 else b_all   # 末尾 50
```

= R の WLS は "連続的 fade" / N の median は "binary cut (全 vs 末 50)"。WLS のほうが **smooth** (= overfit 抑制) で、特に short known zone で効くと推定。

### 5.3 R は b_well を **3 種出力** (median + WLS + 末尾 50) して GBM に全部渡す

```python
# per formation 6 個 × 3 種 = 18 features
'bw_{fn}': b_all,         # median (N と同じ)
'bw50_{fn}': b_50,        # 末尾 50 median
'bww_{fn}': b_wls,        # WLS recent-weighted
```
([R:491-493](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py))

GBM に「どの b_well 推定が正確か」を選ばせる戦略。**この複数推定を全部突っ込む流儀が R の哲学**。

### 5.4 物理的解釈

`b_well` は **formula `TVT = -Z + ANCC + b_well` における well 固有の offset** (= 各 well で formation 位置が global plane からどれだけズレているか)。bit が進むにつれてこの offset は緩やかに drift する可能性 (caliper 校正・温度補正 etc) → 末尾を重くすることで **drift が直近に追従**。decay=0.02 は経験値。

---

## 6. Stacking ensemble

### 6.1 N 流 (= 8 base, 最強構成)

**base models** ([N:619-732](../../_research_kernels/needless090__score-10-081-score-lb-32-rank/score-10-081-score-lb-32-rank.py)):

| # | model | params | seeds | OOF dim |
|---|---|---|---|---|
| 1-3 | LightGBM | num_leaves=127, lr=0.04, n_estimators=5000, GPU | seed=42/7/123 | 3 |
| 4-6 | CatBoost | depth=8, lr=0.04, iter=5000, T4 dual GPU | seed=42/7/123 | 3 |
| 7 | TabICL_A | ctx=4096, n_estimators=4 (= TabICL ensemble member) | 5 seeds (0-4) avg | 1 |
| 8 | TabICL_B | ctx=8192, n_estimators=4 | 1 seed (42) | 1 |

**TabICL** = transformer-based **In-Context Learning** tabular regressor (OpenML 系 SoTA に近い)。`tabicl-regressor*.ckpt` を private dataset 経由で持ち込み (Internet disabled) ([N:644-666](../../_research_kernels/needless090__score-10-081-score-lb-32-rank/score-10-081-score-lb-32-rank.py))。

**TabICL feature 選定** ([N:678-684](../../_research_kernels/needless090__score-10-081-score-lb-32-rank/score-10-081-score-lb-32-rank.py)): LGB quick fit (n_est=300) で feature_importances を取って **top-50 features** だけを TabICL 入力に。TabICL は次元高すぎると遅いので削減必須。

**meta-learner** ([N:629-637](../../_research_kernels/needless090__score-10-081-score-lb-32-rank/score-10-081-score-lb-32-rank.py)):

```python
ridge = Ridge(alpha=1., fit_intercept=False, positive=True)
ridge.fit(Sx, y.values)             # Sx = (n_train, 8)
final = oof_s if r_stk < r_avg else Sx.mean(1)   # simple avg と Ridge stack 比較
```

`positive=True` で **負の重みを禁止** ⇒ 重みが 0 になる base が出るが解釈性高い。`fit_intercept=False` で線形結合 (= 重みの和 ≈ 1)。**Ridge stack vs Simple mean** で OOF が良い方を採用。

### 6.2 R 流 (= 4 base, 軽量だが Top 3 達成)

**base** ([R:78-99](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py)):

| # | model | params | OOF dim |
|---|---|---|---|
| 1-3 | LightGBM | num_leaves=255, lr=[0.025/0.020/0.030] (= 3 種異なる lr), n_est=8000 | 3 |
| 4 | CatBoost | depth=7, lr=0.025, iter=8000, T4 dual GPU | 1 |

**TabICL なし** ⇒ 軽量 (~30 min vs N の 55 min)。LGB の `lr` を 3 種変える流儀 (= seed だけでなく hyperparam 多様化) は R 固有。`num_leaves=255` で **N より深い** モデル (N=127 だが trees 8000 で長く回す)。

**meta-learner** ([R:706-708](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py)): N と同型 (`Ridge positive=True, fit_intercept=False`)。

### 6.3 OOF flow (両者共通)

1. `GroupKFold(5).split(X, y, well_id)` ⇒ 5 splits ([R:667](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py))
2. 各 base × 各 fold で train → val OOF + test pred (avg over 5 folds)
3. `Sx = column_stack(oofs)` (n_train, n_base) / `St = column_stack(test_preds)` (n_test, n_base)
4. Ridge fit on `(Sx, y_residual)` → simple avg と OOF RMSE 比較 → 良い方
5. **post-processing**: `pred *= alpha; pred *= (1 - exp(-md_since/tau))` を grid search (R は w_pf も追加で 3 軸 [R:729-737](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py))
6. **Savitzky-Golay smooth** (window=17, polyorder=3) per well ([R:745-752](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py))

### 6.4 共通エッセンス + 物理的解釈

- **Residual target** (= `TVT - last_known_TVT`) で全 base を fit。abs TVT 直接予測より **GBM が trees の depth を浪費せず** 効率良い学習 (= drift モデリングに集中)。
- **Ridge `positive=True`**: 負の重みは out-of-distribution の symptom (= base が anti-correlated していると negative weight が出る)。それを禁止して "diversity contributing only positively" に制約 → robust。
- **post-processing fade-in `(1 - exp(-md_since / tau))`**: hidden zone の **直近 (last_known の隣)** は予測しやすい / 遠ざかると uncertainty 増 → **遠方を 0 (= last_known continuation) に近づける** 物理的事前。`tau` は grid search でチューニング、典型 25-100 ft。
- **Savitzky-Golay**: 隣接予測同士の HF jitter を物理的に滑らか化 (= TVT は連続関数のはず)。

---

## 7. 移植 map (= 我々の repo への取り込み計画)

| # | 要素 | 移植先 | 推定工数 | 期待 LB 改善 (現 baseline = ?) | 依存 | 元 file:line |
|---|---|---|---|---|---|---|
| **A** | **FormationPlaneKNN + DenseANCCImputer** | `src/rogii/imputers.py` (新規) | **0.5 日** | -1.0 (この formula が無ければ LB 12 帯入りすら不可) | scipy `cKDTree`, numpy lstsq | [R:319-394](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py) |
| **B** | **6-formation per-well b_well 計算 + tvt_formula features** (median + WLS) | `src/rogii/features.py` (拡張) | **0.5 日** | -0.5 (formula が GBM に渡って初めて effective) | A 必須 | [R:483-503](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py) + WLS [R:181-186](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py) |
| **C** | **Self-NCC multi-scale (3 windows + score)** | `src/rogii/self_ncc.py` (新規) | **1 日** | -0.3 〜 -0.5 | numpy 純 | [R:188-209](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py) |
| **D** | **Beam Search (NumPy 5 configs 版を先に)** | `src/rogii/beam.py` (新規) | **1.5 日** | -0.5 | numpy 純 | [N:138-163](../../_research_kernels/needless090__score-10-081-score-lb-32-rank/score-10-081-score-lb-32-rank.py) |
| **E** | **Numba JIT Beam (7 configs, ±2 delta)** | `src/rogii/beam_jit.py` (D を高速化) | **1 日** | -0.1 〜 -0.2 (LB ではなく runtime 改善が主) | numba | [R:108-144](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py) |
| **F** | **Particle Filter (ANCC + Z 両方)** | `src/rogii/pf.py` (新規) | **2 日** | -0.4 〜 -0.6 | scipy interp1d, numpy | [R:223-313](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py) |
| **G** | **xcorr_tvt_features** (tasmim 流、戦略 doc B-1) | `src/rogii/xcorr.py` (新規) | **0.5 日** | -0.5 〜 -0.8 (戦略 doc 推定値) | numpy 純 | [T:236-263](../../_research_kernels/tasmim__lb-11-068-rogii-wellbore-geology-prediction/lb-11-068-rogii-wellbore-geology-prediction.py) |
| **H** | **GR detrend residual** (R 固有) | `src/rogii/features.py` (1 関数追加) | **0.25 日** | -0.2 〜 -0.4 | numpy polyfit | [R:211-216](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py) |
| **I** | **Affine GR calibration** | `src/rogii/features.py` (1 関数追加) | **0.25 日** | -0.1 〜 -0.2 | numpy polyfit | [R:174-179](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py) |
| **J** | **tw_diff multi-anchor families (4 families)** | `src/rogii/features.py` (拡張) | **0.5 日** | -0.2 (D/F 後で初めて effective) | D, F | [R:619-623](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py) |
| **K** | **LGB×3 diverse-lr + CB stacking** | `src/rogii/models/gbm.py` (kagglib 経由) | **1 日** | -0.2 (既存 baseline ある前提) | kagglib | [R:78-99 + 666-717](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py) |
| **L** | **Ridge stack (positive=True, no intercept)** | `kagglib.stacking` 既存利用 | **0.25 日** | -0.05 | sklearn Ridge | [R:706-716](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py) |
| **M** | **Post-proc: alpha × tau × w_pf grid + Savitzky-Golay** | `src/rogii/postproc.py` (新規) | **0.5 日** | -0.1 〜 -0.3 | scipy savgol_filter | [R:725-752](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py) |
| **N** | **TabICL ×2 (4096 / 8192 ctx)** | `src/rogii/models/tabicl.py` (新規、optional) | **1.5 日** | -0.1 〜 -0.3 (N の 8 base 構成での貢献) | tabicl wheel + ckpt as private dataset | [N:644-732](../../_research_kernels/needless090__score-10-081-score-lb-32-rank/score-10-081-score-lb-32-rank.py) |

**工数合計**: A〜M で **10.5 日** (= 約 2 週)。N (TabICL) を入れると **12 日**。

**依存グラフ**:
```
A (Imputers) ─┬─→ B (b_well features) ─┐
              │                          ├→ J (tw_diff) ─┐
D (Beam) ──┬──┤                          │                │
           └──┴→ E (Numba Beam)          │                │
F (PF) ────────┴→ J (tw_diff)            │                │
C (Self-NCC) ─────────────────────────────┤                │
G (xcorr) ────────────────────────────────┤                │
H (GR detrend) ───────────────────────────┤                │
I (Affine cal) ───────────────────────────┤                │
                                          ↓                ↓
                            K (LGB×3 + CB) → L (Ridge stack) → M (post-proc) → submission
                            N (TabICL) ─────┘
```

A → D → F が **critical path** (= 同時並行で進めると 2.5 日で「Beam + PF + Plane-fit」最低限に到達)。残り C/G/H/I/J/K/L/M を 2 週目で積み上げ。

---

## 8. 注意点 / 落とし穴

### 8.1 leak

- **`FormationPlaneKNN.impute(self_wid=...)`** で **train 時に自分自身を除外する** ([R:341](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py))。`is_train=False` 時は `self_wid=None` でフル参照。**移植時に self exclusion を絶対に省略しない** (= leak で OOF 異常に良くなる、LB 乖離発生)。
- **`DenseANCCImputer`** も同様 ([R:386](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py))。

### 8.2 数値安定性

- **`A[:,i,i] += 1e-9`** ([R:351](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py)) を入れないと plane-fit が特異になる well 出現 → `linalg.solve` で例外。except 内で `pinv` fallback あり ([R:354-358](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py)) を必ず移植。
- **PF likelihood `lk = np.maximum(lk, 1e-300)`** ([R:250](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py)) を入れないと particle 全死で `ws=0` 除算。
- **`np.clip(pos, tmin-50, tmax+50)`** ([R:247](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py)) で typewell range 外を許す (50 ft buffer) — buffer 削ると tail particle 切られて bias。

### 8.3 メモリ

- **NCC matrix `Hn @ Cn.T / win`** = (nh, M) の ndarray。1 well の hidden が 1000 ft で stride=3, known 1000 ft なら M ≈ 333 → 1000 × 333 = 333K float32 = 1.3 MB / well。773 wells で 1 GB 程度 (joblib threads で分散すれば OK)。
- **Beam Search の `hI/hP` 2D arrays** = (n_steps, beam_size) ([R:113](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py)) でわずか。
- **PF 全 particle の per-step storage は不要** (mean/std だけ pickle)。

### 8.4 計算量サマリ

| 要素 | 1 well あたり (典型 hidden=600 ft) | 773 wells × 4 thread |
|---|---|---|
| FormationPlaneKNN.impute | <100ms (3x3 lstsq vectorized) | ~20s (分散) |
| Self-NCC multi-scale (3 win) | 200-500ms | 1-2 min |
| Beam (1 config, NumPy) | 500ms-1s | 2-4 min |
| Beam (1 config, Numba JIT) | 50-200ms (warm 後) | 30s-1 min |
| ANCC PF (N=500) | 500ms-1s | 2-4 min |
| Z PF (N=500) | 500ms-1s | 2-4 min |
| **合計 (全 features)** | **3-5s** | **15-30 min on Kaggle T4×2** |

### 8.5 Joblib threads が効く理由

両 kernel とも `prefer='threads'` ([R:639](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py))。理由:
- imputer (FI/DI) を **process 間 pickle するとコスト高** (Dense の `cKDTree` + 60 pts/well × 773 wells = メモリ大)
- threads は GIL に縛られるが **NumPy/scipy 関数は GIL release** で実効並列
- numba `@njit` も GIL release ⇒ thread-friendly

### 8.6 移植時の hidden bug

- **`hw['GR'].rolling(...).iloc[hw.index.get_loc(idx)]`** ([R:294](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py)) — `hw.index` が unique でないと爆死。pandas DataFrame の index は CSV load 後に必ず `reset_index(drop=True)` で連番化を確認。
- **`np.interp` の `tw_tvt` は sorted 必須**。`tw = pd.read_csv(...).sort_values('TVT')` ([R:416](../../_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py)) を必ず移植。
- **TabICL の wheel + ckpt は private dataset 経由** ([N:644-666](../../_research_kernels/needless090__score-10-081-score-lb-32-rank/score-10-081-score-lb-32-rank.py))。Internet disabled なので。我々の repo で導入するなら `~/projects/kaggle/_assets/tabicl/` に保管 → Kaggle datasets push 必要。

---

## 9. ライセンスと再配布性

### 9.1 Kaggle public kernel の license

両 kernel の `kernel-metadata.json` に license tag 欠落 ⇒ Kaggle Terms 8.B により **default は Apache 2.0 相当の "non-exclusive license to use the Notebook for the Competition"** (出典: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/rules)。

### 9.2 我々の取り扱い

- **そのまま fork して提出するのは禁止** (= 各 kernel 著者の同意が必要)
- **アルゴリズムの copy + 自分の repo で再実装は OK** (アルゴリズム自体は copyrightable でない)
- **コード snippet を docs/research/ に引用する場合は必ず出典 URL + 著者名を残す** ⇒ 本 doc では §出典 で URL + file:line を記載済み
- **再実装 commit message に "inspired by [author]/[notebook]" を明記**

### 9.3 TabICL について

`tabicl` package は **Apache 2.0** (出典: https://github.com/soda-inria/tabicl LICENSE)。Kaggle private dataset 経由で持ち込み可能 (両 kernel が既に実証)。

---

## 10. 戦略 doc Phase 1.6 §B 優先順位 update 提案 (= トレードオフ表のみ)

> **CRITICAL: 推奨案ではなく「選択軸」のみ提示する。判断は開発者に委ねる**。

現行 (`docs/strategy/winning-strategy.dense.md` line 19-29):

```
| 1 | xcorr_tvt_features              | -0.5〜0.8  | exp003 |
| 2 | wls_b_well + multi-scale SC     | -0.3〜0.5  | exp003 |
| 3 | gr_detrend_resid                | -0.2〜0.4  | exp003 |
| 4 | 7 beam configs + Numba JIT      | -0.2〜0.4  | exp004 |
| 5 | Sequence Transformer ...        | -0.3〜1.0  | Phase 4 |
...
```

### 10.1 update 候補 1: 「核心 6 要素を最優先化」

戦略 doc の現行は B-1=xcorr / B-2=wls / B-3=gr_detrend / B-4=Beam が並んでいるが、**FormationPlaneKNN + b_well + Beam + PF が無いと LB 12 帯すら不可** (= R/N が 10.0 帯に居るのは 6 要素全部入ってるから)。我々の現状 baseline (= exp001 で RMSE 30+ failed) はそもそも formula を持ってない可能性大 → §7 移植 map の **A/B (Imputer + b_well features)** を **「絶対外せない 3 原則」相当** に格上げ。

| 観点 | 現行 (xcorr 1 位) | update 候補 1 (Imputer + b_well 1 位) |
|---|---|---|
| LB 12 帯到達速度 | 遅い (xcorr 単独では formula がないと LB 25 帯) | **早い** (A+B+D で LB 12 入り) |
| 公開上位の形と一致 | 部分的 | **完全一致** (R/N どちらも A+B+D+F が core) |
| 我々の独自性 | 高 (xcorr は tasmim 由来で R/N にない) | 中 (Plane-fit は誰でもやる) |
| 工数 | 0.5 日 (G) | 1.5 日 (A+B+D 並列で 1 日でも可能) |

### 10.2 update 候補 2: 「Imputer + b_well を Phase 1.6 §A "絶対外せない" に追加」

現行 §A:
1. target = TVT - last_known_TVT (residual) ★
2. tvt_formula = -Z + ANCC + b_well を features に ★
3. typewell matching を model に教える ★

§A.2 は formula を入れろ と書いてあるが **「ANCC を test では plane-fit imputer で imput」「b_well は per-formation × per-well で計算 (median + WLS + 末尾 50)」までは記述なし**。これを §A.2 に追加で「(= FormationPlaneKNN + DenseANCCImputer + 6-formation b_well 3 種)」と注記すべき。

| 観点 | 現行 §A.2 | update 候補 2 (具体化) |
|---|---|---|
| 実装ガイド明確性 | "feature に入れろ" だけで具体実装が抽象 | "Imputer + b_well 3 種 features" まで具体化 |
| 新規メンバー (= 我々の future Claude session) の迷い | 大きい | 小さい |
| 戦略 doc の length 増 | 0 | +3 行 |

### 10.3 update 候補 3: 「公開 baseline LB との差を § D に明記」

現行 §D で LB 一覧があるが、**我々の現 exp001 LB が RMSE 30+ で公開 baseline (12.602) からも大幅に遠い** ことを明示すると、Phase 2 GO 前に「core 6 要素を入れずに submit するな」 hard constraint が立つ。

### 10.4 トレードオフ表 (update する / しない)

| 軸 | 現行のまま | 候補 1+2+3 を全部 apply | 候補 1 のみ |
|---|---|---|---|
| 戦略の方向修正 | 0 | 大 | 中 |
| 既存 narrative との整合性 | 完璧 | 一部 break (xcorr 1 位降格) | 一部 break |
| 我々の現状 (exp001 failed) との整合 | 弱い (xcorr で勝てる空気感が強い) | 強い (まず formula 入れろ という流れ) | 強い |
| update 工数 (doc 編集) | 0 min | 15-30 min | 5-10 min |
| 開発者 (= ユーザー) の re-review 必要 | No | Yes | Yes |

### 10.5 選択軸

- **公開 6 要素にどれだけ忠実に再現するか** (高 = 候補 1+2+3 / 低 = 現行)
- **B-1 xcorr (= tasmim 流の独自 edge) を最優先に置くか** (はい = 現行 / いいえ = 候補 1)
- **戦略 doc を hard constraint (= core 入れずに submit 禁止) として使うか** (はい = 候補 3 / いいえ = 現行)

判断は開発者に委ねる。

---

## 付録: 残り 4 件の kernel との位置関係 (補助参照)

| kernel | 6 要素のうち持つもの | 独自 edge | 我々が拾うべきもの |
|---|---|---|---|
| K (karnakbaev, LB 10.784) | A/B/C (1 win) /D (5)/F/I/L/M | XGB を base に追加 (5 base stack) | XGB 追加 base としての構成 (移植 map K の variant) |
| T (tasmim, LB 11.068) | A/B/D (2)/L | **xcorr_tvt_features** (= G), tw_boundary_dist | G (移植 map に含む) |
| `konbu17` 11.912 | A/B/D (4) | XGB 追加 + KNN baseline | (低優先) |
| `pilkwang` (LB 不明) | -- | super stack 設計 | meta strategy のみ |
| `shinyanagai 12.388 / triple-signal` | D 系 | triple beam + dual PF | LB 帯低いので主参照は R/N |

> **結論**: §1-9 の 6 要素 + §7 移植 map A〜M (12 種) で **R/N の上位 LB 帯を再現** + tasmim G で **独自 edge** を確保できる。N (TabICL) は optional だが 8 base stack 完成版を狙うなら必要。
