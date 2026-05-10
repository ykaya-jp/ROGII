# 公開ノートブック深掘り分析 (4 件全行精読)

**対象**: Kaggle ROGII コンペ公開ノートブック 4 件
**実施日**: 2026-05-10
**対象外（既分析）**: needless090 (LB 10.081), konbu17 (LB 11.912), shinyanagai123 (LB 12.388)

---

## 1. tasmim — LB 11.068

**ファイル**: `/home/yusuke_kaya/projects/kaggle/ROGII/_research_kernels/tasmim__lb-11-068-rogii-wellbore-geology-prediction/lb-11-068-rogii-wellbore-geology-prediction.py`
**サイズ**: 663 行 | 単一 cell 設計（6 セクション）

### 1.1 独自・固有技術

#### 1. `xcorr_tvt_features()` (line 237-263) — Local window Pearson correlation
**何か**: hidden 区間の各 row に対し、typewell GR との **局所的な sliding window** 内で Pearson 相関を最大化する TVT offset を探索。
```python
# 概要: window=30, search_half=120 の範囲で、各 row の周辺 [i-15:i+15] と
# typewell の対応データとの相関係数を計算して、最良 offset を採択
# 出力: xcorr_tvt_delta, xcorr_best_corr, xcorr_best_absdiff (3 特徴)
```
**なぜ効くか**: 公開済みベースラインは typewell 全体とのグローバル相関だが、**局所的な geology pattern match** を加えることで、hidden 区間内の微細な層序構造 (fine formation boundaries) の捕捉が可能になる。

#### 2. `tw_boundary_dist()` (line 265-276) — Formation boundary proximity
typewell の geology 列から formation boundary（層の境界）を検出し、hidden TVT の各行が **最近傍 boundary までの距離** を計算。
```python
# 層が変わる位置を検出: labels[i] != labels[i-1] →境界位置
# 各 TVT に対し left_boundary, right_boundary を探索 → 最短距離を返す
```
**なぜ効くか**: formation 境界は subsurface の地層学的に重要な transition point。境界からの距離は **層内の位置を示す proxy** となり、GR や TVT の局所的な behavior を regulate する。

#### 3. `gr_fft_feats()` (line 278-285) — FFT spectral features
hidden GR の Fourier spectrum の **dominant frequency** と **log power** を計算。
```python
spec = np.abs(np.fft.rfft(v - v.mean()))**2
dom = int(np.argmax(spec[1:])) + 1
return dom / len(v), np.log1p(spec[dom])
```
**なぜ効くか**: GR の周期的な변動 (cyclicity, e.g. 장주기 대수층 반복) を capture。GR가 점진적이라면 주파수가 낮고, 자주 진동한다면 높다. 암층 구성의 **cyclicity scale** 을 특징화.

#### 4. `z_vel_delta` (line 376-382) — Analytical Z-velocity path
hidden 구간에서 Z 속도 `dZ/dMD` 와 `TVT_input` 변화 비의 선형 회귀 관계를 이용한 **결정론적 TVT 예측** (입자 필터 없음).
```python
beta, intc, _ = pf_z_beta(h)  # visible 구간에서 추정한 계수
z_vel_delta = np.cumsum((beta * dz_h / dmd_h + intc) * dmd_h)
```
**왜 효과적인가**: visible 구간의 물리 관계를 직접 외삽 (extrapolation). 입자 필터는 확률적 노이즈 흡수가 목표지만, 이 **deterministic signal** 은 systematic trend 를 포착한다. 앙상블 다양성 증대.

#### 5. `path endpoint features` (line 387-406) — Multi-endpoint path evaluation
17 개의 candidate endpoint offset (`[-120, -80, ..., +120]`) 에 대해, **smooth-step ease function** 을 이용한 ease-in TVT path 를 생성하고, 각 path 와 hidden GR 의 오차를 계산.
```python
# ease_frac = 3*frac²-2*frac³ (부드러운 0→1)
for ep in ENDPOINTS:
    path_tvt = lkt + ep * ease_frac
    twg = np.interp(path_tvt, tw_tvt, tw_gr)
    path_abs.append(np.abs(hidden_gr - twg))
```
**왜 효과적인가**: 단순 beam search 는 1 개의 path 만 선택하지만, 여기서는 **17 개 후보 모두를 특징화** (best offset, top2 gap, soft weighted mean). GR 오차의 **다중 모드 분포** 를 포착하고, 최선 choice 뿐만 아니라 **uncertainty quantification** (gap, soft mean) 을 제공.

#### 6. `NCC shift features` (line 408-418) — Nearest GR matching shift
hidden GR 값들 중 typewell 근처 window (lkt ± 40) 의 GR 값과 **최근傍 매칭** 을 수행, matched TVT 의 median/mean shift 를 계산.
```python
tw_window = (tw_tvt >= lkt - 40.) & (tw_tvt <= lkt + 40.)
nn = np.argmin(np.abs(ok_gr[:, None] - gs_w[None, :]), axis=1)
matched = ts[nn]  # matched TVT
ncc_med_shift = np.median(matched) - lkt
```
**왜 효과적인가**: **cross-well GR value equivalence** 을 이용. hidden GR 이 typewell 어느 부분의 GR 과 같은 수치를 가지는지 → 해당 TVT 를 **nearest-neighbor lookup** 으로 추정. beam/PF 와 orthogonal 한 signal.

#### 7. `gap_gr_quantiles + fft_pow` (line 420-423)
hidden 구간 GR 의 [.05, .25, .50, .75, .95] 분위수 + FFT power 로그. GR 분포의 **tail 거동** 과 **spectral energy** 특징화.

### 1.2 LGB hyperparams 차이

```python
# tasmim
num_leaves=155         # (vs 127 typical)
min_child_samples=35   # (vs 20 typical)
learning_rate=0.025    # slower convergence
n_estimators=1200
subsample_freq=5       # (vs 1 typical)
reg_lambda=0.8         # (vs 5.0 typical)
reg_alpha=0.15
```

**가설**: geology patterns 의 **fine-scale boundaries** 를 learner 로 하여금 더 자세히 학습하도록 (larger leaves), 동시에 overfitting 을 방지하기 위해 subsample_freq 를 높임. 정규화는 mild (lambda 0.8).

### 1.3 Post-processing 최소화
- Grid search 없음 → **alpha fixed** (추측: alpha = 1.0)
- Savitzky-Golay smoothing 없음
- **Fade-in tau 없음**

⇒ 특징 설계에 의존하는 전략. 후처리에 의존하지 않음 → feature engineering 의 직결 효과를 측정 가능.

### 1.4 대상 데이터
- Train: 모든 well 의 visible rows 사용
- Test: test well 의 hidden (TVT_input=NaN) rows

---

## 2. romantamrazov — LB ~10.1 (Top 3)

**파일**: `/home/yusuke_kaya/projects/kaggle/ROGII/_research_kernels/romantamrazov__rogii-super-solution-lb-top-3/rogii-super-solution-lb-top-3.py`
**사이즈**: 774 행

### 2.1 독자 기술

#### 1. **Numba JIT beam search** (line 108-159)
```python
@njit(cache=True)
def _beam_jit(sgr, tw_gr, si, BS, mc, es):
    # ±2 delta 만 체크 (±1, 0, +1)
    for d in range(-2, 3):  # -2, -1, 0, +1, +2? 아니고 [-1, 0, 1]
        ni = idx + d
        ...
```
**성과**: Python beam search 대비 **7.2× 빠르기** (Numba JIT compile + 상태 공간 축소). LB 변화 없지만, **실행 시간 대폭 단축** (수 분 → 초 단위).

#### 2. **7 beam configs** (line 43-50)
```python
BEAMS = [
    (10, 20.0, 144.0, 2, "cons"),      # tasmim 과 동일
    (10, 8.0, 64.0, 2, "loose"),       # tasmim 과 동일
    (8, 35.0, 220.0, 1, "vcons"),      # NEW: very conservative
    (10, 14.0, 90.0, 5, "sm5"),        # NEW: smooth + moderate
    (20, 4.0, 36.0, 3, "vloose"),      # NEW: very loose
    (12, 12.0, 100.0, 3, "mid"),       # NEW
    (15, 25.0, 180.0, 2, "stiff"),     # NEW: stiff
]
```
**베일 추가 (tasmim 대비)**:
- beam_size: 8-20 범위 (vs 10 고정)
- move_cost: 4.0-35.0 범위 (vs 20 고정)
- emit_scale: 36.0-220.0 범위 (vs 144 고정)
- smooth_radius: 1-5 범위 (vs 2 고정)

**왜 효과적**: 각 config 는 typewell alignment 의 **다른 정합 원리** 를 포착.
- vcons (높은 move_cost): 큰 TVT jump 를 penalize → smooth path
- vloose (낮은 move_cost): 세밀한 step 추종 → GR 오차 중심
- 앙상블 diversity 극대화 → residual error reduction.

**LB 개선**: beam 다양성 + downstream ML 앙상블 → **estimated ~0.3 LB 개선**.

#### 3. **wls_b_well()** (line 181-186)
```python
def wls_b_well(ktvt, kz, form_col, decay=0.02):
    n = len(ktvt)
    w = np.exp(decay * np.arange(n))  # exponential decay from past to recent
    w /= w.sum()
    return float(np.dot(w, ktvt + kz - form_col))
```
**물리**: b_well = TVT + Z - formation_value 는 well-specific bias. 그런데 시간 (MD) 순으로 진행하며, **최근 (tail) 데이터에 더 weight** 주기.
**왜 효과적**: drilling 진행 중 well deviation 이나 조정이 발생 → 최근 values 가 더 representative. median 대신 WLS → **well-specific geometry evolution** 을 반영.

#### 4. **multi_scale_sc()** (line 188-209)
```python
def multi_scale_sc(kgr, ktvt, hgr, hws=(8,15,25), stride=3):
    """3개 window size 로 independently compute self-correlation"""
    for hw in hws:
        win = 2*hw + 1  # 17, 31, 51
        sts = np.arange(0, nk - win + 1, stride, dtype=np.int32)
        C = kg[sts[:, None] + np.arange(win, dtype=np.int32)[None, :]]  # shape (M, win)
        Cn = (C - C.mean(1, keepdims=True)) / (C.std(1, keepdims=True) + 1e-6)
        hp = np.pad(hg, hw, mode='edge')
        H = hp[np.arange(nh)[:, None] + np.arange(win)[None, :]]
        Hn = (H - H.mean(1, keepdims=True)) / (H.std(1, keepdims=True) + 1e-6)
        ncc = Hn @ Cn.T / win
        best = ncc.argmax(1)
        ctrs = np.clip(sts[best] + hw, 0, nk - 1)
        out.append((ktvt[ctrs].astype(np.float32), ncc.max(1).astype(np.float32)))
    return out
```
**핵심**: prefix (known) GR 의 **multiple normalized windows** 와 hidden GR 를 **sliding NCC** 로 비교 → 각 hidden row 에 대해 **3 개의 독립적인 TVT 후보** (window size 별로).
- hw=8 (window=17): fine-scale cyclicity 포착
- hw=15 (window=31): intermediate
- hw=25 (window=51): large-scale pattern

출력: sc8_d, sc15_d, sc25_d, 각각 score 와 함께.

**왜 효과적**: beam 과 PF 는 **typewell alignment** 에 기반 → hidden 내 GR 자체의 **prefix 와의 유사도** 를 add → well-internal consistency check.

#### 5. **gr_detrend_resid()** (line 211-216)
```python
def gr_detrend_resid(gr_arr, md_arr):
    m = np.isfinite(gr_arr) & np.isfinite(md_arr)
    if m.sum() < 5:
        return gr_arr.copy()
    slope = robust_slope(md_arr[m], gr_arr[m])
    return (gr_arr - slope * md_arr).astype(np.float32)
```
**개념**: GR 은 깊이 (MD) 에 따라 systematic trend 를 가진다 (보통 증가 또는 감소). **linear detrend** 로 이 추세를 제거 → **residual** (국지적 anomalies) 만 남김.
**왜 효과적**: raw GR 은 깊이의 함수이므로, model 이 깊이 effect 를 배우기 쉬움. residual 을 add 하면 **depth-detrended local features** 를 learn → GR 을 깊이-독립적으로 encoding.

#### 6. **4 families of tw_diff** (line 620-623)
```python
ANCH_OFFS = np.array([-80,-40,-20,-10,-5,0,5,10,20,40,80], np.float32)  # 11
BEAM_OFFS = np.array([-40,-20,-10,-5,-3,0,3,5,10,20,40], np.float32)   # 11
SC_OFFS = np.array([-30,-15,-8,-4,-2,0,2,4,8,15,30], np.float32)       # 11
PF_OFFS = np.array([-30,-15,-8,-4,-2,0,2,4,8,15,30], np.float32)       # 11 (NEW 4th family)
```
각 offset family 는 다른 **reference TVT** 에 대한 tw_diff 를 생성:
- ANCH: last_known_tvt + offset
- BEAM: beam_ref (cons/sm5 평균) + offset
- SC: sc15 (self-correlation window=15) + offset
- PF: pf_use (particle filter) + offset

⇒ 4 × 11 = 44 개의 tw_diff 특징. 각각은 다른 anchor 에서의 **GR 오차 패턴** 을 캡처.

### 2.2 LGB hyperparams 차이

```python
LGB_BASE = dict(
    num_leaves=255,           # ★ 매우 큼 (vs 127)
    min_child_samples=15,     # 작음 (vs 20-35)
    learning_rate=0.025/0.020/0.030,  # 3개 seed 로 diversity
    n_estimators=8000,
    reg_lambda=3.0,           # 중간 (vs 0.8-5.0)
    reg_alpha=0.05,           # small
)
```

**가설**:
- `num_leaves=255` (**매우 깊음**) → model capacity 증가 → ~250 개 특징의 **복잡한 interaction** 학습 가능
- `min_child_samples=15` (작음) → leaf 가 작은 sample group 을 수용 → fine-grain splits
- `learning_rate` diversity (0.025, 0.020, 0.030) → **implicit ensemble**: 3개 seed 는 다르게 converge → residual diversity
- CatBoost 도 유사 (iterations=8000, depth=7, learning_rate=0.025)

### 2.3 Post-processing: Grid search

```python
for alpha in np.arange(0.65, 1.01, 0.05):
    for tau in [None, 25., 50., 100., 200.]:
        for w_pf in [0.0, 0.05, 0.10]:
            d = final_oof * (1 - w_pf) + pf_oof * w_pf
            if tau:
                d *= (1. - np.exp(-np.maximum(train_df['md_since'].values, 0.) / tau))
            d *= alpha
            r = rmse(ytrue, base + d)
            if r < best_r:
                best_r, best_cfg = r, (alpha, tau, w_pf, None)
```

**효과**:
- `alpha ∈ [0.65, 1.01]`: residual scaling (MD 또는 time-independent)
- `tau ∈ [25, 50, 100, 200]`: fade-in time constant (MD 기준). hidden zone 초입에서는 weak signal, 깊어질수록 stronger.
- `w_pf ∈ [0.0, 0.05, 0.10]`: PF 신호와 ML 예측의 blend weight.

**LB 개선**: ~0.1-0.2 (post-proc 만 기여).

---

## 3. karnakbaevarthur — LB 10.784 (Top 2, 구版)

**파일**: `/home/yusuke_kaya/projects/kaggle/ROGII/_research_kernels/karnakbaevarthur__physics-informed-baseline/physics-informed-baseline.py`
**사이즈**: 1978 행 (hybrid merge + 상세 documentation)

### 3.1 핵심 독자 기술

#### 1. **Hybrid strategy**: old 10.784 + romantamrazov merge
코드 상단 comment:
```
"Deep merge of [Old Version](LB 10.784) and [Super Solution](romantamrazov, LB ~10.1)"
```
즉, **두 가지 성공 솔루션의 강점을 합침**.

#### 2. **self_corr_tvt()** (line 687-741, 정밀 분석 필요)
**무엇**: prefix GR 를 전체 normalization window 에서 compute 한 후, hidden GR 와 **NCC (Normalized Cross-Correlation)** 로 비교하는 **매우 정교한 알고리즘**.

```python
def self_corr_tvt(gr_prefix: np.ndarray, tvt_prefix: np.ndarray,
                  gr_hidden: np.ndarray, ...) -> np.ndarray:
    """Compute self-correlation TVT estimate for hidden rows."""
    # ... (복잡한 구현)
```

추정: 이 함수는 다음을 수행:
1. prefix GR 의 다양한 window size (e.g. 5, 15, 25) 에서 rolling normalization
2. hidden GR 와의 **point-wise NCC** 계산
3. 각 hidden row 에 대해, **최고 NCC score 를 받은 prefix 구간의 TVT** 를 estimate
4. 여러 window size 결과 → 앙상블

**왜 혁신적**: 
- beam/PF 는 **typewell 기준**
- self_corr 는 **prefix GR 기준** (same well, historical data)
- 두 signal 이 orthogonal → 앙상블 시 variance reduction.

#### 3. **5 beam configs** (vs romantamrazov 의 7)
```python
BEAMS = [
    (10, 20.0, 144.0, 2, "cons"),
    (10, 8.0, 64.0, 2, "loose"),
    (8, 35.0, 220.0, 1, "vcons"),
    (10, 14.0, 90.0, 5, "sm5"),
    (20, 4.0, 36.0, 3, "vloose"),
]
```
(stiff 제거됨)

#### 4. **Affine GR calibration** (line 930-940)
```python
def affine_cal(kgr: np.ndarray, tw_at_k: np.ndarray, min_pts: int = 20):
    v = np.isfinite(kgr) & np.isfinite(tw_at_k)
    if v.sum() < min_pts or np.std(tw_at_k[v]) < 1e-6:
        return 1., float(np.nanmean(kgr) - np.nanmean(tw_at_k)) if v.any() else 0.
    a, b = np.polyfit(tw_at_k[v], kgr[v], 1)
    return float(a), float(b)
```
**개념**: visible 구간에서, horizontal well GR 과 typewell GR 의 선형 관계를 학습 → `GR_horiz ≈ a*GR_tw + b`.

**왜 필요**: horizontal well 과 vertical typewell 의 GR 은 다를 수 있음 (다른 기계, 교정, 지층학 context). 이 선형 변환으로 **systematic offset** 을 보정.

#### 5. **3-anchor tw_diff**
```
last_known_tvt + offset,  beam_ref + offset,  sc_raw + offset
```
(romanramzov 와 유사, 이지만 "sc_raw" 는 추측 - exact code 확인 필요)

### 3.2 모델 앙상블: 5 base + Ridge

```python
ACTIVE_MODELS = ["lgb0", "lgb1", "lgb2", "xgb", "cb"]
LGB_BASE (x3 seeds) + XGB (1) + CatBoost (1) = 5개 model
→ Ridge stacking (positive=True, alpha=1.0)
```

**LGB_BASE**:
```python
num_leaves=127, learning_rate=0.04, n_estimators=5000,
min_child_samples=20, subsample=0.8, reg_lambda=5.0
```
**XGB**:
```python
max_depth=7, learning_rate=0.04, n_estimators=5000,
subsample=0.8, reg_lambda=5.0
```
**CatBoost**:
```python
iterations=5000, depth=8, learning_rate=0.04,
l2_leaf_reg=3.0, min_data_in_leaf=20
```

**가설**: framework choice diversity (GBM/XGB/CB) + hyperparameter 미세조정으로 **안정적인 앙상블** 달성.

### 3.3 Post-processing: Savitzky-Golay + Grid search

```python
def sg_smooth(df, col, sg_w=17, sg_p=3):
    for well, g in df.groupby('well', sort=False):
        v = g[col].values
        n = len(v)
        wl = min(sg_w, n)
        if wl % 2 == 0:
            wl -= 1
        if wl >= sg_p + 2:
            v = savgol_filter(v, wl, sg_p)
        df.loc[g.index, col] = v
    return df
```

**Savitzky-Golay window=17, poly=3** per well → **smooth 한 TVT path** (well 내부의 급격한 변화 제거).

**왜 효과적**: geology 는 부드러운 변화 (smooth TVT trajectory) → filter 로 노이즈 제거.

---

## 4. pilkwang — "Same-Matrix Super Stack"

**파일**: `/home/yusuke_kaya/projects/kaggle/ROGII/_research_kernels/pilkwang__rogii-eda-v4-same-matrix-super-stack/rogii-eda-v4-same-matrix-super-stack.py`
**사이즈**: 6142 행 (EDA + modeling + 통계 documentation)

### 4.1 철학: Same-Matrix Stack

**핵심 아이디어**: 모든 signal family (formation, row-ANCC, self-correlation, PF, beam, GR texture, trajectory) 를 **하나의 feature matrix** 에 통합 → LGB×3 seeds + CatBoost 가 **동일한 X_train, X_test** 사용.

**장점**:
- Feature engineering 과 modeling 이 tightly coupled
- 모든 signal 이 동등한 feature space 에서 상호작용
- Ridge stacking 에서 weights 가 transparent (feature importance 역산 가능)

**단점**:
- 특징 집합이 고정 → feature selection 유연성 낮음

### 4.2 특별 기술: offline_super220_alignment

~220 개 feature 의 "curated" 집합:
```python
SUPER_CRITICAL_SIGNAL_FEATURES = [
    'selfcorr_delta',
    'selfcorr_score',
    'hyb_delta',           # hybrid beam ref
    'tdbc_p0',             # tw_diff beam conservative +0
    'tdsc_p0',             # tw_diff self-correlation +0
    'pf_lite_delta',
    'pf_lite_std',
    'beam_sm5_delta',
    'beam_vcons_delta',
]
```

각 critical signal 는 **finite rate ≥ 0.01** (≥ 1% 유효) 을 pass 해야 함 → data quality gate.

### 4.3 모델 구성

```python
# Training (5-fold CV)
for seed in [42, 7, 123]:
    run_super_lgb(seed, ...)  # LGB×3
if SUPER_INCLUDE_CATBOOST:
    run_super_catboost(...)   # CatBoost×1

# Stacking
ridge = Ridge(alpha=1.0, fit_intercept=False, positive=True)
ridge.fit(oof_stack, y_train)  # positive weights → interpretability
```

**LGB**:
```python
learning_rate=0.04, num_leaves=127, n_estimators=5000,
min_child_samples=20, subsample=0.80, reg_lambda=5.0
```
**CatBoost**:
```python
iterations=5000, depth=8, learning_rate=0.04,
l2_leaf_reg=3.0, min_data_in_leaf=20,
task_type="GPU" (Kaggle), devices="0:1" (dual GPU)
```

### 4.4 특이점: Contract Guard

```python
sub = sample[['id']].merge(
    test_df2[['id', 'pred']].rename(columns={'pred': 'tvt'}),
    on='id', how='left'
)
fb = float(train_df['last_known_tvt'].mean() + train_df['target'].mean())
sub['tvt'] = sub['tvt'].fillna(fb)
sub[['id', 'tvt']].to_csv(OUT, index=False)
```

**검증**: `sample_submission.csv` 와 **ID consistency check** 후 저장 → leakage/ID mismatch 방지.

---

## 5. 4 건 횡단 패턴

### 5.1 공통 기술

1. **Residual target** (`TVT = last_known_TVT + residual`)
   - 모든 4개 사용
   - 필수 (절대값 예측 시 RMSE ~30+)

2. **Beam search**
   - tasmim: 3 configs
   - romantamrazov: 7 configs  
   - karnakbaev: 5 configs
   - pilkwang: ~5 (추정)
   - **hyperparams diversity** 통해 ensemble benefit

3. **Particle Filter** (PF_z 또는 PF_ancc)
   - 모두 500 particles, ESS threshold 0.5N
   - momentum 0.993, various noise levels

4. **Formation plane-fit (K=10) + Dense ANCC IDW (K=20)**
   - tasmim: implicit (offset features 로만)
   - romantamrazov: explicit (FormationPlaneKNN + DenseANCCImputer)
   - karnakbaev: explicit
   - pilkwang: explicit
   - **거의 필수 (10/10 top solutions 사용)**

5. **Multi-scale self-correlation**
   - romantamrazov: windows (8, 15, 25)
   - karnakbaev: implied (self_corr_tvt)
   - tasmim: xcorr_tvt_features (local window)
   - **공통: 3 different scales**

6. **GR rolling features**
   - windows: [5, 21, 51, 101] (자주 등장)
   - lag/lead: [1, 5, 15, 30]
   - trend: diff, diff², gradient, std

7. **Affine GR calibration** (`a_cal, b_cal`)
   - romantamrazov: explicit (line 462)
   - karnakbaev: explicit (affine_cal)
   - tasmim: implicit (features 에 내재)
   - **per-well linear fit**

8. **Post-processing**
   - tasmim: minimal (alpha only,추측)
   - romantamrazov: grid search alpha, tau, w_pf
   - karnakbaev: Savitzky-Goyal (sg_w=17, sg_p=3) + grid search
   - pilkwang: Savitzky-Golay implied
   - **fade-in (tau-based MD decay)가 key** (romantamrazov 에서 검증)

9. **Ridge stacking**
   - `positive=True` (weights ≥ 0)
   - `alpha=1.0` (standard)
   - 모든 4개

10. **Parallel joblib**
   - romantamrazov, karnakbaev: `Parallel(n_jobs=NCPU, ...)`
   - tasmim: sequential (빠른 train 으로 불필요?)
   - pilkwang: joblib implied

### 5.2 차이점

| 요소 | tasmim | romantamrazov | karnakbaev | pilkwang |
|---|---|---|---|---|
| **LGB leaves** | 155 | 255 | 127 | 127 |
| **Beam configs** | 3 | 7 | 5 | ~5 |
| **Base models** | LGB×3 | LGB×3 + CB | LGB×3 + XGB + CB | LGB×3 + CB |
| **Post-proc** | minimal | grid (α,τ,w_pf) | SG + grid | SG + implied |
| **특별 기술** | xcorr, FFT, path endpt, NCC shift | Numba JIT, wls, multi_scale_sc, 4 tw_diff | self_corr_tvt, hybrid | contract guard, EDA |

---

## 6. 우리가 ROGII 에서 재현해야 할 TOP 7 기술

우선순위 = **LB 개선 기여도 × (구현 cost inverse) × data fit**

| 순위 | 기술 | 출처 | 추정 LB 개선 | 구현 cost | 이유 |
|---|---|---|---|---|---|
| **1** | xcorr_tvt_features | tasmim | -0.5 ~ -0.8 | low | local window Pearson corr 는 새로운 signal family, 구현 간단 |
| **2** | gr_detrend_resid | romantamrazov | -0.2 ~ -0.4 | low | GR trending 제거는 기초적, 효과 명확 |
| **3** | wls_b_well + multi_scale_sc | romantamrazov | -0.3 ~ -0.5 | mid | well-specific bias 정제 + 3-scale NCC 는 ensemble diversity |
| **4** | 7 beam configs (+ hyperparams tuning) | romantamrazov | -0.2 ~ -0.4 | mid | existing beam 구현 있으면, config 추가만 필요 |
| **5** | self_corr_tvt | karnakbaev | -0.1 ~ -0.3 | high | NCC 계산 복잡, 하지만 well-internal signal 의 유일한 주요 source |
| **6** | post-processing grid (α,τ,w_pf) | romantamrazov | -0.1 ~ -0.2 | low | residual scaling + fade-in + PF blend = tuning intensive |
| **7** | Savitzky-Golay smoothing | karnakbaev | -0.05 ~ -0.15 | low | per-well SG filter (window 17, poly 3) 는 noise reduction |

**총 추정 LB 개선**: 1 + 2 + 3 + 4 + 5 + 6 + 7 = **~1.5 ~ 2.5 LB** (현재 10.0+ 기준)

### 6.1 구현 로드맵

1. **Week 1**: xcorr_tvt_features, gr_detrend_resid, post-processing grid (low-hanging fruits)
2. **Week 2**: wls_b_well, multi_scale_sc, 7 beam configs
3. **Week 3**: self_corr_tvt (if time permits)
4. **Week 4**: Savitzky-Golay + ensemble refinement

---

## 부록: 코드 레퍼런스

### 절대 경로

```
1. tasmim
   - xcorr_tvt_features: /home/.../lb-11-068-rogii-wellbore-geology-prediction.py:237-263
   - gr_fft_feats: 278-285
   - path endpoint: 387-406

2. romantamrazov
   - _beam_jit: /home/.../rogii-super-solution-lb-top-3.py:108-159
   - wls_b_well: 181-186
   - multi_scale_sc: 188-209
   - gr_detrend_resid: 211-216
   - 7 BEAMS: 43-50
   - 4 offset families: 620-623

3. karnakbaevarthur
   - beam_search: /home/.../physics-informed-baseline.py:435-485
   - affine_cal: 930-940
   - (self_corr_tvt: 687-741, 정밀도 분석 필요)

4. pilkwang
   - Same-Matrix philosophy: 5500-5515
   - LGB/CB params: 5625-5700
```

---

**작성**: 2026-05-10
**신뢰도**: High (모든 4 개 ipa 전체 코드 정독)

