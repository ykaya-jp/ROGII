# exp009 — 案 E (Sparse GP for ANCC posterior) + Edge O (direction-aware Beam) (Phase 4 第 2 層、9 切り路線)

## 0. 状況 (2026-05-11)

- exp005 = LB **10.317** (Silver、自前 best)
- exp006 = LB **10.503** (TabICL 失敗、postmortem 済)
- exp007 = `feat/phase-3-exp007-edge-q-m` (Edge Q + Edge M + 自前 4 base、Kaggle 上で RUNNING、想定 LB 9.5-9.7)
- exp008 = `feat/phase-4-exp008-case-d-kalman` (= 案 D Kalman 7 features、push 直前)
- exp009 = 本 doc (= 案 E + Edge O、exp008 から派生、`feat/phase-4-exp009-case-e-edge-o`)
- 公開 LB Top 1 = 9.256、Gold cutoff = 9.919、賞金圏 (Top 4) = 9.415
- ユーザー指示「9 切らないと優勝は無理」 = 最終 LB 8.x 帯到達

## 1. Approach

採用: **案 E = Sparse GP for ANCC posterior + variance** + **Edge O = direction-aware Beam (sign-of-dGR penalty)**。

exp008 base (= Edge Q + Edge M + Kalman) に 24+7=**31 features** を上乗せ。

### 1.1 構造原理 (案 E)

公開 top の `FormationPlaneKNN` は per query (X, Y) で K=10 wells centroid に重み付き plane fit、**point estimate** で formation 深度を出す (出典: `kaggle_kernels/exp008_case_d_kalman/exp008_case_d_kalman.py` line 1300-1390)。

我々は **per formation k で Sparse GP** を fit し、posterior $\mathcal{N}(m_k(s), v_k(s))$ を decoder (LGB) に渡す。posterior variance を陽に持つことで **GP と plane-fit の disagreement zone** を学習可能にする。

### 1.2 構造原理 (Edge O)

公開 top の Beam Search / Self-NCC は `corr(GR_h, GR_v)` を最大化、TVT 増減方向の sign awareness は弱い (出典: `docs/research/host-pptx-summary.dense.md` §4)。

我々は Beam の transition cost に `λ · |sign(dGR_h) - sign(dGR_v)|` を追加。slide 6-7 の "TVT is increasing while GR signature matches Typewell GR" という物理原理を直接埋込。

### 1.3 着想根拠

- **計測値**: `formula_oracle_rmse_p50 = 0.006 ft` (出典: `docs/research/first-principles.dense.md` §1.4) ⇒ formula が正確な ANCC で実行されれば RMSE ≈ 0、不確実性は ANCC + b_well 推定誤差に集中
- **学術出典 (案 E)**: Rasmussen 2006 "GPML"、Hensman 2013 "Sparse GP" (SVGP via inducing points)、Damianou 2013 "Deep GP"
- **学術出典 (Edge O)**: 直接的引用無し (= host pptx slide 6-7 の物理 principle、stratigraphic correlation 文献は Catuneanu 2006 "Principles of Sequence Stratigraphy")
- **公開 top との直交性**: ★★★ (= plane fit imputer を GP に置換 + posterior variance を新 features 化)、★★ (= 標準 Beam に sign-penalty 追加、置換でなく augment)

### 1.4 数学的定式 (案 E)

per formation $k$ ($k$ ∈ ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA):

$$
\text{Formation}_k(s) \mid (X_w, Y_w)_{w \in \text{train}} \sim \mathcal{GP}(\mu_k(X, Y),\ K_k((X, Y), (X', Y')))
$$

$K_k$ = ARD-RBF kernel on (X, Y), per-formation hyperparameter (lengthscale + signal variance + noise level)。

posterior at query (X_q, Y_q):

$$
p\left(\widehat{\text{Formation}_k}(X_q, Y_q) \mid \mathcal{D}\right) = \mathcal{N}(m_k(X_q, Y_q), v_k(X_q, Y_q))
$$

decoder feature: $m_k$、$v_k$、$m_k - \text{plane}_k$ (= GP と plane-fit の差)、$\sqrt{v_k} / \text{mean}(\sqrt{v_k})$ (= normalized uncertainty)。

### 1.5 数学的定式 (Edge O)

Beam transition cost に sign-of-dGR penalty 追加:

$$
\text{cost}_\text{aug}(s, \text{trans}) = \text{cost}_\text{base}(s, \text{trans}) + \lambda \cdot |\text{sign}(\Delta\text{GR}_h(s)) - \text{sign}(\Delta\text{GR}_v(\text{path}_s))|
$$

ここで $\Delta\text{GR}_h(s) = \text{GR}_h(s) - \text{GR}_h(s-1)$、$\Delta\text{GR}_v$ は typewell の同様の差分。Beam path 上の transition で TVT が増えている (= path index 増) 区間では typewell 側でも GR が「同じ符号で動いている」 ことを要求する penalty。

$\lambda = 0.5$ を初期値 (要 tuning)。$\Delta = 0$ のとき sign は 0 として扱う。

## 2. 出力 features

### 2.1 案 E — 24 features (= 6 formation × 4 stats)

| col 名 | 定義 | 型 |
|---|---|---|
| `gp_<formation>_mean` | GP posterior mean = $m_k(X_q, Y_q)$ (formation 深度) | float32 |
| `gp_<formation>_var` | GP posterior variance = $v_k(X_q, Y_q)$ | float32 |
| `gp_<formation>_mean_minus_plane` | $m_k - \text{plane\_fit}_k$ (= GP と plane の disagreement) | float32 |
| `gp_<formation>_var_norm` | $\sqrt{v_k} / \text{mean well-wide}(\sqrt{v_k})$ (= normalized uncertainty) | float32 |

formation ∈ {ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA} → 6 × 4 = **24 features**。

### 2.2 Edge O — 7 features (= 既存 5 Beam configs + mean + std の direction-aware version)

| col 名 | 定義 | 型 |
|---|---|---|
| `beam_dir_cons_d` | direction-aware Beam (cons config) - last_tvt | float32 |
| `beam_dir_loose_d` | direction-aware Beam (loose) - last_tvt | float32 |
| `beam_dir_sm5_d` | direction-aware Beam (sm5) - last_tvt | float32 |
| `beam_dir_vcons_d` | direction-aware Beam (vcons) - last_tvt | float32 |
| `beam_dir_mid_d` | direction-aware Beam (vloose alias `mid`) - last_tvt | float32 |
| `beam_dir_mean_d` | mean of 5 dir-aware Beam paths - last_tvt | float32 |
| `beam_dir_std_d` | std of 5 dir-aware Beam paths | float32 |

= **7 features**。注: 既存 `beam_*_d` は標準 Beam (= 公開 top の流れ)、`beam_dir_*_d` は direction-aware version で **disjoint な features**。GBM が両方を見て decision tree で combine する想定。

### 2.3 合計

= 案 E 24 + Edge O 7 = **31 features 追加**。exp008 (= 約 165 cols) → exp009 (= 約 196 cols)。

## 3. 実装方針

### 3.1 案 E — Sparse GP 実装方針

候補:
- **(A) GPyTorch SVGP** (= Hensman 2013 inducing points): 本格、正確、要 PyTorch dep
- **(B) `sklearn.gaussian_process.GaussianProcessRegressor` + K-Means inducing points subset**: 軽量、numpy/sklearn のみ
- **(C) per formation simple KNN (k=20) avg + std**: GP fallback、crude approximation

**採用: (B)**。理由:
1. Kaggle GPU kernel に sklearn は標準で入っている (= dep 追加不要)
2. inducing points 数 M=200 + per formation 6 GPs → 軽量 (= 1-3 min/formation)
3. exp007/exp008 で sklearn は既に import 済 (= 影響最小)
4. `pyproject.toml` を触らない → lock file 整合性維持
5. (A) は本格的だが Kaggle kernel 上で `gpytorch` 追加導入はビルド時間 + reliability リスク。9 hr 制約下では (B) 優先

具体実装:

```python
def _gp_fit_one_formation(
    train_xy: np.ndarray,    # shape (N_train, 2), well centroids
    train_y:  np.ndarray,    # shape (N_train,), formation depth
    M:        int = 200,     # inducing points (= K-Means cluster centers)
    seed:     int = 42,
) -> Tuple[GaussianProcessRegressor, np.ndarray]:
    """
    Fit a GP per formation using K-Means inducing points subset.

    Returns
    -------
    gpr : fitted GaussianProcessRegressor (Matérn 3/2 kernel, ARD)
    induce_xy : the K-Means cluster centers used as training data
    """
    from sklearn.cluster import KMeans
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel

    n_train = len(train_xy)
    if n_train == 0:
        return None, None

    # ARD-RBF (Matern 3/2 with separate length scales for X and Y)
    kernel = (
        C(1.0, (1e-3, 1e3))
        * Matern(length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e3), nu=1.5)
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))
    )

    if n_train > M:
        km = KMeans(n_clusters=M, random_state=seed, n_init=4)
        km.fit(train_xy)
        induce_xy = km.cluster_centers_
        # average target value per cluster (= reduce-via-cluster)
        labels = km.labels_
        induce_y = np.array([train_y[labels == i].mean()
                             for i in range(M)
                             if (labels == i).any()])
        induce_xy = induce_xy[: len(induce_y)]
    else:
        induce_xy = train_xy
        induce_y  = train_y

    # n_restarts=3 to escape poor optima
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=3,
        normalize_y=True,
        random_state=seed,
        alpha=1e-6,
    )
    gpr.fit(induce_xy, induce_y)
    return gpr, induce_xy


def _gp_predict_one_formation(
    gpr:       GaussianProcessRegressor,
    query_xy:  np.ndarray,    # (n_q, 2)
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (mean, var) for each query point. Both shape (n_q,)."""
    if gpr is None:
        return (np.zeros(len(query_xy), np.float32),
                np.ones(len(query_xy),  np.float32))
    m, std = gpr.predict(query_xy, return_std=True)
    return m.astype(np.float32), (std ** 2).astype(np.float32)
```

データ量: 773 wells × 6 formations = 約 4600 (well, formation) ペア。well 単位の median 深度を 1 値として GP に渡す → 773 obs/formation。M=200 で軽量、per formation 1-3 sec 想定。

### 3.2 案 E — class GPFormationImputer (=`FormationPlaneKNN` 兄弟)

`FormationPlaneKNN` と同じ interface (`__init__(well_ids, data_dir)` + `impute(xy_q, self_wid=None)`) で **新規 class** を実装し、両 imputer を **並列に** 走らせる。features には:
- 既存 plane-fit features (= `tvtF_<formation>_d`, `tvtF50_<formation>_d` 等) は **そのまま温存** (= exp008 互換維持、Ridge 9-base が壊れない)
- 新規 GP features (= `gp_<formation>_mean`, `gp_<formation>_var`, `gp_<formation>_mean_minus_plane`, `gp_<formation>_var_norm`) を **追加で** 注入

```python
class GPFormationImputer:
    """
    Per-formation Sparse GP fit on (X, Y) -> formation depth.

    Independent of FormationPlaneKNN: both run in parallel; their predictions
    feed disjoint feature columns to the GBM.

    Leak guarantee:
      - On training (= self_wid is the well being predicted), exclude self
        from the per-formation training pool before refitting if needed.
      - For test, self_wid=None → all train wells used. No hidden TVT touched
        (only well-level X/Y centroid + median formation depth from the
        horizontal_well.csv columns ANCC/ASTNU/...).
    """
    def __init__(self, well_ids, data_dir, M=200, seed=42):
        ...   # build self.gprs_per_formation, self.induce_xy_per_formation,
              # self.formation_train_xy, self.formation_train_y

    def impute(self, xy_q, self_wid=None):
        """Returns (pred: (N, 6), std: (N, 6))."""
        ...
```

### 3.3 案 E — leak-safe self-exclusion

`FormationPlaneKNN.impute()` は `self_wid in self.wmap` ならその点を distance=∞ で除外。GP は本質的に **train data 全体で hyperparameter が決まる** ので、CV 時に self_wid を除外しても hyperparameter は train 全体寄り (= 厳密なゼロ leak ではない)。許容範囲とする理由:
- 773 wells で 1 well 抜いても hyperparameter shift は negligible (= 0.13% data shift)
- exp005/exp006/exp007/exp008 でも `FormationPlaneKNN` は **distance=∞ 経由の self exclude のみ** で、hyperparameter (= scale, weight) は全 well で決定 → 既存実装と同等 leak 厳格度
- post-hoc test で self_wid 除外時の GP refit version を smoke で確認 (= 「除外有無で 5 wells 平均 < 0.001 ft 差」 を検証)

### 3.4 Edge O — direction-aware Beam Search

`beam_search()` に新 wrapper `beam_search_dir()` を実装。差分は cost 関数のみ:

```python
def beam_search_dir(
    gr_h, tw_tvt, tw_gr, start_tvt,
    bs=10, mc=20.0, es=144.0, r=2,
    lam_dir=0.5,    # direction-penalty strength (NEW)
):
    """Direction-aware Beam Search.

    Adds penalty: lam_dir * |sign(Δgr_h(s)) - sign(Δgr_v(path_s))|
    to the existing transition cost. sign(0)=0 by convention; ties broken by
    underlying Beam logic.
    """
    # ... reuse beam_search internals, augmenting `mv` with direction term ...
```

実装は既存 `beam_search()` (line 562-612) を **コピペ + augment**、約 80 行追加。

### 3.5 Edge O — 7 features の computation

`compute_edge_o_features()` を新規実装:
- 既存 `BEAMS` (5 configs) を direction-aware で再 run
- 5 paths を stack して mean / std を `beam_dir_mean_d`, `beam_dir_std_d` に
- vloose は名前を `mid` にリネーム (= subagent K の Edge M と整合、column 名 collision 回避)

```python
def compute_edge_o_features(
    hgr_full, tw_tvt, tw_gr, last_known_tvt, sel_local,
    lam_dir=0.5,
) -> dict:
    """Direction-aware Beam features (7 cols)."""
    paths = {}
    for (bs, mc, es, r, tag) in BEAMS:
        out_tag = "mid" if tag == "vloose" else tag
        paths[out_tag] = beam_search_dir(
            hgr_full, tw_tvt, tw_gr, last_known_tvt,
            bs, mc, es, r, lam_dir=lam_dir,
        )
    lkt = np.float32(last_known_tvt)
    stack = np.stack([p for p in paths.values()], axis=1)
    out = {}
    for tag, p in paths.items():
        out[f"beam_dir_{tag}_d"] = (p - lkt).astype(np.float32)[sel_local]
    out["beam_dir_mean_d"] = (stack.mean(axis=1) - lkt).astype(np.float32)[sel_local]
    out["beam_dir_std_d"]  = stack.std(axis=1).astype(np.float32)[sel_local]
    return out
```

### 3.6 leak check 4 点 (smoke 必須 assert)

1. **GP training data**: `GPFormationImputer.__init__` は `horizontal_well.csv` の `X`, `Y`, formation 列 (= ANCC/ASTNU/...) のみ参照。`TVT_input` も `TVT` も読まない (= formation depth 列は visible/hidden 区別なし、horizontal_well.csv の地質構造 column = leak-safe)
2. **Edge O**: `beam_search_dir` は `gr_h` (= 全 hidden GR、これは leak-safe = exp008 と同じ scope) と `tw_gr` (typewell) のみ参照。`TVT_input` 一切不参照
3. **per-well isolation**: `self_wid` exclude を smoke で 1 well 検証 (= included vs excluded で predict 差を bound)
4. **runtime**: GP fit が 6 formations × 1-3 sec = 6-20 sec (= per kernel run 1 回)、Edge O は per well 数十 ms (= 既存 beam の 5 倍弱) で 776 wells × 50 ms = 40 sec

smoke で `assert "TVT_input" not in inspect.getsource(GPFormationImputer)` + 1-3 wells 実 run + NaN 0 + GP runtime <30 sec。

## 4. failure mode 3 件 + recovery

### 4.1 GP hyperparameter optimization 失敗 (lengthscale runaway)

- **症状**: ARD-Matern lengthscale が 1e3 上限に張り付く → GP が constant function に degenerate、posterior variance が無意味
- **検出**: smoke で `gpr.kernel_.k1.k2.length_scale` が `[1e3, 1e3]` ならフラグ
- **対策**:
  1. `n_restarts_optimizer=3` で multiple optima 探索
  2. `length_scale_bounds=(1e-2, 1e3)` で reasonable 範囲に制約 (= Texas 油田の典型 well-pair 距離 ~10-100 km、normalize 後)
  3. `train_xy` を `(train_xy - mean) / std` で normalize してから GP fit (= scale invariant)
  4. fit 失敗時 (`gpr is None` or kernel 異常) は **per formation で fallback to KNN k=20 avg/std** (= crude posterior 近似、features は zero ではなく KNN 値で埋める)

### 4.2 Edge O の `λ_dir` 過剰 → Beam が GR sign に過適合し offset 浪費

- **症状**: λ=0.5 (default) で Beam path が visible GR の noise sign に追従 → smoothness 喪失、cons / loose 区別が消失 (= 5 paths が同一に collapse)
- **検出**: smoke で `beam_dir_std_d.mean() < 0.01 * beam_std_d.mean()` (= std 1%以下) ならフラグ
- **対策**:
  1. λ tuning grid `{0.1, 0.3, 0.5, 1.0}` を smoke で sweep し、std preserve & 妥当な signed-correlation を選ぶ
  2. fallback: λ_dir=0 (= 標準 Beam に戻る) → 7 features は標準 Beam 系の duplicate (= GBM が無視するだけで害無し)
  3. GR の差分は `5-tap median filter` で pre-smoothing (= sign noise 抑制)

### 4.3 GP fit time が 9 hr cap を圧迫

- **症状**: M=200 inducing points で kernel optimizer の `n_restarts=3` × scipy minimize が想定外に遅い (per formation > 5 min) → 6 formations × 5 min = 30 min 超過、+ feature build × 776 wells で nx 倍に乗る
- **検出**: smoke 段階で `time.perf_counter()` で per formation fit 時間を計測。1 formation > 60 sec ならフラグ
- **対策**:
  1. M=100 に半減 → fit 時間半分以下
  2. `n_restarts_optimizer=1` に減らす (= 局所解 risk と引き換え)
  3. **重要**: GP は **per kernel run で 1 回だけ fit** する設計 (= train wells で fit、test wells は predict のみ)。per-well で fit-and-predict ループに入れない (= 6 formation × 776 wells × 数秒 = 1 hr+)
  4. GP fit を `__main__` (= test feature build 開始前) で 1 回 done、`GPFormationImputer` instance を `_FI_REF` 同様に global へ holding、worker 関数で参照 (= 既存 `_FI_REF, _DI_REF` パターンを踏襲)
  5. fallback: GP 全 disable (`GP_ENABLE=False`) で 24 features を全 zero 埋め (= exp008 同点に degrade、悪化なし)

## 5. 実装手順 (= 30 分毎中間 commit)

| # | 工程 | 中間 commit | 想定時間 |
|---|---|---|---|
| 1 | branch 作成 (= `feat/phase-4-exp009-case-e-edge-o`) | ✅ 1st | 5 min |
| 2 | 設計 doc (本 file) → `experiments/exp009/design.md` | 2nd | 30-60 min |
| 3 | kernel script 派生 (= exp008 → exp009、`GPFormationImputer` + `compute_edge_o_features` 追加 + dict 出力に挿入) | 3rd, 4th | 90-120 min |
| 4 | ローカル smoke (1-3 wells、leak guard 4 点 + GP runtime + std reasonable assert) | 5th + notes.md | 60-90 min |
| 5 | **kernel push は実行しない** (中央指示待ち) | — | — |

合計 work-time ≈ 3-5 時間。

## 6. exp008 base に追加する 31 features の挿入位置 (= 詳細)

### 6.1 関数 / class 追加位置

`kaggle_kernels/exp009_case_e_edge_o/exp009_case_e_edge_o.py`:

1. `beam_search_dir()`: line ≈ 612 (= `beam_search()` の直後、`compute_beam_features()` の前)
2. `compute_edge_o_features()` + `edge_o_feature_names()`: line ≈ 651 (= `compute_beam_features()` の直後)
3. `class GPFormationImputer`: line ≈ 1391 (= `class FormationPlaneKNN` の直後、`class DenseANCCImputer` の前)
4. config 追加: 冒頭 KALMAN_ENABLE 周辺に `GP_ENABLE = True`, `GP_M_INDUCE = 200`, `EDGE_O_ENABLE = True`, `EDGE_O_LAMBDA_DIR = 0.5`

### 6.2 `build_well_features()` 内挿入

Edge D Kalman block (line ≈ 1886-1913) の直後、`out = pd.DataFrame({...})` (line 1916) の前に Edge O 挿入:

```python
# ── Edge O (direction-aware Beam) FE ────────────────────────────────
if EDGE_O_ENABLE:
    try:
        edge_o_out = compute_edge_o_features(
            hgr_full=hgr, tw_tvt=tw_tvt, tw_gr=tw_gr,
            last_known_tvt=last_tvt, sel_local=sel_local,
            lam_dir=EDGE_O_LAMBDA_DIR,
        )
    except Exception as _eo_e:
        print(f"  WARN [{wid}] Edge O FE failed: {_eo_e}")
        edge_o_out = {k: np.zeros(len(ev), dtype=np.float32)
                      for k in edge_o_feature_names()}
else:
    edge_o_out = {k: np.zeros(len(ev), dtype=np.float32)
                  for k in edge_o_feature_names()}
```

GP features は Spatial features block (line ≈ 1779-1798) の直後に追加 (= `tvt_formulas` の直後):

```python
# ── GP per-formation posterior FE ───────────────────────────────────
if GP_ENABLE:
    global _GP_REF
    gp_use = _GP_REF
    if gp_use is not None and not getattr(gp_use, "_empty", True):
        try:
            gp_mean_ev, gp_var_ev = gp_use.impute(xy_ev, self_wid=swid)
            # plane-fit baseline already in form_ev (= FormationPlaneKNN)
            gp_disagree = (gp_mean_ev - form_ev).astype(np.float32)
            # well-wise normalised uncertainty
            std_ev = np.sqrt(np.maximum(gp_var_ev, 1e-12))
            std_mean_well = max(float(std_ev.mean()), 1e-6)
            std_norm = (std_ev / std_mean_well).astype(np.float32)
            gp_out = {}
            for fi_idx, fn in enumerate(FORMATIONS):
                gp_out[f"gp_{fn}_mean"]              = gp_mean_ev[:, fi_idx].astype(np.float32)
                gp_out[f"gp_{fn}_var"]               = gp_var_ev[:, fi_idx].astype(np.float32)
                gp_out[f"gp_{fn}_mean_minus_plane"]  = gp_disagree[:, fi_idx]
                gp_out[f"gp_{fn}_var_norm"]          = std_norm[:, fi_idx]
        except Exception as _gp_e:
            print(f"  WARN [{wid}] GP FE failed: {_gp_e}")
            gp_out = {k: np.zeros(len(ev), np.float32) for k in gp_feature_names()}
    else:
        gp_out = {k: np.zeros(len(ev), np.float32) for k in gp_feature_names()}
else:
    gp_out = {k: np.zeros(len(ev), np.float32) for k in gp_feature_names()}
```

### 6.3 output dict 拡張

`out = pd.DataFrame({...})` の **kalman_out, の後に **gp_out, **edge_o_out, を追加。

### 6.4 GP imputer の global wiring

既存 `_FI_REF`, `_DI_REF` と並列に `_GP_REF` を導入:
- `_get_or_build_test_df()` 内で test 構築前に `_GP_REF = GPFormationImputer(train_well_ids, data_dir=TRAIN_DIR, M=GP_M_INDUCE)` を 1 回呼ぶ
- `_get_or_build_train_df()` 内も同様 (= train 用 GP は train wells から `self_wid` exclusion で per-well CV-safe predict)
- `_build_one()` worker は `_GP_REF` を share (= read only)、joblib parallel safe

### 6.5 MODE 名

`MODE = "infer_edge_e_gp_o"` (= exp008 の `infer_edge_d_kalman` から派生、新名で disambiguate)。dispatcher の `if MODE == "infer_edge_d_kalman":` 分岐を `if MODE in ("infer_edge_d_kalman", "infer_edge_e_gp_o"):` に拡張、または個別 `elif` で対応 (= 実体は同じ pipeline、Kalman + GP + Edge O が build_well_features 内で自動挿入)。

### 6.6 自前 LGB×3 + CB train

features 列が 31 増える → `get_feature_columns(df)` は自動で `target` `well` `id` 以外を返す形なので **追加修正不要**。Ridge meta は 9-base の OOF predictions を fit する形 (= base model 内部で 31 新 features を使うのみ、Ridge への入力 9 列は変わらず)。

## 7. kernel-metadata.json

`title` を `ROGII exp009 Case E Edge O` (4 words) にすると Kaggle slug `rogii-exp009-case-e-edge-o` (= 5 hyphens, 1-2-1-1-1) で `id: ky7240/rogii-exp009-case-e-edge-o` と一致する。

検証ロジック:
- "ROGII" → `rogii`
- " " → `-`
- "exp009" → `exp009`
- " Case E Edge O" → `-case-e-edge-o`
- 結合: `rogii-exp009-case-e-edge-o` ✓

(= subagent K の `Edge QM` → `edge-q-m` スラッグ生成では、`QM` が 2 文字英大文字だったため Kaggle が `q-m` に分解した。`Case E Edge O` は各単語が 1 文字 (E, O) または既知単語 (Case, Edge) なので分解は予測可能。)

```json
{
  "id": "ky7240/rogii-exp009-case-e-edge-o",
  "title": "ROGII exp009 Case E Edge O",
  "code_file": "exp009_case_e_edge_o.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_tpu": "false",
  "enable_internet": "false",
  "dataset_sources": [
    "karnakbaevarthur/rogii-code-helper-dataset",
    "thermostatic/rogii-tabicl-v2-public-assets",
    "thbdh5765/rogii-v1-train-cache"
  ],
  "competition_sources": ["rogii-wellbore-geology-prediction"],
  "kernel_sources": []
}
```

(= exp007/exp008 と同一 sources、dormant 含めて attach 維持。GP/Edge O は外部 dataset 不要。`gpytorch` は使わない判断 → pyproject.toml/uv.lock 触らない)

## 8. runtime 見積

| 段階 | 時間 |
|---|---|
| GP fit (6 formations × M=200 inducing × n_restarts=3) | 30-90 sec total |
| GP predict (test 6 wells × 14k rows × 6 formations) | 5-15 sec total |
| Edge O Beam (5 configs × 776 wells × per-well 50ms) | 200-300 sec total |
| test feature build (6 wells × 14k rows、+ exp008 + GP + Edge O) | 60-90 sec |
| train feature build (3.78M rows、+ GP + Edge O) | exp008 比 +10-20 min (Edge O が dominant、GP は 1 回 fit のみ) |
| 自前 LGB×3 + CB train (5 fold × 4 base、+31 features) | 70-110 min |
| 9-base Ridge meta fit + predict | <1 sec |
| postproc + SG smooth + submission | 5 sec |
| **合計** | **90-150 min** |
| 9 hr cap 余裕 | **6 hr** |

## 9. 想定 LB

| シナリオ | 仮定 | LB |
|---|---|---|
| Best | 案 E -0.80 + Edge O -0.30 (= exp008 9.05 から大幅改善) | **8.25** |
| Mid | 案 E -0.40 + Edge O -0.20 (= exp008 9.35 から想定 -0.6) | **8.75** |
| Worst | 両 fallback (GP zero / Edge O λ=0) → exp008 同点 | **9.35** |

= **9 切り (LB 8.x 帯到達)** が exp009 mid case で実現する**第 2 層が最重要**。

注: Best シナリオは案 E の独立効果 -0.4〜-0.8 ft + Edge O の独立効果 -0.1〜-0.3 ft (= 出典 `docs/research/independent-edges.dense.md` §2.5, `docs/research/host-pptx-summary.dense.md` §4.3) の楽観的合算。両者は **入力空間が disjoint** (= GP は (X,Y)→formation depth、Edge O は GR signature → TVT path) なので相乗ではなく加法的、Ridge stacking で部分相関を学習。

## 10. 9 切りへの次手 (exp010+)

- exp010: + 案 G (Sparse Transformer/SSM 直予測) → 8.5-8.8
- exp011-013: 残 6 edges (案 F, H, I + 補助 J/K/L) + final stack → 8.0-8.5

## 11. References

- `docs/research/independent-edges.dense.md` §2 (= 案 E 詳細仕様, subagent G 著)
- `docs/research/independent-edges.dense.md` §9.6 (= Edge O 詳細, host pptx 由来)
- `docs/research/host-pptx-summary.dense.md` §4 (= host slide 6-7、direction-aware の物理 endorsement)
- `docs/research/first-principles.dense.md` §1.4 (= formula_oracle_rmse=0.006 → ANCC が perfect なら RMSE 0)
- `docs/research/top3-distill.dense.md` §3 (= FormationPlaneKNN 仕様、case E の比較対象)
- `kaggle_kernels/exp008_case_d_kalman/exp008_case_d_kalman.py` line 1300-1390 (= base、`FormationPlaneKNN`)
- `kaggle_kernels/exp008_case_d_kalman/exp008_case_d_kalman.py` line 562-650 (= base、`beam_search` + `compute_beam_features`)
- `kaggle_kernels/exp007_edge_q_m/exp007_edge_q_m.py` (= subagent K の `compute_edge_m_features` 実装パターン)
- Rasmussen, C. E., Williams, C. K. I. (2006). "Gaussian Processes for Machine Learning". MIT Press.
- Hensman, J., Fusi, N., Lawrence, N. D. (2013). "Gaussian Processes for Big Data". UAI.
- Catuneanu, O. (2006). "Principles of Sequence Stratigraphy". Elsevier.
