# exp008 — 案 D: Kalman/PF on dTVT (Phase 4 最初の独自 edge、9 切り路線)

## 0. 状況 (2026-05-11)

- exp005 = LB **10.317** (Silver、自前 best)
- exp006 = LB **10.503** (TabICL 失敗、postmortem 済)
- exp007 = `feat/phase-3-exp007-edge-q-m` (= LB 9.5-9.7 想定、Edge Q + Edge M + 自前 4 base、現 RUNNING、base camp)
- 公開 LB Top 1 = 9.256、Gold cutoff = 9.919、賞金圏 (Top 4) = 9.415
- ユーザー指示: 「9 切らないと優勝は無理」 = 最終 LB 8.x 帯到達

## 1. Approach

採用: **案 D = State-Space Model on dTVT (Kalman filter + per-well AR(1) MLE + global shrinkage)**

### 1.1 構造原理

公開 top kernel は `dTVT` を i.i.d. 仮定で features に渡す。我々は visible region から per-well φ + σ_ε を MLE 推定し、AR(1) state-space で hidden region を **Kalman filter で MAP estimate**。さらに predicted variance を MD (= prediction horizon) で grow させて、long extrapolation での不確実性を陽に表現。

これは exp007 が encode できなかった「dTVT の自己相関構造」を **追加 7 features** として GBM に渡す orthogonal edge。

### 1.2 着想根拠

- 計測値: `outputs/eda/first_principles/per-well-stats.parquet` (n=773) で **`ar1_phi_p50 = 0.99887`、`ar1_phi_p25 = 0.99671`、`ar1_eps_std_p50 = 0.01556`** を確認 (出典: `docs/research/first-principles.dense.md` §2.1)
- **dTVT は ほぼ random walk** (= AR(1) phi ≈ 1) → state-space で陽に modeling する利得が大きい
- 学術出典: Kalman 1960, Doucet et al. 2001 (PF), Vandewiele 2021 (Ventilator 1 位は LSTM だが PID 物理を post-proc に組込)
- 公開 top の Particle Filter (= `run_pf_ancc`, `run_pf_z` in exp007) は ANCC + Z 上で動かしているが **dTVT 上の AR(1) state-space は無い** (= 直交 edge)

### 1.3 数学的定式

per-well AR(1) on dTVT(s) := TVT(s+1) − TVT(s):

$$
\Delta\text{TVT}(s+1) = \phi \cdot \Delta\text{TVT}(s) + \varepsilon(s), \quad \varepsilon \sim \mathcal{N}(0, \sigma_\varepsilon^2)
$$

$$
\text{TVT}(s+1) = \text{TVT}(s) + \Delta\text{TVT}(s+1)
$$

φ + σ_ε は per-well で visible region のみから MLE 推定 (Yule-Walker、後述 §3.2)。

Kalman 1D の予測共分散は MD (= prediction step t) で grow:

$$
\text{Var}[\Delta\text{TVT}(s+t)] = \frac{\sigma_\varepsilon^2 (1 - \phi^{2t})}{1 - \phi^2}
$$

φ ≈ 1 では上式が ill-conditioned なので、実装では **φ < 0.99999 にクリップ** + φ→1 limit では **`Var ≈ t · σ_ε^2`** に切り替える (= random walk 近似)。

TVT の累積分散:

$$
\text{Var}[\text{TVT}(s+t) - \text{TVT}(s)] \approx \sum_{k=1}^{t} k \cdot \sigma_\varepsilon^2 \cdot \frac{1 - \phi^{2k}}{1 - \phi^2}
$$

実装は per-step accumulator で計算 (closed form と numerical issue 回避)。

## 2. 出力 features (7 列、GBM に渡す)

| col | 定義 | 型 |
|---|---|---|
| `kalman_d` | Kalman MAP estimate of `TVT(s) − last_known_TVT` at each hidden row | float32 |
| `kalman_std` | Kalman posterior 1σ standard deviation of TVT(s) | float32 |
| `kalman_phi` | per-well φ MLE (after global shrinkage) | float32 (constant per well) |
| `kalman_sigma_eps` | per-well σ_ε MLE (after global shrinkage) | float32 (constant per well) |
| `kalman_t_md` | (MD − last_known_MD) / median_dMD = prediction horizon in steps | float32 |
| `kalman_vs_pf_a_d` | `kalman_d − pf_a_d` (= PF ANCC との agreement、disagreement zone 検出) | float32 |
| `kalman_vs_beam_cons_d` | `kalman_d − beam_cons_d` (= Beam consensus との agreement) | float32 |

= **7 features 追加**、合計 ~165 列。

挿入位置: `build_well_features()` 内、Edge M FE block の直後 (line 1658 後)、final DataFrame assemble (line 1661) の前。output dict (line 1660-1753) に `**kalman_out` を `**edge_m_out` の後に追加。

## 3. 実装方針

### 3.1 全体構成 (= 関数追加 1 つ + dict 出力 1 つ)

```
def compute_kalman_features(
    ktvt:        np.ndarray,    # visible TVT_input series, shape (nk,)
    kmd:         np.ndarray,    # visible MD series, shape (nk,)
    last_tvt:    float,
    last_md:     float,
    hmd:         np.ndarray,    # hidden MD at evaluated rows, shape (n_hidden,)
    pf_a_d:      np.ndarray,    # exp007 PF ancc delta, shape (n_hidden,)
    beam_cons_d: np.ndarray,    # exp007 beam consensus delta, shape (n_hidden,)
    phi_global:  float = 0.999, # global shrinkage prior (= ar1_phi_p50)
    sigma_global:float = 0.0156,# global prior (= ar1_eps_std_p50)
    lam_phi:     float = 0.3,
    lam_sigma:   float = 0.3,
) -> Dict[str, np.ndarray]:
```

### 3.2 per-well φ + σ_ε MLE (Yule-Walker、純 numpy)

visible region の dTVT 系列 `dt = np.diff(ktvt)` から:

```
# Yule-Walker AR(1) MLE (closed form)
dt_centered = dt - dt.mean()
phi_local = (dt_centered[:-1] * dt_centered[1:]).sum() / (dt_centered[:-1] ** 2).sum()
phi_local = np.clip(phi_local, 0.0, 0.99999)  # numerical guard
resid = dt_centered[1:] - phi_local * dt_centered[:-1]
sigma_local = resid.std(ddof=1)  # MLE-ish
sigma_local = max(sigma_local, 1e-4)  # floor
```

参照: `outputs/eda/first_principles/per-well-stats.parquet` の `ar1_phi`, `ar1_eps_std` 列が同一 formula で 773 wells 計測済 → 期待される範囲 (φ ≈ 0.99, σ_ε ≈ 0.016) を smoke で確認。

### 3.3 hierarchical Bayesian shrinkage

短い visible (`p10 = 0.18` の visible_ratio) で MLE が不安定。global prior に shrinkage:

```
n_eff = max(len(dt) - 2, 1)    # effective sample size
w_local = n_eff / (n_eff + lam_phi * 100)  # 100 = pseudo-count, λ で controllable
phi = w_local * phi_local + (1 - w_local) * phi_global
```

同様に σ_ε も:

```
w_local_sig = n_eff / (n_eff + lam_sigma * 100)
sigma = w_local_sig * sigma_local + (1 - w_local_sig) * sigma_global
```

`λ_phi = λ_sigma = 0.3` を初期値。global は parquet で計測済 p50 をハードコード (0.999, 0.0156)。

### 3.4 Kalman filter forward pass on dTVT

state = dTVT(s) を 1D scalar として AR(1) で propagate:

```
# Initialize: state at s = nk-1 (= last visible step)
dt_k = ktvt[-1] - ktvt[-2]   # last observed dTVT
P_k  = 0.0                    # zero variance at boundary (= observed)

# Step from last visible MD to first hidden MD (gap in MD sense)
median_dmd = np.median(np.diff(kmd))
median_dmd = max(median_dmd, 1e-3)

# Per hidden row, predict forward
tvt_pred = np.zeros(n_hidden, dtype=np.float32)
std_pred = np.zeros(n_hidden, dtype=np.float32)

# Cumulative dTVT estimate (= sum of expected dTVT increments)
cum_dt = 0.0
cum_var = 0.0
prev_dt = dt_k
prev_md = kmd[-1]   # = last_md

for i in range(n_hidden):
    # Prediction step count from last visible row
    md_now = hmd[i]
    n_steps = max(1, int(round((md_now - prev_md) / median_dmd)))

    for _ in range(n_steps):
        # AR(1) propagate
        prev_dt = phi * prev_dt
        # cum_dt = sum of expected dTVT (mean-zero AR(1) drift to 0)
        cum_dt += prev_dt
        # variance accumulation
        # Var[dTVT(s+k)] = phi^(2k) * 0 + sigma^2 * (1 - phi^(2k)) / (1 - phi^2)
        # Var[sum of dTVT] grows by Var[dTVT(s+k)] per step
        cum_var += sigma ** 2  # AR(1) Var[dTVT_k] ≈ sigma^2/(1-phi^2) at stationary; for our φ≈1 use sigma^2 per step

    prev_md = md_now
    tvt_pred[i] = cum_dt              # MAP estimate of (TVT_hidden − last_TVT)
    std_pred[i] = np.sqrt(cum_var)    # 1σ posterior std

# Std floor (numerical guard for tiny visible)
std_pred = np.maximum(std_pred, sigma * 0.1)
```

注: φ ≈ 1 では `prev_dt → prev_dt` (drift 維持)、φ < 1 では mean-zero に regress (= 末端 drift)。両者の妥協として **φ_shrunk** を使う。Var の累積は `t · σ²` 近似 (random walk limit) に統一 (= φ ≈ 1 で正確、φ < 0.99 では多少 overestimate するが std floor で実用上問題なし)。

### 3.5 disagreement features

```
out["kalman_vs_pf_a_d"]      = (tvt_pred - pf_a_d).astype(np.float32)
out["kalman_vs_beam_cons_d"] = (tvt_pred - beam_cons_d).astype(np.float32)
```

これらは GBM に「PF と Kalman が一致するか / 大きく食い違うか」を直接シグナルとして渡す (= meta-disagreement、ensemble の弱点を補う重要シグナル)。

### 3.6 leak check 4 点 (smoke 必須 assert)

1. **visible only**: `compute_kalman_features` の引数に `hidden TVT` を一切渡さない (= `pf_a_d`, `beam_cons_d` は exp007 既存 hidden features だが、これらは leak-safe 既存実装)
2. **no hidden TVT**: 関数内部で `ev["TVT"]`, `ev["TVT_input"]` を一切参照しない (= visible `ktvt` のみ使用)
3. **no future MD**: hidden MD は monotonic 増加で参照のみ (= forward filter のみ、smoother 無し)
4. **per-well isolation**: φ + σ_ε は当該 well の visible のみで fit、global prior は 773 wells 全体 p50 (= contamination 無し、計測済値の埋め込み)

smoke で `assert "TVT" not in inspect.getsource(compute_kalman_features)` (= 文字列 check)、+ 1-3 wells で実 run + NaN 0 + std monotonic increasing。

## 4. failure mode 3 件 + recovery

### 4.1 末端 drift (φ < 1 で 0 へ regress)

- **症状**: φ_shrunk = 0.95 の well では cum_dt が exponential に 0 へ収束、long extrap で bias 発生
- **検出**: smoke で `kalman_d[−1]` が `pf_a_d[−1]` と大きく乖離する well を flag
- **対策**: shrinkage で `phi_global = 0.999` に pull する (= λ_phi = 0.3 で短い visible は global に寄る)、+ kernel script で `if abs(kalman_d[−1]) < 0.5 * abs(pf_a_d[−1]):` の場合に Ridge weight が 0 で済むよう GBM に決断を委ねる (= feature として渡すだけ、強制 blend しない)

### 4.2 非ガウス noise (fat tail)

- **症状**: `dtvt_p95 = 1.01` vs `dtvt_std = 0.42` → kurt > 3、Kalman 最適性失う
- **検出**: per-well で `dt.std() / dt.abs().mean() > 1.5` を flag
- **対策**: 本 exp008 では Kalman のみ実装。Particle Filter version は v2 候補として `compute_kalman_features` の docstring に TODO 記載。GBM が disagreement features (`kalman_vs_pf_a_d`) を見て hidden zone で Kalman を信頼しなくなる挙動に期待

### 4.3 visible 末端 σ_ε MLE が短い visible で不安定

- **症状**: `visible_ratio_p10 = 0.18` で σ_local が桁違い (e.g., 0.5 vs global 0.016)
- **検出**: `n_eff < 50` の well で `sigma_local > 5 * sigma_global` を flag
- **対策**: hierarchical Bayesian shrinkage (§3.3) で global prior 0.0156 に λ=0.3 で pull、+ `sigma_local = max(sigma_local, 1e-4)` の floor

## 5. 実装手順 (= 30 分毎中間 commit)

| # | 工程 | 中間 commit | 想定時間 |
|---|---|---|---|
| 1 | branch 作成 (= `feat/phase-4-exp008-case-d-kalman`) | ✅ 1st | 5 min |
| 2 | 設計 doc (本 file) → `experiments/exp008/design.md` | 2nd | 30 min |
| 3 | kernel script 派生 (= exp007 → exp008、`compute_kalman_features` 追加 + dict 出力に挿入) | 3rd | 90-120 min |
| 4 | ローカル smoke (1-3 wells、leak guard 4 点 + Kalman std 単調増加 assert) | 4th + notes.md | 60-90 min |
| 5 | **kernel push は実行しない** (中央指示待ち) | — | — |

合計 work-time ≈ 3-4 時間。

## 6. exp007 base に追加する 7 features の挿入位置 (= 詳細)

`kaggle_kernels/exp008_case_d_kalman/exp008_case_d_kalman.py` (= exp007 派生):

1. **新規関数追加**: line ≈ 1053 (`edge_m_feature_names()` の直後) に:
   ```
   def kalman_feature_names() -> List[str]:
       return ["kalman_d", "kalman_std", "kalman_phi", "kalman_sigma_eps",
               "kalman_t_md", "kalman_vs_pf_a_d", "kalman_vs_beam_cons_d"]

   def compute_kalman_features(...) -> Dict[str, np.ndarray]:
       ...
   ```

2. **`build_well_features()` 内挿入**: Edge M block (line 1622-1658) の直後、`out = pd.DataFrame({...})` (line 1661) の前に:
   ```
   # ── Edge D (AR(1) Kalman on dTVT) FE ───────────────────────────────
   if KALMAN_ENABLE:
       try:
           pf_a_d_full      = (pf_a_sel - last_tvt).astype(np.float32)
           beam_cons_d_full = bf.get("beam_cons_d", np.zeros(len(ev), np.float32))
           kalman_out = compute_kalman_features(
               ktvt=ktvt, kmd=kmd, last_tvt=last_tvt, last_md=last_md,
               hmd=hmd, pf_a_d=pf_a_d_full, beam_cons_d=beam_cons_d_full,
           )
       except Exception as _kal_e:
           kalman_out = {k: np.zeros(len(ev), np.float32) for k in kalman_feature_names()}
   else:
       kalman_out = {k: np.zeros(len(ev), np.float32) for k in kalman_feature_names()}
   ```

3. **output dict 拡張**: `**edge_m_out,` の後に `**kalman_out,` を追加 (= line 1752 後)。

4. **config 追加**: 冒頭の MODE / EDGE_Q_ENABLE / EDGE_M_ENABLE 付近に `KALMAN_ENABLE = True` を追加。`MODE = "infer_edge_d_kalman"` に変更し、dispatcher の `elif MODE == 'infer_edge_qm':` を `infer_edge_d_kalman` にも対応 (実体は同じ pipeline、Kalman features は build_well_features 内で自動挿入)。

5. **自前 LGB×3 + CB の features 列が 7 増える**: `get_feature_columns(df)` は自動で `target` `well` `id` 以外を返す形なので追加修正不要。Ridge meta は 9-base の OOF predictions を fit する形 (= base model 内部で 7 新 features を使うのみ、Ridge への入力 9 列は変わらず)。

## 7. kernel-metadata.json

```json
{
  "id": "ky7240/rogii-exp008-case-d-kalman",
  "title": "ROGII exp008 Case D Kalman",
  "code_file": "exp008_case_d_kalman.py",
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

(= exp007 と同一 sources、dormant 含めて attach 維持。Kalman は外部 dataset 不要)

## 8. runtime 見積

| 段階 | 時間 |
|---|---|
| test feature build (6 wells × 14k rows、+Kalman) | 30-35 sec (Kalman は per-well 数十 ms × 6) |
| train feature build (3.78M rows、+Kalman) | exp007 比 +20-40 sec (= Kalman 1D 軽量) |
| 自前 LGB×3 + CB train (5 fold × 4 base、+7 features) | 60-90 min (= exp007 同等、列数微増のみ) |
| 9-base Ridge meta fit + predict | <1 sec |
| postproc + SG smooth + submission | 5 sec |
| **合計** | **75-125 min** |
| 9 hr cap 余裕 | **6-7 hr** |

## 9. 想定 LB

| シナリオ | 仮定 | LB |
|---|---|---|
| Best | exp007 base = 9.5 + Kalman -0.3 + disagreement -0.1 | **9.0-9.1 帯** |
| Mid | exp007 base = 9.6 + Kalman -0.2 + disagreement -0.05 | **9.35** |
| Worst (Kalman 寄与なし) | exp007 base = 9.7 + Kalman ≈ 0 (Ridge weight 微小) | **9.7** (悪化なし) |

**9 切りまで残 0.0-0.1**: exp008 単独で 9.0 帯到達は楽観、9.3-9.5 帯が現実的。9 切りには exp009 (案 E: GP for ANCC + Edge O) との合成が必要。

## 10. 9 切りへの次手 (exp009+)

- exp009: + 案 E (Bayesian GP for ANCC) + Edge O → 8.7-9.0
- exp010: + 案 G (Sparse Transformer/SSM 直予測) → 8.5-8.8
- exp011-013: 残 6 edges + final stack → 8.0-8.5

## 11. References

- `docs/research/independent-edges.dense.md` §1 (= 案 D 詳細仕様、subagent G 著)
- `docs/research/first-principles.dense.md` §2.1 (= ar1_phi p50=0.999、ar1_eps_std p50=0.0156 計測元)
- `outputs/eda/first_principles/per-well-stats.parquet` (= 773 wells × 22 cols、ar1_phi/ar1_eps_std 計測済)
- `docs/research/top3-distill.dense.md` §2 (= Numba PF for ANCC + Z 詳細、参考)
- `docs/research/strategy-critique.dense.md` §1 (= 案 D の構造評価)
- `kaggle_kernels/exp007_edge_q_m/exp007_edge_q_m.py` line 1395-1758 (= base、build_well_features)
- Kalman 1960: "A New Approach to Linear Filtering and Prediction Problems"
- Doucet et al. 2001: "Sequential Monte Carlo Methods in Practice"
