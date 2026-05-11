# 3 Hack Implementation Spec (= GeostatsPy Kriging + TimeXer + MEMO、2026-05-11)

> **目的**: subagent W (`docs/research/academic-literature-deeper.dense.md` @ branch `docs/paradigm-deeper-2026-05-11`、 752 行) で提示された即時試行可 3 hack を ROGII kernel script レベルで **実装 spec + smoke 設計** まで refine する。 数式 / dependencies / inject 位置 (file:line) / Kaggle 9 hr runtime / failure modes を必ず添える。
> **適用範囲**: branch `docs/3hack-spec-2026-05-11` (= `feat/phase-5-edge-r-online` から派生、 subagent X が active な `docs/paradigm-deeper-2026-05-11` とは disjoint)
> **base kernel**:
> - `kaggle_kernels/exp005_cache_blend/exp005_cache_blend.py` (= 2320 行、 5-base + Ridge stacking + path b blend)
> - `kaggle_kernels/exp009_case_e_edge_o/exp009_case_e_edge_o.py` (= 4503 行、 Case E Sparse GP + Edge O dir-aware Beam + Edge D Kalman + Edge R Online)
> **CRITICAL**: 推奨は出さない (= `~/.claude/CLAUDE.md` 中立指示原則)。 各 hack の選択軸 + 失敗モード + 工数で判断軸のみ提示する。

---

## 0. 序: subagent W 提示 3 hack の優先順位

### 0.1 「即時試行可 5 問」 通過確認 (= `kaggle/CLAUDE.md §11.2`)

各 hack を 5 問で再評価 (= subagent W §7.7 を更に kernel script レベルで refine):

| hack | Q1 数理本質 | Q2 優勝寄与 | Q3 代替比較 | Q4 rule 耐性 | Q5 datapoint 価値 | 通過 |
|---|---|---|---|---|---|---|
| #1 GeostatsPy Kriging | BLUE = 数学的最適、 anisotropic 構造を陽に encode | -0.1〜-0.3 ft + UQ 副産物 (= 下流 Beta-NLL 連携余地) | 既存 FormationPlaneKNN k=10 等方より厳密 | 完全静的、 host fix 影響 0 | per-formation variogram param で isolate ablation 可 | ◎ |
| #2 TimeXer | 物理事前 `TVT = -Z + ANCC + b` を attention で直接 encode | -0.2〜-0.5 ft、 12 benchmark SOTA 実績 | PatchTST cross-attention 自前より benchmark 実績で確度高 | clean static 解、 host fix 耐性 ◎ | endogenous/exogenous 分離 ablation で isolate 可 | ◎ |
| #3 MEMO TTT | per-well transductive adaptation = 数理本質 | -0.1〜-0.3 ft、 augmentation diversity に依存 | TTT++ TFA より augmentation diversity に依存、 random_stretch 多様性が key | augmentation rule 内、 host fix 影響なし | augmentation N + niter ablation で entropy 寄与 isolate | ◎ |

3 件すべて通過。 推奨せず、 選択軸のみ提示する (= 中央 / 開発者判断)。

### 0.2 着手順序の **選択軸** (= 軽さでなく数理本質、 `kaggle/CLAUDE.md §11`)

| 軸 | 候補 hack | 根拠 |
|---|---|---|
| **確度 × 工数** が小 | Hack 1 (Kriging) | variogram fit が通れば必ず BLUE、 工数 2-3d、 GeostatsPy 既成 |
| **9 切り upper bound** | Hack 2 (TimeXer) | -0.5 ft 上限が最大、 12 benchmark SOTA、 ただし工数 2-3d + Kaggle GPU 必須 |
| **既存 Edge R との直交性** | Hack 3 (MEMO) | Edge R (= LGB continued training) は instance-level fine-tune、 MEMO は augmentation marginal entropy で完全直交、 同時適用可 |
| **9 hr runtime 制約余裕** | Hack 1 > Hack 3 > Hack 2 | Hack 1 ≤30 min, Hack 3 ≈1.7 hr, Hack 2 ≈3-5 hr (= GPU 前提) |
| **host fix 耐性** | Hack 1 = Hack 2 > Hack 3 | Kriging / TimeXer は完全静的、 MEMO は augmentation rule 内で内部完結 |

---

## 1. Hack 1: GeostatsPy Kriging × ANCC imputation

### 1.1 数式 + ROGII 適用

**variogram model** (= 2 点間距離 $h$ の半分散):
$$
\gamma(h) = \frac{1}{2} \mathbb{E}\left[(Z(x) - Z(x+h))^2\right]
$$

実用 spherical model (= GeostatsPy `it=1`):
$$
\gamma(h) = c_0 + c \cdot \left[\frac{3h}{2a} - \frac{1}{2}\left(\frac{h}{a}\right)^3\right] \quad (h \le a)
$$
$$
\gamma(h) = c_0 + c \quad (h > a)
$$
ここで $c_0$ = nugget、 $c$ = contribution、 $a$ = range。

**anisotropic geometric** (= 主軸方向 vs 直交方向で range が異なる):
$$
h_{\text{anis}}(\Delta x, \Delta y) = \sqrt{\left(\frac{h_\parallel}{a_{\text{maj}}}\right)^2 + \left(\frac{h_\perp}{a_{\text{min}}}\right)^2} \cdot a_{\text{maj}}
$$
ROGII の地層は水平方向に長く相関 (= `academic-literature-deeper.dense.md §5.3`)、 NE-SW azimuth で `a_maj / a_min ≈ 2-3` を仮定。

**ordinary kriging** (= 局所平均 stationarity、 mean 未知):
$$
\hat{Z}(x_0) = \sum_{i=1}^n \lambda_i Z(x_i), \quad \sum_{i=1}^n \lambda_i = 1
$$
weight $\lambda_i$ は以下 system から solve:
$$
\begin{pmatrix}
C(x_1, x_1) & \cdots & C(x_1, x_n) & 1 \\
\vdots & \ddots & \vdots & \vdots \\
C(x_n, x_1) & \cdots & C(x_n, x_n) & 1 \\
1 & \cdots & 1 & 0
\end{pmatrix}
\begin{pmatrix} \lambda_1 \\ \vdots \\ \lambda_n \\ \mu \end{pmatrix}
=
\begin{pmatrix} C(x_1, x_0) \\ \vdots \\ C(x_n, x_0) \\ 1 \end{pmatrix}
$$
ここで $C(x_i, x_j) = C(0) - \gamma(\|x_i - x_j\|)$ は covariance、 $\mu$ は Lagrange 乗数。

**kriging variance** (= 予測点 epistemic):
$$
\sigma_K^2(x_0) = C(0) - \sum_{i=1}^n \lambda_i C(x_i, x_0) - \mu
$$
= 予測点が観測点群から離れるほど増加、 下流 Beta-NLL の prior precision として使用可。

**ROGII 適用**: 既存 `FormationPlaneKNN` (= 6 formation × KNN k=10、 等方 weight、 plane fit、 file:line `exp005_cache_blend.py:799-889`) を per-formation の `kb2d` ordinary kriging に置換。 同様に `DenseANCCImputer` (= `exp005_cache_blend.py:892-953`) も IDW from KNN を kriging に置換可。

### 1.2 実装 spec

**関数 signature** (= 新規 `FormationKriging` クラス、 `FormationPlaneKNN` を drop-in replace):

```python
class FormationKriging:
    """anisotropic 2D ordinary kriging per formation。
    GeostatsPy.GSLIB.make_variogram + geostatspy.geostats.kb2d 経由。
    既存 FormationPlaneKNN と同一 signature (= impute(xy_q, self_wid) → (pred, dist))
    で drop-in replace 可。
    """
    def __init__(
        self,
        well_ids: List[str],
        data_dir: Path,
        # variogram defaults (= 既存 KNN scale 約 1.0 = normalized、 host EDA 経由)
        nug: float = 0.0,
        nst: int = 1,           # 1 構造で開始 (= 2 構造は overfit risk)
        it1: int = 1,            # 1=spherical (= 物理直感に最適合)、 3=Gaussian は smooth 過ぎ
        cc1: float = 1.0,        # contribution は per-formation で variance match 後 fit
        azi1: float = 0.0,       # NE-SW を後で sample variogram で fit
        hmaj1: float = 5000.0,   # major range (ft) (= subagent W §5.2 推定)
        hmin1: float = 2000.0,   # minor range (ft)
        # kriging defaults
        ktype: int = 1,          # ordinary (= local mean 自由)
        ndmin: int = 4,
        ndmax: int = 10,         # 既存 PLANE_K=10 と一致
        radius: float = 10000.0,
    ):
        ...

    def impute(
        self,
        xy_q: np.ndarray,         # (N, 2)
        self_wid: Optional[str] = None,
        # variogram override (= per-CV-fold で再 fit する場合)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (predictions: (N, 6), kriging_variance: (N, 6), min_dist: (N,))"""
        ...
```

**kernel script への inject 位置** (= `exp005_cache_blend/exp005_cache_blend.py`):

| 位置 (file:line) | 改修内容 |
|---|---|
| L799-889 (= `FormationPlaneKNN` クラス全体) | **既存維持**、 そのまま残す (= fallback 用) |
| L892-953 (= `DenseANCCImputer` クラス全体) | 同上維持 |
| L955-971 (= global imputer build) | 新規 `_FK_GLOBAL = FormationKriging(...)` を追加 (= 並列、 既存 `_FI_GLOBAL` 隣) |
| L982-983 (= `_FI_REF`, `_DI_REF`) | 新規 `_FK_REF` を追加、 per-CV-fold 上書き対応 |
| L998-1336 (= `build_well_features` の signature) | 引数追加 `fk: FormationKriging = None` (= `fi`, `di` の隣) |
| L1161 (= `form_ev, knn_d = fi_use.impute(...)`) | **置換**: `form_ev, krg_var_ev, knn_d = fk_use.impute(xy_ev, self_wid=swid)` |
| L1162 (= `form_kn, _ = fi_use.impute(...)`) | **置換**: `form_kn, krg_var_kn, _ = fk_use.impute(xy_kn, self_wid=swid)` |
| L1165-1176 (= per-formation TVT formulas) | **拡張**: `krg_var_ev[:, fi_idx]` を新 feature `tvtF_<formation>_krgvar` として追加、 下流 LGB が epistemic を学習 |

**dependencies** (= `requirements.txt` / Kaggle env):

```text
# Kaggle Docker image (= python:3.10) は numpy>=1.23 を含む
# GeostatsPy は numba を要求、 numpy<2.2 制約あり
geostatspy==0.0.71
numba>=0.58.0
# 既存 numpy, scipy, pandas は満たす
# pip install geostatspy が Kaggle で blocked の場合、 wheel を offline upload (= Datasets として add) する手順
```

**Kaggle dataset preload** (= internet blocked 環境用):
- private dataset として `geostatspy-0.0.71-py3-none-any.whl` を upload
- kernel notebook 先頭で `!pip install --no-deps /kaggle/input/geostatspy-wheel/geostatspy-0.0.71-py3-none-any.whl`

### 1.3 Kaggle runtime budget

**実測予測**:
- 773 wells × 6 formations × kriging call = 4638 call
- 1 call = `kb2d` で N_query=1 (= ROGII の test query は per-row not grid) を 1 回、 0.05 sec (= numba JIT 後、 KDTree 4-10 neighbor)
- 合計: **4638 × 0.05 = 232 sec ≈ 4 min**
- variogram fit (= sample variogram 計算 + spherical model fit) は per-formation で one-shot、 6 × 30 sec = 3 min
- **総計 ≈ 7 min** (= 9 hr 制約に対し余裕大)

注意: `kb2d` は grid 推論向け実装、 ROGII の per-row 推論には wrapper が必要 (= `nx=1, ny=1` で grid を query 点 1 個に潰す or 自前 simple kriging 実装に置換)。 **後者の方が 10x 高速**、 詳細は §1.5 failure mode #2 を参照。

### 1.4 smoke 設計 (= 1-3 wells で動作確認)

**smoke target**: `experiments/exp005/smoke_kriging.py` を新規作成、 以下を確認:

1. **Stage A (= 単体 variogram fit)**: train wells 全 773 well の ANCC formation top depth を抽出 → `geostatspy.geostats.gamv` で sample variogram → `make_variogram` で spherical fit → 出力 `range_maj`, `range_min`, `azi`, `nug`, `cc1` を CSV save
2. **Stage B (= 単 well kriging predict)**: 任意 test well 1 個で `xy_q` 100 点に対し `FormationKriging.impute` 実行 → `(predictions, kriging_var, min_dist)` shape 確認、 NaN / inf 不在確認
3. **Stage C (= 既存 KNN との数値比較)**: 同 test well で `FormationPlaneKNN.impute` と数値比較、 中央差 |Δ| < 5 ft (= 単位 ft 想定) を許容。 大きく外れる well は variogram parameter が不適切な signal
4. **Stage D (= LGB feature 寄与)**: train 全件で kriging predict を `tvtF_<formation>_d` に流し込み、 LGB 1 fold で OOF RMSE 計測 → 既存 KNN feature 比 ±1% 内なら sanity OK

**実行コマンド** (= Kaggle local 互換):
```bash
cd /home/yusuke_kaya/projects/kaggle/ROGII
python experiments/exp005/smoke_kriging.py --stage A --max_wells 50
python experiments/exp005/smoke_kriging.py --stage B --well_id <pick_one>
python experiments/exp005/smoke_kriging.py --stage C --max_wells 5
python experiments/exp005/smoke_kriging.py --stage D --fold 0 --max_wells 100
```

### 1.5 failure modes 3 件 + recovery

**failure mode #1: variogram fit が degenerate (= nugget=cc 状態、 spatial structure なし)**
- **兆候**: `make_variogram` 後の `sample_variogram` 値が flat (= sill 直上で開始)、 range estimation が radius を超える
- **原因**: ROGII の (X, Y) が host EDA で normalized 済みで raw ft 単位ではなく、 想定 range (5000 ft) と座標 scale 不整合
- **recovery**: (a) `host EDA` doc で normalization 因子を再確認、 (b) sample variogram の経験的 range を実測してから model fit、 (c) 失敗時は既存 `FormationPlaneKNN` に fallback (= class-level `if self._degenerate: return knn_impute(...)`)

**failure mode #2: `kb2d` の grid 仕様が ROGII の per-row query に不適合 (= 推論が遅すぎる)**
- **兆候**: well 1 件で kb2d 呼び出し 10+ sec、 773 well で 9 hr 超過 risk
- **原因**: `kb2d` は `(nx, ny)` grid で全点 prediction、 ROGII は per-row scattered query を要求
- **recovery**: (a) 自前 simple_ok_kriging wrapper を `numpy.linalg.solve` ベースで実装 (= scipy.spatial.cKDTree + variogram model evaluation で 1 query 0.005 sec target)、 (b) `kb2d` を batch 化 (= 数 100 query を bounding box grid に統合、 不要点 discard)、 (c) fallback で existing KNN を使う

**failure mode #3: kriging variance が下流 LGB で feature explosion**
- **兆候**: `tvtF_<formation>_krgvar` 6 列追加で feature 総数が +6、 LGB の `feature_fraction` 衝突、 fold variance σ 増加
- **原因**: 既存 feature 100+ 列に 6 列追加で over-fit risk、 特に well 数 < feature 数領域で
- **recovery**: (a) krgvar を 6 列でなく 1 列に集約 (= `mean(krgvar)`, `max(krgvar)`)、 (b) `feature_fraction=0.7` を維持して LGB の bagging で吸収、 (c) krgvar を direct feature でなく **TVT predict の weight** として使用 (= `tvtF_d` の inverse-variance weighted ensemble)

---

## 2. Hack 2: TimeXer × ROGII

### 2.1 数式 + ROGII 適用

**TimeXer 構造** (= Wang 2024-2025 arXiv:2402.19072):

1. **endogenous patch embedding**: target $\mathbf{x} \in \mathbb{R}^{T_{\text{en}}}$ を非重複 patch $\{\mathbf{x}_i\}_{i=1}^P$ (= $\mathbf{x}_i \in \mathbb{R}^{L_p}$, $P = T_{\text{en}} / L_p$) に分割、 各 patch を $\mathbf{e}_i = W_{\text{patch}} \mathbf{x}_i + \mathbf{p}_i \in \mathbb{R}^D$ で embed
2. **global endogenous token**: 学習可能 token $\mathbf{g}_{\text{en}} = \text{Learnable}(\mathbf{x}) \in \mathbb{R}^D$ を patch tokens の先頭に concat (= $\mathbf{H} = [\mathbf{g}_{\text{en}}; \mathbf{e}_1; \ldots; \mathbf{e}_P]$、 $\mathbf{H} \in \mathbb{R}^{(P+1) \times D}$)
3. **exogenous variate embedding**: 各 exogenous variate $\mathbf{y}^{(k)} \in \mathbb{R}^{T_{\text{ex}}}$ を $\mathbf{v}_k = W_{\text{variate}} \mathbf{y}^{(k)} \in \mathbb{R}^D$ で embed (= patch せず、 変量ごとに 1 token、 $K$ exogenous で $\mathbf{V} \in \mathbb{R}^{K \times D}$)
4. **encoder block** (= $L$ 層、 各層で):
   - **self-attention** (= endogenous patches 間): $\mathbf{H}' = \text{SelfAttn}(\mathbf{H})$
   - **cross-attention** (= global token → exogenous): $\mathbf{g}'_{\text{en}} = \text{CrossAttn}(Q=\mathbf{g}_{\text{en}}, K=\mathbf{V}, V=\mathbf{V})$
   - exogenous information flow back: $\mathbf{H}[0] \leftarrow \mathbf{g}'_{\text{en}}$、 next self-attention で endogenous patches へ伝播
5. **prediction head**: 最終層 endogenous patch tokens を flatten → 線形射影で $\hat{\mathbf{x}}_{\text{future}} \in \mathbb{R}^{T_{\text{pred}}}$

**self-attention 数式** (= scaled dot-product、 endogenous patches):
$$
\text{SelfAttn}(\mathbf{H}) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{D}}\right) V, \quad Q, K, V \in \mathbb{R}^{(P+1) \times D}
$$

**cross-attention 数式** (= global token query、 variate key/value):
$$
\text{CrossAttn}(\mathbf{g}_{\text{en}}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{g}_{\text{en}} (W_K \mathbf{V})^\top}{\sqrt{D}}\right) (W_V \mathbf{V}) \in \mathbb{R}^{D}
$$

**ROGII 適用**:
- **endogenous** = horizontal well の MD ordered (TVT_input) sequence、 hidden region で NaN を含む (= TVT_input の visible 部分のみ embed、 hidden は target)
- **exogenous variates** = GR_horizontal, GR_typewell (re-sampled to endogenous T_en), Z_horizontal, X_well, Y_well, ANCC_imputed (= existing FormationPlaneKNN or Hack 1 Kriging output) の 6 variate
- **target**: hidden region の TVT prediction (= 既存 path b の `test_delta` 相当)
- 物理事前 `TVT = -Z + ANCC + b_well` は exogenous variate に Z と ANCC を含む構造で **自然に attention で encode 可能** (= subagent W §7.1)

### 2.2 実装 spec

**ROGII adapter file** (= 新規):
```text
src/rogii/models/timexer_adapter.py
├── class ROGIITimeXerConfig    # configs.* 全 17 attr (= seq_len, patch_len, d_model, n_heads, e_layers, d_ff, dropout, enc_in, ...)
├── class ROGIITimeXerDataset   # PyTorch Dataset (= per-well sample、 endogenous + exogenous 同期)
└── class ROGIITimeXerWrapper   # fit() / predict() で既存 pipeline と統一
```

**model 取り込み path**:
- `thuml/TimeXer` repo を git submodule で添付 (= `external/TimeXer/`)、 license 確認後 commit
- import: `from external.TimeXer.models.TimeXer import Model as TimeXerModel`
- Kaggle dataset preload で `TimeXer.tar.gz` を upload (= internet blocked 対策)

**hyperparameters** (= subagent W §2.4 + arXiv §Implementation Details 確認):

| param | value | 根拠 |
|---|---|---|
| `seq_len` (= $T_{\text{en}}$) | 256 (= ROGII visible region 中央値 580 / 2 を patch_len 16 で割り切れる近似) | 既存 well-level data の visible 区間統計 |
| `pred_len` (= $T_{\text{pred}}$) | 128 (= hidden region 中央値 250 / 2、 padding で吸収) | 既存 hidden mask 統計 |
| `patch_len` (= $L_p$) | 16 (= long-term forecast default) | TimeXer paper Impl Details |
| `d_model` (= $D$) | 256 (= {128, 256, 512} 中央、 ROGII feature scale に適応) | TimeXer paper search range |
| `n_heads` | 8 | conventional Transformer |
| `e_layers` (= $L$) | 2 (= {1, 2, 3} 中央) | TimeXer paper search range |
| `d_ff` | 512 (= 2 × d_model) | conventional |
| `dropout` | 0.1 | TimeXer default |
| `enc_in` | 1 (= endogenous は target TVT のみ、 multivariate ではない) | ROGII の 1 target / multi exogenous 構造 |
| `c_out` | 1 | TVT 1 値予測 |
| `features` | "MS" (= multi-input, single-output) | TimeXer の exogenous 分離 mode |
| `learning_rate` | $1 \times 10^{-4}$ | TimeXer Adam default |
| `batch_size` | 16 (= Kaggle GPU メモリ依存) | per-well sample、 well 数 776 で `776/16 = 49 step/epoch` |
| `train_epochs` | 10 (= early stop) | TimeXer default |
| loss | L2 | TimeXer default、 後で Huber に変更可 |
| optimizer | Adam | TimeXer default |

**kernel script への inject 位置** (= `exp009_case_e_edge_o.py`):

| 位置 (file:line) | 改修内容 |
|---|---|
| L1-30 (= imports) | `from external.TimeXer.models.TimeXer import Model as TimeXerModel` 追加 |
| L260-280 (= Edge R knobs 隣) | `TIMEXER_ENABLE = True / TIMEXER_WEIGHT = 0.3 / TIMEXER_CKPT = '/kaggle/working/timexer.pt' / TIMEXER_PRED_LEN = 128` knobs 定義 |
| L2664 (= 9.5 Edge R section の前) | **新 section 9.4 "TimeXer 公開 path c blend"** 追加、 `class TimeXerPathC` 定義、 `fit_timexer(train_df, feature_cols) -> torch.nn.Module` + `predict_timexer(model, test_df) -> np.ndarray` 関数を実装 |
| L4343 (= Step F.5 Edge R の前) | **新 Step F.4 "TimeXer path c blend"** 追加、 既存 `test_delta = w_path_b * test_delta_b + (1 - w_path_b) * test_delta_a` の後段に `test_delta = w_c * test_delta_c + (1 - w_c) * test_delta` で blend |

(=  既存 path b blend と並列の **path c** として実装、 Edge R / TimeXer / path b の 3-way weighted blend を `w_path_b, w_path_c` で制御。 NM ensemble に追加することも可)

### 2.3 Kaggle runtime budget

**実測予測** (= Kaggle GPU T4 x2、 16 GB VRAM):

- **train**: 776 well × 10 epoch × 49 step (= bs=16) × 1 forward+backward 0.05 sec = **19000 sec ≈ 5.3 hr**
- **inference**: 776 well × 1 forward 0.02 sec = **15.5 sec**
- **合計 ≈ 5.3 hr** (= 9 hr 制約に対し余裕は 3.7 hr、 他 hack と合算は risk)

**runtime 短縮 path**:
- (a) `train_epochs=5` に短縮 (= 約 2.7 hr)、 ただし収束不足で LB drop 可能性
- (b) `d_model=128` に半減 (= 約 1.4 hr)、 representation 容量低下
- (c) **pretrain on train+val 全件 → freeze + per-well fine-tune 1 epoch のみ** (= 約 2 hr + 0.5 hr) ← **数理本質的に MEMO / Edge R と統合可能 path**
- (d) `e_layers=1` に削減 (= 約 2.6 hr)、 depth 不足

### 2.4 smoke 設計

**smoke target**: `experiments/exp009/smoke_timexer.py` を新規作成。

1. **Stage A (= dataset loader sanity)**: ROGIITimeXerDataset で 5 well を load、 `(endogenous_patch, exogenous_variate, target)` の shape 確認、 NaN 不在確認
2. **Stage B (= forward pass)**: bs=4、 1 batch forward + backward、 loss tensor が finite 確認、 GPU memory < 4 GB 確認 (= Kaggle T4 16 GB の 25% 上限)
3. **Stage C (= 1 epoch overfitting on 5 wells)**: 5 well で 50 epoch train、 train loss が monotonically decrease 確認 (= モデルが少なくとも memorize 能力ありの sanity)
4. **Stage D (= CV fold 0 train + OOF predict)**: GroupKFold fold 0 (= train 620 well / val 156 well) で 5 epoch train、 OOF RMSE を計測。 既存 path b OOF RMSE 比 ±10% 内なら baseline 通過。

**実行コマンド**:
```bash
python experiments/exp009/smoke_timexer.py --stage A --max_wells 5
python experiments/exp009/smoke_timexer.py --stage B
python experiments/exp009/smoke_timexer.py --stage C --epochs 50 --max_wells 5
python experiments/exp009/smoke_timexer.py --stage D --fold 0 --epochs 5
```

### 2.5 failure modes 3 件 + recovery

**failure mode #1: 9 hr runtime 超過 (= train 5.3 hr + Edge R 1.7 hr + Hack 1 0.1 hr + 既存 pipeline 1-2 hr = 8.1-9.1 hr)**
- **兆候**: kernel が timeout 直前で submission.csv 不出力、 過去 exp009 v4 で 8.5 hr 実績ありで余裕薄
- **原因**: TimeXer train が 5 hr 占有
- **recovery**: (a) train_epochs=5 + d_model=128 で 1.4 hr に圧縮、 (b) **pretrain offline** (= Kaggle Notebook で先に train、 checkpoint を private dataset で kernel に preload)、 (c) inference のみ kernel で実行 (= 15 sec で完了)、 (d) 失敗時は `TIMEXER_ENABLE=False` で disable

**failure mode #2: hidden region が pred_len=128 を超える well で predict 切り捨て**
- **兆候**: 一部 well で predict tail が NaN、 fallback で last_known_tvt 補填
- **原因**: ROGII の hidden region 長さは well 毎に異なる、 中央値 250 だが分散大
- **recovery**: (a) **rolling window predict** (= 128 step 進めて 256 visible に shift、 再 predict、 256 visible の更新は previous predict を input に concat)、 (b) `pred_len=256` に変更で対応、 ただし memory 増、 (c) fallback で長すぎる部分は既存 path b prediction で埋める

**failure mode #3: license / Kaggle internet block で TimeXer repo install 不可**
- **兆候**: `pip install -r requirements.txt` blocked、 `git clone` blocked
- **原因**: Kaggle kernel は default internet off、 dataset 経由でのみ external code 取込み
- **recovery**: (a) `thuml/TimeXer` を fork、 必要 files (= `models/TimeXer.py`, `layers/*.py`) を ROGII repo の `external/TimeXer_minimal/` に直接 copy、 license = (TimeXer リポは README に license 明記なし、 後で確認 / 著者連絡)、 (b) Kaggle Datasets として `timexer-minimal-2026-05-11` を private upload

---

## 3. Hack 3: MEMO 流 random_stretch × entropy minimization fine-tune

### 3.1 数式 + ROGII 適用

**MEMO 原典** (= Zhang 2022 NeurIPS、 classification context):

各 test sample $\mathbf{x}$ に対し $N$ augmentation $\{a_i(\mathbf{x})\}_{i=1}^N$ を作り、 **marginal entropy** を最小化 (= MEMO 公式実装は `logits.logsumexp(dim=0)` で marginal average → entropy、 = classification 系):
$$
\mathcal{L}_{\text{MEMO}}(\mathbf{x}; \theta) = -\sum_{y} \bar{p}_\theta(y | \mathbf{x}) \log \bar{p}_\theta(y | \mathbf{x})
$$
ここで $\bar{p}_\theta(y | \mathbf{x}) = \frac{1}{N} \sum_{i=1}^N p_\theta(y | a_i(\mathbf{x}))$。 SGD lr=0.005, niter=1 (= per sample 1 step)、 N=batch_size=32 (= MEMO default)。

**regression での適応** (= ROGII):

ROGII の target は continuous TVT、 classification ではない。 entropy の continuous version は **differential entropy** $H(p) = -\int p(t) \log p(t) dt$ だが、 deterministic prediction では degenerate。 以下 2 path:

**path α (= heteroscedastic regression を導入、 真の MEMO 流)**:
- network 出力を $(\hat{\mu}, \hat{\sigma}^2)$ の 2-head に拡張 (= Beta-NLL or Gaussian NLL)
- augmentation $a_i$ 適用後の予測分布 $\mathcal{N}(\hat{\mu}_i, \hat{\sigma}_i^2)$ を marginal Gaussian で近似:
  $$
  \bar{\mu} = \frac{1}{N} \sum_i \hat{\mu}_i, \quad \bar{\sigma}^2 = \frac{1}{N} \sum_i \hat{\sigma}_i^2 + \frac{1}{N} \sum_i (\hat{\mu}_i - \bar{\mu})^2
  $$
- entropy loss: $\mathcal{L} = \frac{1}{2} \log(2\pi e \bar{\sigma}^2)$
- minimize → augmentation 後 predict 分布が **sharp** になる方向に encoder 学習

**path β (= prediction variance を直接最小化、 simplified MEMO)**:
- deterministic $\hat{\mu}_i = f_\theta(a_i(\mathbf{x}))$ のみ計算
- loss: $\mathcal{L} = \frac{1}{N} \sum_i (\hat{\mu}_i - \bar{\mu})^2 = \text{Var}_i(\hat{\mu}_i)$
- 数学的 interp: augmentation-invariance を強制 (= 同じ well の random_stretch 違いで予測が一致するよう encoder 学習)
- entropy formulation を回避、 簡潔だが MEMO 原論の差別化 (= marginal vs per-sample entropy) なし

**選択軸**: path α は理論厳密だが既存 LGB / NN は variance head なし → reconstruction 工数大。 path β は既存 R kernel / LGB 出力を直接使え、 工数小、 ただし MEMO の本質 (= aleatoric/epistemic 分離 by augmentation marginalization) を捨てる。

**random_stretch augmentation** (= SPWLA 2023 Dreamstar):
$$
a_s(\mathbf{x}_i) = \mathbf{x}_{i \cdot s}, \quad s \sim \text{Uniform}(0.95, 1.05)
$$
= MD 方向に 5% stretch / compress (= 地層厚みの well-to-well variation を data augmentation で encode)。 N=16 augmentation 程度 (= subagent W §6.3)。

**ROGII 適用** (= 既存 Edge R との直交性):
- 既存 Edge R (= `exp009_case_e_edge_o.py:2664-2796`) は **LGB continued training** で test visible region を pseudo-hidden に変換し、 instance-level fine-tune
- MEMO は **augmentation marginal entropy 最小化** で **encoder-level** fine-tune (= LGB の場合は tree structure を変更しない、 leaf value のみ stretch-invariant に reweight)
- 数理的に直交、 同時適用可

### 3.2 実装 spec

**path β (= simplified MEMO) を採用** (= path α は既存 LGB に variance head なしで工数大、 §3.5 failure mode で path β に絞る根拠を提示):

**新規関数** (= `exp009_case_e_edge_o.py` に追加):

```python
def memo_random_stretch(
    hw_visible: pd.DataFrame,
    stretch_factor: float,  # ~ Uniform(0.95, 1.05)
) -> pd.DataFrame:
    """MD 方向に stretch、 TVT_input / GR 等を index 再 sample。
    end point は linear interp で保つ。 leak 防止: hidden TVT には触れない。"""
    n = len(hw_visible)
    new_n = int(n * stretch_factor)
    idx_new = np.linspace(0, n - 1, new_n)
    hw_stretched = pd.DataFrame()
    for col in ["TVT_input", "GR", "X", "Y", "Z"]:
        if col in hw_visible.columns:
            hw_stretched[col] = np.interp(idx_new, np.arange(n), hw_visible[col].to_numpy())
    return hw_stretched

def memo_marginal_variance_loss(
    booster,                    # LGB booster (= already Edge R fine-tuned or base)
    X_R: np.ndarray,             # (N_aug, n_features)
    X_test: np.ndarray,          # (N_test, n_features)
    feature_cols: List[str],
    n_aug: int = 16,
) -> Tuple[np.ndarray, float]:
    """augmentation N=16 で test_delta の variance を計算、 marginal predict 返す。
    LGB の場合は leaf-value update を `init_model` + 短 iteration 続行で実装。"""
    preds_aug = np.zeros((n_aug, X_test.shape[0]))
    for i in range(n_aug):
        # X_R_aug = stretched-augmented visible features (= memo_random_stretch 経由)
        ...
        preds_aug[i] = booster.predict(X_test_aug_i)
    pred_mean = preds_aug.mean(axis=0)
    pred_var  = preds_aug.var(axis=0)
    return pred_mean, float(pred_var.mean())  # mean prediction + 平均 variance (= loss proxy)

def memo_edge_r_extend(
    base_models: dict,
    visible_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    n_aug: int = 16,
    n_iter: int = 1,            # MEMO default = 1
    lr_mul: float = 0.3,         # MEMO SGD lr=0.005、 Edge R lr_mul=0.5 の半分
    stretch_range: Tuple[float, float] = (0.95, 1.05),
) -> dict:
    """既存 edge_r_continued_train_lgb を augmentation marginal で extend。
    各 aug iter で X_R を stretch、 全 aug iter の平均で online_preds を返す。"""
    ...
```

**kernel script への inject 位置** (= `exp009_case_e_edge_o.py`):

| 位置 (file:line) | 改修内容 |
|---|---|
| L268-280 (= Edge R knobs) | 新 knob 追加: `MEMO_ENABLE = True / MEMO_N_AUG = 16 / MEMO_NITER = 1 / MEMO_STRETCH_LO = 0.95 / MEMO_STRETCH_HI = 1.05 / MEMO_LR_MUL = 0.3` |
| L2664-2796 (= Edge R section) | 同 section 内に **subsection 9.5.1 "MEMO augmentation extension"** 追加、 上記 3 関数を定義 |
| L4343-4469 (= Step F.5 Edge R) | **拡張**: `edge_r_continued_train_lgb` 呼び出しを `memo_edge_r_extend` で wrap (= MEMO_ENABLE=True のとき N=16 aug の augmented training + marginal predict、 False で従来 Edge R) |
| L4444-4451 (= blend) | `EDGE_R_BLEND_W` を MEMO 用に分離: `test_delta = w_R * (memo_marginal_predict if MEMO_ENABLE else online_predict) + (1 - w_R) * test_delta` |

### 3.3 Kaggle runtime budget

**実測予測**:
- 既存 Edge R (= no MEMO): 773 well × 1 continued train × 100 boost round × 0.01 sec/round = **773 sec ≈ 13 min** (= `EDGE_R_HARD_TIMEOUT_S = 900` 内)
- MEMO N=16 augmentation 追加: 773 × 16 × continued train + predict = **13 min × 16 = 3.5 hr** (= 9 hr 制約に対し余裕中)
- subagent W §6.3 推定 (= 1.7 hr) との差: subagent W は 1 forward 0.1 sec 想定、 LGB 100 boost で 1 sec/aug = 16 sec/well の概算。 実測でこれより 10x 大きい (= 100 sec/well) なら 21 hr で OUT

**runtime 短縮 path**:
- (a) `MEMO_N_AUG = 8` に半減 (= 1.75 hr)
- (b) `EDGE_R_NUM_BOOST = 50` に半減 (= 1.75 hr)
- (c) augmentation を well 内で MD shuffle ではなく **per-batch stretch** に変更 (= 1 boost round で N_aug を同時処理、 順次でない、 N_eff 1)
- (d) `MEMO_ENABLE` を **subset of wells (= visible 短すぎる well のみ)** に限定 (= 全体推論時間 1/3 想定)

### 3.4 smoke 設計

**smoke target**: `experiments/edge_r_online/smoke_memo.py` を新規作成。

1. **Stage A (= random_stretch sanity)**: 1 well で `memo_random_stretch` を s=0.95, 1.0, 1.05 の 3 値で実行、 stretched DataFrame の shape / NaN 不在 / TVT_input range 一致を確認
2. **Stage B (= marginal predict variance 計測)**: 5 well で N_aug=16 の augmented predict variance を計測、 predict_var の中央値 < 0.5 ft² (= predict が augmentation で大きく揺れないこと、 揺れ過ぎ = degenerate signal)
3. **Stage C (= MEMO-extended Edge R vs base Edge R)**: 1 fold で MEMO-extended Edge R predict と通常 Edge R predict の OOF RMSE 差を計測、 ±5% 内なら sanity 通過
4. **Stage D (= 9 hr runtime check)**: 773 well 全件で MEMO_N_AUG=16, NITER=1 を実行、 wall clock < 4 hr 確認 (= timeout buffer 残し)

**実行コマンド**:
```bash
python experiments/edge_r_online/smoke_memo.py --stage A --well_id <pick_one>
python experiments/edge_r_online/smoke_memo.py --stage B --max_wells 5 --n_aug 16
python experiments/edge_r_online/smoke_memo.py --stage C --fold 0 --max_wells 100
python experiments/edge_r_online/smoke_memo.py --stage D --max_wells 773
```

### 3.5 failure modes 3 件 + recovery

**failure mode #1: random_stretch で hidden mask 位置がズレ、 leak 発生 risk**
- **兆候**: stretched visible region の末尾 index が hidden mask の境界を侵食、 hidden region の GR / Z が augmented input に混入
- **原因**: stretch は MD 軸全体に作用、 visible / hidden の境界が float index になり境界処理が曖昧
- **recovery**: (a) **stretch を visible region 内に閉じる** (= stretch 後の hidden region 境界を strict re-compute、 stretched visible の末尾 row を新 hidden start に再設定)、 (b) hidden region の row を stretched dataset から **完全 drop** (= MEMO は visible reconstruction のみで loss 計算)、 (c) augmentation 前後で `TVT_input.notna()` mask を bitwise 一致確認、 不一致なら drop

**failure mode #2: path β (= simple variance min) では MEMO 本質 (= marginal entropy) を捨てるため LB 寄与限定的**
- **兆候**: smoke Stage C で OOF RMSE 改善が ±0.5% 程度 (= noise 範囲)、 期待 -0.1〜-0.3 ft に到達せず
- **原因**: variance min は augmentation-invariance を強制するだけ、 epistemic uncertainty marginalize の効果なし
- **recovery**: (a) **path α (= LGB を Beta-NLL に置換)** に move、 ただし工数 +3-5 day、 (b) 既存 LGB を MEMO 流の代わりに **single 大 N_aug の bagging** に置換 (= N=64 で aggressive)、 (c) MEMO は 9 切り contribution 小と判定し disable、 他 2 hack に集中

**failure mode #3: visible 短すぎ wells で stretch 後の visible 行数が `EDGE_R_MIN_VISIBLE_ROWS=100` を下回り skip 連鎖**
- **兆候**: well 数 773 の 30%+ で MEMO skip (= visible row 100 未満で MEMO_extend 適用不可)、 全体 LB contribution 1/3 に縮小
- **原因**: stretch 0.95 倍で visible n=110 → 104、 ぎりぎり閾値、 well の visible 統計が想定より短い
- **recovery**: (a) `EDGE_R_MIN_VISIBLE_ROWS` を 80 に下げる (= MEMO のみ別 threshold)、 (b) stretch_lo を 0.97 に narrow (= 縮小幅縮小)、 (c) short-visible wells に対しては stretch でなく **GR noise injection** に切替 (= MEMO は augmentation 形式に依存、 noise でも N_aug marginal は数学的に等価)

---

## 4. 3 hack 統合 implementation order (= 5/13 以降の exp 設計)

### 4.1 exp 番号 + base kernel + inject 順序

> 注: 推奨 exp 順序ではなく **構造的選択肢**。 中央 / 開発者が `kaggle/CLAUDE.md §11.2` 5 問で判断する。

| exp | base kernel | inject hack | 工数 (d) | 9 hr 制約余裕 | LB upper |
|---|---|---|---|---|---|
| **exp010_a** | `exp005_cache_blend` | Hack 1 (Kriging) 単体 | 2-3 | 8 hr | -0.3 |
| **exp010_b** | `exp005_cache_blend` | Hack 1 + Hack 3 (MEMO) | 4-5 | 5 hr | -0.5 |
| **exp011** | `exp009_case_e_edge_o` v4 | Hack 3 (MEMO) のみ (= Edge R 拡張) | 2-3 | 5 hr (= 既 8 hr 実績 + MEMO 3.5 hr で 11 hr OUT、 N_aug=8 で吸収) | -0.3 |
| **exp012_a** | `exp009_case_e_edge_o` v4 | Hack 1 + Hack 3 (Kriging + MEMO) | 4-6 | 5 hr | -0.5 |
| **exp012_b** | `exp009_case_e_edge_o` v4 | Hack 2 (TimeXer) 単体 | 3-4 | 3 hr (= Edge R + TimeXer + Kalman で 9 hr 危険) | -0.5 |
| **exp013** | `exp012_a` (= Hack 1 + Hack 3 通過後) | + Hack 2 (TimeXer offline pretrain) | 5-7 | 6 hr (= TimeXer pretrain offline) | -0.7 |

### 4.2 dependency 整理 (= 3 hack 間の互換性 / conflict)

| pair | 互換性 | conflict / synergy |
|---|---|---|
| Hack 1 (Kriging) × Hack 2 (TimeXer) | 互換 | Kriging output を TimeXer の exogenous variate に直接渡せる (= ANCC_imputed 列が新 kriging 値に置換)、 数理的 synergy 大 |
| Hack 1 (Kriging) × Hack 3 (MEMO) | 互換 | Kriging は feature build 段、 MEMO は test-time fine-tune 段、 完全直交 |
| Hack 2 (TimeXer) × Hack 3 (MEMO) | 互換 | TimeXer は NN encoder、 MEMO の augmentation marginal entropy は NN に **直接適用可** (= path α、 NN は `(μ, σ)` head 自然) |
| Hack 1 × Hack 2 × Hack 3 全合体 | 互換 | TimeXer encoder + Kriging exogenous + MEMO augmentation marginal で **完全 stack**、 ただし 9 hr 制約が boundary case、 GPU + offline pretrain 前提 |

**conflict なし**。 ただし runtime 9 hr boundary は全合体で boundary、 offline pretrain + inference-only kernel で対応必須。

---

## 5. failure recovery plan (= 各 hack 失敗時の fallback)

| hack | 主要 failure | 即時 fallback | 段階 fallback |
|---|---|---|---|
| Hack 1 Kriging | variogram degenerate (= §1.5 #1) | `FormationKriging._degenerate = True` で `FormationPlaneKNN.impute` に切替 (= class-level fallback) | sample variogram fit を per-CV-fold ではなく global one-shot に変更、 anisotropy を等方に縮退 |
| Hack 1 Kriging | runtime 超過 (= §1.5 #2) | `kb2d` の grid 仕様を回避し自前 simple_ok_kriging を使用 | KNN k=10 のままで kriging variance だけ近似 (= per-neighbor distance ベース proxy) |
| Hack 2 TimeXer | 9 hr 超過 (= §2.5 #1) | `TIMEXER_ENABLE=False` で disable、 既存 path b で完結 | offline pretrain + checkpoint preload → inference-only kernel に切替 |
| Hack 2 TimeXer | hidden > pred_len (= §2.5 #2) | rolling window predict (= 128 step shift) で対応 | `pred_len=256` に変更、 memory 増容認 |
| Hack 3 MEMO | leak risk (= §3.5 #1) | stretch を visible region 内に strict 閉じる、 hidden row 完全 drop | MEMO 自体を disable、 既存 Edge R のみで運用 |
| Hack 3 MEMO | variance min が noise (= §3.5 #2) | path α (= Beta-NLL head) に move、 工数 +3-5d | MEMO disable、 N_aug=64 bagging で代替 |

---

## 6. 残課題

### 6.1 検証できなかった claim

- **GeostatsPy `kb2d` の grid 仕様 vs scattered query 性能**: §1.3 で「per-row 推論で 0.05 sec」と推定したが、 実測未実施。 §1.5 #2 で fallback 提示済だが、 smoke Stage B で wall clock 実測が必要
- **TimeXer の Kaggle GPU T4 上 train 時間**: §2.3 で 5.3 hr 推定、 paper Impl Details の batch_size 不明 (= ETT default は 32 想定だが、 ROGII の sequence length 256 で T4 16 GB OOM risk あり、 実測必要)
- **MEMO の path α (= Beta-NLL) を LGB で実装する手順**: LGB 自体は heteroscedastic objective を内蔵せず、 `objective='regression'` + 別 head 追加が必要 → NN 系 base に limit する可能性。 §3.5 #2 で path α move 言及済だが、 工数 +3-5 day の根拠は LGB → NN 移植の実測必要

### 6.2 さらなる deeper 余地

- **Hack 1 + Hack 2 統合**: kriging variance を TimeXer の exogenous variate として渡す (= per-row epistemic を attention で encode) は数理的に未検証、 -0.5 ft 上限超過可能性あり (= 別 doc で深掘り余地)
- **Hack 3 を WLFM pretrain (= subagent W §1.1 hack #5) と統合**: WLFM checkpoint 上に MEMO 流 fine-tune を適用、 cross-well generalization × per-well transductive の二段構成、 ただし WLFM weights public availability 未確認 (= subagent W §8.2)
- **Hack 1 の variogram parameter を per-CV-fold で再 fit**: 現 spec は global one-shot fit、 fold 毎の anisotropy ばらつき吸収余地

### 6.3 並列 worker (= subagent X) との conflict 範囲

- 本 doc は branch `docs/3hack-spec-2026-05-11` で書込、 subagent X は branch `docs/paradigm-deeper-2026-05-11` (= Mamba/TabPFN v2/PINN/PySR) で書込 → **file-level disjoint**、 git conflict なし
- ただし将来 merge 時に `docs/research/` 配下が 2 branch で同時生成、 同 file 衝突は **subagent W 既存 doc (= `academic-literature-deeper.dense.md`)** のみ、 本 doc とは別 file で安全

---

## 7. 関連 doc / source

- subagent W: `docs/research/academic-literature-deeper.dense.md` @ branch `docs/paradigm-deeper-2026-05-11` (= 752 行、 §2.4 TimeXer / §5 Geostats / §6.3 MEMO / §7 hack priority)
- exp005 base: `kaggle_kernels/exp005_cache_blend/exp005_cache_blend.py` (= 2320 行)
- exp009 base (= Edge R 含): `kaggle_kernels/exp009_case_e_edge_o/exp009_case_e_edge_o.py` (= 4503 行)
- 一次資料: TimeXer https://arxiv.org/html/2402.19072v3 / GeostatsPy https://github.com/GeostatsGuy/GeostatsPy / MEMO https://github.com/zhangmarvin/memo / GeostatsPy demo https://geostatsguy.github.io/GeostatsPyDemos_Book/GeostatsPy_kriging.html
- Context7 GeostatsPy: `/geostatsguy/geostatspy` (= 378 snippets, Score 84.9)
- 主道原則: `~/.claude/CLAUDE.md` 主道フレームワーク、 `kaggle/CLAUDE.md §11` 「優勝本質性」 criterion

---

## 8. 更新履歴

- 2026-05-11 初版 (= branch `docs/3hack-spec-2026-05-11`、 subagent W 提示 3 hack を kernel script レベルまで refine)
