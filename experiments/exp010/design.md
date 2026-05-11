# exp010 設計書 — Stratified Edge Q Fold + Adversarial Validation Drop

> 起源: subagent V (= `docs/data-mining-2026-05-11`) H10 発見 = exp007 per-fold RMSE = [10.06, 9.20, **12.07**, 10.55, **11.67**] で σ_fold ≈ 1.175 ft、σ_base ≈ 0.121 ft、比 **9.7 倍**。Jensen 下限 CV ≈ **10.77 ft が天井**。
> 根拠 doc: `docs/dev/2026-05-11-h10-postmortem-and-fold-reform.dense.md` §1-2
> 中央指示: ユーザー指示「9 切らないと優勝は無理」 + H10 = **fold 構造改革が 9 切り core**
> 期待 LB: **9.5-9.7** (= base improve なしで -1.0 ft 改善射程)
> base: `kaggle_kernels/exp009_case_e_edge_o/exp009_case_e_edge_o.py` (= subagent T の v3、Huber+hetero+MEDIAN+path b)
> branch: `feat/phase-6-exp010-fold-reform` ← `feat/phase-5-edge-r-online`

---

## 1. H10 mechanism と数学的天井

### 1.1 観測 evidence (= subagent V 抽出)

exp007 自前 4 base per-fold valid RMSE (`outputs/kernel_logs/exp007/rogii-exp007-edge-qm.log`):

| fold | lgb_own0 | lgb_own1 | lgb_own2 | cb_own | per-fold mean |
|---:|---:|---:|---:|---:|---:|
| 0 | 9.86 | 10.03 | 10.19 | 10.16 | 10.06 |
| 1 | **9.02** | **9.14** | 9.40 | **9.22** | **9.19** (最良) |
| 2 | **12.05** | **12.02** | **12.19** | **12.03** | **12.07** (最悪) ⚠️ |
| 3 | 10.44 | 10.64 | 10.67 | 10.46 | 10.55 |
| 4 | **11.67** | **11.70** | **11.95** | **11.35** | **11.67** ⚠️ |

統計:
- σ_fold = **1.175 ft** (= fold 間 std)
- σ_base = **0.121 ft** (= 同 fold 内 base 間 std)
- 比 σ_fold / σ_base = **9.7 倍**

### 1.2 Jensen 下限 (= 数学的天井)

CV (overall OOF RMSE) は fold 間で MSE 平均:

$$
\overline{\text{MSE}} = \overline{\text{RMSE}}^2 + \sigma_{\text{fold}}^2
$$

- $\overline{\text{RMSE}} \approx 10.71$
- $\sigma_{\text{fold}}^2 \approx 1.38$
- $\overline{\text{MSE}} \approx 114.7 + 1.38 = 116.1$
- $\text{CV} \approx \sqrt{116.1} \approx$ **10.77**

→ **base improve だけで CV を下げても σ_fold が dominant、10.77 が天井**。
→ CV-LB ratio ≈ 1 (= subagent T path b で fold-misalign 解消後) なら LB ≈ 10.4-10.6 帯固定。
→ **9 切り (LB 8.x) には fold 構造改革が必須**。

---

## 2. 実装方針 (= 2 step 改修)

### 2.1 Step 1: stratified Edge Q fold

**目的**: fold 2/4 で RMSE 12 帯になる原因 = **困難 wells の偏在**。`per-well-stats.parquet § tw_gr_resid_std` (= typewell vs horizontal GR residual std、p50=13.89, p90=18.88, max=63.48) で stratify、各 fold に困難 wells を分散配置。

**stratify key 選定根拠**:
- `tw_gr_resid_std` は per-well-stats.parquet の 22 列中、**well 難度の代表指標**として first-principles.dense.md §2.7 §B で同定済
- 高値 (> p90 = 18.88) wells = typewell と horizontal GR pattern alignment が困難 = TVT prediction error 大
- exp007 fold 2/4 で RMSE 12 帯になる wells はこの indicator で識別可能 (= 仮説 H1 と整合)

**実装 (src/rogii/cv.py に追加)**:

```python
def build_stratified_edge_q_folds(
    train_df: pd.DataFrame,
    train_dir: Path,
    per_well_stats_df: pd.DataFrame,  # per-well-stats.parquet
    n_splits: int = 5,
    seed: int = 42,
    stratify_key: str = "tw_gr_resid_std",
    n_bins: int = 4,  # quartile
    fallback_by_well: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """typewell content-hash group + stratify (by tw_gr_resid_std quartile)
    でフォールドを構築。

    1. typewell hash で各 well を group 化 (= 既存 Edge Q ロジック踏襲)
    2. group ごとに代表 well の tw_gr_resid_std を取り、quartile bin (0..3) に
       分類
    3. quartile 毎に group を shuffle 後 round-robin で fold に割り当てる
    4. 各 fold に各 quartile が均等に入る → fold variance 削減
    """
```

**数学的根拠**: stratified group sampling で fold variance を best case で $\sigma_{\text{fold}}^2 \to \sigma_{\text{fold}}^2 / K$ に削減 (= K = quartile 数 = 4)。
**期待**: σ_fold 1.18 → **0.5 以下**、CV 10.77 → 10.0-10.3、LB 10.677 → 9.8-10.1。

**leak 制約**:
- typewell hash group の制約 (= 同 hash の wells は同 fold に閉じ込める) は **stratify より優先**。stratify は group 単位で行う、行単位ではない
- per-well-stats.parquet は train wells (773) のみ参照、test 一切非参照
- 既存 `verify_edge_q_no_leak()` を継続使用

### 2.2 Step 2: adversarial validation drop

**目的**: train (773 wells) vs test (200 wells、host hidden) の **distribution shift** を計測し、test と離れた train wells を sample_weight 0.5 で reduce or drop。

**実装方針**:

1. **特徴量**: per-well-stats.parquet の 22 列のうち、train/test 両方で計算可能な **well-level features** のみ採用 (= visible_ratio, gr_noise_std, dtvt_std, ar1_phi, gr_mean, gr_std, n_rows, tvt_range, z_range)
2. **classifier**: LightGBM binary (class 0 = train、class 1 = test)、5-fold stratified CV で AUC 計測
3. **threshold**:
   - AUC < 0.6: shift 軽微、drop しない (= adv_drop_disabled)
   - AUC ≥ 0.6: 各 train well の test-class predict probability を取得、**下位 q=20% wells** を sample_weight 0.5 (= reduce、drop ではない)
4. **kernel inject 位置**: 既存 `# ── Step E.1: build heteroscedastic sample_weight` の **直後**、`w_train_full` を再度 reweight する

**数学的根拠**:
- domain adaptation theory: target risk ≤ source risk + $d_{\mathcal{H}}(P_S, P_T)$ (= H-divergence)
- adversarial AUC = $2 \cdot d_{\mathcal{H}}(P_S, P_T) - 1$ ≒ classifier の wand 値
- test-likely wells を upweight (= source-only training の生 RMSE は若干悪化、しかし test 上 RMSE は改善)

**leak guard (= CRITICAL)**:
- AV classifier は **per-well feature の分布**を classify、TVT_input (= train hidden) と TVT (= target) は一切参照しない
- test wells (200) は host data なので test/*__horizontal_well.csv (= visible-only feature) からのみ抽出
- per-well-stats は train wells のみだが、test wells は **kernel 内 build_dataset(TEST_DIR)** で同じ schema で再計算する (= 同 feature の独立計算)
- adversarial fit に target column を含めない (= sklearn `model.fit(X_adv, y_adv)` で X に target 含まない)

**期待**: AUC 0.6-0.75 想定 (= Eagle Ford / Austin Chalk / Buda 地層の geographic 偏在で軽度 shift)、drop 後の OOF RMSE 改善 + LB ratio 安定化。

### 2.3 path b 互換性 (= subagent T の 4 改修との接続)

base kernel `exp009_case_e_edge_o.py` には既に以下が inject 済:
- (a) **Huber loss** (= robust regression)
- (b) **heteroscedastic sample_weight** (= per-well-stats `b_well_resid_std` の 1/sigma)
- (c) **multi-seed MEDIAN** (= OWN_USE_MEDIAN_SEEDS=True で lgb 3 seed + cb 2 seed の MEDIAN)
- (d) **path b blend** (= kb simple-avg + own Ridge + 1D grid)

exp010 で追加するのは:
- **Step 1**: `build_edge_q_folds` を `build_stratified_edge_q_folds` に置換 (= fold_id 配分のみ変更)
- **Step 2**: `w_train_full` の構築直後に **adversarial reweight** を append (= multiplicative re-scaling)

→ 既存 4 改修と **直交** (= 互いに independent)、組合せ可能。

---

## 3. 完了条件 (= acceptance criteria)

| # | 項目 | 検証手段 |
|---|---|---|
| 1 | `src/rogii/cv.py` に `build_stratified_edge_q_folds` 関数追加 | grep + import test |
| 2 | `kaggle_kernels/exp010_fold_reform/exp010_fold_reform.py` 作成、stratified fold + AV drop inject 済 | grep `build_stratified_edge_q_folds`、`adversarial` keyword |
| 3 | kernel-metadata.json に 4 dataset_sources (per-well-stats 含む) attach | json parse |
| 4 | ローカル smoke で **σ_fold が 1.18 → 0.5 以下**を mock data で観測 | smoke script output (notes.md に記録) |
| 5 | leak guard PASS (= AV classifier が hidden TVT/target を参照しない) | smoke script assert |
| 6 | runtime cap 内 (= 既存 6-8 hr に +30 min 以内) | smoke で per-fold step 数を換算 |
| 7 | 中間 commit 5+ 件、origin push 済 | git log |
| 8 | experiments/exp010/{design.md, notes.md} 完備 | ls |

**kernel push は実行しない** (= 中央指示)。

---

## 4. failure modes 3 件 + recovery

### 4.1 failure mode A: typewell hash group 数 (= 13 unique groups + 738 singleton) で stratify が機能しない

**兆候**: smoke で 4 quartile が各 fold に出現せず、ある fold で 1 quartile が空。

**原因**: stratify は group 単位で行うが、small singleton group が多すぎると quartile 分散が破綻。

**対策**:
1. group 単位の stratify ではなく、まず group 内 well の **代表 well** (= 最初の well_id) で quartile を計算
2. quartile ごとに group list を shuffle、round-robin で fold に分配
3. fallback: stratify が成立しない場合 (= n_quartile × n_splits < n_groups)、**naive Edge Q (= 既存ロジック) に戻る**、log warning

### 4.2 failure mode B: adversarial classifier が AUC ~0.5 (= shift なし)

**兆候**: 自前計算 AUC が 0.50-0.55 で stable、test と train が区別不能。

**原因**: per-well features の分布が同質 = ROGII の data split が random 配分された。

**対策**:
1. AUC < 0.6 で `adv_drop_disabled = True` flag を set、AV drop logic を skip
2. log "AUC=0.55 → AV drop disabled、stratified fold のみで進行"
3. それでも exp010 = stratified fold 効果 (= σ_fold 削減) は得られる、LB 改善幅は -0.5 程度に縮小

### 4.3 failure mode C: stratified fold で逆に fold variance 増加

**兆候**: smoke で σ_fold 1.18 → 1.5 とか悪化。

**原因**: stratify key (`tw_gr_resid_std`) が fold variance の真因ではなく、別の隠れ変数 (= hidden_len, formation 構成) が dominant。

**対策**:
1. fallback stratify key として `hidden_len` quartile を試す (= second-best 候補)
2. local smoke で 2 key を A/B 比較、低い σ_fold を採用
3. 両 key で改善なしの場合 = H1 (= tw_gr_resid_std による偏在) 仮説誤り、subagent V に再分析依頼 + exp010 push 中止

---

## 5. 残課題 / 未消化

| # | 項目 | impact | 担当 |
|---|---|---|---|
| 1 | smoke で full 773 wells を回さない (= 5 wells × 200 row のみ) | σ_fold の真の値は kernel push 後 Kaggle log で確認必要 | 中央 (= kernel push 判断時) |
| 2 | AV classifier が overfit する可能性 (= train 773 wells で 5-fold CV、test 200 wells を class 1 単一) | n_train >> n_test で class imbalance、AUC inflation 警告 | 設計段階で **stratified by class** を採用、AUC を OOF で評価 |
| 3 | exp010 + Edge S + Edge R との互換性 (= 次 step exp011) | 加算的 inject 復活で σ_fold 改革効果が dilute する可能性 | subagent W/X 完了報告と統合検討 |
| 4 | submit slot 5/day で AB test 可能か | exp008 v3 / exp009 v3 LB 待ち、exp010 dedicated slot 必要 | 中央が決定 |

---

## 6. 関連 doc

- `docs/dev/2026-05-11-h10-postmortem-and-fold-reform.dense.md` (= H10 + 4 candidates)
- `docs/research/first-principles.dense.md` §2.2 (= per-well-stats.parquet 数値)
- `docs/research/gm-wisdom.dense.md` §1.1 (= Adversarial Validation Bojan Tunguz 流、AUC 0.7 threshold)
- `kaggle_kernels/exp009_case_e_edge_o/exp009_case_e_edge_o.py` (= base、line 1183 `build_edge_q_folds`、line 4000-4050 `Step E.1` hetero_w)
- `src/rogii/cv.py` (= 拡張対象、line 14 `make_well_folds`)
