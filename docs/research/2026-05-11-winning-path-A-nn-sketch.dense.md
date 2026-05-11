# 優勝路手法 A — NN sequence model on (GR + dTVT) (= 予備 sketch)

> 起源: docs/dev/2026-05-11-cv-lb-correlation.md § 5.2 + GM §1.2「single paradigm では gold は無理」
> 親 plan: kaggle-rogii-winning-candidates-cv-test-2026-05-12 (= 本タスクの後続)
> 数理本質: TVT は per-row autoregressive sequence。 LGB が陽に model できない **dTVT の auto-correlation 構造** と **GR-TVT cross signal** を、 Mamba / Transformer / 1D-CNN で **end-to-end 学習**。 expected LB lift -0.3 〜 -0.6 ft。

---

## 1. 数理本質 (= なぜ NN が必要か)

### 1.1 既存 LGB stack の limit

exp008 v2 (= LB 9.957、 現 best) の base 構成:
- karnakbaev pretrained LGB×3 + XGB + CB (= 5 base、 row-level tabular)
- 自前 LGB×3 + CB (= 4 base、 row-level tabular)
- 案 D Kalman/PF (= row-level features from sequential model、 LGB に注入)
- Ridge meta (= 9-base linear blend)

= **全 base が tabular GBM**、 sequence model ゼロ。

LGB は per-row prediction が前提。 dTVT の **temporal coherence** (= ar1_phi = 0.999 = ほぼ unit root)、 **GR の trailing-window 依存性** (= gr_ac10 ≈ 0.7、 gr_ac50 ≈ 0.5) を陽に encode できない。 Kalman / Beam / NCC で **partial sequence info** を tabular feature 化しているが、 これは **fixed pre-processing** = end-to-end 学習ではない。

### 1.2 NN sequence model が解く問題

TVT prediction を **sequence-to-sequence** で定式化:

入力 (= per-well per-row、 length n_row ≈ 5000-8000):
- `GR[t]`、 `dTVT[t] = TVT[t] - TVT[t-1]` (visible region のみ、 hidden は mask)
- `Z[t]`、 `MD[t]`、 trajectory features (= incl, azi 等)
- `typewell GR aligned at t` (= Beam 等の交差信号)
- `formation imputed b values` (= 6 formation × tvt_formula_F)
- `visible_mask[t]` (= 1 if visible, 0 if hidden)

出力 (= per-row):
- `predicted TVT[t]` for hidden rows (= visible_mask == 0)

Loss: `L = MSE(pred_TVT[hidden], target_TVT[hidden])` + auxiliary `L_recon = MSE(pred_TVT[visible], TVT_input[visible])` で smoothness 強制。

### 1.3 候補 architecture (= 3 構造原理)

#### A-1: Mamba (= state-space sequence model)
- O(n) 計算量 = 5000-row well で feasible
- Selective scan で long-range dependency capture
- 既存 implementation: `state-spaces/mamba` (= PyTorch、 GPU 必須)
- 期待 LB lift: -0.3 〜 -0.6 ft

#### A-2: Transformer (= attention-based)
- O(n²) = 5000-row well で 25M attention scores = GPU メモリ厳しい
- Block-sparse attention (= longformer 系) で O(n × window) に削減
- 既存 implementation: huggingface `LongformerModel` (= PyTorch)
- 期待 LB lift: -0.3 〜 -0.5 ft (= Mamba と同帯)

#### A-3: TimexER (= time-series specific decomposition)
- 季節性 + trend + residual の独立 model
- ROGII の TVT には季節性なし (= horizontal well で MD = time proxy ではない)
- = 期待効果 limited、 reject 候補

#### A-4: 1D-CNN + residual blocks (= simpler baseline)
- per-row convolution、 receptive field を multi-scale で拡張
- 軽量 (= LGB の半分 GPU memory)、 fast iteration
- 既存 implementation: 自前 PyTorch (= 100-200 行)
- 期待 LB lift: -0.2 〜 -0.4 ft (= Mamba/Transformer よりは弱いが、 simple)

### 1.4 推奨せず軸提示 (= 主道原則)

| 観点 | A-1 Mamba | A-2 Transformer | A-3 TimexER | A-4 1D-CNN |
|---|---|---|---|---|
| 数理本質 | state-space sequence | attention global | seasonal decompose | local convolution |
| 計算量 | O(n) | O(n×window) | O(n) | O(n×K) |
| GPU memory | 中 | 中-大 | 小 | 小 |
| 実装容易度 | 中 (= 既存 mamba lib) | 中-高 (= sparse attention) | 低 (= ROGII に inappropriate) | 高 (= 自前 100 行) |
| 期待 LB lift | -0.3〜-0.6 | -0.3〜-0.5 | unknown (= reject) | -0.2〜-0.4 |
| Kaggle dataset 必要 | mamba pretrained (= 公開あり) | longformer pretrained (= 公開あり) | - | - |

選択軸:
- **最高 LB lift 重視** → A-1 (Mamba)
- **既存 libs の組合せで安全** → A-2 (Transformer + longformer)
- **fast iteration + baseline** → A-4 (1D-CNN)
- **reject** → A-3 (TimexER は ROGII の non-temporal MD axis に不適)

---

## 2. 実装方針 (= A-1 Mamba 採用案、 数理的最強候補)

### 2.1 architecture

```python
class TVTMamba(nn.Module):
    def __init__(self, d_model=256, n_layers=8, d_state=16):
        super().__init__()
        self.input_proj = nn.Linear(N_FEATURES, d_model)
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_model, 1)  # → predicted TVT per row
        self.aux_visible_proj = nn.Linear(d_model, 1)  # auxiliary visible-region recon
    
    def forward(self, x, visible_mask):
        # x: (B=1, T=n_row, F=N_FEATURES)
        h = self.input_proj(x)
        for block in self.mamba_blocks:
            h = block(h) + h  # residual
        tvt_pred = self.output_proj(h).squeeze(-1)  # (B, T)
        return tvt_pred
```

N_FEATURES ≈ 20-40 (= GR + dTVT + Z + MD + typewell + formation imputed)。
B=1 (= per-well batch、 length varies)、 T ≈ 5000-8000 (= per-well rows)。

### 2.2 training

- Per-well GroupKFold (= 本タスクで確定した最良 CV を流用)
- Optimizer: AdamW lr=1e-4、 warmup 1000 steps → cosine decay
- Loss: `L = w_hidden * MSE(pred[hidden], target[hidden]) + w_vis * MSE(pred[visible], TVT_input[visible])` with w_hidden=1.0, w_vis=0.3
- 1 epoch = 1 pass over 618 train wells (= 5-fold で 1 fold val のみ excl)
- Total epochs: 20-50 = 1-3 hr/fold on RTX 4090 or A100
- 5 fold × 1 architecture = 5-15 hr total

### 2.3 inference

- Per test well: forward pass + extract `pred[hidden]` for hidden rows
- Post-proc: Edge S (= 0.01 ft round-to-grid、 既存) + smooth filter
- 5 fold mean for ensemble robustness

### 2.4 ensemble との接続

NN OOF は既存 9-base Ridge meta に **10 番目の base** として追加:
```python
Sx = np.column_stack([
    kb_oof_for_stack[k] for k in ACTIVE_MODELS  # karn 5
] + [
    own_oof[k] for k in own_active_keys  # own 4
] + [
    nn_oof,  # ← Mamba TVT pred OOF
])
ridge_ex = Ridge(alpha=1.0, fit_intercept=False, positive=True)
ridge_ex.fit(Sx, y_kb)
```

NN base の Ridge 係数が正 (= weight > 0.05) なら effective、 0 なら NN は inferior。

---

## 3. リスクと mitigation

| Risk | Mitigation |
|---|---|
| GPU メモリ不足 (= per-well 5000+ rows × Mamba 8 layers) | gradient checkpointing + d_model 縮小 (256 → 128) |
| Convergence 不安定 (= ROGII の TVT scale 11000+ ft、 大きい値) | target normalization (= y_z = (y - mean) / std) + denorm at inference |
| LGB に対する diversity ゼロ (= 同じ feature set + 同じ target) | input feature subset (= GR + dTVT のみ、 formation imputed を除く) で意図的 diversity |
| Compute コスト > LB gain (= 5-15 hr/fold × 5 fold = 25-75 hr で -0.3 ft なら efficiency 疑問) | A-4 1D-CNN で fast iteration、 promising なら A-1 へ scale up |

---

## 4. 本タスクとの関係

| layer | 本タスク (= cv-strategies) | 拡張タスク (= winning-path A) |
|---|---|---|
| CV 戦略確定 | ✓ 最良 CV 確定 | (流用) |
| 自前 base OOF | ✓ 4 SCORED × 4 CV | (流用、 NN base の OOF は別途) |
| NN sequence model 実装 | × | ✓ TVTMamba + training + OOF |
| ensemble に NN base 追加 | × | ✓ Ridge meta 10-base 拡張 |

= 本タスクは「最良 CV を確立して noise から信号を分離」、 拡張タスクは「最良 CV で NN base の真の effect を測定」。

---

## 5. 実装着手の前提

本 sketch は **planning doc only**。 着手前に必要:
1. 本タスク (cv-strategies-2026-05-11) の AC-7 全 pass + 最良 CV 確定
2. GPU 環境 (= Colab Pro A100 / Kaggle Notebook T4 / local RTX 4090) 確保
3. `state-spaces/mamba` パッケージ install + smoke test (= dummy data で 1 fold pass)
4. ROGII per-well data loader (= variable-length sequence、 PyTorch DataLoader 互換)
5. `.criteria/kaggle-rogii-winning-path-A-nn-2026-05-XX.yaml` 起票

---

## 6. 関連

- `~/projects/kaggle/CLAUDE.md` § 4 (= 4 paradigm 強制ルール、 sequence model = paradigm 2 拡張)
- `~/projects/kaggle/CLAUDE.md` § 5 (= 8000 路 5 必要条件、 #3 「self-compiled task 数 75%+」 を満たす)
- docs/dev/2026-05-11-cv-lb-correlation.md § 5.2 (= 優勝路手法候補 A の origin)
- docs/research/2026-05-11-deepest-eda.dense.md (= TVT autoregressive 構造の data evidence)
- Mamba paper: https://arxiv.org/abs/2312.00752
- state-spaces/mamba github: https://github.com/state-spaces/mamba
