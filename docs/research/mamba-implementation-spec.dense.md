# Mamba Implementation Spec for ROGII (2026-05-11)

> 担当 AC: subagent (Auto-mode), branch `docs/mamba-spec-2026-05-11` (派生元 `feat/phase-5-edge-r-online`)
> 残コンペ: **86 日** / 賞金 $50K / **5/13 着手** prerequisites をここで凍結する
> 文脈: subagent X §5.3 (= `docs/research/paradigm-deeper.dense.md` Mamba §1, 906 行) の **推奨組合せ #2 = Mamba 単独で −0.7〜−1.2 ft、 LB 9.0 切り射程** を kernel push 直前まで refine
> 中立指示原則: ここで推奨は出さない。 5 章 ごとに **選択軸 + トレードオフ** を表記し、 最終 go/no-go は開発者に委ねる (~/.claude/CLAUDE.md "去好去悪")
> 出典 URL を **§ 末尾の参考文献** で集約、 本文中の `file:line` 参照は branch HEAD 基準

---

## 0. 序: subagent X §5.3 #2 推奨 + Mamba の 9 切り射程根拠

### 0.1 採択論理 (= なぜ Mamba 単独を 5/13 の prerequisite に置くか)

subagent X §5.3 で 3 scenario が提示された:
- scenario A: **Mamba 一点集中** (Day 1-7、 LB 9.95 → 8.75-9.25、 単独 fail で 1 週間ロス)
- scenario B: 短工数先行 (PySR → PINN → TabPFN v2 → Mamba、 LB 7.65-8.25、 順次依存で 5/30 ギリ)
- scenario C: 並列 4 worker (LB 7.5-8.5、 並列 kill rollback risk)

「**推奨組合せ #2** = Mamba 単独で −0.7〜−1.2 ft、 9 切り射程」は scenario A に最も近く、 5/13 時点で完了している既存資産 (= exp008 case D Kalman、 exp010 fold reform) と原理的に **加法的** (= AR(1) N=1 → N=64 拡張)。

`docs/research/paradigm-deeper.dense.md:590` 表より:
| paradigm | 単独 LB 寄与 | 工数 (日) | 既存 layer との合成 |
|---|---|---|---|
| **Mamba / S4** | **−0.7〜−1.2 ft** | 5-7 | karnakbaev blend に feature, または full replace candidate |

→ 5/13 着手で 5/18-5/20 に submit 可能、 残 70 日に **3 つの直交 paradigm (TabPFN v2 / PINN / PySR)** を上乗せする時間が残る。

### 0.2 「9 切り」 までの定量分解 (= 数理本質 criterion / kaggle CLAUDE.md §11.2)

現状: exp005 v2 LB 10.203 ft (`docs/dev/submission-postmortems.dense.md` 参照、 Edge S +0.114 ft 改善後)。

| 期待 component | LB 寄与 (ft) | 累積 LB |
|---|---|---|
| 現在 (exp005 v2) | baseline | 10.203 |
| exp008 case D Kalman + Huber + hetero MEDIAN + path b (= subagent T v3) | −0.4 〜 −0.6 | 9.60-9.80 |
| exp010 stratified Edge Q fold + adversarial drop (= 5/12 submit 予定) | −0.05 〜 −0.10 | 9.55-9.75 |
| **+ Mamba (本 spec)** | **−0.7 〜 −1.2** | **8.35-9.05** |

**9 切り確率**: 中央値で LB 8.7 帯 → 8 切りまでは届かないが 9 切り (= LB <9.5) は **65-75%** (= subagent X §1.4.1 試算と整合)。

### 0.3 数理本質: Mamba が抑制する「誤差源」 と既存 Edge との非重複

`docs/research/paradigm-deeper.dense.md:712-722` の orthogonality matrix を ROGII 既存 Edge と再 mapping:

| 誤差源 | Edge S | Edge Q (exp010) | 案 D Kalman (exp008) | **Mamba (本案)** |
|---|---|---|---|---|
| AR(1)-like noise の short-range | ○ | △ | ◎ | ◎ (= overlap だが + path b で吸収) |
| **hidden zone MD 4000+ ft 末端 systematic bias** | × | × | △ (= N=1 上限) | **◎ (= N=64 で log-linear 抑制)** |
| fault / fold での ANCC 不連続 | × | × | △ | ○ (= selective $\Delta_t$ で local 検出) |
| label noise (= formula_oracle_rmse 0.006) Huber 化 | × | × | ○ | ○ (= Huber loss 継承) |

→ **Mamba 主目標 = MD 4000+ ft 末端 systematic bias 抑制** (`docs/research/first-principles.dense.md:159` §A: extrap curve mean が **-16** で大きい)。 これは案 D Kalman が原理的に拾えない領域 (= variance grow が $\frac{1}{1-\phi^2} \to \infty$ as $\phi \to 0.999$)。

### 0.4 反証思考: 「Mamba 単独 9 切り達成確率 65-75%」 の脆さ

`docs/research/paradigm-deeper.dense.md:194-200` failure modes 3 件 + 本 spec で独自に追加した 2 件:
1. **scratch train で 376 well overfit** (= CV-LB 乖離 +0.5 ft 以上) → §8.1
2. **mamba-ssm install 失敗 (= Internet disabled + ABI mismatch)** → §1.3, §8.2
3. **fault zone での $\Delta_t$ saturate** → §8.3 (本 spec で新規)

→ 確率 65-75% は **install 失敗 risk 25%** を引いた値。 install pre-check (= §6.1 CPU smoke) で risk を 10% 未満に下げれば確率は 75-85% へ。

---

## 1. mamba-ssm install (= Kaggle Internet disabled 対応)

### 1.1 wheel + cuda artifacts list

Kaggle GPU notebook image (= v168 GPU release 2026-03-20、 `https://github.com/Kaggle/docker-python/releases`):
- Python **3.12**
- torch **2.10.0** (+ cu128, GPU image 側)
- CUDA toolkit **12.8**
- cxx11 ABI: **TRUE** (= torch ≥ 2.7 から default)
- 推定 compute capability: T4 = **7.5**、 P100 = **6.0** (= mamba-ssm が要求する CUDA 11.6+ 満足)

→ **正確に match する wheel** (state-spaces/mamba v2.3.2.post1 release 2026-05-09 から):
```
mamba_ssm-2.3.2.post1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl  (322 MB)
causal_conv1d-1.6.2.post1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl  (194 MB)
```

検証元: `https://api.github.com/repos/state-spaces/mamba/releases/tags/v2.3.2.post1` および `https://api.github.com/repos/Dao-AILab/causal-conv1d/releases/tags/v1.6.2.post1` (2026-05-11 取得)

Python 依存 (= no-deps install で wheel から自動引出される):
- `einops>=0.6.1`
- `triton>=2.0` (Kaggle image 既存)
- `ninja` (= compile fallback、 wheel 直 install なら不要)
- `transformers` (= optional、 mixer 系で参照、 model 構築のみなら不要)

### 1.2 Kaggle private dataset 化手順

#### Step A: ローカル機で wheel + manifest 準備 (= offline build host)
```
# ローカル WSL / Linux x86_64 で実行 (= 開発機)
mkdir -p /tmp/rogii-mamba-runtime
cd /tmp/rogii-mamba-runtime

# wheel 直 DL
wget https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/causal_conv1d-1.6.2.post1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# einops の wheel も同梱 (= 既存 Kaggle image にあるが念のため pin)
pip download --no-deps -d . "einops>=0.6.1,<1.0"

# manifest (= sha256 + size verification)
sha256sum *.whl > manifest.sha256.txt
ls -la
```

#### Step B: dataset upload (= kaggle CLI)
```
# Kaggle dataset metadata
cat > dataset-metadata.json <<EOF
{
  "title": "ROGII Mamba Runtime (cu12 torch2.10 cp312)",
  "id": "ky7240/rogii-mamba-runtime",
  "licenses": [{"name": "MIT"}]
}
EOF

kaggle datasets create -p . -r zip
# → dataset slug "ky7240/rogii-mamba-runtime"
```

#### Step C: kernel 側 install (= Internet disabled でも動く)
```
# kaggle_kernels/exp016_mamba/exp016_mamba.py 冒頭
import subprocess, sys
WHEELS = [
    "/kaggle/input/rogii-mamba-runtime/causal_conv1d-1.6.2.post1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
    "/kaggle/input/rogii-mamba-runtime/mamba_ssm-2.3.2.post1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
]
for w in WHEELS:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", w])
# import 試験
import causal_conv1d, mamba_ssm
print("mamba_ssm", mamba_ssm.__version__, "causal_conv1d", causal_conv1d.__version__)
```

### 1.3 工数 + dependency risk

| 工程 | 工数 | risk |
|---|---|---|
| local wheel DL + manifest | 30 min | low (= GitHub Releases 直 DL) |
| Kaggle dataset upload | 15 min | low (= 515 MB、 20 GB 上限の 2.5%) |
| kernel install 検証 (= CPU smoke) | 30 min | medium (= ABI mismatch なら ImportError、 §8.2 復旧) |
| **dependency risk level** | | **medium** (= ABI / torch minor / CUDA driver の 3 軸で 1 つでも外れると runtime ImportError) |

#### 1.3.1 ABI 確認 method (= 失敗 mode 早期検出)
```
# kernel 冒頭で 1 行確認
python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
# Expected: True  (= cxx11abiTRUE wheel と一致)
```
→ False が出たら **mamba-ssm の abiFALSE wheel に差替え**: `mamba_ssm-2.3.2.post1+cu12torch2.10cxx11abiFALSE-*` (= mamba release に **2.10 abiFALSE は存在しない**、 2.6 まで遡る必要あり、 §8.2)

#### 1.3.2 wheel-tag 不整合 fallback
- **A**: cu12torch2.10 が動かない → cu12torch2.9 wheel に降格 (= torch 2.9 を pip-install --no-deps、 重い)
- **B**: ABI 衝突 → `MAMBA_FORCE_BUILD=TRUE pip install --no-build-isolation git+...` で source build (= 5-10 min、 nvcc/ninja 必要、 Kaggle build tool あり)
- **C**: 完全 fail → §8.2 torch SSM 自前実装 fallback (= LB 寄与 −0.3〜−0.5 ft に劣化)

### 1.4 LB 寄与推定 (本 step 単独)
- install そのもの: **0 ft** (= 単に走るかどうか)
- ただし install 失敗時の機会損失: **−0.7〜−1.2 ft の全消滅**
- → install の **数理本質寄与は indirect**、 「Mamba を Kaggle 上で動かすための prerequisite gate」

---

## 2. Mamba model architecture

### 2.1 input encoding (= per-well sequence)

#### 2.1.1 sequence 構成
1 well = 1 sequence、 sequence の各 row が **MD-aligned bit step** (= visible + hidden で連続)。

input features (per row, total **d_in ≈ 32**):
| feature | 説明 | source |
|---|---|---|
| `Z` | bit Z (deep) | 公式 csv |
| `X`, `Y` | bit XY | 公式 csv |
| `MD` | measured depth | 公式 csv |
| `GR` | gamma ray | 公式 csv |
| `last_known_TVT` | hidden region は visible 末端値で fill | `build_well_features` exp008:1673 |
| `tvt_input` | visible は実値、 hidden は NaN → 0 fill + mask | 公式 csv (= visible のみ実値) |
| `is_visible_mask` | 0/1 mask、 hidden = 0 | derived |
| `kalman_mean`, `kalman_std` | 案 D Kalman の posterior (= 既存 feature 流用) | exp008 KALMAN_ENABLE |
| `formation_onehot_6` | ANCC / ASTNU / ASTNL / EGFDU / EGFDL / BUDA | exp008:270 FORMATIONS |
| `b_well_prior` | per-well WLS estimate (= heteroscedastic weight と整合) | exp008 HETERO_W_STATS_PATH |
| `md_normalized` | (MD - md_well_start) / 10000 (= scale invariant) | derived |
| `delta_md` | MD step interval (= 通常 0.5 ft) | derived |
| `gr_smooth_w5` | 5-row moving avg GR | derived |
| `gr_resid_from_tw` | GR - typewell GR (= alignment residual) | exp008 Edge M 系 |
| `visible_ratio_well` | per-well constant (= 0.18-0.35) | exp008 stat |

seq_len: 各 well の n_rows (= **max ~12000**、 `docs/research/first-principles.dense.md:69` hidden_len_p90 = 6349 ft + visible)。

#### 2.1.2 padding + batching 戦略
- batch_size = **8 wells** (= 12000 × 32 × 4 byte × 2 (grad) × 8 = 24 MB、 T4 16 GB 余裕)
- **left-pad** (= visible 末端を sequence 末尾に揃える、 Mamba は causal なので未来情報 leak 無し)
- attention mask 不要 (= Mamba は selective $\Delta_t$ で自然に位置を見る、 ただし padding token 用 `is_valid_mask` を loss で使う)

#### 2.1.3 normalization
- per-feature **global z-score** (= train all-well で μ, σ 計算、 inference に固定)
- `tvt_input` のみ **per-well z-score** (= well 間の bias 吸収、 visible 末端値で normalize は cheat、 一律 well-mean で)
- 出力 `dTVT_residual` は **無 scaling** (= ft 単位そのまま、 Huber loss と整合)

### 2.2 Mamba block × 4 layer

#### 2.2.1 architecture
```
input  : (B, L, d_in=32)
   │
   ▼
Linear(32 → d_model=128)
   │
   ▼
[ Mamba(d_model=128, d_state=64, d_conv=4, expand=2) ] × 4
+ residual + RMSNorm  per block
   │
   ▼
Linear(128 → 1)   = dTVT_residual prediction
```

各 Mamba block の param 数: $3 \cdot 2 \cdot 128^2 \approx 98K$、 × 4 layer = **~400 K params** (= 376 well × 4500 row avg = 1.7M token に対する param ratio 0.025 = 過学習しにくい範囲、 `docs/research/paradigm-deeper.dense.md:198` 過学習 mode 対応)

#### 2.2.2 hyperparameter (= MambaTS 慣行から)
| 名前 | 値 | 根拠 |
|---|---|---|
| `d_model` | **128** | `docs/research/paradigm-deeper.dense.md:165` time-series 慣行 |
| `d_state` | **64** | Mamba-2 で d_state 拡張可能 (= 同 doc:113) |
| `d_conv` | 4 | local smoothing (= 0.5 ft × 4 = 2 ft window、 ROGII bit step) |
| `expand` | 2 | block expansion factor (= 同 doc:160) |
| `n_layers` | **4** | scratch train なので深すぎ NG、 4 で過学習耐性 + 末端 bias 抑制両立 |
| dropout | **0.1** | Mamba 公式 implementation には dropout 無、 自前で `nn.Dropout(0.1)` を RMSNorm 後に挿入 |

### 2.3 出力 head (= dTVT residual)

#### 2.3.1 prediction target
$$
\text{dTVT}_{\text{residual}}(t) = \text{TVT}(t) - \text{last\_known\_TVT}_{\text{well}}
$$

→ visible 末端からの **incremental TVT 変化** を学習。 cumsum で TVT 復元は不要 (= row-level の direct 値)。

#### 2.3.2 mask 処理
loss は **hidden region のみ** で計算:
```
y_pred = head(mamba_out)   # (B, L, 1)
loss = huber(y_pred[hidden_mask], y_true[hidden_mask], delta=0.5)
```
visible region の predict は loss に入らないが、 Mamba の hidden state は visible 通過で「conditioning」 されているため、 学習信号は visible 経由で間接的に流れる。

#### 2.3.3 uncertainty 出力 (= future 拡張、 v1 では skip)
- v1 は **point estimate のみ**
- v2 候補: Monte Carlo dropout (= forward 10 回 → std)、 quantile head (= q10/q50/q90 head 3 つ) → `docs/research/paradigm-deeper.dense.md:174` 参照

---

## 3. training pipeline

### 3.1 Edge Q stratified fold (subagent Y 整合)

#### 3.1.1 fold 定義の継承
exp010 (`docs/dev/exp010-design.dense.md`、 5/12 submit 予定) で確立した **stratified Edge Q fold** をそのまま使う:
- typewell content hash で group
- adversarial validation drop で test-like 376 wells 除外
- 5 fold OOF、 σ_fold 0.526 → 0.303 (= 42% 削減、 local smoke `df6e1c6` 検証済)

#### 3.1.2 Mamba 専用 fold 上書き禁止理由
別 fold を使うと own_oof (= path b stack) との column merge で fold-misalign leak が再発 (= subagent T v3 の v2 9-base Ridge leak と同型、 exp008:209 参照)。 **必ず exp010 fold を継承**。

#### 3.1.3 train / val split
```
for fold in range(5):
    train_wells = wells where adv_drop=False and fold_id != fold
    val_wells   = wells where adv_drop=False and fold_id == fold
    # adversarial-dropped wells は train にも val にも入らない
```

### 3.2 Huber loss + heteroscedastic (subagent T 整合)

#### 3.2.1 Huber loss
```python
loss = F.huber_loss(y_pred, y_true, delta=0.5, reduction="none")
# delta = 0.5 (exp008 CB Huber:delta=0.5 と一致、 dtvt_std p50 = 0.42)
```
→ fat-tail noise (= `dtvt_p95/std = 2.4 > Gaussian 1.645`、 `docs/research/first-principles.dense.md:72`) を Laplace-like 扱い。

#### 3.2.2 heteroscedastic sample weight
```python
sigma_well = b_well_resid_std[well_id]   # exp008 HETERO_W_FILL_P50 = 0.00835
w_well = (1.0 / sigma_well)
w_well = w_well / w_well.mean()          # normalize per batch
w_well = w_well.clamp(0.5, 2.0)          # HETERO_W_CLIP_[LO,HI]
loss = (loss * w_well[:, None, None]).mean()
```
→ exp008:197-206 と完全に同じ weight 設計、 fold misalign なし。

### 3.3 hyperparameters (lr, batch_size, epochs)

| 名前 | 値 | 根拠 |
|---|---|---|
| optimizer | **AdamW** | β = (0.9, 0.95), weight_decay = 0.01 (= Mamba paper §A.1) |
| lr | **3e-4** | warmup 500 step → cosine decay (= MambaTS 2024 §4.2) |
| warmup | 500 step | linear 0 → 3e-4 |
| schedule | cosine | 0 epoch から min_lr=3e-5 まで |
| batch_size | **8 wells** | T4 16 GB、 §2.1.2 |
| n_epochs | **20** | 376 well × 20 = 7520 sample/fold、 1 fold = 940 step (batch 8) → 18.8K step total |
| grad_clip | **1.0** | norm clip (= Mamba paper §A.1) |
| ema | optional v2 | (β=0.999、 v1 では skip) |
| seed | **(42, 123, 2024)** | MEDIAN ensemble (= subagent T v3 慣行) |

#### 3.3.1 multi-seed ensemble (= subagent T v3 流)
exp008:175-194 と整合:
- mamba_seeds = (42, 123, 2024) で 3 run
- 各 fold で 3 seed × 5 fold = **15 train run**
- OOF: `mamba_oof_med` = MEDIAN over 3 seed (per-fold)
- test: `mamba_test_med` = MEDIAN over 3 seed (per-fold) → 5 fold mean

### 3.4 LB 寄与推定 (本 step 単独)
- Huber + hetero: **−0.1 ft** (= 既存 own と直交ではない、 重複あり)
- MEDIAN over 3 seed: **−0.05 〜 −0.10 ft** (= seed variance 抑制)
- 純 Mamba state-space 効果は §4.3 で blend 後に確認

---

## 4. ROGII inject 位置

### 4.1 base kernel = exp008 v3 (subagent T)

`kaggle_kernels/exp008_case_d_kalman/exp008_case_d_kalman.py` (3652 行、 subagent T v3) を base に **新 kernel `kaggle_kernels/exp016_mamba/exp016_mamba.py` を fork**。

差分:
1. dataset 追加: `rogii-mamba-runtime` を kernel-metadata.json `dataset_sources` に
2. import block: §1.2 Step C の wheel install を冒頭に
3. Mamba train + predict module を新規 (= `src/rogii/mamba_model.py`、 spec のみ、 本 task で実装しない)
4. OWN_USE_MEDIAN_SEEDS は維持、 own bases に Mamba を追加

### 4.2 Mamba を base model #5 として追加

#### 4.2.1 own_oof / own_test の column 拡張
exp008:3262-3469 の own 学習 loop の **後** に Mamba block を挿入:

```
own_keys (v3 既存): ["lgb_own_med", "cb_own_med"]                # 2 base
own_keys (v3 + Mamba): ["lgb_own_med", "cb_own_med", "mamba_med"]  # 3 base
```

→ `own_oof["mamba_med"]` および `own_test["mamba_med"]` を新規 column として追加。 既存 LGB / CB の MEDIAN seed loop と同型。

#### 4.2.2 train / inference の挿入位置
```
exp008 既存 flow:
  Step F: meta blend (path b)
   ├─ kb_avg_oof = mean(kb_5)
   ├─ own_oof_blend = Ridge(own_2)             ← own_2 に Mamba 追加で own_3 に
   └─ 1D grid search w_kb
```

→ Step E (own train) の中で **追加 1 段**: Mamba train + predict → MEDIAN aggregate → own_oof["mamba_med"] 書込。

### 4.3 path b Ridge separate での blend weight 学習

#### 4.3.1 数式
$$
y_{\text{final}} = w_{\text{kb}} \cdot \frac{1}{5}\sum_{k \in \text{kb}} y_k + (1 - w_{\text{kb}}) \cdot \text{Ridge}_{\text{own}}(y_{\text{lgb\_med}}, y_{\text{cb\_med}}, y_{\text{mamba\_med}})
$$

- Ridge は positive=True、 fit_intercept=False (= exp008:3530)
- Mamba の Ridge coef が **支配的 (>0.5)** になれば Mamba が path b を主導、 さもなくば LGB/CB が支配
- $w_{\text{kb}}^*$ は 1D grid search (= exp008:3540-3548)

#### 4.3.2 期待 weight
| 想定 sigma 関係 | own Ridge coef (Mamba) | $w_{\text{kb}}^*$ |
|---|---|---|
| Mamba が末端 bias を強く抑制 | **0.5-0.7** | 0.3-0.4 |
| Mamba 効果薄 (= LGB/CB と同程度) | 0.3-0.4 | 0.5 |
| Mamba 過学習 (= CV-LB 乖離) | 0.1 以下 | 0.7+ |

→ Ridge coef を **logging で必ず確認** (= exp008:3536 と同じ `log.info("own Ridge weights:")` パターン)。

### 4.4 LB 寄与推定 (本 step 単独)
- Mamba を base #5 で path b blend: **−0.5 〜 −0.9 ft** (= subagent X §1.4.1 の中域、 重複考慮で 0.2 ft 割引)
- 重要観察: Mamba 単独 OOF RMSE が LGB MEDIAN より **0.3+ ft 良い** ことを確認できれば LB 寄与確度 80%+

---

## 5. Kaggle runtime budget

### 5.1 train 1.5 hr + inference 80 sec

| component | runtime (T4 / 776 wells) | 累積 |
|---|---|---|
| exp008 既存 (= karnakbaev pretrained predict + Edge Q/M/D + own LGB×3 seed + CB×3 seed + path b) | 6-7 hr | 6-7 hr |
| **Mamba train: 4 layer × 128 dim × 3 seed × 5 fold = 15 run** | **1.5 hr** | **7.5-8.5 hr** |
| Mamba inference: 776 wells × ~12000 row × 4 layer batch=8 | **80 sec** | **+1.3 min** |
| **合計** | | **7.5-8.5 hr** |

→ Kaggle 9 hr cap の **83-94%** 占有、 余裕 6-17%。

#### 5.1.1 train runtime 詳細試算
- 1 fold per seed: 940 step × 20 epoch × 8 batch × (12000 seq_len, 32 d_in) ≈ 1.5e8 input token
- T4 (= 8.1 TFLOPS fp16) で Mamba forward + backward 通過: ~6 min / fold-seed
- 5 fold × 3 seed = 15 run × 6 min = **90 min = 1.5 hr** ✓

### 5.2 8 hr 想定 (9 hr cap ぎりぎり)

#### 5.2.1 ぎりぎり問題と対策
| 状況 | 累積 hr | 対策 |
|---|---|---|
| 全 component nominal | 7.5-8.5 hr | OK |
| Mamba train が 2x 遅い (= T4 slot busy) | 9-9.5 hr | **TIMEOUT_S を 9.0 hr に設定し、 train phase early stop → mamba_med は当該 fold-seed のみで MEDIAN** |
| exp008 既存 phase が 1 hr 超過 | 8.5-9.5 hr | TabICL dormant (= TABICL_ENABLE=False) で +1 hr 確保 (exp008:129 既存) |

#### 5.2.2 timeout 設計
exp008:195 OWN_TRAIN_HARD_TIMEOUT_S = 6\*3600 と同型で:
```python
MAMBA_TRAIN_HARD_TIMEOUT_S = 1.8 * 3600  # 1.8 hr (1.5 hr nominal + 20% buffer)
```
→ 超過したら mamba_oof は **手元 fold-seed の MEDIAN 部分集合** で fall back (= own_oof["mamba_med"] 部分埋め + 残 NaN を全体 mean で imputation)。

### 5.3 fallback: 1 layer or fold 削減

| fallback level | 内容 | 削減時間 | LB 寄与劣化 |
|---|---|---|---|
| **L1: Mamba 4→2 layer** | n_layers = 2 | -45 min | −0.7 → −0.5 ft |
| L2: 3 seed → 1 seed (= MEDIAN 無) | seeds = (42,) | -60 min | −0.7 → −0.6 ft (seed variance +0.1) |
| L3: 5 fold → 3 fold | n_splits=3 | -36 min | −0.7 → −0.55 ft (但し path b stack 整合性悪化) |
| **L4: full disable** | Mamba skip | -90 min | −0.7 → 0 (exp008 v3 のまま) |

→ L1 が最も影響軽微、 L4 は Mamba 全否定。 中間 L2/L3 は path b stack 整合性悪化で **避ける**。

---

## 6. smoke 設計

### 6.1 5 wells × 200 row CPU smoke

#### 6.1.1 目的
1. mamba-ssm install が動く (= ImportError 無、 CUDA call 無 path)
2. Mamba model が forward + backward 通過 (= shape / dtype mismatch 無)
3. loss が NaN にならない

#### 6.1.2 実装
```python
# tests/smoke_mamba_cpu.py (= 本 task では作成しない、 spec のみ)
DEBUG_MAX_WELLS = 5
DEBUG_MAX_ROWS_PER_WELL = 200
DEBUG_DEVICE = "cpu"   # nvfuser 系 fallback、 ただし Mamba CUDA kernel が無いと NotImplementedError
```

#### 6.1.3 CPU 上の壁
mamba-ssm は **CUDA-only**: `selective_scan_cuda` が CPU で `NotImplementedError`。 CPU smoke は **import 確認 + Linear/RMSNorm 単独 forward** までしかできない (= Mamba block 飛ばし)。

→ CPU smoke の **真の用途**: Kaggle にアップする前の **wheel install 確認** + **データ前処理パイプライン確認** (= §2.1.1 input encoding)。

### 6.2 1 fold GPU smoke

#### 6.2.1 Kaggle T4 上で初回確認
- DEBUG_MAX_WELLS = 50 wells
- DEBUG_ONE_FOLD = True (= 5 fold のうち 1 fold のみ)
- 1 seed (= 42 only)
- 期待 runtime: 5-8 min

#### 6.2.2 確認 checklist
- [ ] `nvidia-smi` で T4 認識
- [ ] `torch._C._GLIBCXX_USE_CXX11_ABI == True`
- [ ] `import mamba_ssm; mamba_ssm.__version__ == "2.3.2.post1"`
- [ ] forward / backward 通過、 loss NaN なし
- [ ] OOF RMSE が **5-15 ft** (= 50 wells subset、 過学習しても 15 以下、 ぶっ壊れ検知用)
- [ ] GPU memory peak < 12 GB (= 16 GB 余裕)

### 6.3 full 5 fold

#### 6.3.1 fold 順次
exp010 fold (= stratified Edge Q + adv drop) で fold 0 → 1 → 2 → 3 → 4 を seed (42, 123, 2024) で逐次 train。

#### 6.3.2 OOF aggregate
```python
mamba_oof_med = MEDIAN over 3 seed per row    # shape (n_train_rows,)
mamba_test_med = MEDIAN over 3 seed per row, mean over 5 fold   # shape (n_test_rows,)
```

#### 6.3.3 path b blend 確認 (= LB submit 直前)
- OOF RMSE: kb_avg / lgb_med / cb_med / mamba_med 4 列
- own Ridge coef: lgb / cb / mamba の 3 weight
- $w_{\text{kb}}^*$: grid search 結果

→ **mamba_med が own Ridge で coef >= 0.4 にならない場合、 §5.3 fallback または investigation 強化**。

---

## 7. 5/13 以降の exp 設計

### 7.1 exp016 (= subagent T v3 + Mamba)

#### 7.1.1 構成
- kernel: `kaggle_kernels/exp016_mamba/exp016_mamba.py` (= exp008 fork)
- dataset: `rogii-mamba-runtime` (新規 §1.2)
- branch: `feat/phase-6-exp016-mamba` (= 5/13 作成)
- 設計 doc: `docs/dev/exp016-mamba-design.dense.md` (= 5/13 作成、 本 spec を要約)

#### 7.1.2 work plan (= 5/13 着手 + 7 day 想定)
| 日 | 内容 | 確証 |
|---|---|---|
| 5/13 | wheel local DL + Kaggle dataset upload + CPU smoke | wheel import 通過 |
| 5/14 | exp008 fork → exp016 kernel skelton + Mamba module 配線 | local syntax check pass |
| 5/15 | 1 fold GPU smoke (Kaggle T4) | OOF RMSE 5-15 ft, GPU memory OK |
| 5/16-5/17 | 5 fold × 3 seed full train | full OOF parquet 完成、 own Ridge coef 確認 |
| 5/18 | path b blend + LB submit | LB score 記録 (期待: 8.7-9.0) |
| 5/19-5/20 | postmortem + fallback 試行 (L1 layer reduce 等) | 仮説検証完了 |

### 7.2 AB test (= Mamba on/off)

#### 7.2.1 isolate 設計 (= kaggle CLAUDE.md §11.2 datapoint 価値)
| submit | 差分 | 期待効果 isolate |
|---|---|---|
| exp008 v3 (= 既存) | baseline | 9.6-9.8 ft 想定 (= subagent T 試算) |
| **exp016 (= +Mamba)** | own_3 base に Mamba 追加 | +Mamba 単独 effect = exp016 − exp008 v3 |
| exp016-fallback-L1 (= Mamba 2 layer) | n_layers 4 → 2 | layer depth effect |

→ 3 submit 並列で **Mamba 効果を ±0.1 ft 精度で isolate** (= LB σ 0.05 想定、 GM CLAUDE.md §4.3 ratio drift 監視)。

#### 7.2.2 ratio / trend coef 監視
- `docs/research/2026-05-xx-local-vs-lb-correlation.md` (= kaggle CLAUDE.md §8.1 #3) に append
- exp008 v3 / exp016 / exp016-L1 の 3 件で LB - est_oof_rmse の **trend coef σ** を集計

---

## 8. failure modes 3 件 + recovery

### 8.1 mamba-ssm install 失敗 → torch SSM 自前実装 fallback

#### 兆候
- kernel boot 直後の `import mamba_ssm` で `ImportError: undefined symbol: ...` または `OSError: libtorch...`
- ABI / torch minor mismatch が原因の 80%

#### 1st recovery (= ABI 切替)
```python
# wheel を abiFALSE に差替えて再 install
WHEELS_FALLBACK = [
    "/kaggle/input/rogii-mamba-runtime/causal_conv1d-1.6.2.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
    "/kaggle/input/rogii-mamba-runtime/mamba_ssm-2.3.2.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
]
```
ただし torch 2.6 wheel は torch 2.10 環境では **torch 直 import 衝突**、 別 venv 必要 → Kaggle では事実上不可。

#### 2nd recovery (= source build)
```python
import subprocess, os
os.environ["MAMBA_FORCE_BUILD"] = "TRUE"
subprocess.check_call(["pip", "install", "--no-build-isolation", "--no-deps",
                        "/kaggle/input/rogii-mamba-runtime/mamba-ssm-2.3.2.post1.tar.gz"])
# 5-10 min compile、 nvcc/ninja 利用、 Kaggle image にあり
```
risk: compile 中の memory spike (= T4 image で OOM 可能)、 build cache が /tmp/ で session 跨ぎ消える。

#### 3rd recovery (= torch SSM 自前実装)
```python
# minimal S4 / Mamba-like block を pure torch で
class TorchSSM(nn.Module):
    def __init__(self, d_model=128, d_state=64):
        super().__init__()
        self.A_log = nn.Parameter(torch.zeros(d_state))      # HiPPO 初期化省略
        self.B = nn.Linear(d_model, d_state, bias=False)
        self.C = nn.Linear(d_state, d_model, bias=False)
        self.D = nn.Parameter(torch.zeros(d_model))
    def forward(self, x):                # (B, L, d_model)
        A = -torch.exp(self.A_log)
        B, C = self.B(x), x              # selective skip、 input-dependent 無
        out = torch.zeros_like(x)
        h = torch.zeros(x.shape[0], A.shape[0], device=x.device)
        for t in range(x.shape[1]):     # O(L) Python loop、 遅いが動く
            h = h * torch.exp(A) + B[:, t]
            out[:, t] = self.C(h) + self.D * x[:, t]
        return out
```
- Python loop で L=12000 は **20 sec/forward**、 train 不可、 ただし predict のみなら間に合う
- LB 寄与劣化: −0.7 → **−0.3〜−0.5 ft** (= HiPPO 初期化無、 selective parameterization 無)

### 8.2 OOM → batch_size 削減 or seq_len chunking

#### 兆候
- `torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate ...`
- T4 16 GB の peak usage > 14 GB

#### 1st recovery (= batch_size 削減)
```python
BATCH_SIZE = 4    # was 8
```
runtime +30%、 lr は同じ (= 大 batch ほど lr 大が定石だが Mamba は relatively insensitive、 paper §5.1)。

#### 2nd recovery (= seq_len chunking)
12000 row sequence を 6000 + 6000 に分割、 hidden state を chunk 間で carry-over (= `inference_params` 使用、 mamba-ssm API `https://github.com/state-spaces/mamba#inference`)。
- train: chunk truncation で BPTT 6000 step、 末端 1000 row は state 引継ぎあり
- inference: chunked autoregressive、 numerical drift 小

#### 3rd recovery (= d_model 削減)
```python
D_MODEL = 64    # was 128
```
param count 半減 (= 200K) で memory 半減、 LB 寄与 −0.1 ft 劣化 (= 過学習耐性向上の trade-off `docs/research/paradigm-deeper.dense.md:198`)。

### 8.3 overfit → dropout / weight decay 強化

#### 兆候
- CV-LB gap > 0.5 ft (= `docs/research/paradigm-deeper.dense.md:198` の overfit signal)
- OOF RMSE 3-5 ft で LB 9.5-10 ft (= 異常 gap)

#### 1st recovery (= dropout 強化)
```python
DROPOUT = 0.3    # was 0.1
```
LB 寄与 maintain、 OOF 0.2-0.3 ft 悪化、 LB-OOF gap 縮小。

#### 2nd recovery (= weight_decay 強化)
```python
WEIGHT_DECAY = 0.05    # was 0.01
```
Mamba 系は AdamW WD 0.01 が default だが 0.05 まで上げて oscillation 抑制。

#### 3rd recovery (= heavy augmentation)
- visible window random crop (= MD start を ±20% でずらす)
- GR noise injection (= per-row Gaussian σ=2)
- `docs/research/paradigm-deeper.dense.md:198` (b) と整合

### 8.4 fault zone での $\Delta_t$ saturate (= 本 spec 新規追加)

#### 兆候
- fault layer 跨ぎで $\Delta_t \to 0$ または $\infty$
- prediction に sudden jump (= 公開 top の Beam Search 系 fault 検出と比較)

#### 1st recovery (= $\Delta_t$ clamp)
mamba-ssm 公式 API には直接 clamp 引数なし → forward hook で `Mamba.dt` 出力に `clamp(0.01, 100)` を挟む。

#### 2nd recovery (= Edge O fault 検出を pre-process)
- ANCC slope discontinuity を per-row mask として feature 追加 (= `is_fault_zone` binary)
- fault zone では Mamba 出力を Kalman posterior に reset (= subagent X §1.4.2 (c))

---

## 9. 残課題

### 9.1 spec で凍結できなかった項目

1. **MambaTS / Mamba4Cast pretrained 利用判断**: subagent X §1.3.4 で言及あり、 license 要確認。 v1 では scratch train で push、 v2 で fine-tune 検証
2. **uncertainty 出力 (quantile head)**: §2.3.3、 v1 では point estimate のみ、 v2 で q10/q50/q90 + Edge S への wire
3. **Mamba-2 vs Mamba 採択**: subagent X §1.1.5 で Mamba-2 は 2-8x 高速だが d_state 64 で十分なら Mamba-1 でも OK。 wheel 名 `mamba_ssm` は両方含む (= class `Mamba2` あり)
4. **Edge R online との衝突**: 本 spec は exp008 v3 base、 Edge R online (= phase-5-edge-r-online) との merge 検討は 5/14 以降
5. **fold sigma 計測**: 5 fold + 3 seed = 15 run の fold-seed sigma を OOF で計測、 0.3 以下なら安定、 0.5 超なら seed 8 まで増設

### 9.2 5/13 着手 prerequisite checklist

- [ ] 開発機で wheel 4 file (mamba_ssm + causal_conv1d × abiTRUE/FALSE 各 1) を DL
- [ ] `manifest.sha256.txt` 生成 (= §1.2 Step A)
- [ ] `dataset-metadata.json` 作成 + `kaggle datasets create -p . -r zip` 実行
- [ ] Kaggle Web UI で `ky7240/rogii-mamba-runtime` が public visible (= 必ず Private 設定確認、 wheel 公開判断は subagent X §1.3 license 確認後)
- [ ] exp010 stratified Edge Q fold (= 5/12 LB submit) が **safe な結果** で帰ってきている (= 9.55-9.75 帯)
- [ ] Edge R online との base branch 確定 (= feat/phase-5-edge-r-online or feat/phase-6-exp010-fold-reform のどちら)

### 9.3 datapoint 価値の自問 (= kaggle CLAUDE.md §11.2 #5)

submit する場合の datapoint 価値:
- exp008 v3 vs exp016: Mamba 単独 effect isolate (= 0.5-0.9 ft 期待)
- exp016 vs exp016-L1: layer depth effect (= 0.1-0.2 ft 期待)
→ 両方とも **LB σ 0.05** より十分大、 noise 範囲埋もれ無し、 datapoint 価値 high

### 9.4 「優勝本質性」 自問 (= kaggle CLAUDE.md §11.2 #1-#5)

1. **数理本質**: state-space model で AR(1) を任意 order に拡張、 hidden zone 末端 systematic bias を log-linear 抑制 → core 数理問題に直接寄与 ✓
2. **優勝寄与**: 単独 -0.7〜-1.2 ft、 9 切り射程、 8.x 帯目標で線形以上の貢献 ✓
3. **代替比較**: 案 D Kalman N=1 (= 既存) では理論上限あり、 PINN/TabPFN との比較は subagent X §5 で実施済 (= 単独最大寄与) ✓
4. **rule 耐性**: pure ML approach、 host fix で死ぬ exploit 系ではない ✓
5. **datapoint 価値**: §9.3 で明示、 noise 範囲埋もれ無し ✓

→ 5/13 着手で **本質性 criterion 全 5 項目満たす**。

---

## 参考文献 (= 本 spec 内引用 URL)

### 一次資料 (= 論文 + 公式 GitHub + API)
- Mamba (Gu & Dao 2023): https://arxiv.org/abs/2312.00752
- Mamba-2 (Dao & Gu 2024): https://arxiv.org/abs/2405.21060
- HiPPO (Gu et al. 2020): https://arxiv.org/abs/2008.07669
- S4 (Gu et al. 2022): https://arxiv.org/abs/2111.00396
- MambaTS (2024): https://arxiv.org/abs/2405.16440
- TSMamba (2024): https://arxiv.org/abs/2411.02941
- state-spaces/mamba GitHub: https://github.com/state-spaces/mamba
- mamba-ssm PyPI: https://pypi.org/project/mamba-ssm/
- mamba release v2.3.2.post1 (2026-05-09): https://github.com/state-spaces/mamba/releases/tag/v2.3.2.post1
- mamba release JSON: https://api.github.com/repos/state-spaces/mamba/releases/tags/v2.3.2.post1
- causal-conv1d GitHub: https://github.com/Dao-AILab/causal-conv1d
- causal-conv1d release v1.6.2.post1 (2026-05-09): https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.6.2.post1
- causal-conv1d release JSON: https://api.github.com/repos/Dao-AILab/causal-conv1d/releases/tags/v1.6.2.post1
- Kaggle docker-python: https://github.com/Kaggle/docker-python
- Kaggle GPU image v168 (2026-03-20): https://github.com/Kaggle/docker-python/releases

### 内部 doc (= 同 repo 別 branch + 同 branch)
- `docs/research/paradigm-deeper.dense.md` §1 Mamba (branch `docs/paradigm-deeper-2026-05-11`)
- `docs/research/first-principles.dense.md` §2.1, §2.7 (現 branch)
- `docs/research/mathematical-formulation.dense.md` §4.4 (branch `docs/cv-breakthrough-2026-05-11`)
- `docs/research/independent-edges.dense.md` (Edge S/Q/M/O/R/T 系)
- `docs/research/problem-essence.dense.md` (= generative model)
- `kaggle_kernels/exp008_case_d_kalman/exp008_case_d_kalman.py` (= subagent T v3, base kernel, 3652 行)
- `docs/dev/exp010-design.dense.md` (= 5/12 stratified Edge Q fold)
- `docs/dev/submission-postmortems.dense.md` (= exp005/v3 系 LB 記録)

### kaggle CLAUDE.md 関連
- `/home/yusuke_kaya/projects/kaggle/CLAUDE.md` (= GM-level workflow、 §11 優勝本質性 criterion)
- `~/.claude/rules/kaggle-research-workflow.md`
- `~/.claude/CLAUDE.md` (= 主道フレームワーク、 自動発火 trigger、 Plan Mode 出力ルール)
