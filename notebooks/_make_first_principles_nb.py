"""nbformat で notebooks/01_first_principles_eda.ipynb を生成."""
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

REPO = Path("/home/yusuke_kaya/projects/kaggle/ROGII")
NB_PATH = REPO / "notebooks/01_first_principles_eda.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def main():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        md(
            """# 01 First-Principles EDA — ROGII Wellbore Geology Prediction

> **目的**: Track E-1 (生成過程モデル化) + Track E-2 (Irreducible Error 上限見積もり) の数値根拠を実データから取る。

このノートブックは `notebooks/_first_principles_measure.py` / `_first_principles_extras.py` / `_first_principles_lower_bound.py` の**結果を可視化**する。実計測は parquet/json として永続化済み。

| 計測対象 | 出力 file | 主な指標 |
|---|---|---|
| GR sensor noise std / dTVT/dMD / b_well resid / typewell GR resid / AR(1) | `outputs/eda/first_principles/per-well-stats.parquet` + `summary.json` | per-well & cross-well 集計 |
| best lag / b_well drift / extrapolation curve | `outputs/eda/first_principles/per-well-extras.parquet` + `extrapolation-residual-curve.parquet` + `summary-extras.json` | naive last-value broadcast の理論 RMSE と md_since 関数 |
| GR Cramér-Rao lower bound / formula oracle / synthetic GR matching | `outputs/eda/first_principles/per-well-bound.parquet` + `summary-bound.json` | 理論的 RMSE 下限 3 種 |

> 数値の物理的解釈・irreducible error の理論議論は `docs/research/first-principles.dense.md` に書く。ここはあくまでデータ確認用。
"""
        )
    )

    cells.append(
        code(
            """import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path('/home/yusuke_kaya/projects/kaggle/ROGII')
OUT = REPO / 'outputs/eda/first_principles'

with open(OUT / 'summary.json') as f:
    summary_main = json.load(f)
with open(OUT / 'summary-extras.json') as f:
    summary_extras = json.load(f)
with open(OUT / 'summary-bound.json') as f:
    summary_bound = json.load(f)

print('=== Main (per-well-stats) ===')
for k, v in summary_main.items():
    print(f'  {k}: {v}')
print()
print('=== Extras (best-lag / b drift / extrapolation) ===')
for k, v in summary_extras.items():
    print(f'  {k}: {v}')
print()
print('=== Bound (lower bound estimates) ===')
for k, v in summary_bound.items():
    print(f'  {k}: {v}')
"""
        )
    )

    cells.append(
        md(
            """## 1. per-well 統計 (`per-well-stats.parquet`)

主な観測:

- **GR noise std (rolling 11 mean 残差)**: median **8.4** API, p10 6.2 / p90 11.7 → well 間で 2x 差
- **dTVT/dMD std (visible 内)**: median **0.42**, p95 1.01 → bit は 1ft 進むごとに TVT が ±0.42 ft 程度動く
- **b_well residual std**: median **0.0083 ft** ≈ 2.5mm → formula 単独の理論誤差はほぼ 0 (公開 doc の 0.007 ft と一致)
- **typewell GR resid std (TVT 軸対応点での GR 差)**: median **13.9 API** → typewell 単独で horizontal の GR を完全説明はできない (= 地理的 facies 差で 14 API のズレ)
- **AR(1) phi (dTVT/dMD の AR(1) 係数)**: median **0.999** → 実質 random walk (= 1 ft 前の dTVT は 1 ft 後とほぼ独立にはならない、強い persistence)
"""
        )
    )

    cells.append(
        code(
            """df_main = pd.read_parquet(OUT / 'per-well-stats.parquet')
print('shape:', df_main.shape)
print(df_main.describe().T[['count','mean','50%','min','max']])
"""
        )
    )

    cells.append(
        code(
            """fig, axes = plt.subplots(2, 3, figsize=(14, 8))
df_main['gr_noise_std'].dropna().hist(ax=axes[0,0], bins=40); axes[0,0].set_title('GR noise std (API)')
df_main['dtvt_std'].dropna().hist(ax=axes[0,1], bins=40); axes[0,1].set_title('dTVT/dMD std (ft/ft)')
df_main['b_well_resid_std'].dropna().hist(ax=axes[0,2], bins=40); axes[0,2].set_title('b_well resid std (ft)')
df_main['tw_gr_resid_std'].dropna().hist(ax=axes[1,0], bins=40); axes[1,0].set_title('typewell GR resid std (API)')
df_main['hidden_len'].dropna().hist(ax=axes[1,1], bins=40); axes[1,1].set_title('hidden_len (rows = ft)')
df_main['visible_ratio'].dropna().hist(ax=axes[1,2], bins=40); axes[1,2].set_title('visible_ratio')
fig.tight_layout()
plt.show()
"""
        )
    )

    cells.append(
        md(
            """## 2. extrapolation residual curve (= naive last-value broadcast の md_since 別 RMSE)

key: hidden 区間の bit が「last_known TVT のまま動かなかった」ら residual = TVT_true - last_known。これを md_since (= visible 末端からの距離) で 50ft bin して std を計算。

これは **「何もしない baseline」の理論 RMSE**。md_since の関数として grow rate を見ると **物理的 uncertainty の本質** が見える。

| md_since (ft) | std (ft) |
|---|---|
| 25 | 0.94 |
| 75 | 2.33 |
| 475 | 8.82 |
| 975 | 12.30 |
| 1975 | 15.36 |
| 3975 | 18.80 |

→ **平方根に近い grow rate** (= random walk と整合) で、4000 ft で std 18.8 ft。
全 hidden 区間 RMSE = **15.9 ft** (= 何もせずに last_known を broadcast した時の RMSE)。
"""
        )
    )

    cells.append(
        code(
            """ec = pd.read_parquet(OUT / 'extrapolation-residual-curve.parquet')
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(ec['md_since_center'], ec['std'], lw=2, label='residual std')
ax[0].plot(ec['md_since_center'], ec['p95_abs'], lw=1, alpha=0.5, label='|residual| p95')
# random-walk 基準: std ~ sqrt(md_since) * dTVT_std
md = ec['md_since_center'].to_numpy()
dtvt_std = float(df_main['dtvt_std'].median())
rw = dtvt_std * np.sqrt(np.maximum(md, 1.0))
ax[0].plot(md, rw, '--', alpha=0.7, label=f'random-walk: dTVT_std={dtvt_std:.2f} sqrt(md)')
ax[0].set_xlabel('md_since (ft)'); ax[0].set_ylabel('std (ft)')
ax[0].set_title('Extrapolation residual std vs md_since')
ax[0].legend(); ax[0].grid(alpha=0.3); ax[0].set_xlim(0, 5000)

ax[1].plot(ec['md_since_center'], ec['n'])
ax[1].set_xlabel('md_since (ft)'); ax[1].set_ylabel('n (sample count per 50ft bin)')
ax[1].set_title('Sample density vs md_since (生存バイアス)')
ax[1].grid(alpha=0.3); ax[1].set_xlim(0, 5000)
fig.tight_layout()
plt.show()
"""
        )
    )

    cells.append(
        md(
            """## 3. 理論 RMSE 下限見積もり (3 仮説)

### 仮説 A — Formula Oracle (= ANCC が完全観測)

`TVT = -Z + ANCC + b_well`, b_well を visible 区間 median で推定。train で ANCC が完璧なら、hidden 区間の TVT 予測 RMSE = **per-well median 0.0055 ft / 全体 ≈ 0.006 ft**。

これは **train 上の理論的下限**。test では ANCC が NaN なので Plane-fit imputer を入れる必要があり、imputer 単独の RMSE は公開 doc (konbu17 は 17 ft / R kernel は ~11 ft) → **test 実際の formula 単独 RMSE は 11-17 ft**。

### 仮説 B — GR Cramér-Rao (1 点 GR 観測の Fisher Information)

bit が 1 点 GR 観測 → typewell GR(TVT) との比較で TVT を推定。CR lower bound = `noise_std / |dGR/dTVT|`.

- GR noise std: median **8.77 API**
- |dGR/dTVT| median: **1.41 API/ft**
- → 1 点 CR lower bound = **6.4 ft (median)**, p10 4.1 / p90 8.8

ただしこれは **連続観測時 1/sqrt(N) で縮む**。例: 100 ft の連続観測なら 6.4/sqrt(100) = 0.64 ft が下限。

### 仮説 C — Synthetic GR matching (visible 末尾 200 ft holdout)

visible 末尾 200 ft を hidden 模擬。残り visible で affine cal → typewell GR(TVT) と grid search matching → RMSE = **median 301 ft**。

これは **multi-modal posterior が悪さする例**: 1 点 argmin だと地層内 GR pattern の repeating で完全に外す。**Beam Search/PF の temporal coherence が必要**な根拠。

公開 top kernel が LB **10 ft 帯** に居るのは、仮説 A (formula 11-17 ft) を base に、仮説 B (連続 GR Bayesian fusion = Beam/PF) で **TVT 制約を 1 ft 以下** にし、仮説 C の 300 ft を回避している証拠。
"""
        )
    )

    cells.append(
        code(
            """df_bound = pd.read_parquet(OUT / 'per-well-bound.parquet')
print('shape:', df_bound.shape)
print(df_bound.describe().T[['count','mean','50%','min','max']])

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
df_bound['gr_cr_lb_1pt'].dropna().hist(ax=axes[0], bins=40)
axes[0].set_title('GR CR LB 1-point (ft)')
df_bound['formula_oracle_rmse'].dropna().clip(upper=0.05).hist(ax=axes[1], bins=40)
axes[1].set_title('Formula oracle RMSE (ft, train)')
df_bound['gr_match_synth_rmse'].dropna().clip(upper=600).hist(ax=axes[2], bins=40)
axes[2].set_title('Synthetic GR matching RMSE (1-pt argmin)')
fig.tight_layout()
plt.show()
"""
        )
    )

    cells.append(
        md(
            """## 4. 数値的下限の総合まとめ — `docs/research/first-principles.dense.md` 参照

| 階層 | 下限 (RMSE ft) | 出典 |
|---|---|---|
| 物理的 floor (Plane-fit ANCC + b_well median) | **~11-17** | konbu17/Roman kernel report |
| Beam + PF + Self-NCC (公開 top 実績) | **~10.0-10.8** | needless090 LB 10.081 / Roman ~10.1 |
| 1 位 (= 我々狙い) | **~7-9** ? | 推測 (private LB 通用するか別問題) |
| GR 100ft 区間連続観測の Fisher info 下限 | **~0.64** | 仮説 B |
| ANCC が完全観測の formula RMSE | **~0.006** | 仮説 A |
| Naive last-known broadcast | **~15.9** | extrapolation curve |
| 1 点 argmin GR matching (= temporal coherence なし) | **~301** | 仮説 C |

→ 公開 top 10.0 帯は **plane-fit imputer の 11-17 ft 帯のすぐ下** で停滞している。**imputer の精度を直接改善できれば 1 ft オーダーで LB が動く可能性大**。これが我々の主戦場の 1 つ。
"""
        )
    )

    cells.append(
        md(
            """## 5. 後続作業

- 各計測値の物理的解釈と確率モデル定式化 → `docs/research/first-principles.dense.md`
- 公開 top が捕捉していない情報源の同定 → `docs/research/independent-edges.dense.md`
- 戦略 doc 案 A/B/C の批判的再評価 → `docs/research/strategy-critique.dense.md`
"""
        )
    )

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NB_PATH, "w") as f:
        nbf.write(nb, f)
    print(f"wrote: {NB_PATH}")


if __name__ == "__main__":
    main()
