"""exp019 = Colab で生成した ravaghi LB 9.43 submission.csv をそのまま提出する pass-through kernel.

Plan §8.2 / §10 Track 2.1-2.4 の最終 step。
Colab で ravaghi NB を A100 GPU run → submission.csv 生成 →
Kaggle dataset `ky7240/rogii-ravaghi-colab-output` に upload →
本 kernel が dataset から submission.csv を読んで output に出すだけ → submit。

Expected LB: 9.43 (= ravaghi original 同等) → defense ライン確実獲得
真因切分: もし LB 9.5 以下なら fork 再現成功、 exp017 LB 10.030 の真因は env 差。
             もし LB > 9.7 なら fork 再現失敗、 真因は paradigm 自体の non-determinism。
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

DATASET_ROOT = Path("/kaggle/input/rogii-ravaghi-colab-output")
candidates = [
    DATASET_ROOT / "submission.csv",
    Path("/kaggle/input") / "rogii-ravaghi-colab-output" / "submission.csv",
]

src = None
for c in candidates:
    if c.exists():
        src = c
        break
if src is None:
    raise FileNotFoundError(
        f"submission.csv not found. Upload Colab output as Kaggle dataset "
        f"`ky7240/rogii-ravaghi-colab-output` (see notebooks/2026-05-13-ravaghi-colab-rerun.README.md)."
    )

print(f"Loading: {src}")
sub = pd.read_csv(src)
print(f"Rows: {len(sub)}, columns: {list(sub.columns)}")
assert len(sub) == 14151, f"expected 14151 rows, got {len(sub)}"
assert set(sub.columns) == {"id", "tvt"}, f"unexpected cols: {sub.columns.tolist()}"

# Pass-through
sub.to_csv("submission.csv", index=False)
print(f"submission.csv written ({len(sub)} rows)")
print(sub.head())
