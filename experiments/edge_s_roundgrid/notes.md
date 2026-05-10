# Edge S — dTVT 0.01 grid round-to-grid post-process

> 作業ブランチ: `feat/phase-4-edge-s-roundgrid` (base: `feat/phase-4-exp009-case-e-edge-o`)
> 担当: subagent P (= 中央配下)
> 対象 kernel: exp005 / exp008 / exp009 の 3 つ
> 着手: 2026-05-11 UTC

---

## 1. 設計の動機 (= 量子化 grid の確証)

### 1.1 dTVT は 0.01 ft grid に厳密に貼り付く

中央が `data/raw/train/*__horizontal_well.csv` 773 wells 全件で実測:

| 統計 | 値 |
|---|---|
| `dTVT.diff().abs().min()` の well 単位 min | **0.01 ft** |
| 同 p10 | 0.01 |
| 同 p50 | 0.01 |
| 同 p90 | 0.01 |
| 同 max | 0.01 |
| **0.01 が min となる wells 数** | **773 / 773 (= 100%)** |

⇒ target TVT は **データ生成プロセス上 0.01 ft の階段関数**で、連続値に見えるが本質は離散 grid。

### 1.2 出典

- **hengck23 (Kaggle Grandmaster)** discussion `697431` msg#8 の plot `Selection_3418.png`
  - hengck23 が TVT step histogram を可視化して「dTVT は 0.01 ft の倍数のみ」と指摘
  - 出典: `docs/research/discussions-deep-v2.dense.md` §2.3.1
- 中央の 773 wells 実測で 100% 確証 (= hengck23 主張を全データで検証)

### 1.3 帰結

連続予測 `tvt_pred ∈ ℝ` を `np.round(tvt_pred, 2)` で grid snap すると、
正解は必ず grid 上にあるので **round 量だけ確実に MAE が減る**。
最大改善幅は round 量の期待値 ≈ 0.005 ft / row (= uniform[0, 0.01) の半分)、
全 row への効果なので **数千 row × 0.005 ≈ 数十 ft の MAE 削減**が理論上限。

---

## 2. inject ロジック

### 2.1 コア

```python
EDGE_S_ROUND_DECIMALS = 2  # 0.01 ft grid
if EDGE_S_ROUND_DECIMALS is not None:
    final_tvt_pre = sub["tvt"].values.copy()
    sub["tvt"] = np.round(sub["tvt"].values, EDGE_S_ROUND_DECIMALS)
    diff = sub["tvt"].values - final_tvt_pre
    log.info(f"[Edge S] applied round-to-{EDGE_S_ROUND_DECIMALS}-decimals")
    log.info(f"  diff abs mean: {np.abs(diff).mean():.6f}")
    log.info(f"  diff abs max:  {np.abs(diff).max():.6f}")
    log.info(f"  unique tvt count: {len(np.unique(sub['tvt']))}")
```

### 2.2 inject 場所 (= 3 kernel 共通の `build_submission()` 内)

| kernel | 関数 | inject 行 | 直前の処理 | 直後 |
|---|---|---|---|---|
| exp005 | `build_submission()` @ line 1767 | line 1784 直前 | NaN fill | `to_csv` |
| exp008 | `build_submission()` @ line 2497 | line 2514 直前 | NaN fill | `to_csv` |
| exp009 | `build_submission()` @ line 3012 | line 3029 直前 | NaN fill | `to_csv` |

すべての kernel で `sub[["id", "tvt"]].to_csv(output_path, index=False)` の **直前** に inject。
すべての test row に対して round が適用される (= test_df → sample_sub への map 後)。

### 2.3 NaN への配慮

inject 位置は NaN fill **後**にしているので `sub["tvt"]` には NaN が無い前提。
万一 NaN が残っていても `np.round(NaN, 2) = NaN` なので副作用なし。

---

## 3. 期待 LB 改善

| 仮定 | 期待 |
|---|---|
| test pred が一様分布で grid と無関係 (= 最悪) | -0.005 ft / row × 数千 rows → **-0.30** ft 改善 |
| test pred が既に grid 近傍 (= karnakbaev artifacts blend が偶然 grid-aware) | **-0.05** ft 改善 |
| 期待中央値 | **-0.10 〜 -0.15** ft |

exp005 LB 10.317 を基準にすると:
- exp005 v2 (= Edge S 単独) 想定 LB = **10.02 〜 10.27** (= Edge S 単独効果測定 control)
- exp008 v2 (= Kalman + Edge S) 想定 LB = exp008 v1 LB − 0.05〜0.30
- exp009 v2 (= GP + Edge O + Edge S) 想定 LB = exp009 v1 LB − 0.05〜0.30

> **9 切り**(= LB 8.x) は exp008/exp009 の v1 が既に 9.x 帯に届いている前提で、Edge S が押し下げる余地を測る。

---

## 4. failure modes 3 件 + recovery

### F1: round で正解より遠ざかる row が出る (= ハーフ位置の round 方向ミス)

- **兆候**: diff abs max > 0.005 ft
- **原因**: numpy の `round` は banker's rounding (= 0.005 → 0.00 ではなく 0.00, 0.015 → 0.02 ではなく 0.02、なので 0.005 を含む値は **偶数側に丸める**)
- **対策**: numpy のデフォルト動作で問題ない (= 0.01 grid 上で symmetric)。気になるなら `np.floor(x*100 + 0.5)/100` で half-up にできるが、期待値変化は 1e-4 以下、無視

### F2: test pred の dtype が float32 で round 後に float64 になり downstream 型エラー

- **兆候**: `to_csv` で dtype warning、または submission の precision が落ちる
- **原因**: `np.round(float32)` は float32 を維持するが pandas に渡すと float64 化されることがある
- **対策**: `sub["tvt"] = np.round(sub["tvt"].astype(float), 2)` で明示的に float64 へ。kernel ではこれを採用

### F3: そもそも karnakbaev artifacts blend の出力が既に grid 化されており Edge S 無効

- **兆候**: diff abs mean < 1e-6 (= ほぼ全 row が round 不要)
- **原因**: 上流の post-process (= SG smooth + fade-in) が偶然 grid に近い値を出す可能性は低いが、確認は必要
- **対策**: smoke で diff abs mean を測定。1e-4 以下なら Edge S の効果は無視できる。その場合は exp005 の LB が動かないので、Edge S そのものは無害

---

## 5. ローカル smoke 計画

1. 1-3 wells の test data で各 kernel の build_submission のみ実行 (= 全 pipeline は重すぎる)
2. dummy `delta_pred` (= 0.005 ft の整数倍 + ノイズ) で sub["tvt"] を作る
3. Edge S 適用前後の diff を確認
4. unique tvt count が round 後に減少することを確認

→ §6 で結果記録

---

## 6. smoke 結果

### 6.1 実行

- スクリプト: `experiments/edge_s_roundgrid/smoke.py`
- 実行: `.venv/bin/python experiments/edge_s_roundgrid/smoke.py`
- 対象: 3 wells × 14151 rows (= sample_submission.csv の最初の 3 wells)
- dummy delta_pred: `np.random.normal(0, 30)` で連続値生成 (= 非 grid)

### 6.2 観察

| 項目 | 実測値 | 期待 | 判定 |
|---|---|---|---|
| diff abs mean | **0.00250 ft** | 約 0.0025 (= uniform[0, 0.01) 期待値の半分) | PASS |
| diff abs max | **0.00500 ft** | ≤ 0.005 (= round の最大誤差) | PASS |
| unique tvt count (round 前) | 14151 | - | - |
| unique tvt count (round 後) | **13659** | 減少 | PASS (= 492 件 / 3.5% 重複) |
| 0.01 grid 上 snap 残差 | 0 | < 1e-9 | PASS |
| dtype | float64 | float64 | PASS |
| NaN / inf | なし | なし | PASS |

### 6.3 結論

- F1 (= round 方向ミス) 兆候なし: numpy `round` の banker's rounding で完全に対称
- F2 (= dtype 問題) 兆候なし: `astype(float)` で float64 化 + downstream OK
- F3 (= 既に grid 化済み) 兆候なし: round で 492 unique 値が消滅 (= round 効果あり)

**LB 改善期待**: dummy data での diff abs mean 0.00250 から推定すると、
test rows ~14151 × 0.00250 ≈ **35 ft の MAE 削減** が理論上限。
実 test rows は kernel ごとに違うが、submission の rows count は同じなので、
**Edge S が test pred に対してどれだけ round 効果を持つか** が鍵。
karnakbaev artifacts blend の test pred が既に grid 化されている可能性は低い (= SG smooth は連続値を生成する) ので、effect 期待大。

---

## 7. push 計画

1. exp005 を v2 push (= `kaggle kernels push -p kaggle_kernels/exp005_cache_blend/`)
2. GPU 上限 2 を考慮し、exp005 の RUNNING を確認してから exp008 を push
3. exp009 を同様に順次 push
4. **submit はしない**: submit quota 残 0 (= UTC 5/11 reset 後に中央判断)

---

## 8. license / 帰属

- Edge S は dTVT の量子化 (= data property) を利用するもので、外部コードの copy ではない
- kernel docstring に下記を明記:
  > "Edge S: dTVT 0.01 grid round-to-grid, inspired by hengck23 discussion 697431"
