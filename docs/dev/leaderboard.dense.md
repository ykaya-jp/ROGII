# ROGII LB Tracking

> CV (TVT hidden RMSE, GroupKFold by well, 5-fold) と Kaggle public LB を **submission 毎に必ず両方記録** する。
> CV-LB diff > 0.5 ft なら overfit 警戒。

## Submission Log

| date (UTC) | exp | model | CV | public LB | rank | diff | note |
|---|---|---|---|---|---|---|---|
| 2026-05-10 | exp002 | LGB residual baseline (tvt_formula + 6-formation FormationPlaneKNN) | **13.82** | _RUNNING_ | - | - | first sub. kernel: `ky7240/rogii-exp002-lgb-residual-tvt-formula` v3 |

## 主要 LB ベンチマーク (2026-05-10 時点)

| 順位帯 | LB | 著者 / kernel |
|---|---|---|
| Top 3 | ~10.1 | romantamrazov `rogii-super-solution-lb-top-3` |
| Top 32 | 10.081 | needless090 `score-10-081-score-lb-32-rank` (Beam×5 + PF×2 + Self-NCC + LGB×3 + CB×3 + TabICL + Ridge) |
| - | 10.784 | karnakbaev `physics-informed-baseline` |
| - | 11.068 | tasmim |
| - | 11.912 | konbu17 |
| - | 12.388 | shinyanagai123 |
| 公開 baseline | 12.602 | romantamrazov `rogii-super-baseline-lb` (LGB 系) |
| NN starter | CV 15.5 | cdeotte `nn-starter-cv-15-5` |

## 自前 exp 目標 LB

| Phase | exp | LB 目標 | 仕掛け |
|---|---|---|---|
| 2 | exp002 | 12.x | tvt_formula + LGB baseline |
| 2 | exp003 | 11.x | + xcorr_tvt + wls_b_well + multi-scale SC + gr_detrend_resid |
| 2 | exp004 | 10.5-11 | + Beam Search × 5 configs + 6-formation plane-fit residual |
| 3 | exp005 (blend) | **10.3-10.5** | exp004 × karnakbaev artifact blend (Ridge meta) |
| 3 | exp006 | 10.1 | + Self-NCC + Numba PF + TabICL stack |
| 4 | exp007 (DL) | 9.x | PatchTST + Cross-Attn + 15-seed × 5-fold MEDIAN |
| 5 | exp008 (final stack) | **8.x (1 位射程)** | A+B+C+karnakbaev OOF + TabICL OOF + post-proc |

## CV-LB Diff 監視

CV と public LB の diff を毎回算出。

| exp | CV | LB | diff | 解釈 |
|---|---|---|---|---|
| exp002 | 13.82 | TBD | TBD | TBD |

> diff > 0.5 ft → overfit 警戒。Sub 確定時 (Phase 6) は CV best と LB best の **両方** を Sub 1/2 に選択。
