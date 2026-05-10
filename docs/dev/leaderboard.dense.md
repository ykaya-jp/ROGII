# ROGII LB Tracking

> CV (TVT hidden RMSE, GroupKFold by well, 5-fold) と Kaggle public LB を **submission 毎に必ず両方記録** する。
> CV-LB diff > 0.5 ft なら overfit 警戒。

## Submission Log

| date (UTC) | sub id | exp | model | CV | public LB | private LB | rank | diff | note |
|---|---|---|---|---|---|---|---|---|---|
| 2026-05-10 13:25 | 52515207 | exp002 | LGB residual baseline (tvt_formula + 6-formation FormationPlaneKNN) | **13.82** | **14.695** | - | 圏外 (>1000?) | **+0.875** ⚠️ | kernel: `ky7240/rogii-exp002-lgb-residual-tvt-formula` v3。CV-LB diff +0.875 ft で **悪化方向**、公開 baseline 12.602 にも届いていない |
| 2026-05-10 15:58 | 52519856 | exp003 | LGB tysig (xcorr_tvt + multi-scale SC + WLS b_well + Beam x5 + Self-NCC + tw_diff 33 anchor x offset + GR detrend resid) | TBD | _PENDING_ (1.5h+) | - | - | - | kernel: `ky7240/rogii-exp003-lgb-tysig` v1。Kaggle scoring queue 遅延中 |
| 2026-05-10 16:14 | 52520314 | exp005 | karnakbaev pretrained LGB3+XGB+CB blend (Apache-2.0) + live test FE + Ridge meta | TBD | _PENDING_ (~30 min) | - | - | - | kernel: `ky7240/rogii-exp005-cache-blend` v1。subagent G 構築、kernel runtime 70 秒、Approach B = karnakbaev artifact blend (LB 10.784 base) |

## 主要 LB ベンチマーク (2026-05-11 取得 = 公開 LB Gold cutoff の実態)

公開 leaderboard top 20 (`kaggle competitions leaderboard --show` 確認):

| rank | team | submitDate | public LB |
|---|---|---|---|
| 1 | Virtute | 2026-05-10 11:03 | **9.256** |
| 2 | Silogram | 2026-05-09 19:53 | 9.301 |
| 3 | Mr.キノコ | 2026-05-10 08:45 | 9.346 |
| 4 | anshul3501 | 2026-05-10 06:37 | 9.415 |
| 5 | zihao qi | 2026-05-10 14:55 | 9.575 |
| 6 | Chris Deotte | 2026-05-09 22:15 | 9.576 |
| 7 | Takahiro Saito | 2026-05-10 02:43 | 9.584 |
| 8 | Samith Chimminiyan | 2026-05-10 13:00 | 9.718 |
| 9 | Ayodeji | 2026-05-10 12:33 | 9.728 |
| 10 | Proteek Chaudhuri | 2026-05-10 09:43 | 9.729 |
| 11 | Descendants of General Giap | 2026-05-10 12:49 | 9.803 |
| 12 | theredbluepill | 2026-05-10 06:13 | 9.810 |
| 13 | Praxel | 2026-05-09 21:02 | 9.824 |
| 14 | Pavlo Ivanin | 2026-05-10 13:30 | 9.845 |
| 15 | Андрей Четверяков | 2026-05-10 10:37 | 9.848 |
| 16 | Kazuki Harada | 2026-05-08 13:12 | 9.865 |
| 17 | Mudit Chaturvedi | 2026-05-09 17:31 | 9.892 |
| 18 | Scott Willis | 2026-05-10 05:58 | 9.897 |
| 19 | Kevin E R MILLE | 2026-05-10 14:55 | 9.914 |
| 20 | spbforce | 2026-05-10 08:41 | **9.919** |

> **Gold 圏 cutoff (= Top 20)**: LB **9.919** (推定)。Top 1: **9.256**。
> 賞金圏 (1-4 位) cutoff: **9.415**。
> 我々の現状 LB 14.695 から Gold 圏まで **-4.776 ft**、Top 4 賞金圏まで **-5.280 ft**。
> 713 teams 中 Top 20 はあと 86 日でも到達可能、Top 4 は独自 edge 必須。

## 公開 kernel ベンチマーク (主要)

| 帯 | LB | 著者 / kernel | 備考 |
|---|---|---|---|
| Top 3 (推定) | ~10.1 | romantamrazov `rogii-super-solution-lb-top-3` | Numba PF + Beam x7 + plane-fit + Self-NCC + LGB x3 + CB |
| - | 10.081 | needless090 `score-10-081-score-lb-32-rank` | + TabICL + 8-base stack |
| - | 10.784 | karnakbaev `physics-informed-baseline` (Top 2 Rank claim) | physics formula + 5-base stack、artifacts 公開 = `karnakbaevarthur/rogii-code-helper-dataset` |
| - | 11.068 | tasmim `lb-11-068-rogii-wellbore-geology-prediction` | xcorr_tvt features (= 我々の exp003 でも使用) |
| - | 11.912 | konbu17 `rogii-plane-fit-formation-top-knn` | plane-fit + KNN |
| - | 12.388 | shinyanagai123 `12-388-a-better-baseline-particle-filter` | Beam + Particle Filter |
| 公開 baseline | 12.602 | romantamrazov `rogii-super-baseline-lb` | LGB 系 baseline |
| 上位 vote | ? | `pilkwang/rogii-eda-v4-same-matrix-super-stack` (267KB, 80 votes, 5/10 update, GPU enabled) | **Same-Matrix Super Stack**、9 帯候補。`docs/research/pilkwang-distill.dense.md` 参照 |
| 上位 vote #2 | ? | `nina2025/rogii-h-blend-v2` (280KB, 39 votes, GPU) | h-blend |
| 上位 vote #3 | ? | `ravaghi/wellbore-geology-prediction-lightgbm` (105 votes 2 位、GPU) | reference 多数: romantamrazov + karnakbaev + shinyanagai + konbu17 + vishwasmishra + cdeotte |
| - | ? | `svanikkolli/beamstack-255-engine` (34 votes, GPU) | BeamStack-255 |
| NN starter | CV 15.5 | cdeotte `nn-starter-cv-15-5` | NN 例 |

## 自前 exp 目標 LB (= 改修 b A+D+E+G の roadmap)

> 改修 b 採択 (`docs/dev/schedule-2026-05-10.dense.md` §0)。Public LB Top 20 = LB 9.919、Top 4 = LB 9.415、Top 1 = LB 9.256。

| Phase | exp | LB 目標 | 仕掛け | 状態 |
|---|---|---|---|---|
| 2 | exp002 | 12.x | tvt_formula + LGB baseline | ✅ LB **14.695** (悪化、CV-LB diff +0.875) |
| 2 | exp003 | 11.x | + xcorr_tvt + wls_b_well + multi-scale SC + gr_detrend_resid | 🟡 PENDING |
| 3 | **exp005** | **9-10** | **karnakbaev artifacts blend** (LB 10.784 base + live test FE + Ridge meta) | 🟡 PENDING |
| 3 | **exp006** | **9.0-9.5** | exp005 + **TabICL** + **PF-lite** (pilkwang stateless ensemble) | subagent I 準備中 |
| 4 | exp007 | 9.0-9.3 | + 案 D (Kalman/PF on dTVT, AR(1) MAP feature) | 未着手 |
| 4 | exp008 | 8.7-9.0 | + 案 E (Bayesian GP for ANCC posterior + var feature) | 未着手 |
| 5 | exp009 | 8.5-8.8 | + 案 G (Per-Well MoE wrapper) | 未着手 |
| 5 | exp010 | **8.0-8.5 (1 位射程)** | + final stack A+D+E+G OOF Ridge + post-proc | 未着手 |
| 6 | exp011 | 7.8 | + pseudo-label round | 未着手 |
| 6 | exp012-13 | - | final 2 sub | 未着手 |

## CV-LB Diff 監視

CV と public LB の diff を毎回算出。

| exp | CV | LB | diff | 解釈 |
|---|---|---|---|---|
| exp002 | 13.82 | **14.695** | **+0.875** | **CV-LB +0.875 ft 悪化方向**。原因仮説: (1) GroupKFold by well の visible/hidden 分割が test の hidden パターンを過小推定、(2) tvt_formula の 6-formation FormationPlaneKNN の test 側 imputation が train より弱い、(3) hidden zone 末端の systematic bias (= `first-principles.dense.md` §2.4 でのMD 9000+ で mean -16) を捕捉できていない |
| exp003 | TBD | _PENDING_ | TBD | 期待: tysig features で improve、CV-LB diff < 0.5 を願う |
| exp005 | (no 自前 CV) | _PENDING_ | TBD | karnakbaev artifacts は OOF 既知 (= 約 10.78)、自前 CV 取らずに test pred 直接生成 |

> CV-LB diff > 0.5 ft → overfit 警戒。Sub 確定時 (Phase 6) は CV best と LB best の **両方** を Sub 1/2 に選択。
> 今回 exp002 の diff +0.875 は **CV 戦略の見直し signal**。`src/rogii/cv.py` を typewell content-hash groups で再設計 (subagent G が `1bb6caf feat(cv): typewell content-hash groups + data-spec TVT definition` で着手済)。

## Submit quota 管理

Kaggle 規定: **5 submissions per day** (UTC reset 00:00)。

| date (UTC) | submissions used | 残 |
|---|---|---|
| 2026-05-10 | 3 (exp002 + exp003 + exp005) | 2 残 |
| 2026-05-11 | 0 | 5 残 |
