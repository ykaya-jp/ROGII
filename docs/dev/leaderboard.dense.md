# ROGII LB Tracking

> CV (TVT hidden RMSE, GroupKFold by well, 5-fold) と Kaggle public LB を **submission 毎に必ず両方記録** する。
> CV-LB diff > 0.5 ft なら overfit 警戒。

## Submission Log

| date (UTC) | sub id | exp | model | CV | public LB | private LB | rank | diff | note |
|---|---|---|---|---|---|---|---|---|---|
| 2026-05-10 13:25 | 52515207 | exp002 | LGB residual baseline (tvt_formula + 6-formation FormationPlaneKNN) | **13.82** | **14.695** | - | 圏外 (>1000?) | **+0.875** ⚠️ | kernel: `ky7240/rogii-exp002-lgb-residual-tvt-formula` v3。CV-LB diff +0.875 ft で **悪化方向**、公開 baseline 12.602 にも届いていない |
| 2026-05-10 15:58 | 52519856 | exp003 | LGB tysig (xcorr_tvt + multi-scale SC + WLS b_well + Beam x5 + Self-NCC + tw_diff 33 anchor x offset + GR detrend resid) | TBD | _PENDING_ (1.5h+) | - | - | - | kernel: `ky7240/rogii-exp003-lgb-tysig` v1。Kaggle scoring queue 遅延中 |
| 2026-05-10 16:14 | 52520314 | exp005 | karnakbaev pretrained LGB3+XGB+CB blend (Apache-2.0) + live test FE + Ridge meta | (no 自前 CV) | **10.317** | - | Silver (~30-50 位) | - | kernel: `ky7240/rogii-exp005-cache-blend` v1。subagent G 構築、kernel runtime 70 秒、Approach B = karnakbaev artifact blend (LB 10.784 base) → +0.467 ft 改善で 10.317、**Gold まで +0.398** |
| 2026-05-10 16:51 | 52521223 | exp006 | exp005 + TabICL 6th base (4096 ctx, n_est=4) + Ridge 6-base re-fit (positive=True) | (no 自前 CV) | **10.503** ⚠️ | - | exp005 より下落 | - | kernel: `ky7240/rogii-exp006-tabicl-pflite` v1。subagent I 構築、Kaggle GPU 上で完走。**TabICL 投入が negative** = Ridge 6-base re-fit で blend weights 劣化 (推測: TabICL OOF が exp005 5-base より悪く、positive 制約下で fold 内 weight 配分が exp005 base 単独より劣化)。**postmortem 必要** |

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

## 9 切り戦略 (= LB 8.x 帯到達、優勝条件、2026-05-11 採択)

ユーザー指示 (2026-05-11):「9 切らないと優勝は無理」 = **public LB 9.0 を切る (= 8.x 帯)** 必要。
public LB Top 1 = 9.256 だが、private LB との order 反転リスクを見越して **8.0-8.5 帯** が安全圏。

### 現状から 8.x 帯までの距離

| 起点 | LB | Top 1 (9.256) まで | 8.5 まで | 8.0 まで |
|---|---|---|---|---|
| exp005 (現 best) | **10.317** | -1.061 | **-1.817** | **-2.317** |
| exp007 想定 (Edge Q + M) | 9.5-9.7 | -0.244 〜 +0.444 | -1.0〜-1.2 | -1.5〜-1.7 |

= **exp005 から -2.0 ft 以上の改善**が必要。これは複数の独自 edge を**順次積み上げ + final stack** が必須。

### 9 切り roadmap (= 改修 b 拡張、残 86 日)

| Phase | exp | LB 目標 | 投入 edge | 状態 |
|---|---|---|---|---|
| 3 (現) | **exp005** | **9-10** | karnakbaev artifacts blend | ✅ **LB 10.317** (Silver) |
| 3 | exp006 | 9.0-9.5 | + TabICL + Ridge 6-base re-fit | ✅ **LB 10.503** ⚠️ (悪化、postmortem 必要) |
| 3 | **exp007** | **9.5-9.7** | exp005 base + **Edge Q (typewell hash CV)** + **Edge M (visible-as-typewell)** + 自前 LGB×3+CB | subagent K 準備中 (kernel push 直前) |
| 4 | exp008 | 9.0-9.4 | + **案 D (Kalman/PF on dTVT, AR(1) MAP feature)** | 未着手 |
| 4 | exp009 | 8.7-9.0 | + **案 E (Bayesian GP for ANCC posterior + var feature)** + **Edge O (direction-aware Beam/NCC)** | 未着手 |
| 5 | exp010 | 8.4-8.7 | + **Edge N (offset well retrieval / dip continuity)** + **Edge P (pseudo-typewell linkage prior)** | 未着手 |
| 5 | exp011 | 8.2-8.5 | + **Edge R (Online TTA, +0.37 ft 実測)** | 未着手 |
| 5 | exp012 | 8.0-8.3 | + **案 G (Per-Well MoE wrapper)** + final stack 全 OOF Ridge meta | 未着手 |
| 6 | exp013 | 7.9-8.1 | + post-proc (Geology snap / dip continuity / PID smooth) + pseudo-label round | 未着手 |
| 6 | exp014-15 | - | final 2 sub: CV best vs LB best | 未着手 |

### 9 切りに必要な独自 edge stack (= 公開 top にない 8 layers)

1. **Edge Q (Typewell-aware GroupKFold)** — pseudo-typewell leak 解消、+0.05〜+0.20 ft
2. **Edge M (visible-as-typewell)** — host pptx slide 9 公式 endorsement、-0.30〜-0.60 ft
3. **案 D (Kalman/PF on dTVT)** — AR(1) auto-correlation 0.999 を陽にmodel、-0.30〜-0.60 ft
4. **案 E (Bayesian GP for ANCC)** — point estimate → posterior + variance、-0.40〜-0.80 ft
5. **Edge O (direction-aware Beam/NCC)** — host pptx slide 6-7 endorsement、-0.10〜-0.30 ft
6. **Edge N (offset well retrieval)** — host pptx slide 12-13 endorsement、-0.40〜-0.80 ft
7. **Edge P (pseudo-typewell linkage prior)** — subagent J 発見、-0.20〜-0.40 ft
8. **Edge R (Online TTA)** — 公開実測 +0.37 ft、-0.30〜-0.40 ft
9. **案 G (Per-Well MoE)** — visible_ratio gating、-0.20〜-0.50 ft
10. **Final stack + post-proc + pseudo-label** — Ridge meta + Geology snap + PID smooth + 1 round pseudo、-0.30〜-0.60 ft

合計期待値 (= 全部 hit): exp005 (10.317) - 2.55〜-5.20 ft = **LB 5.1〜7.8 帯**。
合計期待値 (= 50% hit): -1.3〜-2.6 ft = **LB 7.7〜9.0 帯**。
**Top 1 (9.256) 超え + private 安全圏 (8.0-8.5) は妥当**。

### exp006 postmortem (TabICL 投入失敗の分析)

- 想定 best 9.7-10.0、実 LB 10.503 = **mid case を下回り** (= exp005 base からの正味マイナス +0.186 ft)
- 仮説 1: TabICL OOF が karnakbaev 5-base より悪く、Ridge 6-base re-fit が positive 制約下で TabICL に低 weight 振っても、**他 5 base への weight 再分配で exp005 の最適 weight からズレ** (= weight constellation の劣化)
- 仮説 2: GroupKFold(5) で TabICL OOF を生成したが、karnakbaev 5-base の OOF と **fold が一致してない** = blend に inconsistent weight
- 仮説 3: TabICL の 4096 ctx + n_est=4 が full sample 学習を再現できず低品質
- **学び**: 単に base model を増やすのではなく、**Edge Q (CV fold leak 解消) を先に入れる**べき。exp007 で先に Edge Q を投入、TabICL は後段で再評価
- **対策**: exp007 では TabICL を一旦 dormant、Edge Q + Edge M + 自前 LGB×3+CB の 4-base re-fit に集中

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
| 2026-05-10 | 4 (exp002 + exp003 + exp005 + exp006) | 1 残 |
| 2026-05-11 | 0 | 5 残 (exp007 + 検証 sub に投入予定) |
