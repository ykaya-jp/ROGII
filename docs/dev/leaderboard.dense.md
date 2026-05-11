# ROGII LB Tracking

> CV (TVT hidden RMSE, GroupKFold by well, 5-fold) と Kaggle public LB を **submission 毎に必ず両方記録** する。
> CV-LB diff > 0.5 ft なら overfit 警戒。

## Submission Log

| date (UTC) | sub id | exp | model | CV | public LB | private LB | rank | diff | note |
|---|---|---|---|---|---|---|---|---|---|
| 2026-05-10 13:25 | 52515207 | exp002 | LGB residual baseline (tvt_formula + 6-formation FormationPlaneKNN) | **13.82** | **14.695** | - | 圏外 (>1000?) | **+0.875** ⚠️ | kernel: `ky7240/rogii-exp002-lgb-residual-tvt-formula` v3。CV-LB diff +0.875 ft で **悪化方向**、公開 baseline 12.602 にも届いていない |
| 2026-05-10 15:58 | 52519856 | exp003 | LGB tysig (xcorr_tvt + multi-scale SC + WLS b_well + Beam x5 + Self-NCC + tw_diff 33 anchor x offset + GR detrend resid) | (no 自前 CV) | **17.510** ⚠️⚠️ | - | exp002 より悪化 +2.815 | - | kernel: `ky7240/rogii-exp003-lgb-tysig` v1。**自前複雑 features 単独路線の劇的失敗** = 33 anchor x offset 等の高次 features 過剰添加で overfit 起こした可能性。CV-LB diff も大きい (= subagent A の CV 計測未完了で正確値不明、想定 12-13 → LB 17.510)。**学び: 自前 features 単独路線は捨て、karnakbaev blend ベース + 補助 features 路線に集中** |
| 2026-05-10 16:14 | 52520314 | exp005 | karnakbaev pretrained LGB3+XGB+CB blend (Apache-2.0) + live test FE + Ridge meta | (no 自前 CV) | **10.317** | - | Silver (~30-50 位) | - | kernel: `ky7240/rogii-exp005-cache-blend` v1。subagent G 構築、kernel runtime 70 秒、Approach B = karnakbaev artifact blend (LB 10.784 base) → +0.467 ft 改善で 10.317、**Gold まで +0.398** |
| 2026-05-10 16:51 | 52521223 | exp006 | exp005 + TabICL 6th base (4096 ctx, n_est=4) + Ridge 6-base re-fit (positive=True) | (no 自前 CV) | **10.503** ⚠️ | - | exp005 より下落 | - | kernel: `ky7240/rogii-exp006-tabicl-pflite` v1。subagent I 構築、Kaggle GPU 上で完走。**TabICL 投入が negative** = Ridge 6-base re-fit で blend weights 劣化 (推測: TabICL OOF が exp005 5-base より悪く、positive 制約下で fold 内 weight 配分が exp005 base 単独より劣化)。**postmortem 必要** |
| 2026-05-10 22:10 | 52526612 | exp007 | exp005 + Edge Q (typewell content-hash GroupKFold) + Edge M (visible-as-typewell self-alignment, 4 features) + 自前 LGB3+CB (Edge Q fold, 165-col features) + Ridge 9-base re-fit | **Ridge OOF 10.3874** | **10.677** ⚠️⚠️ | - | exp005 より +0.36 悪化 | **+0.29** ⚠️ | kernel: `ky7240/rogii-exp007-edge-qm` v1。Kaggle scoring 10h+ PENDING 後 SCORED。**重大な失敗 = 自前 4 base 追加 + Ridge meta が逆効果**。原因仮説: (1) fold-misalign 影響量 +0.12 想定 → 実 +0.29 で 2.4 倍severe、(2) 自前 base OOF 10.66-10.93 = karnakbaev 10.78 と同等以下で diversity 不足、(3) `lgb_own2 weight=0` で Ridge 自体が一部除外したが効果なし。**学び**: (a) 自前 4 base 路線は **karnakbaev 真の OOF 再生成 (path a)** が必須、(b) **path b (= kb simple-avg + own Ridge separate)** で subagent T 改修済みだが、自前 base 質改善 (Multi-seed MEDIAN + Huber + hetero) も同時必要、(c) **karnakbaev base のみ + Edge S/R inject (= 自前 4 base なし) が一番 safe**、これが exp008 v2 / exp009 v2 の運命を左右 |
| 2026-05-11 00:13 | 52528539 | exp005 v2 | karnakbaev blend + **Edge S** (round-to-grid) | (no 自前 CV) | **10.203** ✅ | - | exp005 → -0.114 改善 | - | kernel: `ky7240/rogii-exp005-cache-blend` v2。subagent P 構築、Edge S 単独効果測定 control。**Edge S 単独 = -0.114 ft 改善 (= 予測 10.05-10.27 範囲内、想定 -0.05〜-0.30 の中域)**。Gold cutoff (9.919) まで -0.284 ft 不足、**Edge S 単独では Gold 不可** |
| 2026-05-11 00:32 | 52528863 | exp005 v3 | karnakbaev blend + Edge S + **Edge R** (test-time online learning, continued training on test visible TVT_input) | (no 自前 CV) | **10.387** ⚠️ | - | exp005 v2 → **+0.184 悪化** | - | kernel: `ky7240/rogii-exp005-cache-blend` v3。subagent U 構築、Edge R 単独効果測定。**🚨 想定 9.65-9.95 を大きく外れ +0.435 ft 悪化方向**。kernel log で Edge R phase 正常動作確認 (= diff abs mean 0.177、w_R 0.5、applied True)、実装 bug なし。**Edge R REFUTED 確定 = base-dependent**: 公開 698002 raw LGB tysig (LB 11.068 base) で +0.37 報告だが我々 karnakbaev pretrained 5-base (LB 10.2 base) では +0.184 worse、 posterior 上書き害。 根本仮説 3 件: (H-R1) base model 依存、(H-R2) w_R=0.5 が過大、(H-R3) augmentation 不足 (= 300 row × 3 wells)。**学び: pane 2 「最高 ROI 1 手」revoke、 online TTA 採用禁止、 5/12 plan で Edge R 除外** |
| 2026-05-11 03:15 | 52532035 | exp008 v2 | exp007 base + **案 D (Kalman/PF on dTVT, AR(1) Yule-Walker MLE + global shrinkage, 7 features)** + Edge S | TBD | **9.957** ✅ | - | **9 切り達成!! Silver ギリギリ rank 50** | - | kernel: `ky7240/rogii-exp008-case-d-kalman` v2。subagent V 構築、案 D Kalman/PF 単独効果測定。**最初の 9 帯 SCORED**、9 切り戦略 milestone 1 達成。 Gold cutoff (= Top 10 = 9.728) まで -0.229 ft、 exp008 v3 / exp009 v2 SCORED 待ち |
| 2026-05-11 05:37 | 52534803 | exp009 v2 | exp008 v2 base + **案 E (Sparse GP for ANCC posterior + variance, 24 features, sklearn Matern 3/2 ARD + K-Means inducing M=200)** + **Edge O (direction-aware Beam/NCC, 7 features)** + Edge S | TBD | _PENDING_ | - | - | - | kernel: `ky7240/rogii-exp009-gp-edgeo` v2。expect: 9.957 → **9.5-9.7 帯 (= 8 帯射程、Top 10 圏)** |
| 2026-05-11 08:31 | 52539046 | exp008 v3 | exp008 v2 + **Huber loss + heteroscedastic sample_weight + Multi-seed MEDIAN x3 seed x 5 fold + path b kb-avg + own-Ridge separate blend** | TBD | _PENDING_ | - | - | - | kernel: `ky7240/rogii-exp008-case-d-kalman` v3。expect: 9.957 → **9.5-9.8 帯 (= Gold 確実)** |
| 2026-05-11 14:42 | (= kernel push、 RUNNING) | exp010 | exp008 v3 base + **stratified Edge Q fold (= tw_gr_resid_std quartile stratify) + adversarial validation drop (= top-20% test-likely train wells を sample_weight 0.5)** | smoke: σ_fold 0.526→0.303 (-42%) | _RUNNING_ (19h+) | - | - | - | kernel: `ky7240/rogii-exp010-foldreform-v1`。 H10 = Jensen lower bound 10.77 ft 解消の fold 構造改革。 σ_fold 1.18 → 0.5 で CV 天井 10.0-10.3 ft 想定、 LB expect 9.5-9.7 |

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

> **Medal cutoff (= 777 teams、 Kaggle 250-999 公式ルール、 2026-05-11 09:50 UTC 取得)**:
> - **Gold (= Top 10)**: LB **9.728** (= Proteek Chaudhuri @ 10位)
> - **Silver (= Top 50)**: LB **9.957** (= 我々 Reexel @ rank 50、 ギリギリ Silver)
> - **Bronze (= Top 100)**: LB ≈ 10.5 (推定、 詳細は full LB CSV 参照)
> 賞金圏 (= 1-4 位) cutoff: **9.415** (= anshul3501 @ 4位)。 Top 1: **9.132** (Virtute、 5/11 00:02 更新で前日 9.256 から -0.124 改善)。
> 我々現状 9.957 (rank 50/777) から Gold (Top 10) まで **-0.229 ft**、 Top 4 賞金圏まで **-0.542 ft**、 Top 1 まで **-0.825 ft**。
> Deadline: **2026-08-05 23:59 UTC** (= 86 day 残)。
>
> **修正履歴 (2026-05-11)**: 過去記載「Gold cutoff (= Top 20) = 9.919」 は teams 数未確認 (= 当時 713 teams を想定して Top 20 = 0.1%×713 + 5 ≈ 12 で Gold と推定したが、 Kaggle 公式 medal rule は **250-999 teams で Top 10 = Gold 固定**) で誤り。 9.919 は **Silver 圏内 (= Top 20 付近)** に該当。 表内 cell 中の「Gold cutoff (9.919) まで -X.XXX」 表現は当時のままだが、 真の Gold cutoff は **9.728** (= Top 10) で再評価必要。

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
| exp003 | (no 自前 CV) | **17.510** | very large | **CRITICAL FAILURE**: 自前 LGB tysig 単独路線で features 33+ 過剰添加 → 大幅 overfit。LB 17.510 は公開 baseline (12.602) どころか exp002 baseline (14.695) より悪化 +2.815 ft。**学び**: (a) 自前複雑 features 単独路線は捨て、karnakbaev blend を母法に補助 features を加える路線に集中 (= exp005-009 路線)、(b) 自前 features を加える時は **OOF で必ず CV を測定し** karnakbaev blend と比較してから submit (= 今回 subagent A が CV 取らず submit したのが根本原因) |
| exp005 | (no 自前 CV) | 10.317 | n/a | karnakbaev artifacts blend + live test FE。karnakbaev 公表 OOF ≈ 10.78 → LB **10.317 (= -0.467 ft、LB が OOF より良い**、live test FE が effective)。**仮説**: karnakbaev 元 OOF は固定 fold、test predictions が improve したのは feature engineering 部分。自前 CV を取らずに submit したため、improve の本当の幅は不明 |
| exp006 | (no 自前 CV) | 10.503 | n/a | TabICL phase failure → karnakbaev 5-base NM blend fallback。LB 10.503 は exp005 (10.317) より +0.186 悪化、原因仮説 = Ridge re-fit が NM blend より劣化 (Ridge OOF 不在で確定不能)。**教訓**: fallback path も含めて自前 OOF を取る design に |
| **exp007** | **Ridge 9-stk OOF 10.3874** ⚠️ fold-misalign | _PENDING_ | TBD | Ridge weights: kb-side total 0.620 + 自前 total 0.380、`lgb_own2=0.000` (完全除外)、主要 `lgb2(kb)=0.224 / cb_own=0.227 / xgb(kb)=0.212`。**⚠️ fold alignment 問題**: `kb_oof_q is karnakbaev train-set predict (not true OOF), Edge Q fold とは partition 異なる`。Ridge meta input が fold-misaligned で **Ridge OOF が optimistic biased 可能性**。LB が 10.5-10.7 帯まで悪化するリスクあり |
| exp005 | (no 自前 CV) | _PENDING_ | TBD | karnakbaev artifacts は OOF 既知 (= 約 10.78)、自前 CV 取らずに test pred 直接生成 |

> CV-LB diff > 0.5 ft → overfit 警戒。Sub 確定時 (Phase 6) は CV best と LB best の **両方** を Sub 1/2 に選択。
> 今回 exp002 の diff +0.875 は **CV 戦略の見直し signal**。`src/rogii/cv.py` を typewell content-hash groups で再設計 (subagent G が `1bb6caf feat(cv): typewell content-hash groups + data-spec TVT definition` で着手済)。

## CV-LB Trend 観察 (= 2026-05-11 整理、ユーザー指摘 "CV と LB の改善傾向が合致しているかチェックすべき" への応答)

### 1. 現状の systemic 欠陥

- **exp005 / exp006** = 自前 CV を取らず blind submit (= karnakbaev published OOF のみ参照、自前 4 base なし)
- **exp003** = blind submit 大失敗 (LB 17.510)
- **exp007** = 自前 CV (Ridge OOF 10.3874) を取れたが **fold-misaligned** (= karnakbaev OOF と Edge Q fold の partition 不一致) → optimistic bias 可能性
- **exp008 v2 / exp009 v2** = Kaggle 上で v2 run 中、CV は kernel log を待つ

### 2. trend 観察 (= 改善が CV / LB で integer か)

| from | to | CV 改善 | LB 改善 | 一致? |
|---|---|---|---|---|
| exp002 baseline | exp003 自前 tysig 単独 | 不明 (CV 計測なし) | -2.815 (悪化) | n/a (CV 不明) |
| exp002 baseline | exp005 karnakbaev blend | n/a (karnakbaev OOF base 既知 10.78、自前 CV なし) | +4.378 (-14.695 → -10.317) | n/a (CV 不明) |
| exp005 | exp006 +TabICL | n/a (CV 不明) | -0.186 (悪化) | n/a (CV 不明) |
| exp005 | exp007 +Edge Q+M | Ridge OOF 10.3874 (vs karnakbaev OOF 10.78、-0.39 改善) | TBD (PENDING) | 判定不能 |

= **明確な「CV と LB の trend 一致性」を測定できていない** ⇒ **明日 reset 後の submit 前に必ず自前 CV を測定する仕組み必要**。

### 3. CV gating ルール (= submit 前必須チェック、明日適用)

submit 直前に以下を全て **PASS** することを要件化:

1. **Ridge OOF RMSE が前回 best より良い (= 低い)** — 明日の exp008 v2 は OOF ≤ 10.3874 でないと submit しない、exp009 v2 は OOF ≤ exp008 v2 OOF でないと submit しない
2. **fold alignment**: Ridge meta input の OOF が全て同一 fold partition で生成済み (= 今回 exp007 で発見した misalignment を解消した version か確認)
3. **on-grid 比率** (Edge S): Edge S inject 済 kernel は 100% on-grid を達成しているか
4. **submission.csv sanity**: id 全一致、NaN 0、tvt range 物理的に妥当 (= last_known_TVT 周辺)、unique tvt count 約 30-70% 削減 (Edge S 動作確認)

これらを **kernel log から自動抽出**するスクリプトを後段で書く案あり。

### 4. exp007 fold-misalignment の影響推定

karnakbaev published OOF は karnakbaev の元 5-fold で生成 (= 各 sample は karnakbaev's val fold で predict)。Edge Q fold とは partition が異なるため、Ridge meta input の各 row で:
- 自前 4 base OOF = Edge Q val fold (= 真の OOF)
- karnakbaev 5 base OOF = 元 karnakbaev val fold (= 真の OOF だが partition 違い)

Ridge meta は両者を **同じ row index** で fit。同じ row でも fold が異なる = 自前 base の val fold partition が karnakbaev base の **train fold** と部分的に重なる場合、karnakbaev base はその row で **train data として fit** されている → **leak**。

影響量: Ridge weights を見ると kb-side 0.620 (= dominant)。kb side が leak 込みなら Ridge OOF は **真の OOF より optimistic** で、LB が悪化する。

定量推定: Edge Q fold (5-fold) vs karnakbaev fold (5-fold) の重複率 = 1/5 = 20% → leak 影響 ~0.2 × kb weight 0.62 ≈ **+0.12 ft 程度の overestimate** → 実 LB は Ridge OOF 10.3874 + 0.12 ≈ **10.5 帯予想**。

### 5. 解消 path (= exp008 v2 / exp009 v2 で確認)

subagent L (exp008) + subagent M (exp009) は同じ karnakbaev OOF 取り扱いを継承しているため、**両者とも同じ fold-misalignment 問題を抱える**。これを解消するには:

- (a) **karnakbaev pretrained を Edge Q fold で 真の OOF 再生成** (= 各 fold の train side で karnakbaev を re-fit → val side で predict)、subagent K が "deferred to v2" と note 済み。runtime 30+ min/fold × 5 fold = 2-3 hr 追加
- (b) **karnakbaev OOF を Ridge meta から除外**、自前 4 base のみ Ridge meta、karnakbaev は別途 simple average で blend
- (c) **GroupKFold (well_id) で再 align** (= karnakbaev original fold に戻す、Edge Q 効果は捨てる)

選択軸 = (a) 計算コスト大 / (b) Edge Q 効果分離可能 / (c) Edge Q 効果を諦める。判断は exp007 LB 結果次第。

## Submit quota 管理

Kaggle 規定: **5 submissions per day** (UTC reset 00:00)。

| date (UTC) | submissions used | 残 |
|---|---|---|
| 2026-05-10 | 4 (exp002 + exp003 + exp005 + exp006) | 1 残 |
| 2026-05-11 | 0 | 5 残 (exp007 + 検証 sub に投入予定) |
