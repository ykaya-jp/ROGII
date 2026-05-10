# Edge R — Test-Time Online Learning 設計書 (Phase 5)

## 0. 目的

`feat/phase-5-roi3-v3` を base に、 **test wells の visible region (= PS 前、 `TVT_input` notna)** を擬似教師として
LGB pretrained model に continued training (= `init_model`) をかけ、
test 分布へ self-supervised に適応させる。
exp005 v2 (10.317) → **exp005 v3 9.95** で Gold cutoff 9.919 を割る保険を作り、
exp009 v3 (8.2-8.5 想定) → **exp009 v4 7.8-8.1 帯** で 9 切り達成 + Top 1 圏 (= $25K) を狙う。

## 1. 出典の論拠

| doc | section | claim |
|---|---|---|
| Kaggle discussion topic 698002 | 全文 | online 10.953 vs no-online 11.323 = **-0.370 ft** 改善 (同 setup 比較) |
| `docs/research/kaggle-deepdive.dense.md` | §Edge R (subagent J 出力) | 公開 LB 実測差分。 host 禁止アナウンスなし |
| Gal & Ghahramani 2016 | Dropout as Bayesian approximation | test-time inference の理論基盤 (= 補強) |
| LightGBM `lgb.train` API | `init_model` | continued training (warm start) の正規 API |

pane 2 評価: 工数 1-2 日、 Gold 確率 65% → 85% (= 単独最高 ROI)

## 2. 数理

### 2.1 問題定式化

test well $w$ の visible 行集合を $\mathcal{V}_w = \{i : \text{TVT\_input}_i \text{ is not NaN}\}$、
hidden 行集合を $\mathcal{H}_w = \{i : \text{TVT\_input}_i \text{ is NaN}\}$ とする。
ground truth の隠匿は **hidden** 側にのみ存在し、visible 側の `TVT_input` は host が公開している。

target は absolute TVT ではなく **delta = TVT - last_known_tvt** (= 既存 pipeline 全部入り)。

Edge R は visible 内で擬似 split を切る:

$$
\text{kn}_w = \{i \in \mathcal{V}_w : i < i^* \},\quad
\text{ev}_w = \{i \in \mathcal{V}_w : i^* \leq i < |\mathcal{V}_w|\}
$$

ここで $i^*$ は visible の末尾 $K$ 行 (= 100 or 20%) を擬似 ev に分ける split point。
擬似 ev の target は visible TVT_input そのものから:

$$
y^{\text{R}}_i = \text{TVT\_input}_i - \text{TVT\_input}_{i^* - 1},\quad i \in \text{ev}_w
$$

これで Edge R は **既存 build_well_features を visible-only mode で 1 回叩く** だけで feature + target が揃う (= 実装最小)。

### 2.2 continued training

pretrained LGB booster $M_{\text{pre}}$ (= karnakbaev 5 base) を $\theta^{(0)}$ とし、
test visible-pooled $(X^{\text{R}}, y^{\text{R}})$ で追加 boosting:

$$
M_{\text{online}} = \text{lgb.train}(\text{params}, \text{Dataset}(X^{\text{R}}, y^{\text{R}}),
\text{num\_boost\_round}=R, \text{init\_model}=M_{\text{pre}})
$$

ここで $R = 100\text{-}300$ (= topic 698002 から)。
予測時は $M_{\text{online}}$ を test_df 全行に適用、 既存 blend に $w_R \cdot M_{\text{online}}$ で重ね合わせ。

$w_R$ は OOF grid search で決定 (= 0.3-0.7、推奨 0.5 初期値)。

### 2.3 leak 検査 4 点

1. **visible TVT_input のみ参照**: host が公開している test data に存在する列 = 使用 OK
2. **hidden TVT に絶対触れない**: ev 行は visible 内に切る、 hidden mask 域を target に使わない
3. **karnakbaev pretrained の train data と test visible は別 well**: row id レベルで disjoint 確認 (= train/test split 自体は public)
4. **擬似 split point $i^*$ は well 内のみで完結**: 他 well の情報を参照しない、 fold leak 不可

## 3. option A vs option B trade-off

### option A (= per-well online)

- 各 test well で独立 fine-tune → 776 model
- pros: well-specific 適応、 small-data overfit のリスクはあるが well-by-well の noise structure を取れる
- cons: 推論コスト = 776 × 5 base × 1-2 min ≈ **65-130 hr** = 9 hr cap **不可能**

### option B (= test-pooled online) **推奨**

- 全 test の visible を 1 pool → 各 pretrained base に 1 回 continued training → 5 base × 1 model
- pros: 推論コスト = 5 × 1-3 min = **5-15 min** = 9 hr cap 内余裕
- pros: pool 全体で sample 多 (= 全 test 視界の数十万 row) → overfit 抑制
- cons: well-specific 適応はない (= global shift のみ補正)

### 推奨 = **option B**

理由:
- 9 hr cap (= 公式 runtime 制約) を絶対に守る必要
- topic 698002 の -0.370 ft 改善は pool 系で確認されている (= option B 系)
- option A は将来 inject 余地として残す (= v5 候補)

## 4. exp005 v3 / exp009 v4 への注入位置

### 4.1 exp005 v3 = Edge S + Edge R 単独

base: `kaggle_kernels/exp005_cache_blend/exp005_cache_blend.py` (= v2、 Edge S 注入済)

注入位置:

| step | line range | 変更内容 |
|---|---|---|
| import | 1990 付近 (= `MODE == "infer"` 内、 base_models load 後) | Edge R 用 helper を呼ぶ |
| feature build | `_get_or_build_test_df()` 結果を visible-mode で再生成 (= 新規 helper) | visible 行を ev として feature 化 |
| continued training | base_models load 後、 `predict_all` 前 | 5 base 各々を `init_model` で continued training |
| inference | `predict_all(models_online, ...)` で fine-tuned model を使う | 既存 NM/Ridge blend は変更なし |
| blend with online | 既存 `test_delta` (= NM blend) と online prediction を $w_R$ で重ね合わせ | `w_R = 0.5` 初期値 (= grid search は v3 では skip、 v4 で実装) |

想定 LB: **10.317 → 9.95** (= Edge S が +0.0 ft 寄与は smoke で済、 Edge R 単独で -0.37 ft 期待)

### 4.2 exp009 v4 = 既存 v3 + Edge R 重ね合わせ

base: subagent T が `feat/phase-5-roi3-v3` で push 済の
`kaggle_kernels/exp009_case_e_edge_o/exp009_case_e_edge_o.py` (= v3、 Huber + hetero + MEDIAN + path b)

注入位置:

| step | line range | 変更内容 |
|---|---|---|
| import / config | 220-240 付近 (= Phase 5 v3 multi-seed config 直後) | `EDGE_R_ENABLE`, `EDGE_R_NUM_BOOST`, `EDGE_R_BLEND_W` 追加 |
| visible feature build | `MODE == "infer_edge_e_gp_o"` 内、 build_dataset (test) 直後 (= line 3587 付近) | 新規 helper `build_visible_dataset` で test visible-as-ev を rebuild |
| continued training | karnakbaev 5 base load 後 (= line 3656 付近) | 5 base 各々を `init_model` で continued training |
| 自前 base はそのまま | 4 own base (LGB/CB MEDIAN) は触らない | own base は train data で十分に flexible、Edge R を kb base のみに適用 (= 過剰 fine-tune 回避) |
| Step F path b blend 後 | `test_delta` 算出後 (= line 4115 付近) | `test_delta_with_R = w_R × online_pred + (1 - w_R) × test_delta` |
| post-process は不変 | apply_postproc + sg_smooth_per_well + Edge S round | 既存 step G/H をそのまま使う |

想定 LB: **exp009 v3 想定 8.2-8.5 → exp009 v4 7.8-8.1 帯**

## 5. ハイパーパラメータ

| param | 値 | 根拠 |
|---|---|---|
| `EDGE_R_VISIBLE_TAIL_K` | 100 (or 20% min) | visible 内の擬似 ev サイズ。 hidden length 平均 ~110 行と同等 |
| `EDGE_R_NUM_BOOST` | 200 | topic 698002 で 100-300 推奨、 中央値採用 |
| `EDGE_R_LR_MUL` | 0.5 | continued training は元 lr × 0.5 (= 過適応抑制、 LightGBM 慣習) |
| `EDGE_R_BLEND_W` | 0.5 | OOF grid search 余裕がない exp005 v3 では 0.5 固定、 exp009 v4 で grid search 推奨 (= 余力なければ 0.5) |
| `EDGE_R_MIN_VISIBLE_ROWS` | 30 | visible 行数 30 未満の well は skip (= 教師信号不十分) |

## 6. runtime 制約

### 6.1 exp005 v3 (CPU)

| step | runtime |
|---|---|
| 既存 v2 base (Edge S 込) | 70 sec |
| Edge R: visible feature build | +30-60 sec (= 776 well × 100ms parallel) |
| Edge R: 5 base continued training × 200 round | +3-5 min (= LGB CPU、 XGB/CB は同じく warm start API あり) |
| Edge R: 5 base predict | +20 sec |
| **合計** | **約 5-7 min** = 9 hr cap 圧倒的余裕 |

### 6.2 exp009 v4 (GPU)

| step | runtime |
|---|---|
| 既存 v3 base (GP + Edge O + MEDIAN + path b) | 想定 6-8 hr |
| Edge R: visible feature build (test only) | +30-60 sec |
| Edge R: 5 kb base continued training × 200 round | +5-10 min (LGB CPU + XGB warm + CB warm) |
| Edge R: 5 base predict | +20 sec |
| **合計** | **約 8-8.5 hr** = 9 hr cap **タイト**、 timeout monitor 必須 |

### 6.3 timeout 対策

- exp009 v4 で **continued training に 15 min 超** かかった場合、 `EDGE_R_NUM_BOOST` を 200 → 100 にダウングレード (= fallback)
- `EDGE_R_ENABLE = False` の env override を入れて緊急 disable 可能化

## 7. blend 戦略

### 7.1 exp005 v3 (= v2 5-base NM blend + Edge R)

```
test_delta_pre = NM_blend(5 base original predict)   # 既存 v2
test_delta_R   = NM_blend(5 base online predict)     # Edge R
test_delta     = 0.5 × test_delta_pre + 0.5 × test_delta_R
```

### 7.2 exp009 v4 (= 既存 path b + Edge R)

```
test_delta_pre = path_b_blend(kb 5 + own 4)          # 既存 v3
test_delta_R   = NM_blend(kb 5 online predict)       # Edge R (kb 系のみ online)
test_delta     = 0.5 × test_delta_pre + 0.5 × test_delta_R
```

注: 自前 base は **Edge R を適用しない** (= train data + Edge Q fold で既に flexible)。
過剰 fine-tune による degenerate を回避。

## 8. 失敗 mode と対策

| mode | 兆候 | 対策 |
|---|---|---|
| visible 行数不足 well | feature build で None | `EDGE_R_MIN_VISIBLE_ROWS=30` 未満は skip、 train 側 visible で代替 |
| continued training の overfit | online predict が absurd value | `num_boost_round=200` 固定、 lr_mul=0.5 で抑制、 NaN/inf chk |
| Edge S と Edge R の干渉 | round-to-grid 後の 0.01 snap で online の細かい変化が消える | 既存通り Edge S は最後段で適用、 online は delta 段階で混ぜる (= 干渉なし) |
| timeout (exp009 v4) | 8.5 hr 超 | `EDGE_R_NUM_BOOST` 100 にダウングレード、 fallback ロジック内蔵 |
| visible-train gap | test の visible と train の visible で分布が違う (= covariate shift) | Edge R 本来の目的 (= test 分布適応) なので feature ではなく問題そのもの。 過大に持つと弊害だが、 $w_R = 0.5$ で抑える |
| LB regression | 9.95 想定が 10.5 に悪化 | $w_R$ を grid で再探索、 fallback `EDGE_R_BLEND_W = 0.3` |

## 9. 累積期待効果

| 改修 | CV Δ (推定) | LB Δ (推定) | 根拠 |
|---|---|---|---|
| Edge R 単独 (exp005 v3) | -0.30 | **-0.37** | topic 698002 実測 |
| Edge R + v3 改修 (exp009 v4) | -0.20 | **-0.30** | v3 既存改修と一部 overlap、 漸減則 |

想定 LB:
- exp005 v3 = **10.317 → 9.95** (Gold cutoff 9.919 を辛うじて割る)
- exp009 v4 = **8.4 → 8.05** (mid 推定、 Top 1 9.256 圏)

## 10. submit 戦略 (= 中央判断、 本 worker は実装まで)

| slot | kernel | 役割 |
|---|---|---|
| safe | exp005 v3 | Gold 圏入り保険 (= LB 9.95 想定) |
| risky | exp009 v4 | 賞金 1 位 圏 ($25K、 LB 7.8-8.1 想定) |

7-day final freeze ルールに従い、 残 86 day はまだ middle game。 即 push 推奨 (= 中央指示待ち)。

## 11. 関連 file

- 本 doc: `experiments/edge_r_online/design.md`
- 実装 notes: `experiments/edge_r_online/notes.md`, `experiments/exp005_v3/notes.md`, `experiments/exp009_v4/notes.md`
- kernel: `kaggle_kernels/exp005_cache_blend/exp005_cache_blend.py` (v3), `kaggle_kernels/exp009_case_e_edge_o/exp009_case_e_edge_o.py` (v4)
- 上位 lesson 出典: ~/.claude/CLAUDE.md "[2026-05-10] orbit-wars lesson" = host dataset を確認する習慣を持つ

## 12. 更新履歴

- 2026-05-11: 初版 (= subagent U が pane 2 提案を受けて作成)
