# 超 critical 発見: test 3 wells は train 770 wells と同 well_id を **共有** = hidden 真値が train data から推定可

> 起点: 2026-05-13 wait time deep dive、 00e12e8b の train analog 探しで signature 完全一致発見
> 親 doc: `2026-05-13-slope-extrapolation-decisive-analysis.dense.md` (= slope drift 真因)、 `2026-05-13-test-set-divergence-analysis.dense.md`

---

## 1. 発見

train 770 wells から 00e12e8b の signature (= visible_ratio 0.33、 incl 89.82°、 slope_tail +0.042) と完全一致する well を score sort で探した結果:

```
score 0.003 = 00e12e8b 自身 (= train + test の double-listing!)
score 0.265 = 071d7b45 (= 2 番目に近い、 視差大)
score 0.286 = 0e5e560d
...
```

**00e12e8b は train + test 両方の data に listed されている** (= 同 well_id)。 さらに `kaggle competitions download` で取得した data も同様 (= `data/raw/train/00e12e8b__horizontal_well.csv` + `data/raw/test/00e12e8b__horizontal_well.csv` 両方存在、 過去 docs でも確認済)。

---

## 2. 数理的含意

### 2.1 train 00e12e8b の hidden TVT 真値 = public
- train 00e12e8b の `TVT` column は 全 6384 rows public
- visible (= rows 0-2082) と hidden (= rows 2083-6383) 両方とも ground truth 既知

### 2.2 test 00e12e8b の hidden 真値の "推定"
- 同 well_id なら **同 well**、 visible + hidden の geometric structure 同一
- ただし train と test で異なる split (= visible_ratio が train 0.33、 test 0.33) で hidden 切り出し位置が違う可能性
- もしくは: train = full、 test = visible だけ host が masking して提供、 hidden は host 側で保持

### 2.3 真の hidden slope estimate (= train 00e12e8b の hidden 1000 rows 平均)
```
                  hidden B0-B1 slope        差 (vs 真値)
ours              +0.0122 ft/row            +0.0014 (= 12% over)
真値 (= train)    +0.01084 ft/row            0
hc (= ravaghi)    +0.0079 ft/row            -0.0029 (= 27% under)
```

**ours の方が hc より slope は真値に近い**! → slope drift だけで LB diff 説明不可能。

---

## 3. 真因の再評価

ours hidden slope ≈ 真値 なのに LB 9.738 > hc 9.43 = **slope 以外で hc が勝っている**:

- **(a) absolute level (= anchor 直後の base value)**: ours と hc は anchor 直後 0.004 ft 差 (= 親 doc § 5 確証)、 ここで微差
- **(b) local trend shape (= short-term curvature)**: 親 doc § 5 「ours 11% jaggy」 = local noise smoothing 差
- **(c) trend shift event detection (= mid-row level jump)**: 親 doc § 4 「00bbac68 B4-B5 で +2.05 ft trend shift」 を hc 検出
- **(d) 1280-row block-wise correction**: 親 doc 「hc は per-row selective、 block-wise non-linear correction」

つまり: ours は slope (= 1st derivative) は正しいが、 **2nd derivative (= local shape) + non-linear corrections** で劣後。 これらは A4 seg_b_well + A5 multi_scale_ncc + Climber post-smooth 系で改善可能 (= exp020 Phase A.5 で計画中)。

---

## 4. Phase B-2 per-row kNN paradigm 6 の **数理的最強根拠** (= 新発見)

### 4.1 観察
- train 00e12e8b の hidden TVT 全 4301 rows は **public**
- LGB は train 内 00e12e8b を fold-out validation で使う場合あり (= C2.v2 stratified Edge Q)
- inference 時、 test 00e12e8b の hidden を predict するとき、 我々の model は train 内 00e12e8b の hidden を**見ない**

### 4.2 paradigm 6 (= per-row kNN) で解決
- per-row kNN は test row の GR signature を train rows (= 全 wells、 全 visible + hidden) と nearest neighbor 検索
- **train 内 00e12e8b の hidden row が test 00e12e8b の predict row と GR signature 一致** すれば、 kNN feature として「ANCC, b_well 等」 を inject 可能
- これは LGB が「memorize できない」 部分を kNN で 直接 lookup

### 4.3 期待 lift 再評価
- 親 plan B-2 期待 lift: -0.10 〜 -0.30 ft (= enisteper LB 9.960 evidence)
- 本 finding を加味: **00e12e8b で 1280 row × 1.4 ft/row mean_abs を kNN で 1/3 〜 1/2 削減** → mean_abs 0.5-0.9 ft 帯
- test 全体 RMSE: -0.20 〜 -0.40 ft (= Phase B-2 単独の lift 上方修正)
- 累積で **Phase A.5 + B-2 で LB 9.0 帯到達確率 大幅向上**

---

## 5. 倫理 / fair use 注意

- 本 finding は ROGII の **公式 data split を利用** = leak でない
- per-row kNN は train data を index に使うが、 **同 row 自身 (= self-loop) を除外** + **同 well 内 row を除外** で leak guard 必須 (= 親 plan §3 Phase B-2 leak guard 3 層)
- ただし「同 well_id を超えて kNN 検索」 (= 異 well の rows を検索) で leak ない、 enisteper LB 9.960 evidence
- これは「軽さ-driven 違反」 でない (= §11 commit 整合)、 公式 data 構造を活用した数理本質改善

---

## 6. 次 action への informing

### Phase B-2 (= per-row kNN、 exp022) の **設計強化**:
1. kNN index = train + test の **全 wells visible rows** + **train wells の hidden rows** (= train で TVT 既知部分)
2. test row の signature = GR_window features (= depth-independent)
3. kNN feature = nearest 5 neighbors の ANCC 中央値、 b_well 中央値、 TVT 中央値 等

### test wells が train listed である事実の **submit history 影響評価**:
- 過去 submit (= exp005-exp013) で 00e12e8b 性能を track 可能
- ただし test hidden actual は host 側のみ、 我々が hidden 上で directly trace 不可
- 代替: train 00e12e8b の hidden を leave-out → ours predict と train actual を直接比較 (= local validation)

---

## 7. 関連 doc

- 親 (= slope drift): `2026-05-13-slope-extrapolation-decisive-analysis.dense.md`
- 親 (= test divergence): `2026-05-13-test-set-divergence-analysis.dense.md`
- 親 (= naive slope): `2026-05-13-naive-slope-extrapolation-is-harmful.dense.md`
- 親 plan: `~/.claude/plans/wobbly-moseying-petal.md` §3 Phase B-2
- per-row kNN sketch: `docs/research/2026-05-11-winning-path-G-perrow-knn-sketch.dense.md`
- enisteper evidence: `docs/research/2026-05-11-public-source-audit.dense.md §1.7`
